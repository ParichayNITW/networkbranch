import pyomo.environ as pyo
from pyomo.opt import SolverManagerFactory
from math import pi, log10

def solve_batch_pipeline(
    nodes, edges, pumps, peaks, demands, time_horizon_hours,
    dra_cost_per_litre, diesel_price, grid_price,
    min_velocity=0.5, max_velocity=3.0
):
    m = pyo.ConcreteModel()

    # ---- SETS ----
    m.T = pyo.RangeSet(1, time_horizon_hours)
    m.N = pyo.Set(initialize=[n['id'] for n in nodes])
    m.E = pyo.Set(initialize=[e['id'] for e in edges])
    m.PUMP = pyo.Set(initialize=[p['id'] for p in pumps])

    edge_from = {e['id']: e['from_node'] for e in edges}
    edge_to   = {e['id']: e['to_node'] for e in edges}
    pump_node = {p['id']: p['node_id'] for p in pumps}
    pump_branch_to = {p['id']: p.get('branch_to', edge_to.get(p['main_edge'], None)) for p in pumps}

    length = {e['id']: e['length_km']*1000 for e in edges}
    diameter = {e['id']: e['diameter_m'] for e in edges}
    thickness = {e['id']: e['thickness_m'] for e in edges}
    roughness = {e['id']: e.get('roughness', 0.00004) for e in edges}
    max_dr = {e['id']: e.get('max_dr', 0.0) for e in edges}

    elev = {n['id']: n['elevation'] for n in nodes}
    density = {n['id']: n['density'] for n in nodes}
    viscosity = {n['id']: n['viscosity'] for n in nodes}

    # Costs and bounds
    m.dra_cost = pyo.Param(initialize=dra_cost_per_litre)
    m.diesel_price = pyo.Param(initialize=diesel_price)
    m.grid_price = pyo.Param(initialize=grid_price)
    m.min_v = pyo.Param(initialize=min_velocity)
    m.max_v = pyo.Param(initialize=max_velocity)

    demand_nodes = set(demands.keys())
    m.demand_nodes = pyo.Set(initialize=demand_nodes)
    m.demand_vol = pyo.Param(m.demand_nodes, initialize=demands)
    edge_peaks = peaks

    # ---- VARIABLES ----
    m.flow = pyo.Var(m.E, m.T, domain=pyo.NonNegativeReals)
    def dra_bounds(m, e, t): return (0, max_dr[e])
    m.dra = pyo.Var(m.E, m.T, domain=pyo.NonNegativeReals, bounds=dra_bounds)
    m.rh = pyo.Var(m.N, m.T, domain=pyo.NonNegativeReals)

    pump_max = {p['id']: p['n_max'] for p in pumps}
    def n_bounds(m, p, t): return (0, pump_max[p])
    m.num_pumps = pyo.Var(m.PUMP, m.T, domain=pyo.NonNegativeIntegers, bounds=n_bounds)

    pump_min_rpm = {p['id']: p['min_rpm'] for p in pumps}
    pump_max_rpm = {p['id']: p['max_rpm'] for p in pumps}
    def rpm_bounds(m, p, t): return (0, pump_max_rpm[p])
    m.pump_rpm = pyo.Var(m.PUMP, m.T, domain=pyo.NonNegativeReals, bounds=rpm_bounds)
    m.pump_on = pyo.Var(m.PUMP, m.T, domain=pyo.Binary)

    # ---- CONSTRAINTS ----

    # Flow continuity at each node, each hour
    def flow_continuity_rule(m, n, t):
        inflow = sum(m.flow[e, t] for e in m.E if edge_to[e] == n)
        outflow = sum(m.flow[e, t] for e in m.E if edge_from[e] == n)
        return inflow == outflow
    m.flow_balance = pyo.Constraint(m.N, m.T, rule=flow_continuity_rule)

    # Demand fulfillment (over horizon)
    def demand_satisfaction_rule(m, n):
        inflow = sum(m.flow[e, t] for e in m.E for t in m.T if edge_to[e] == n)
        outflow = sum(m.flow[e, t] for e in m.E for t in m.T if edge_from[e] == n)
        net = inflow - outflow
        return net == m.demand_vol[n]
    m.demand_fulfillment = pyo.Constraint(m.demand_nodes, rule=demand_satisfaction_rule)

    # Pump logic: OFF => RPM and num_pumps zero
    def pump_rpm_status(m, p, t):
        return m.pump_rpm[p, t] <= m.pump_on[p, t] * pump_max_rpm[p]
    m.pump_rpm_status = pyo.Constraint(m.PUMP, m.T, rule=pump_rpm_status)
    def pump_num_on(m, p, t):
        return m.num_pumps[p, t] <= m.pump_on[p, t] * pump_max[p]
    m.pump_num_on = pyo.Constraint(m.PUMP, m.T, rule=pump_num_on)

    # Velocity limits
    def velocity_limit(m, e, t):
        d = diameter[e]
        area = pi * d**2 / 4
        v = m.flow[e, t] / 3600.0 / area
        return pyo.inequality(m.min_v, v, m.max_v)
    m.velocity_limits = pyo.Constraint(m.E, m.T, rule=velocity_limit)

    # Head balance, SDH (including peaks)
    g = 9.81
    def sdh_head_rule(m, e, t):
        from_n = edge_from[e]
        to_n = edge_to[e]
        d = diameter[e]
        rough = roughness[e]
        L = length[e]
        rho = density[from_n]
        kv = viscosity[from_n]
        Q = m.flow[e, t]
        area = pi * d**2 / 4
        v = Q / 3600.0 / area
        Re = v * d / (kv * 1e-6) if kv > 0 else 1e6
        if Re < 4000:
            f = 64 / max(Re, 1e-6)
        else:
            arg = rough/d/3.7 + 5.74/(Re**0.9)
            f = 0.25 / (log10(arg)**2) if arg > 0 else 0.015
        DR_frac = m.dra[e, t]/100.0
        loss = f * (L/d) * (v**2/(2*g)) * (1-DR_frac)
        # Peaks logic
        if e in edge_peaks:
            for pk in edge_peaks[e]:
                pk_loss = f * ((pk['location_km']*1000)/d) * (v**2/(2*g)) * (1-DR_frac)
                loss = max(loss, (pk['elevation_m'] - elev[from_n]) + pk_loss + 50)
        # Pump head
        pump_head = 0
        for p in pumps:
            if pump_node[p['id']] == from_n and pump_branch_to[p['id']] == to_n:
                a, b, c = p['A'], p['B'], p['C']
                pump_rpm = m.pump_rpm[p['id'], t]
                dol = p['max_rpm']
                H = (a*Q**2 + b*Q + c)*(pump_rpm/dol)**2 if dol > 0 else 0
                n_pump = m.num_pumps[p['id'], t]
                pump_head += H * n_pump
        return m.rh[from_n, t] + pump_head >= m.rh[to_n, t] + loss
    m.head_balance = pyo.Constraint(m.E, m.T, rule=sdh_head_rule)

    # ---- OBJECTIVE FUNCTION ----
    # Minimize total cost (power/fuel + DRA) over all hours
    def total_cost_rule(m):
        total_cost = 0
        for p in pumps:
            pid = p['id']
            node_id = pump_node[pid]
            power_type = p['power_type']
            a, b, c = p['A'], p['B'], p['C']
            P, Qc, R, S, T = p['P'], p['Q'], p['R'], p['S'], p['T']
            dol = p['max_rpm']
            sfc = p.get('sfc', 0)
            rate = p.get('grid_rate', 0)
            for t in range(1, time_horizon_hours+1):
                # Identify main edge
                edges_out = [e for e in edges if edge_from[e['id']] == node_id and edge_to[e['id']] == pump_branch_to[pid]]
                if not edges_out: continue
                eid = edges_out[0]['id']
                Q = m.flow[eid, t]
                pump_rpm = m.pump_rpm[pid, t]
                n_pump = m.num_pumps[pid, t]
                H = (a*Q**2 + b*Q + c)*(pump_rpm/dol)**2 if dol > 0 else 0
                Qe = Q * dol/pump_rpm if pump_rpm > 0 else Q
                eff = (P*Qe**4 + Qc*Qe**3 + R*Qe**2 + S*Qe + T)/100.0 if pump_rpm > 0 else 0.5
                eff = max(0.05, eff)
                rho_val = density[node_id]
                pwr_kW = (rho_val * Q * 9.81 * H * n_pump)/(3600.0 * 1000.0 * eff * 0.95)
                if power_type.lower() == "grid":
                    cost = pwr_kW * grid_price
                else:
                    fuel_per_kWh = (sfc*1.34102)/820.0
                    cost = pwr_kW * fuel_per_kWh * diesel_price
                total_cost += cost
        # Add DRA cost
        for e in edges:
            eid = e['id']
            for t in range(1, time_horizon_hours+1):
                # Assume dra (%) = ppm (for illustration; can add proper DRA curve logic)
                dra_ppm = m.dra[eid, t]
                Q = m.flow[eid, t]
                dra_vol = dra_ppm * Q * 1000.0 / 1e6  # L/hr (if ppm = mg/L and flow in m3/hr)
                dra_cost = dra_vol * dra_cost_per_litre
                total_cost += dra_cost
        return total_cost
    m.total_cost = pyo.Objective(rule=total_cost_rule, sense=pyo.minimize)

    # ---- SOLVE ----
    results = SolverManagerFactory('neos').solve(m, solver='bonmin', tee=True)
    m.solutions.load_from(results)

    # ---- EXTRACT RESULTS ----
    # All variables returned as: results_dict["flow"][(edge_id, t)] = value, etc.
    results_dict = {
        "flow": {(e, t): pyo.value(m.flow[e, t]) for e in m.E for t in m.T},
        "dra": {(e, t): pyo.value(m.dra[e, t]) for e in m.E for t in m.T},
        "residual_head": {(n, t): pyo.value(m.rh[n, t]) for n in m.N for t in m.T},
        "pump_on": {(p, t): int(pyo.value(m.pump_on[p, t])) for p in m.PUMP for t in m.T},
        "pump_rpm": {(p, t): pyo.value(m.pump_rpm[p, t]) for p in m.PUMP for t in m.T},
        "num_pumps": {(p, t): int(pyo.value(m.num_pumps[p, t])) for p in m.PUMP for t in m.T},
        "total_cost": pyo.value(m.total_cost)
    }
    return results_dict
