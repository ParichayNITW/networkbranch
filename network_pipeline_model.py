import pyomo.environ as pyo
from math import log10, pi

def solve_network_pipeline(nodes, edges, pump_curves, dra_cost, diesel_price):
    """
    nodes: list of dicts, each with all properties per node (name, elev, demand, etc.)
    edges: list of dicts, each with From, To, Length, Diameter, Wall Thickness, Roughness, Max Achievable DR
    pump_curves: dict of node_name -> (df_flow_head, df_flow_eff)
    dra_cost: DRA cost (INR/L)
    diesel_price: Fuel (INR/L)
    """

    # Build node/edge index mappings
    node_names = [n["Node Name"] for n in nodes]
    node_idx = {name: i for i, name in enumerate(node_names)}
    edge_list = [(e["From Node"], e["To Node"]) for e in edges]
    N = len(nodes)
    E = len(edges)

    model = pyo.ConcreteModel()
    model.Nodes = pyo.Set(initialize=node_names)
    model.Edges = pyo.Set(initialize=edge_list, dimen=2)

    # Per-node input data
    node_elev = {n["Node Name"]: float(n["Elevation (m)"]) for n in nodes}
    node_demand = {n["Node Name"]: float(n["Demand (m3/hr)"]) for n in nodes}
    node_rho = {n["Node Name"]: float(n["Density (kg/m3)"]) for n in nodes}
    node_KV = {n["Node Name"]: float(n["Viscosity (cSt)"]) for n in nodes}
    is_pump = {n["Node Name"]: n["Is Pump?"] == "Yes" for n in nodes}
    min_rpm = {n["Node Name"]: float(n.get("Pump Min RPM", 1000.0)) for n in nodes}
    max_rpm = {n["Node Name"]: float(n.get("Pump DOL", 1500.0)) for n in nodes}
    n_pumps = {n["Node Name"]: int(n.get("No. of Pumps", 1)) for n in nodes}

    # Per-edge input data
    edge_len = {(e["From Node"], e["To Node"]): float(e["Length (km)"]) for e in edges}
    edge_D = {(e["From Node"], e["To Node"]): float(e["Diameter (m)"]) for e in edges}
    edge_t = {(e["From Node"], e["To Node"]): float(e["Wall Thickness (m)"]) for e in edges}
    edge_rough = {(e["From Node"], e["To Node"]): float(e["Roughness (m)"]) for e in edges}
    edge_max_dr = {(e["From Node"], e["To Node"]): float(e.get("Max Achievable DR (%)", 40.0)) for e in edges}

    # Pyomo sets/params
    model.elev = pyo.Param(model.Nodes, initialize=node_elev)
    model.demand = pyo.Param(model.Nodes, initialize=node_demand)
    model.rho = pyo.Param(model.Nodes, initialize=node_rho)
    model.KV = pyo.Param(model.Nodes, initialize=node_KV)

    model.length = pyo.Param(model.Edges, initialize=edge_len)
    model.D = pyo.Param(model.Edges, initialize=edge_D)
    model.t = pyo.Param(model.Edges, initialize=edge_t)
    model.rough = pyo.Param(model.Edges, initialize=edge_rough)
    model.max_dr = pyo.Param(model.Edges, initialize=edge_max_dr)

    # Decision variables
    model.flow = pyo.Var(model.Edges, domain=pyo.NonNegativeReals)
    model.res_head = pyo.Var(model.Nodes, domain=pyo.NonNegativeReals)
    model.dra = pyo.Var(model.Edges, bounds=(0, 1), initialize=0.0)  # 0 to 1 (fractional DR)

    # For pump nodes: speed, n_pumps
    model.speed = pyo.Var(model.Nodes, domain=pyo.NonNegativeReals)
    model.nop = pyo.Var(model.Nodes, domain=pyo.NonNegativeIntegers, bounds=(0, 8))

    # Flow balance at each node
    def flow_balance_rule(m, n):
        inflow = sum(m.flow[i, j] for (i, j) in m.Edges if j == n)
        outflow = sum(m.flow[n, j] for (n2, j) in m.Edges if n2 == n)
        return inflow + m.demand[n] == outflow
    model.flow_balance = pyo.Constraint(model.Nodes, rule=flow_balance_rule)

    # Hydraulic calculations
    g = 9.81

    def head_loss_rule(m, i, j):
        Q = m.flow[i, j]
        D = m.D[i, j]
        rough = m.rough[i, j]
        L = m.length[i, j]
        t = m.t[i, j]
        rho = m.rho[i]
        kv = m.KV[i]
        DR = m.dra[i, j]
        area = pi * (D ** 2) / 4.0
        v = Q / 3600.0 / area
        Re = v * D / (kv * 1e-6)
        if Re.value < 4000:
            f = 64.0 / max(Re.value, 1e-6)
        else:
            arg = (rough / D / 3.7) + (5.74 / (max(Re.value, 1e-6) ** 0.9))
            f = 0.25 / (log10(arg) ** 2) if arg > 0 else 0.0
        # Apply DR
        DH = f * ((L * 1000.0) / D) * ((v ** 2) / (2 * g)) * (1 - DR)
        return m.res_head[i] >= m.res_head[j] + (m.elev[j] - m.elev[i]) + DH
    model.head_loss = pyo.Constraint(model.Edges, rule=head_loss_rule)

    # Pump station constraints
    for n in node_names:
        if is_pump[n]:
            model.nop[n].setub(n_pumps[n])
            model.nop[n].setlb(1)
            model.speed[n].setlb(min_rpm[n])
            model.speed[n].setub(max_rpm[n])
        else:
            model.nop[n].fix(0)
            model.speed[n].fix(0.0)

    # Objective: minimize total cost (power + DRA)
    power_costs = []
    dra_costs = []
    for e in edge_list:
        i, j = e
        if is_pump[i]:
            # Use pump curve polynomial (from uploaded curves)
            if i in pump_curves:
                dfh, dfe = pump_curves[i]
                # Quadratic fit: Head = A Q^2 + B Q + C (at rated speed)
                Qh, Hh = dfh.iloc[:, 0].values, dfh.iloc[:, 1].values
                A, B, C = np.polyfit(Qh, Hh, 2)
                flow = model.flow[e]
                speed = model.speed[i]
                rated = max_rpm[i]
                head = (A * flow ** 2 + B * flow + C) * (speed / rated) ** 2
                # Efficiency fit (quartic)
                Qe, Ee = dfe.iloc[:, 0].values, dfe.iloc[:, 1].values
                P, Qc, R, S, T = np.polyfit(Qe, Ee, 4)
                eff = (P * flow ** 4 + Qc * flow ** 3 + R * flow ** 2 + S * flow + T)
                eff = max(0.01, eff / 100)
            else:
                head = 50
                eff = 0.7
            pwr = (node_rho[i] * model.flow[e] * 9.81 * head * model.nop[i]) / (3600.0 * 1000.0 * eff * 0.95)
            power_costs.append(pwr * 24 * diesel_price)
        # DRA cost (DR is fraction, apply to flow * 24 hr)
        dra_amt = model.dra[e] * model.flow[e] * 1000.0 * 24.0 / 1e6
        dra_costs.append(dra_amt * dra_cost)
    model.Obj = pyo.Objective(expr=sum(power_costs) + sum(dra_costs), sense=pyo.minimize)

    # Solve using Ipopt (for Streamlit Cloud)
    solver = pyo.SolverFactory('ipopt')
    results = solver.solve(model, tee=False)
    model.solutions.load_from(results)

    # Build output
    node_out = []
    for n in node_names:
        node_out.append({
            "Node": n,
            "Elevation": node_elev[n],
            "Demand": node_demand[n],
            "Density": node_rho[n],
            "Viscosity": node_KV[n],
            "Residual Head": pyo.value(model.res_head[n]),
            "Is Pump": is_pump[n],
            "Pump RPM": float(pyo.value(model.speed[n])),
            "No. Pumps": int(pyo.value(model.nop[n]))
        })
    edge_out = []
    for e in edge_list:
        i, j = e
        D = edge_D[e]
        t = edge_t[e]
        area = pi * (D ** 2) / 4.0
        Q = pyo.value(model.flow[e])
        v = Q / 3600.0 / area
        kv = node_KV[i]
        Re = v * D / (kv * 1e-6)
        edge_out.append({
            "From": i, "To": j, "Length (km)": edge_len[e],
            "Diameter (m)": D, "Wall Thk (m)": t,
            "Flow (m3/hr)": Q, "Velocity (m/s)": v,
            "Reynolds": Re,
            "Drag Reduction (%)": 100 * pyo.value(model.dra[e]),
        })
    total_cost = float(pyo.value(model.Obj))
    return node_out, edge_out, total_cost, None, None, None, None
