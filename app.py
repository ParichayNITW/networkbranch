
import streamlit as st
import pandas as pd
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverManagerFactory
from math import log10, pi
import os

os.environ['NEOS_EMAIL'] = os.environ.get('NEOS_EMAIL', 'youremail@example.com')

def solve_pipeline(stations, terminal, FLOW, KV_list, rho_list, RateDRA, Price_HSD):
    model = pyo.ConcreteModel()
    N = len(stations)
    model.I = pyo.RangeSet(1, N)
    model.Nodes = pyo.RangeSet(1, N+1)
    kv_dict = {i: float(KV_list[i-1]) for i in range(1, N+1)}
    rho_dict = {i: float(rho_list[i-1]) for i in range(1, N+1)}
    model.KV = pyo.Param(model.I, initialize=kv_dict)
    model.rho = pyo.Param(model.I, initialize=rho_dict)
    model.FLOW = pyo.Param(initialize=FLOW)
    model.Rate_DRA = pyo.Param(initialize=RateDRA)
    model.Price_HSD = pyo.Param(initialize=Price_HSD)

    length = {}; d_inner = {}; roughness = {}; thickness = {}; smys = {}; design_factor = {}; elev = {}
    Acoef = {}; Bcoef = {}; Ccoef = {}
    Pcoef = {}; Qcoef = {}; Rcoef = {}; Scoef = {}; Tcoef = {}
    min_rpm = {}; max_rpm = {}
    sfc = {}; elec_cost = {}
    pump_indices = []; diesel_pumps = []; electric_pumps = []
    inj_source = {}
    max_dr = {}
    last_pump_idx = None

    default_t = 0.007
    default_e = 0.00004
    default_smys = 52000
    default_df = 0.72

    for i, stn in enumerate(stations, start=1):
        length[i] = stn.get('L', 0.0)
        if 'D' in stn:
            D_out = stn['D']
            thickness[i] = stn.get('t', default_t)
            d_inner[i] = D_out - 2*thickness[i]
        elif 'd' in stn:
            d_inner[i] = stn['d']
            thickness[i] = stn.get('t', default_t)
        else:
            d_inner[i] = 0.7
            thickness[i] = default_t
        roughness[i] = stn.get('rough', default_e)
        smys[i] = stn.get('SMYS', default_smys)
        design_factor[i] = stn.get('DF', default_df)
        elev[i] = stn.get('elev', 0.0)
        has_pump = stn.get('is_pump', False)
        if has_pump:
            pump_indices.append(i)
            Acoef[i] = stn.get('A', 0.0)
            Bcoef[i] = stn.get('B', 0.0)
            Ccoef[i] = stn.get('C', 0.0)
            Pcoef[i] = stn.get('P', 0.0)
            Qcoef[i] = stn.get('Q', 0.0)
            Rcoef[i] = stn.get('R', 0.0)
            Scoef[i] = stn.get('S', 0.0)
            Tcoef[i] = stn.get('T', 0.0)
            min_rpm[i] = stn.get('MinRPM', 0)
            max_rpm[i] = stn.get('DOL', 0)
            if stn.get('sfc', 0) not in (None, 0):
                diesel_pumps.append(i)
                sfc[i] = stn.get('sfc', 0.0)
            else:
                electric_pumps.append(i)
                elec_cost[i] = stn.get('rate', 0.0)
            max_dr[i] = stn.get('max_dr', 0.0)
            last_pump_idx = i
        inj_source[i] = last_pump_idx

    elev[N+1] = terminal.get('elev', 0.0)
    model.L = pyo.Param(model.I, initialize=length)
    model.d = pyo.Param(model.I, initialize=d_inner)
    model.e = pyo.Param(model.I, initialize=roughness)
    model.SMYS = pyo.Param(model.I, initialize=smys)
    model.DF = pyo.Param(model.I, initialize=design_factor)
    model.z = pyo.Param(model.Nodes, initialize=elev)
    model.pump_stations = pyo.Set(initialize=pump_indices)
    if pump_indices:
        model.A = pyo.Param(model.pump_stations, initialize=Acoef)
        model.B = pyo.Param(model.pump_stations, initialize=Bcoef)
        model.C = pyo.Param(model.pump_stations, initialize=Ccoef)
        model.Pcoef = pyo.Param(model.pump_stations, initialize=Pcoef)
        model.Qcoef = pyo.Param(model.pump_stations, initialize=Qcoef)
        model.Rcoef = pyo.Param(model.pump_stations, initialize=Rcoef)
        model.Scoef = pyo.Param(model.pump_stations, initialize=Scoef)
        model.Tcoef = pyo.Param(model.pump_stations, initialize=Tcoef)
        model.MinRPM = pyo.Param(model.pump_stations, initialize=min_rpm)
        model.DOL = pyo.Param(model.pump_stations, initialize=max_rpm)
    def nop_bounds(m, j):
        lb = 1 if j == 1 else 0
        ub = stations[j-1].get('max_pumps', 2)
        return (lb, ub)
    model.NOP = pyo.Var(model.pump_stations, domain=pyo.NonNegativeIntegers,
                        bounds=nop_bounds, initialize=1)
    speed_min = {}; speed_max = {}
    for j in pump_indices:
        lo = max(1, (int(model.MinRPM[j]) + 9)//10) if model.MinRPM[j] else 1
        hi = max(lo, int(model.DOL[j])//10) if model.DOL[j] else lo
        speed_min[j], speed_max[j] = lo, hi
    model.N_u = pyo.Var(model.pump_stations, domain=pyo.NonNegativeIntegers,
                       bounds=lambda m,j: (speed_min[j], speed_max[j]),
                       initialize=lambda m,j: (speed_min[j]+speed_max[j])//2)
    model.N = pyo.Expression(model.pump_stations, rule=lambda m,j: 10*m.N_u[j])
    dr_max = {j: int(max_dr.get(j, 40)//10) for j in pump_indices}
    model.DR_u = pyo.Var(model.pump_stations, domain=pyo.NonNegativeIntegers,
                        bounds=lambda m,j: (0, dr_max[j]), initialize=0)
    model.DR = pyo.Expression(model.pump_stations, rule=lambda m,j: 10*m.DR_u[j])
    model.RH = pyo.Var(model.Nodes, domain=pyo.NonNegativeReals, initialize=50)
    model.RH[1].fix(stations[0].get('min_residual', 50.0))
    for j in range(2, N+2):
        model.RH[j].setlb(50.0)
    g = 9.81
    flow_m3s = pyo.value(model.FLOW)/3600.0 if FLOW is not None else 0.0
    v = {}; Re = {}; f = {}
    for i in range(1, N+1):
        area = pi * (d_inner[i]**2) / 4.0
        v[i] = flow_m3s / area if area > 0 else 0.0
        kv = kv_dict[i]
        rho = rho_dict[i]
        if kv > 0:
            Re[i] = v[i]*d_inner[i]/(float(kv)*1e-6)
        else:
            Re[i] = 0.0
        if Re[i] > 0:
            if Re[i] < 4000:
                f[i] = 64.0/Re[i]
            else:
                arg = (roughness[i]/d_inner[i]/3.7) + (5.74/(Re[i]**0.9))
                f[i] = 0.25/(log10(arg)**2) if arg > 0 else 0.0
        else:
            f[i] = 0.0

    TDH = {}
    EFFP = {}
    model.SDH = pyo.Var(model.I, domain=pyo.NonNegativeReals, initialize=0)
    model.sdh_constraint = pyo.ConstraintList()
    for i in range(1, N+1):
        DR_frac = 0
        if inj_source.get(i) in pump_indices:
            DR_frac = model.DR[inj_source[i]]/100.0
        DH_next = f[i] * ( (length[i]*1000.0) / d_inner[i] ) * (v[i]**2 / (2*g)) * (1 - DR_frac)
        expr_next = model.RH[i+1] + (model.z[i+1] - model.z[i]) + DH_next
        model.sdh_constraint.add(model.SDH[i] >= expr_next)
        for peak in stations[i-1].get('peaks', []):
            L_peak = peak['loc'] * 1000.0
            elev_k = peak['elev']
            DR_frac_peak = 0
            if inj_source.get(i) in pump_indices:
                DR_frac_peak = model.DR[inj_source[i]]/100.0
            DH_peak = f[i] * ( (L_peak) / d_inner[i] ) * (v[i]**2 / (2*g)) * (1 - DR_frac_peak)
            expr_peak = (elev_k - model.z[i]) + DH_peak + 50.0
            model.sdh_constraint.add(model.SDH[i] >= expr_peak)
        if i in pump_indices:
            TDH[i] = (model.A[i]*model.FLOW**2 +
                      model.B[i]*model.FLOW +
                      model.C[i]) * ((model.N[i]/model.DOL[i])**2)
            flow_eq = model.FLOW * model.DOL[i]/model.N[i]
            EFFP[i] = (
                model.Pcoef[i]*flow_eq**4 +
                model.Qcoef[i]*flow_eq**3 +
                model.Rcoef[i]*flow_eq**2 +
                model.Scoef[i]*flow_eq   +
                model.Tcoef[i]
            ) / 100.0
        else:
            TDH[i] = 0.0
            EFFP[i] = 1.0

    model.head_balance = pyo.ConstraintList()
    model.peak_limit = pyo.ConstraintList()
    model.pressure_limit = pyo.ConstraintList()
    for i in range(1, N+1):
        kv = kv_dict[i]
        rho = rho_dict[i]
        if i in pump_indices:
            model.head_balance.add(model.RH[i] + TDH[i]*model.NOP[i] >= model.SDH[i])
        else:
            model.head_balance.add(model.RH[i] >= model.SDH[i])
        D_out = d_inner[i] + 2*thickness[i]
        MAOP_head = (2*thickness[i]*(smys[i]*0.070307)*design_factor[i]/D_out)*10000.0/rho
        if i in pump_indices:
            model.pressure_limit.add(model.RH[i] + TDH[i]*model.NOP[i] <= MAOP_head)
        else:
            model.pressure_limit.add(model.RH[i] <= MAOP_head)
        peaks = stations[i-1].get('peaks', [])
        for peak in peaks:
            loc_km = peak['loc']
            elev_k = peak['elev']
            L_peak = loc_km*1000.0
            DR_frac_peak = 0
            if inj_source.get(i) in pump_indices:
                DR_frac_peak = model.DR[inj_source[i]]/100.0
            loss_no_dra = f[i] * (L_peak/d_inner[i]) * (v[i]**2/(2*g))
            if i in pump_indices:
                expr = model.RH[i] + TDH[i]*model.NOP[i] - (elev_k - model.z[i]) - loss_no_dra*(1-DR_frac_peak)
            else:
                expr = model.RH[i] - (elev_k - model.z[i]) - loss_no_dra*(1-DR_frac_peak)
            model.peak_limit.add(expr >= 50.0)
    total_cost = 0
    for i in pump_indices:
        rho_i = rho_dict[i]
        kv_i = kv_dict[i]
        power_kW = (rho_i * FLOW * 9.81 * TDH[i] * model.NOP[i])/(3600.0*1000.0*EFFP[i]*0.95)
        if i in electric_pumps:
            power_cost = power_kW * 24.0 * elec_cost.get(i,0.0)
        else:
            fuel_per_kWh = (sfc.get(i,0.0)*1.34102)/820.0
            power_cost = power_kW * 24.0 * fuel_per_kWh * Price_HSD
        dra_cost = (model.DR[i]/4) * (FLOW*1000.0*24.0/1e6) * RateDRA
        total_cost += power_cost + dra_cost
    model.Obj = pyo.Objective(expr=total_cost, sense=pyo.minimize)
    results = SolverManagerFactory('neos').solve(model, solver='bonmin', tee=False)
    model.solutions.load_from(results)
    result = {}
    for i, stn in enumerate(stations, start=1):
        name = stn['name'].strip().lower().replace(' ', '_')
        if i in pump_indices:
            num_pumps = int(pyo.value(model.NOP[i]))
            speed_rpm = float(pyo.value(model.N[i])) if num_pumps > 0 else 0.0
            eff = float(pyo.value(EFFP[i])*100.0) if num_pumps > 0 else 0.0
        else:
            num_pumps = 0; speed_rpm = 0.0; eff = 0.0
        if i in pump_indices and num_pumps > 0:
            rho_i = rho_dict[i]
            power_kW = (rho_i * FLOW * 9.81 * float(pyo.value(TDH[i])) * num_pumps)/(3600.0*1000.0*float(pyo.value(EFFP[i]))*0.95)
            if i in electric_pumps:
                rate = elec_cost.get(i,0.0)
                power_cost = power_kW * 24.0 * rate
            else:
                sfc_val = sfc.get(i,0.0)
                fuel_per_kWh = (sfc_val*1.34102)/820.0
                power_cost = power_kW * 24.0 * fuel_per_kWh * Price_HSD
        else:
            power_cost = 0.0
        if i in pump_indices:
            dra_cost = (float(pyo.value(model.DR[i]))/4)*(FLOW*1000.0*24.0/1e6)*RateDRA
            drag_red = float(pyo.value(model.DR[i]))
        else:
            dra_cost = 0.0; drag_red = 0.0
        head_loss = float(pyo.value(model.SDH[i] - (model.RH[i+1] + (model.z[i+1]-model.z[i]))))
        res_head = float(pyo.value(model.RH[i]))
        velocity = v[i]; reynolds = Re[i]
        result[f"num_pumps_{name}"] = num_pumps
        result[f"speed_{name}"] = speed_rpm
        result[f"efficiency_{name}"] = eff
        result[f"power_cost_{name}"] = power_cost
        result[f"dra_cost_{name}"] = dra_cost
        result[f"drag_reduction_{name}"] = drag_red
        result[f"head_loss_{name}"] = head_loss
        result[f"residual_head_{name}"] = res_head
        result[f"velocity_{name}"] = velocity
        result[f"reynolds_{name}"] = reynolds
        result[f"sdh_{name}"] = float(pyo.value(model.SDH[i]))
        if i in pump_indices:
            result[f"coef_A_{name}"] = float(pyo.value(model.A[i]))
            result[f"coef_B_{name}"] = float(pyo.value(model.B[i]))
            result[f"coef_C_{name}"] = float(pyo.value(model.C[i]))
            result[f"dol_{name}"]    = float(pyo.value(model.DOL[i]))
            result[f"min_rpm_{name}"]= float(pyo.value(model.MinRPM[i]))
    term = terminal.get('name','terminal').strip().lower().replace(' ','_')
    result.update({
        f"speed_{term}": 0.0,
        f"num_pumps_{term}": 0,
        f"efficiency_{term}": 0.0,
        f"power_cost_{term}": 0.0,
        f"dra_cost_{term}": 0.0,
        f"drag_reduction_{term}": 0.0,
        f"head_loss_{term}": 0.0,
        f"velocity_{term}": 0.0,
        f"reynolds_{term}": 0.0,
        f"sdh_{term}": 0.0,
        f"residual_head_{term}": float(pyo.value(model.RH[N+1])),
    })
    result['total_cost'] = float(pyo.value(model.Obj))
    return result

# ==================== CHUNK 2: Streamlit UI =========================

st.set_page_config(page_title="Pipeline Network Optimizer", layout="wide")
st.title("Pipeline Network Optimizer")

st.header("Stations (including branches and tap-off points)")
if "stations" not in st.session_state:
    st.session_state["stations"] = []

stations_df = pd.DataFrame(st.session_state["stations"])
columns = [
    "name", "L", "D", "t", "rough", "SMYS", "DF", "elev", "is_pump", "min_residual",
    "A", "B", "C", "P", "Q", "R", "S", "T", "MinRPM", "DOL", "max_pumps", "sfc", "rate", "max_dr", "peaks"
]
display_columns = columns

if stations_df.empty:
    stations_df = pd.DataFrame([{col: "" for col in columns}])

edited = st.data_editor(stations_df[display_columns], num_rows="dynamic", key="station_editor")
if st.button("Save Stations"):
    data = edited.to_dict(orient="records")
    for row in data:
        if isinstance(row.get("peaks"), str) and row["peaks"].strip():
            try:
                row["peaks"] = eval(row["peaks"])
            except Exception:
                row["peaks"] = []
        elif not row.get("peaks"):
            row["peaks"] = []
        row["is_pump"] = str(row.get("is_pump", "")).strip().lower() in ["1", "true", "yes"]
    st.session_state["stations"] = data
    st.success("Stations saved!")

st.header("Terminal Node")
if "terminal" not in st.session_state:
    st.session_state["terminal"] = {"name": "Terminal", "elev": 0.0, "min_residual": 50.0}
terminal = st.session_state["terminal"]
terminal["name"] = st.text_input("Terminal Name", value=terminal["name"])
terminal["elev"] = st.number_input("Terminal Elevation (m)", value=float(terminal.get("elev", 0.0)), step=1.0)
terminal["min_residual"] = st.number_input("Minimum Required RH at Terminal (m)", value=float(terminal.get("min_residual", 50.0)), step=1.0)
st.session_state["terminal"] = terminal

st.header("Fluid Properties (per station except terminal)")
if "KV_list" not in st.session_state or len(st.session_state["KV_list"]) != len(st.session_state["stations"]):
    st.session_state["KV_list"] = [1.0] * len(st.session_state["stations"])
if "rho_list" not in st.session_state or len(st.session_state["rho_list"]) != len(st.session_state["stations"]):
    st.session_state["rho_list"] = [850.0] * len(st.session_state["stations"])
KV_list = []
rho_list = []
for i, stn in enumerate(st.session_state["stations"]):
    col1, col2 = st.columns(2)
    with col1:
        kv = st.number_input(f"Kinematic Viscosity (cSt) for {stn['name']}", value=float(st.session_state["KV_list"][i]), key=f"kv_{i}")
    with col2:
        rho = st.number_input(f"Density (kg/m3) for {stn['name']}", value=float(st.session_state["rho_list"][i]), key=f"rho_{i}")
    KV_list.append(kv)
    rho_list.append(rho)
st.session_state["KV_list"] = KV_list
st.session_state["rho_list"] = rho_list

st.header("Operating Parameters")
FLOW = st.number_input("Flow Rate (m3/hr)", value=1000.0, min_value=0.0, step=10.0)
RateDRA = st.number_input("DRA Cost (per L)", value=0.0, step=0.1)
Price_HSD = st.number_input("Diesel Price (per L)", value=90.0, step=1.0)

st.header("Run Optimization")
if st.button("Optimize Pipeline Network"):
    st.info("Running optimization (please wait for NEOS solution)...")
    try:
        stations = st.session_state["stations"]
        terminal = st.session_state["terminal"]
        KV_list = st.session_state["KV_list"]
        rho_list = st.session_state["rho_list"]
        results = solve_pipeline(stations, terminal, FLOW, KV_list, rho_list, RateDRA, Price_HSD)
        st.success("Optimization Complete!")
        st.subheader("Optimization Results (per station)")
        res_table = []
        for stn in stations:
            name = stn['name'].strip().lower().replace(' ', '_')
            row = {
                "Station": name,
                "Num Pumps": results.get(f"num_pumps_{name}", 0),
                "Pump Speed (RPM)": results.get(f"speed_{name}", 0),
                "Efficiency (%)": results.get(f"efficiency_{name}", 0),
                "Power Cost": results.get(f"power_cost_{name}", 0),
                "DRA Cost": results.get(f"dra_cost_{name}", 0),
                "Drag Reduction (%)": results.get(f"drag_reduction_{name}", 0),
                "Head Loss (m)": results.get(f"head_loss_{name}", 0),
                "Residual Head (m)": results.get(f"residual_head_{name}", 0),
                "Velocity (m/s)": results.get(f"velocity_{name}", 0),
                "Reynolds": results.get(f"reynolds_{name}", 0),
                "SDH (m)": results.get(f"sdh_{name}", 0),
            }
            res_table.append(row)
        st.dataframe(pd.DataFrame(res_table))
        st.subheader("Terminal Node Results")
        term = terminal.get('name','terminal').strip().lower().replace(' ','_')
        st.write({k:v for k,v in results.items() if term in k or k == 'total_cost'})
        st.write(f"Total Cost: {results.get('total_cost',0):,.2f}")
    except Exception as e:
        st.error(f"Optimization failed: {str(e)}")

st.markdown("---")
st.info("""
**Instructions**
- Enter all stations (mainline, branches, tap-off points) above. Use "peaks" column to specify elevation peaks as a list of dicts (e.g., `[{"loc": 12.0, "elev": 304.2}]`).
- Mark stations as pump stations by setting "is_pump" to True.
- Set all pump and cost data as needed for each station.
- Enter fluid properties per segment.
- Enter terminal node details, DRA, flow, and fuel/power costs.
- Click "Optimize Pipeline Network" and wait for results.
- The app will build the full model and use NEOS/Bonmin as per your code.
""")
