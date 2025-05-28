import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components

def parse_pairs(csv_string):
    # Parse "10,40;15,36;20,28" into [(10,40),(15,36),(20,28)]
    if not csv_string or not isinstance(csv_string, str):
        return []
    pairs = []
    for entry in csv_string.split(';'):
        parts = entry.strip().split(',')
        if len(parts) == 2:
            try:
                pairs.append((float(parts[0]), float(parts[1])))
            except:
                continue
    return pairs

def polyfit_head(flow_head):
    if len(flow_head) < 2: return [0,0,0]
    Q = np.array([row[0] for row in flow_head])
    H = np.array([row[1] for row in flow_head])
    coef = np.polyfit(Q, H, 2)
    return coef.tolist()

def polyfit_eff(flow_eff):
    if len(flow_eff) < 2: return [0,0,0,0,0]
    Q = np.array([row[0] for row in flow_eff])
    E = np.array([row[1] for row in flow_eff])
    coef = np.polyfit(Q, E, 4)
    return coef.tolist()

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



st.set_page_config(page_title="Pipeline Network Optimizer", layout="wide")
st.title("Pipeline Network Optimizer")

station_cols = [
    "Name", "Length (km)", "Diameter (m)", "Thickness (m)", "Roughness (m)",
    "SMYS", "DF", "Elevation (m)", "Is Pump", "Power Source", "SFC (Diesel)", "Rate (Grid)", 
    "DOL", "Min RPM", "Max Pumps",
    "Flow-Head Pairs (CSV)", "Flow-Efficiency Pairs (CSV)",
    "Peak Distance (km)", "Peak Elevation (m)"
]
if "stations" not in st.session_state:
    st.session_state["stations"] = [{col: "" for col in station_cols}]

edited = st.data_editor(
    pd.DataFrame(st.session_state["stations"]),
    column_config={
        "Is Pump": st.column_config.SelectboxColumn(options=["No", "Yes"]),
        "Power Source": st.column_config.SelectboxColumn(options=["", "Grid", "Diesel"]),
    },
    num_rows="dynamic", key="station_editor"
)
if st.button("Save Stations"):
    st.session_state["stations"] = edited.fillna("").to_dict(orient="records")
    st.success("Stations Saved!")

# --- Pump Curve CSV Uploaders ---
st.subheader("Upload Pump Curves (Optional, Per Pump Station)")

if "curve_files" not in st.session_state:
    st.session_state["curve_files"] = {}

for i, stn in enumerate(st.session_state["stations"]):
    if str(stn.get("Is Pump", "")).lower() == "yes":
        st.markdown(f"**Pump: {stn['Name'].title()}**")
        col1, col2 = st.columns(2)
        with col1:
            uploaded_fh = st.file_uploader(
                f"Upload Flow vs Head CSV for {stn['Name']}",
                type=["csv"],
                key=f"fh_upload_{i}"
            )
            if uploaded_fh is not None:
                fh_df = pd.read_csv(uploaded_fh)
                # Validate required columns
                if set(fh_df.columns) >= {"Flow", "Head"}:
                    st.session_state["curve_files"][f"fh_{i}"] = fh_df
                    st.write(fh_df)
                else:
                    st.warning("CSV must have columns: Flow, Head")
        with col2:
            uploaded_fe = st.file_uploader(
                f"Upload Flow vs Efficiency CSV for {stn['Name']}",
                type=["csv"],
                key=f"fe_upload_{i}"
            )
            if uploaded_fe is not None:
                fe_df = pd.read_csv(uploaded_fe)
                if set(fe_df.columns) >= {"Flow", "Efficiency"}:
                    st.session_state["curve_files"][f"fe_{i}"] = fe_df
                    st.write(fe_df)
                else:
                    st.warning("CSV must have columns: Flow, Efficiency")

# Suction pressure for first station
st.subheader("Available Suction Pressure At First Station")
suction_press = st.number_input("Available Suction Pressure (m)", value=50.0, step=1.0)

# Fluid properties
st.subheader("Fluid Properties (Per Station Except Terminal)")
if "KV_list" not in st.session_state or len(st.session_state["KV_list"]) != len(st.session_state["stations"]):
    st.session_state["KV_list"] = [1.0]*len(st.session_state["stations"])
if "rho_list" not in st.session_state or len(st.session_state["rho_list"]) != len(st.session_state["stations"]):
    st.session_state["rho_list"] = [850.0]*len(st.session_state["stations"])
KV_list, rho_list = [], []
for i, stn in enumerate(st.session_state["stations"]):
    col1, col2 = st.columns(2)
    with col1:
        kv = st.number_input(f"Kinematic Viscosity (cSt) For {stn['Name'].title()}", value=st.session_state["KV_list"][i], key=f"kv_{i}")
    with col2:
        rho = st.number_input(f"Density (kg/m3) For {stn['Name'].title()}", value=st.session_state["rho_list"][i], key=f"rho_{i}")
    KV_list.append(kv)
    rho_list.append(rho)
st.session_state["KV_list"] = KV_list
st.session_state["rho_list"] = rho_list

# Operating and cost params
st.header("Run Optimization")
FLOW = st.number_input("Flow Rate (m3/hr)", value=1000.0, min_value=0.0, step=10.0)
RateDRA = st.number_input("DRA Cost (per L)", value=0.0, step=0.1)
Price_HSD = st.number_input("Diesel Price (per L)", value=90.0, step=1.0)

# Terminal
st.subheader("Terminal Node")
terminal_name = st.text_input("Terminal Name", value="Terminal")
terminal_elev = st.number_input("Terminal Elevation (m)", value=0.0, step=1.0)
terminal_min_rh = st.number_input("Minimum Required RH At Terminal (m)", value=50.0, step=1.0)
terminal = {"name": terminal_name, "elev": terminal_elev, "min_residual": terminal_min_rh}

# Network visualization
st.header("Pipeline Network Visualization")
try:
    net = Network(height="500px", width="100%", bgcolor="#222222", font_color="white", directed=True)
    for stn in st.session_state["stations"]:
        name = stn.get("Name", "Station")
        is_pump = str(stn.get("Is Pump","")).lower() == "yes"
        net.add_node(name, label=name, color="red" if is_pump else "blue")
    net.add_node(terminal_name, label=terminal_name, color="green")
    for i, stn in enumerate(st.session_state["stations"]):
        if i < len(st.session_state["stations"])-1:
            net.add_edge(stn.get("Name",""), st.session_state["stations"][i+1].get("Name",""), color="gray")
        else:
            net.add_edge(stn.get("Name",""), terminal_name, color="green")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as temp_html:
        net.save_graph(temp_html.name)
        components.html(open(temp_html.name, 'r', encoding='utf-8').read(), height=520)
except Exception as e:
    st.warning(f"Network visualization failed: {e}")

# Optimization and backend data prep
if st.button("Optimize Pipeline Network"):
    stations_for_backend = []
    for i, stn in enumerate(st.session_state["stations"]):
        is_pump = str(stn.get("Is Pump", "")).lower() == "yes"
        power_source = stn.get("Power Source", "").lower()
        sfc = float(stn["SFC (Diesel)"]) if (power_source == "diesel" and stn["SFC (Diesel)"]) else 0
        rate = float(stn["Rate (Grid)"]) if (power_source == "grid" and stn["Rate (Grid)"]) else 0

        # Pump curves from uploaded CSV or fallback to text
        if f"fh_{i}" in st.session_state["curve_files"]:
            fh_pairs = st.session_state["curve_files"][f"fh_{i}"][["Flow", "Head"]].values.tolist()
        else:
            fh_pairs = parse_pairs(stn.get("Flow-Head Pairs (CSV)", ""))

        if f"fe_{i}" in st.session_state["curve_files"]:
            fe_pairs = st.session_state["curve_files"][f"fe_{i}"][["Flow", "Efficiency"]].values.tolist()
        else:
            fe_pairs = parse_pairs(stn.get("Flow-Efficiency Pairs (CSV)", ""))

        [A,B,C] = polyfit_head(fh_pairs)
        [P,Q,R,S,T] = polyfit_eff(fe_pairs)

        # Peak
        peak_dist = float(stn.get("Peak Distance (km)", 0) or 0)
        peak_elev = float(stn.get("Peak Elevation (m)", 0) or 0)
        peaks = []
        if peak_dist and peak_elev:
            peaks = [{"loc": peak_dist, "elev": peak_elev}]
        stations_for_backend.append({
            "name": stn["Name"],
            "L": float(stn["Length (km)"] or 0),
            "D": float(stn["Diameter (m)"] or 0),
            "t": float(stn["Thickness (m)"] or 0),
            "rough": float(stn["Roughness (m)"] or 0),
            "SMYS": float(stn["SMYS"] or 0),
            "DF": float(stn["DF"] or 0),
            "elev": float(stn["Elevation (m)"] or 0),
            "is_pump": is_pump,
            "A": A, "B": B, "C": C, "P": P, "Q": Q, "R": R, "S": S, "T": T,
            "DOL": float(stn["DOL"] or 0),
            "MinRPM": float(stn["Min RPM"] or 0),
            "max_pumps": int(stn["Max Pumps"] or 2),
            "sfc": sfc,
            "rate": rate,
            "max_dr": 40.0,
            "min_residual": suction_press if i == 0 else 50.0,
            "peaks": peaks
        })
    KV_list = st.session_state["KV_list"]
    rho_list = st.session_state["rho_list"]

    try:
        # results = solve_pipeline(stations_for_backend, terminal, FLOW, KV_list, rho_list, RateDRA, Price_HSD)
        st.success("Optimization Complete! (Backend not called in this demo)")
        # st.write(results)
        st.write("Backend call would happen here. Results would be shown below.")
    except Exception as e:
        st.error(f"Optimization failed: {e}")

st.markdown("""
---
**Instructions**
- Edit all station and pump details in the main table.
- For pump curve data, either:
    - Upload a CSV (columns: "Flow,Head" and/or "Flow,Efficiency") for each pump station, **or**
    - Enter text in table cell as `Flow,Head;Flow,Head;...`
- Enter peak distance (km) and elevation (m) in table if there is a peak.
- "Is Pump" and "Power Source" are dropdowns.
- SFC/Rate auto-selected based on power source.
- Click Optimize to process!
""")
