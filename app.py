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
