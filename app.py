import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import tempfile
from pyvis.network import Network
import streamlit.components.v1 as components

st.set_page_config(page_title="Pipeline Network Optimizer (Branch/Loop)", layout="wide")
st.title("Pipeline Network Optimizer with Branch and Loop Support")

# ========== Nodes Table ==========
if "nodes" not in st.session_state:
    st.session_state["nodes"] = []
if "edges" not in st.session_state:
    st.session_state["edges"] = []

st.header("Define Nodes (Stations, Demand Points, etc.)")
node_cols = ["Node Name", "Elevation (m)", "Demand (m3/hr)", "Is Pump"]
node_df = pd.DataFrame(st.session_state["nodes"], columns=node_cols)
node_df = st.data_editor(
    node_df,
    column_config={
        "Is Pump": st.column_config.SelectboxColumn(options=["No", "Yes"]),
    },
    num_rows="dynamic", key="node_editor"
)
if st.button("Save Nodes"):
    st.session_state["nodes"] = node_df.fillna("").to_dict(orient="records")
    st.success("Nodes saved!")

# ========== Edges Table ==========
st.header("Define Edges (Pipeline Segments, Branches, Loops)")
if len(st.session_state["nodes"]) > 1:
    all_node_names = [n["Node Name"] for n in st.session_state["nodes"]]
    edge_cols = ["From Node", "To Node", "Length (km)", "Diameter (m)", "Roughness (m)", "Has Pump", "DRA Max (%)"]
    edge_df = pd.DataFrame(st.session_state["edges"], columns=edge_cols)
    edge_df = st.data_editor(
        edge_df,
        column_config={
            "From Node": st.column_config.SelectboxColumn(options=all_node_names),
            "To Node": st.column_config.SelectboxColumn(options=all_node_names),
            "Has Pump": st.column_config.SelectboxColumn(options=["No", "Yes"]),
        },
        num_rows="dynamic", key="edge_editor"
    )
    if st.button("Save Edges"):
        st.session_state["edges"] = edge_df.fillna("").to_dict(orient="records")
        st.success("Edges saved!")
else:
    st.warning("Define at least 2 nodes before adding edges.")

# ========== Pump Curves Upload ==========
st.header("Upload Pump Curves for Edges with Pumps (Optional)")
if "pump_curves" not in st.session_state:
    st.session_state["pump_curves"] = {}
for idx, edge in enumerate(st.session_state["edges"]):
    if str(edge.get("Has Pump", "")).lower() == "yes":
        st.subheader(f"Edge {idx+1}: {edge['From Node']} ➔ {edge['To Node']}")
        col1, col2 = st.columns(2)
        with col1:
            fh = st.file_uploader(f"Flow vs Head CSV (Edge {idx+1})", type=["csv"], key=f"fh_{idx}")
            if fh is not None:
                fh_df = pd.read_csv(fh)
                st.session_state["pump_curves"][f"fh_{idx}"] = fh_df
                st.write(fh_df)
        with col2:
            fe = st.file_uploader(f"Flow vs Efficiency CSV (Edge {idx+1})", type=["csv"], key=f"fe_{idx}")
            if fe is not None:
                fe_df = pd.read_csv(fe)
                st.session_state["pump_curves"][f"fe_{idx}"] = fe_df
                st.write(fe_df)

# ========== Network Visualization ==========
st.header("Network Visualization")
try:
    net = Network(height="500px", width="100%", bgcolor="#222222", font_color="white", directed=True)
    for n in st.session_state["nodes"]:
        is_pump = str(n.get("Is Pump", "")).lower() == "yes"
        label = f"{n['Node Name']}"
        net.add_node(label, label=label, color="red" if is_pump else "blue")
    for e in st.session_state["edges"]:
        net.add_edge(e["From Node"], e["To Node"], color="green" if e["Has Pump"]=="Yes" else "gray")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as temp_html:
        net.save_graph(temp_html.name)
        components.html(open(temp_html.name, 'r', encoding='utf-8').read(), height=520)
except Exception as e:
    st.warning(f"Network visualization failed: {e}")

# ========== Run Optimization Button ==========
st.header("Run Optimization (Demo)")
if st.button("Optimize Network"):
    st.info("This is a UI demo. Backend to be plugged in below (Pyomo MINLP with network logic).")
    # Here you would call: solve_network(nodes, edges, pump_curves, etc.)
    st.write("Nodes:", st.session_state["nodes"])
    st.write("Edges:", st.session_state["edges"])
    st.write("Pump Curves:", st.session_state["pump_curves"])
    st.success("Plug in the full Pyomo backend for network optimization here.")

st.markdown("""
---
**Instructions:**  
1. Define all stations/demand points as nodes.  
2. Define all segments/branches as edges (you can create loops/branches freely).  
3. For each edge with a pump, upload the required curves.  
4. Click "Optimize Network" to run the optimization (backend to be added).
""")
