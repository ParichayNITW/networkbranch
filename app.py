import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import networkx as nx
import uuid

import network_pipeline_model as netmodel

st.set_page_config(page_title="Pipeline Optima™ Network", layout="wide")
st.markdown("<style>div.block-container{padding-top:1rem;}</style>", unsafe_allow_html=True)

if "nodes" not in st.session_state: st.session_state["nodes"] = []
if "edges" not in st.session_state: st.session_state["edges"] = []
if "pump_curves" not in st.session_state: st.session_state["pump_curves"] = {}
if "results" not in st.session_state: st.session_state["results"] = None

st.title("Pipeline Optima™ Network Edition (Branched/Looped Optimizer)")
st.markdown("By Parichay Das &copy; 2025. All professional outputs for pipelines with branches, loops, and demands at any node.")

with st.sidebar:
    st.header("Global Cost Parameters")
    dra_cost = st.number_input("DRA Cost (INR/L)", value=500.0)
    diesel_price = st.number_input("Diesel Price (INR/L)", value=90.0)
    if st.button("Reset Everything"):
        st.session_state["nodes"] = []
        st.session_state["edges"] = []
        st.session_state["pump_curves"] = {}
        st.session_state["results"] = None
        st.experimental_rerun()

# --- NODES ---
st.header("1️⃣ Define Nodes (Stations, Demand, Junctions)")
if st.button("Add Node"):
    st.session_state["nodes"].append({
        "Node Name": f"Node{len(st.session_state['nodes'])+1}",
        "Elevation (m)": 0.0,
        "Demand (m3/hr)": 0.0,
        "Density (kg/m3)": 850.0,
        "Viscosity (cSt)": 10.0,
        "Is Pump?": "No",
        "Pump Min RPM": 1000.0,
        "Pump DOL": 1500.0,
        "No. of Pumps": 1,
    })
if len(st.session_state["nodes"]) > 0:
    nodes_df = pd.DataFrame(st.session_state["nodes"])
    edited = st.data_editor(
        nodes_df,
        column_config={
            "Is Pump?": st.column_config.SelectboxColumn(options=["No", "Yes"])
        },
        num_rows="dynamic"
    )
    st.session_state["nodes"] = edited.fillna("").to_dict(orient="records")

# --- EDGES ---
st.header("2️⃣ Define Edges (Segments/Branches/Loops)")
if len(st.session_state["nodes"]) > 1:
    nlist = [n["Node Name"] for n in st.session_state["nodes"]]
    if st.button("Add Edge"):
        st.session_state["edges"].append({
            "From Node": nlist[0],
            "To Node": nlist[1],
            "Length (km)": 50.0,
            "Diameter (m)": 0.7,
            "Wall Thickness (m)": 0.007,
            "Roughness (m)": 0.00004,
            "Max Achievable DR (%)": 40.0
        })
    if len(st.session_state["edges"]) > 0:
        edges_df = pd.DataFrame(st.session_state["edges"])
        edited_edges = st.data_editor(
            edges_df,
            column_config={
                "From Node": st.column_config.SelectboxColumn(options=nlist),
                "To Node": st.column_config.SelectboxColumn(options=nlist)
            },
            num_rows="dynamic"
        )
        st.session_state["edges"] = edited_edges.fillna("").to_dict(orient="records")

# --- PUMP CURVES ---
st.header("3️⃣ Upload Pump Curves (for Pump Nodes Only)")
for idx, node in enumerate(st.session_state["nodes"]):
    if node["Is Pump?"] == "Yes":
        st.markdown(f"#### Pump Curves for **{node['Node Name']}**")
        fh = st.file_uploader(f"Flow vs Head CSV ({node['Node Name']})", type="csv", key=f"fh_{idx}")
        fe = st.file_uploader(f"Flow vs Efficiency CSV ({node['Node Name']})", type="csv", key=f"fe_{idx}")
        if fh and fe:
            dfh = pd.read_csv(fh)
            dfe = pd.read_csv(fe)
            st.session_state["pump_curves"][node['Node Name']] = (dfh, dfe)
            st.success(f"Curves loaded for {node['Node Name']}")
        elif node['Node Name'] in st.session_state["pump_curves"]:
            st.info(f"Curves already loaded for {node['Node Name']}")

# --- NETWORK GRAPH ---
st.header("4️⃣ Visualize Network")
try:
    G = nx.DiGraph()
    for n in st.session_state["nodes"]:
        G.add_node(n["Node Name"])
    for e in st.session_state["edges"]:
        G.add_edge(e["From Node"], e["To Node"])
    pos = nx.spring_layout(G, seed=42)
    edge_labels = {(e["From Node"], e["To Node"]): f"{e['Length (km)']} km" for e in st.session_state["edges"]}
    plt.figure(figsize=(7,4))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1300, font_size=11)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='crimson')
    st.pyplot(plt.gcf())
    plt.close()
except Exception:
    st.warning("Matplotlib/networkx required for network visual.")

# --- RUN OPTIMIZATION ---
st.header("5️⃣ Run Optimization")
if st.button("🚀 Run Pipeline Optimization"):
    with st.spinner("Running full Pyomo MINLP..."):
        nodes = st.session_state["nodes"]
        edges = st.session_state["edges"]
        pump_curves = st.session_state["pump_curves"]
        res = netmodel.solve_network_pipeline(
            nodes, edges, pump_curves, dra_cost, diesel_price
        )
        st.session_state["results"] = res
        st.success("Optimization complete!")

# --- OUTPUT TABS ---
if st.session_state["results"] is not None:
    node_out, edge_out, total_cost, _, _, _, _ = st.session_state["results"]
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📋 Summary",
        "🔗 Edge Details",
        "💰 Cost",
        "⚙️ Performance",
        "🌀 Pump/Curve",
        "📉 Pressure",
        "🧊 3D Plots"
    ])

    # --- SUMMARY TAB ---
    with tab1:
        st.markdown("### Node Results")
        df_nodes = pd.DataFrame(node_out)
        st.dataframe(df_nodes)
        st.download_button("Node Results CSV", df_nodes.to_csv(index=False), file_name="node_results.csv")
        st.success(f"Total Optimized Cost: ₹{total_cost:,.2f} per day")

    # --- EDGE TAB ---
    with tab2:
        st.markdown("### Edge Results")
        df_edges = pd.DataFrame(edge_out)
        st.dataframe(df_edges)
        st.download_button("Edge Results CSV", df_edges.to_csv(index=False), file_name="edge_results.csv")

    # --- COST TAB ---
    with tab3:
        st.markdown("### Cost Breakdown")
        df_edges = pd.DataFrame(edge_out)
        df_nodes = pd.DataFrame(node_out)
        if not df_edges.empty:
            st.bar_chart(df_edges.set_index("From")["Flow (m3/hr)"])
        st.write("**Power/DRA costs per station:**")
        cost_rows = []
        for n in df_nodes.itertuples():
            if getattr(n, "Is Pump"):
                cost_rows.append({
                    "Node": n.Node,
                   
                    "Power Cost": getattr(n, "No. Pumps", 0) * diesel_price * 24  # Illustrative, real value from model as needed
                })
        if cost_rows:
            df_cost = pd.DataFrame(cost_rows)
            st.dataframe(df_cost)
            st.download_button("Download Cost Breakdown", df_cost.to_csv(index=False), file_name="cost_breakdown.csv")
        else:
            st.info("No cost data for this run.")

    # --- PERFORMANCE TAB ---
    with tab4:
        st.markdown("### Performance Summary")
        df_edges = pd.DataFrame(edge_out)
        if not df_edges.empty:
            st.line_chart(df_edges[["Flow (m3/hr)", "Velocity (m/s)", "Reynolds"]])
        st.info("Review flow, velocity, and Reynolds for all segments.")

    # --- PUMP/CURVE TAB ---
    with tab5:
        st.markdown("### Pump Characteristic/Efficiency Curves")
        for n in st.session_state["nodes"]:
            if n["Is Pump?"] == "Yes" and n["Node Name"] in st.session_state["pump_curves"]:
                dfh, dfe = st.session_state["pump_curves"][n["Node Name"]]
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=dfh.iloc[:,0], y=dfh.iloc[:,1], mode='lines+markers', name="Head Curve"))
                fig1.update_layout(title=f"Flow vs Head: {n['Node Name']}", xaxis_title="Flow (m3/hr)", yaxis_title="Head (m)")
                st.plotly_chart(fig1, use_container_width=True)

                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=dfe.iloc[:,0], y=dfe.iloc[:,1], mode='lines+markers', name="Efficiency Curve"))
                fig2.update_layout(title=f"Flow vs Efficiency: {n['Node Name']}", xaxis_title="Flow (m3/hr)", yaxis_title="Efficiency (%)")
                st.plotly_chart(fig2, use_container_width=True)

    # --- PRESSURE TAB ---
    with tab6:
        st.markdown("### Pressure/Head Profile")
        # For simplicity, show residual head at each node, can extend to length-based profile later
        df_nodes = pd.DataFrame(node_out)
        if not df_nodes.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_nodes["Node"],
                y=df_nodes["Residual Head"],
                mode="lines+markers",
                name="Residual Head"
            ))
            fig.update_layout(title="Residual Head per Node", xaxis_title="Node", yaxis_title="Head (m)")
            st.plotly_chart(fig, use_container_width=True)

    # --- 3D TAB ---
    with tab7:
        st.markdown("### 3D Parameter Sensitivity & Surface Plots")
        # Only pump nodes are relevant
        pump_nodes = [n["Node"] for n in node_out if n["Is Pump"]]
        if not pump_nodes:
            st.info("No pump stations found in this case.")
        else:
            selected_node = st.selectbox("Select Pump Node for 3D Plot", pump_nodes)
            node_row = next(n for n in node_out if n["Node"] == selected_node)
            if selected_node not in st.session_state["pump_curves"]:
                st.warning("Upload pump curves for this node for advanced plotting.")
            else:
                dfh, dfe = st.session_state["pump_curves"][selected_node]
                # Fit pump curves
                A, B, C = np.polyfit(dfh.iloc[:,0], dfh.iloc[:,1], 2)
                P, Qc, R, S, T = np.polyfit(dfe.iloc[:,0], dfe.iloc[:,1], 4)
                rpm_min = int(node_row["Pump RPM"]) if node_row["Pump RPM"] > 0 else 1000
                rpm_max = int(node_row["Pump RPM"]+200) if node_row["Pump RPM"] > 0 else 1500
                flow_range = np.linspace(dfh.iloc[:,0].min(), dfh.iloc[:,0].max(), 50)
                speed_range = np.linspace(rpm_min, rpm_max, 20)
                X, Y = np.meshgrid(flow_range, speed_range)
                Z_head = (A*X**2 + B*X + C)*(Y/rpm_max)**2
                Q_adj = X * rpm_max / Y
                Z_eff = (P*Q_adj**4 + Qc*Q_adj**3 + R*Q_adj**2 + S*Q_adj + T)
                rho = node_row["Density"]
                g = 9.81
                Z_eff_nonzero = np.where(Z_eff < 0.1, 0.1, Z_eff)
                Z_power = (rho * X * g * Z_head)/(3600*Z_eff_nonzero*0.95*1000) * 24 * diesel_price
                plot_type = st.selectbox("Surface Type", ["Head (m)", "Efficiency (%)", "Power Cost (₹/day)"])
                fig = go.Figure()
                if plot_type == "Head (m)":
                    fig.add_trace(go.Surface(x=flow_range, y=speed_range, z=Z_head, colorscale='Viridis'))
                    fig.update_layout(title="Head vs Flow vs Speed", scene={"xaxis_title":"Flow (m³/hr)","yaxis_title":"RPM","zaxis_title":"Head (m)"})
                elif plot_type == "Efficiency (%)":
                    fig.add_trace(go.Surface(x=flow_range, y=speed_range, z=Z_eff, colorscale='Viridis'))
                    fig.update_layout(title="Efficiency vs Flow vs Speed", scene={"xaxis_title":"Flow (m³/hr)","yaxis_title":"RPM","zaxis_title":"Efficiency (%)"})
                elif plot_type == "Power Cost (₹/day)":
                    fig.add_trace(go.Surface(x=flow_range, y=speed_range, z=Z_power, colorscale='Viridis'))
                    fig.update_layout(title="Power Cost vs Flow vs Speed", scene={"xaxis_title":"Flow (m³/hr)","yaxis_title":"RPM","zaxis_title":"Cost (₹/day)"})
                st.plotly_chart(fig, use_container_width=True)

# --- END ---

st.markdown(
    """
    <div style='text-align: center; color: gray; margin-top: 2em; font-size: 0.9em;'>
    &copy; 2025 Pipeline Optima™ Network Edition. Developed by Parichay Das. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)
