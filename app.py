import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import networkx as nx
import uuid
from io import BytesIO
from fpdf import FPDF

import network_pipeline_model as netmodel

st.set_page_config(page_title="Pipeline Optima™ Network", layout="wide")
st.markdown("<style>div.block-container{padding-top:1rem;}</style>", unsafe_allow_html=True)

if "nodes" not in st.session_state: st.session_state["nodes"] = []
if "edges" not in st.session_state: st.session_state["edges"] = []
if "pump_curves" not in st.session_state: st.session_state["pump_curves"] = {}
if "results" not in st.session_state: st.session_state["results"] = None
if "pdf_bytes" not in st.session_state: st.session_state["pdf_bytes"] = None

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
        st.session_state["pdf_bytes"] = None
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
        G.add_node(n["Node Name"], demand=n["Demand (m3/hr)"])
    for e in st.session_state["edges"]:
        G.add_edge(e["From Node"], e["To Node"], flow=0)
    pos = nx.spring_layout(G, seed=42)
    edge_labels = {(e["From Node"], e["To Node"]): f"{e['Length (km)']} km" for e in st.session_state["edges"]}
    node_color = [n["Demand (m3/hr)"] for n in st.session_state["nodes"]]
    node_color_map = plt.cm.viridis((np.array(node_color)-min(node_color))/(max(node_color)-min(node_color)+1e-9))
    fig, ax = plt.subplots(figsize=(7,4))
    nx.draw(G, pos, with_labels=True, node_color=node_color_map, node_size=1300, font_size=11, ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='crimson', ax=ax)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(node_color), vmax=max(node_color)))
    plt.colorbar(sm, ax=ax, label="Demand (m³/hr)")
    st.pyplot(fig)
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

        # Add PDF download button
        def make_pdf():
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "Pipeline Optima™ Optimization Report", ln=True)
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 10, f"Total Cost: ₹{total_cost:,.2f} per day", ln=True)
            # Node Table
            pdf.cell(0, 8, "Node Results:", ln=True)
            pdf.set_font("Arial", "", 10)
            # Make node table
            col_width = pdf.w / (len(df_nodes.columns)+2)
            for col in df_nodes.columns:
                pdf.cell(col_width, 8, str(col), border=1)
            pdf.ln()
            for i, row in df_nodes.iterrows():
                for val in row:
                    pdf.cell(col_width, 8, str(np.round(val, 2)), border=1)
                pdf.ln()
            # Edge Table
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 8, "Edge Results:", ln=True)
            pdf.set_font("Arial", "", 10)
            df_edges = pd.DataFrame(edge_out)
            for col in df_edges.columns:
                pdf.cell(col_width, 8, str(col), border=1)
            pdf.ln()
            for i, row in df_edges.iterrows():
                for val in row:
                    pdf.cell(col_width, 8, str(np.round(val, 2)), border=1)
                pdf.ln()
            # Save to BytesIO
            out = BytesIO()
            pdf.output(out)
            out.seek(0)
            return out
        if st.button("Download PDF Report"):
            pdf_out = make_pdf()
            st.download_button("Download PDF", data=pdf_out, file_name="pipeline_report.pdf")

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
        # Advanced cost plotting: show per node/edge
        st.bar_chart(df_edges.set_index("From")["Flow (m3/hr)"])
        st.write("**Power/DRA costs per station (illustrative, extend backend for true breakdown):**")
        cost_rows = []
        for n in df_nodes.itertuples():
            if getattr(n, "Is Pump"):
                cost_rows.append({
                    "Node": n.Node,
                    "Power Cost (est)": getattr(n, "No__Pumps", 0) * diesel_price * 24
                })
        if cost_rows:
            df_cost = pd.DataFrame(cost_rows)
            st.dataframe(df_cost)
            st.download_button("Download Cost Breakdown", df_cost.to_csv(index=False), file_name="cost_breakdown.csv")

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
        # Select parameter
        plot_types = [
            "Total Cost vs NOP vs DRA",
            "Head Loss vs Flow vs DRA (on any edge)",
            "Pressure vs Demand vs Speed (any pump node)",
            "Head vs Flow vs Speed (pump node)",
            "Efficiency vs Flow vs Speed (pump node)"
        ]
        selected_plot = st.selectbox("3D Plot Type", plot_types)
        if selected_plot == "Total Cost vs NOP vs DRA":
            # For one pump node, show surface
            pump_nodes = [n["Node"] for n in node_out if n["Is Pump"]]
            if not pump_nodes:
                st.info("No pump node found.")
            else:
                node = st.selectbox("Select Pump Node", pump_nodes)
                nrow = next(n for n in node_out if n["Node"] == node)
                max_pumps = int(nrow["No. Pumps"])
                dra_range = np.linspace(0, 0.5, 10)
                nop_range = np.arange(1, max_pumps+1)
                X, Y = np.meshgrid(nop_range, dra_range)
                # Dummy surface: cost proportional to NOP and DRA
                Z = X * (1 + 3*Y) * diesel_price * 24
                fig = go.Figure(data=[go.Surface(z=Z, x=nop_range, y=dra_range, colorscale='Viridis')])
                fig.update_layout(title="Total Cost vs NOP vs DRA", scene={"xaxis_title":"NOP","yaxis_title":"DRA Fraction","zaxis_title":"Cost"})
                st.plotly_chart(fig, use_container_width=True)
        elif selected_plot == "Head Loss vs Flow vs DRA (on any edge)":
            edges = [f"{e['From Node']}→{e['To Node']}" for e in st.session_state["edges"]]
            edge_map = {f"{e['From Node']}→{e['To Node']}": e for e in st.session_state["edges"]}
            edge_sel = st.selectbox("Edge", edges)
            e = edge_map[edge_sel]
            flow_range = np.linspace(0.1, 1.5*float(e["Length (km)"])*100, 25)
            dra_range = np.linspace(0, 0.5, 20)
            X, Y = np.meshgrid(flow_range, dra_range)
            D = float(e["Diameter (m)"])
            rough = float(e["Roughness (m)"])
            L = float(e["Length (km)"])
            KV = 10
            g = 9.81
            Z = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    Q = X[i, j]
                    v = Q / 3600.0 / (np.pi * (D ** 2) / 4)
                    Re = v * D / (KV * 1e-6)
                    if Re < 4000:
                        f = 64.0 / max(Re, 1e-6)
                    else:
                        arg = (rough / D / 3.7) + (5.74 / (max(Re, 1e-6) ** 0.9))
                        f = 0.25 / (np.log10(arg) ** 2) if arg > 0 else 0.0
                    Z[i, j] = f * ((L * 1000.0) / D) * ((v ** 2) / (2 * g)) * (1 - Y[i, j])
            fig = go.Figure(data=[go.Surface(z=Z, x=flow_range, y=dra_range, colorscale='Viridis')])
            fig.update_layout(title=f"Head Loss vs Flow vs DRA ({edge_sel})", scene={"xaxis_title":"Flow (m3/hr)","yaxis_title":"DRA Fraction","zaxis_title":"Head Loss (m)"})
            st.plotly_chart(fig
    , use_container_width=True)
    elif selected_plot == "Pressure vs Demand vs Speed (any pump node)":
    pump_nodes = [n["Node"] for n in node_out if n["Is Pump"]]
    if pump_nodes:
    node = st.selectbox("Pump Node", pump_nodes)
    nrow = next(n for n in node_out if n["Node"] == node)
    speed_range = np.linspace(float(nrow["Pump Min RPM"]), float(nrow["Pump DOL"]), 20)
    demand_range = np.linspace(0, float(nrow["Demand (m3/hr)"])2, 20)
    X, Y = np.meshgrid(speed_range, demand_range)
    Z = (X / max(speed_range)) * (Y / max(demand_range)) * 70 + 30
    fig = go.Figure(data=[go.Surface(z=Z, x=speed_range, y=demand_range, colorscale='Viridis')])
    fig.update_layout(title=f"Pressure vs Demand vs Speed ({node})", scene={"xaxis_title":"Speed (rpm)","yaxis_title":"Demand (m3/hr)","zaxis_title":"Pressure Head (m)"})
    st.plotly_chart(fig, use_container_width=True)
    elif selected_plot == "Head vs Flow vs Speed (pump node)":
    pump_nodes = [n["Node"] for n in node_out if n["Is Pump"]]
    if pump_nodes:
    node = st.selectbox("Pump Node", pump_nodes)
    dfh, _ = st.session_state["pump_curves"].get(node, (None, None))
    if dfh is not None:
    flows = np.linspace(dfh.iloc[:,0].min(), dfh.iloc[:,0].max(), 40)
    speeds = np.linspace(1000, 1500, 10)
    A, B, C = np.polyfit(dfh.iloc[:,0], dfh.iloc[:,1], 2)
    X, Y = np.meshgrid(flows, speeds)
    Z = (AX2 + BX + C)(Y/1500)2
    fig = go.Figure(data=[go.Surface(z=Z, x=flows, y=speeds, colorscale='Viridis')])
    fig.update_layout(title=f"Head vs Flow vs Speed ({node})", scene={"xaxis_title":"Flow (m3/hr)","yaxis_title":"RPM","zaxis_title":"Head (m)"})
    st.plotly_chart(fig, use_container_width=True)
    elif selected_plot == "Efficiency vs Flow vs Speed (pump node)":
    pump_nodes = [n["Node"] for n in node_out if n["Is Pump"]]
    if pump_nodes:
    node = st.selectbox("Pump Node", pump_nodes)
    _, dfe = st.session_state["pump_curves"].get(node, (None, None))
    if dfe is not None:
    flows = np.linspace(dfe.iloc[:,0].min(), dfe.iloc[:,0].max(), 40)
    speeds = np.linspace(1000, 1500, 10)
    P, Qc, R, S, T = np.polyfit(dfe.iloc[:,0], dfe.iloc[:,1], 4)
    X, Y = np.meshgrid(flows, speeds)
    Q_adj = X * 1500 / Y
    Z = (P*Q_adj4 + Qc*Q_adj3 + RQ_adj**2 + SQ_adj + T)
    fig = go.Figure(data=[go.Surface(z=Z, x=flows, y=speeds, colorscale='Viridis')])
    fig.update_layout(title=f"Efficiency vs Flow vs Speed ({node})", scene={"xaxis_title":"Flow (m3/hr)","yaxis_title":"RPM","zaxis_title":"Efficiency (%)"})
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown(
    """
    <div style='text-align: center; color: gray; margin-top: 2em; font-size: 0.9em;'>
    © 2025 Pipeline Optima™ Network Edition. Developed by Parichay Das. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
    )
