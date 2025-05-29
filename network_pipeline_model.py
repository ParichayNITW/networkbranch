import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
from math import pi
from io import BytesIO
from fpdf import FPDF
import networkx as nx
from network_pipeline_model import solve_pipeline  # Make sure backend is present!

st.set_page_config(page_title="Pipeline Optima™ Network Batch Scheduler", layout="wide")

# ---- HEADER ----
st.markdown(
    "<h1 style='text-align:center;font-size:3.4rem;font-weight:700;color:#232733;margin-bottom:0.25em;margin-top:0.01em;'>Pipeline Optima™ Network Batch Scheduler</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<div style='text-align:center;font-size:2.05rem;font-weight:700;color:#232733;margin-bottom:0.15em;margin-top:0.02em;'>MINLP Pipeline Network Optimization with Batch Scheduling</div>",
    unsafe_allow_html=True
)
st.markdown("<hr style='margin-top:0.6em; margin-bottom:1.2em; border: 1px solid #e1e5ec;'>", unsafe_allow_html=True)

# ---- SIDEBAR INPUT ----
with st.sidebar:
    st.title("Pipeline Network Input")
    st.write("All flow/demand units: m³/hr or m³ (monthly)")
    st.markdown("#### Nodes (Stations/Demand Centers)")
    nodes_df = st.data_editor(
        pd.DataFrame(columns=["ID", "Name", "Elevation (m)", "Density (kg/m³)", "Viscosity (cSt)", "Monthly Demand (m³)"]),
        num_rows="dynamic", key="nodes_df"
    )
    st.markdown("#### Edges (Pipes/Branches)")
    edges_df = st.data_editor(
        pd.DataFrame(columns=["ID", "From Node", "To Node", "Length (km)", "Diameter (m)", "Thickness (m)", "Max DR (%)", "Roughness (m)"]),
        num_rows="dynamic", key="edges_df"
    )
    st.markdown("#### Pumps")
    pumps_df = st.data_editor(
        pd.DataFrame(columns=[
            "ID", "Node ID", "Branch To", "Power Type", "No. Pumps", "Min RPM", "Max RPM",
            "SFC (Diesel)", "Grid Rate (INR/kWh)"
        ]),
        num_rows="dynamic", key="pumps_df"
    )
    st.markdown("#### Peaks (Optional)")
    peaks_df = st.data_editor(
        pd.DataFrame(columns=["Edge ID", "Location (km)", "Elevation (m)"]),
        num_rows="dynamic", key="peaks_df"
    )
    st.markdown("#### Global & Cost")
    dra_cost = st.number_input("DRA Cost (INR/L)", value=500.0, step=1.0)
    diesel_price = st.number_input("Diesel Price (INR/L)", value=70.0, step=0.5)
    grid_price = st.number_input("Grid Electricity (INR/kWh)", value=9.0, step=0.1)
    min_v = st.number_input("Min velocity (m/s)", value=0.5)
    max_v = st.number_input("Max velocity (m/s)", value=3.0)
    time_horizon = st.number_input("Scheduling Horizon (hours)", value=720, step=24)

# ---- PARSE INPUT ----
def parse_nodes(nodes_df):
    nodes = []
    for _, row in nodes_df.iterrows():
        if row['ID']:
            nodes.append({
                "id": str(row['ID']), "name": str(row['Name']),
                "elevation": float(row['Elevation (m)']),
                "density": float(row['Density (kg/m³)']),
                "viscosity": float(row['Viscosity (cSt)'])
            })
    return nodes

def parse_edges(edges_df):
    edges = []
    for _, row in edges_df.iterrows():
        if row['ID'] and row['From Node'] and row['To Node']:
            edges.append({
                "id": str(row['ID']), "from_node": str(row['From Node']), "to_node": str(row['To Node']),
                "length_km": float(row['Length (km)']), "diameter_m": float(row['Diameter (m)']),
                "thickness_m": float(row['Thickness (m)']), "max_dr": float(row['Max DR (%)']),
                "roughness": float(row['Roughness (m)']) if not pd.isna(row['Roughness (m)']) else 0.00004
            })
    return edges

def parse_pumps(pumps_df):
    pumps = []
    for _, row in pumps_df.iterrows():
        if row['ID'] and row['Node ID']:
            pumps.append({
                "id": str(row['ID']), "node_id": str(row['Node ID']),
                "branch_to": str(row['Branch To']) if row['Branch To'] else None,
                "power_type": row['Power Type'], "n_max": int(row['No. Pumps']),
                "min_rpm": int(row['Min RPM']), "max_rpm": int(row['Max RPM']),
                "sfc": float(row['SFC (Diesel)']) if not pd.isna(row['SFC (Diesel)']) else 0.0,
                "grid_rate": float(row['Grid Rate (INR/kWh)']) if not pd.isna(row['Grid Rate (INR/kWh)']) else 0.0,
                # You can add logic to let user upload/fit pump curves here if desired
                "A": -0.0002, "B": 0.25, "C": 40, "P": -1e-8, "Q": 5e-6, "R": -0.0008, "S": 0.17, "T": 55
            })
    return pumps

def parse_peaks(peaks_df):
    edge_peaks = dict()
    for _, row in peaks_df.iterrows():
        if row['Edge ID']:
            e = str(row['Edge ID'])
            pk = {"location_km": float(row['Location (km)']), "elevation_m": float(row['Elevation (m)'])}
            if e not in edge_peaks:
                edge_peaks[e] = []
            edge_peaks[e].append(pk)
    return edge_peaks

def get_demands(nodes_df):
    demands = dict()
    for _, row in nodes_df.iterrows():
        if not pd.isna(row['Monthly Demand (m³)']) and float(row['Monthly Demand (m³)']) > 0:
            demands[str(row['ID'])] = float(row['Monthly Demand (m³)'])
    return demands

# --- NETWORK VISUALIZATION ---
def visualize_network(nodes, edges):
    if not nodes or not edges: return
    G = nx.DiGraph()
    for n in nodes:
        G.add_node(n['id'], label=n['name'])
    for e in edges:
        G.add_edge(e['from_node'], e['to_node'], label=e['id'])
    pos = nx.spring_layout(G, seed=42)
    fig = go.Figure()
    # Edges
    for e in edges:
        x0, y0 = pos[e['from_node']]
        x1, y1 = pos[e['to_node']]
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode='lines', line=dict(width=2, color='black'),
            hoverinfo='none', showlegend=False
        ))
    # Nodes
    for n in nodes:
        x, y = pos[n['id']]
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode='markers+text', marker=dict(size=20, color='#1f77b4'),
            text=n['name'], textposition="bottom center", showlegend=False
        ))
    fig.update_layout(
        plot_bgcolor='#181818', paper_bgcolor='#181818',
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        title="Pipeline Network Visualization", height=400, margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

# ---- MAIN UI ----
st.header("Network Preview")
nodes = parse_nodes(nodes_df)
edges = parse_edges(edges_df)
pumps = parse_pumps(pumps_df)
peaks = parse_peaks(peaks_df)
demands = get_demands(nodes_df)
if len(nodes) >= 2 and len(edges) >= 1:
    visualize_network(nodes, edges)

# ---- RUN OPTIMIZATION ----
if st.button("🚀 Run Batch Network Optimization"):
    with st.spinner("Running MINLP solver..."):
        results = solve_pipeline(
            nodes, edges, pumps, peaks, demands, int(time_horizon),
            dra_cost, diesel_price, grid_price, min_v, max_v
        )
        st.session_state["results"] = results

# ---- OUTPUT TABS ----
if "results" in st.session_state:
    results = st.session_state["results"]
    # 1. Summary Tab
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📋 Summary", "💰 Costs", "⚙️ Performance", "🌀 System Curves",
        "🔄 Pump Scheduling", "📉 DRA Curves", "🧊 3D Analysis"
    ])
    # ---- Tab 1: Summary ----
    with tab1:
        st.subheader("Key Results Table")
        summary = []
        all_nodes = {n['id']: n['name'] for n in nodes}
        all_edges = {e['id']: (e['from_node'], e['to_node']) for e in edges}
        hours = sorted(set(k[1] for k in results["flow"]))
        for e in all_edges:
            tot_flow = sum(results["flow"][(e, t)] for t in hours)
            avg_flow = np.mean([results["flow"][(e, t)] for t in hours])
            tot_dra = sum(results["dra"][(e, t)] for t in hours)
            summary.append({
                "Edge": e, "From": all_edges[e][0], "To": all_edges[e][1],
                "Total Vol (m³)": tot_flow, "Avg Flow (m³/hr)": avg_flow,
                "Avg DRA (%)": tot_dra/len(hours)
            })
        st.dataframe(pd.DataFrame(summary))
        st.info(f"Total Optimized Cost (INR): {results['total_cost']:.2f}")

    # ---- Tab 2: Cost Breakdown ----
    with tab2:
        st.subheader("Cost Breakdown by Node/Edge")
        df_cost = []
        for e in all_edges:
            dra_sum = sum(results["dra"][(e, t)] for t in hours)
            flow_sum = sum(results["flow"][(e, t)] for t in hours)
            df_cost.append({
                "Edge": e, "Total DRA": dra_sum, "Total Flow": flow_sum,
                "DRA Cost": dra_sum * dra_cost,
            })
        st.dataframe(pd.DataFrame(df_cost))

    # ---- Tab 3: Performance ----
    with tab3:
        st.subheader("Performance (Heads, RH, etc)")
        df_perf = []
        for n in all_nodes:
            avg_rh = np.mean([results["residual_head"][(n, t)] for t in hours])
            df_perf.append({
                "Node": all_nodes[n], "Avg RH (m)": avg_rh
            })
        st.dataframe(pd.DataFrame(df_perf))

    # ---- Tab 4: System Curves ----
    with tab4:
        st.subheader("System Curves for Selected Edge")
        e_ids = list(all_edges.keys())
        selected_e = st.selectbox("Select Edge", e_ids)
        x = hours
        y = [results["flow"][(selected_e, t)] for t in x]
        y2 = [results["dra"][(selected_e, t)] for t in x]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name='Flow'))
        fig.add_trace(go.Scatter(x=x, y=y2, mode='lines+markers', name='DRA'))
        fig.update_layout(title=f"System Curves for {selected_e}", xaxis_title="Hour", yaxis_title="Value")
        st.plotly_chart(fig, use_container_width=True)

    # ---- Tab 5: Pump Scheduling ----
    with tab5:
        st.subheader("Pump Operation (ON/OFF, RPM, Num)")
        p_ids = [p['id'] for p in pumps]
        if not p_ids:
            st.warning("No pumps defined.")
        else:
            selected_p = st.selectbox("Select Pump", p_ids)
            x = hours
            y_on = [results["pump_on"][(selected_p, t)] for t in x]
            y_rpm = [results["pump_rpm"][(selected_p, t)] for t in x]
            y_n = [results["num_pumps"][(selected_p, t)] for t in x]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y_on, mode='lines+markers', name='Pump ON'))
            fig.add_trace(go.Scatter(x=x, y=y_rpm, mode='lines+markers', name='Pump RPM'))
            fig.add_trace(go.Scatter(x=x, y=y_n, mode='lines+markers', name='Num Pumps'))
            fig.update_layout(title=f"Pump Schedule: {selected_p}", xaxis_title="Hour")
            st.plotly_chart(fig, use_container_width=True)

    # ---- Tab 6: DRA Curves ----
    with tab6:
        st.subheader("DRA Dosage Across Edges")
        for e in all_edges:
            y = [results["dra"][(e, t)] for t in hours]
            st.line_chart(y, use_container_width=True)

    # ---- Tab 7: 3D Analysis ----
    with tab7:
        st.subheader("3D Visualization")
        # Example: 3D plot of Flow, DRA, and Time for selected edge
        selected_e = st.selectbox("Edge for 3D", e_ids, key="3d_edge")
        flow_3d = [results["flow"][(selected_e, t)] for t in hours]
        dra_3d = [results["dra"][(selected_e, t)] for t in hours]
        fig = go.Figure(data=[go.Scatter3d(
            x=hours, y=flow_3d, z=dra_3d, mode='lines+markers',
            marker=dict(size=3), line=dict(width=2)
        )])
        fig.update_layout(scene = dict(
            xaxis_title='Hour',
            yaxis_title='Flow (m³/hr)',
            zaxis_title='DRA (%)',
            bgcolor='#222'
        ), title=f"3D Surface: {selected_e}")
        st.plotly_chart(fig, use_container_width=True)

    # ---- Download Buttons (CSV, PDF) ----
    st.download_button(
        label="Download Results CSV",
        data=pd.DataFrame(summary).to_csv(index=False),
        file_name="results_summary.csv"
    )
    # You can also add PDF export using FPDF as per your earlier requirement

# ---- FOOTER ----
st.markdown(
    "<div style='text-align: center; color: gray; margin-top: 2em; font-size: 0.9em;'>&copy; 2025 Pipeline Optima™ v2.0. Developed by Parichay Das. All rights reserved.</div>",
    unsafe_allow_html=True
)
