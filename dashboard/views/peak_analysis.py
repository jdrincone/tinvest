import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import local modules
from dashboard.styling import format_currency
from dashboard.charts import plot_horizontal_bar, plot_vertical_bar

def render_peak_analysis(clients: pd.DataFrame, transactions: pd.DataFrame, portfolio: pd.DataFrame):
    """
    Renders the Peak Analysis view for detailed monthly insights.
    """
    st.header("Análisis de Meses Pico")
    st.markdown("Detalle de los drivers de *Net New Money* (NNM) para meses específicos.")

    # 1. Identify Peak Months (Top 10 for selection)
    nnm_monthly = (
        transactions
        .assign(year_month=lambda d: d["date"].dt.to_period("M").astype(str))
        .groupby("year_month")["signed_amount"]
        .sum()
        .reset_index(name="nnm")
        .sort_values("nnm", ascending=False)
    )
    
    top_months = nnm_monthly["year_month"].head(10).tolist()
    
    # Selection Widget (Filter by Month)
    col_sel, col_info = st.columns([1, 2])
    with col_sel:
        peak_month = st.selectbox("Seleccionar Mes Pico", options=top_months, index=0)
    
    # --- LOGIC: Based on user provided 'build_nnm_peak_summary' ---
    
    # Filter transactions for the selected month
    tx_m = transactions[transactions["date"].dt.to_period("M").astype(str) == peak_month].copy()
    
    # Merge Client Info - DETAILS ONLY
    # 'client_type' is already in transactions from load_data
    # We only need registration_date for the "New vs Existing" logic
    cols_to_merge = ["client_id", "registration_date"]
    
    tx_m = tx_m.merge(
        clients[cols_to_merge],
        on="client_id",
        how="left",
    )

    # Calculate Total NNM
    total_nnm = tx_m["signed_amount"].sum()
    
    # --- NNM by Product ---
    nnm_product = (
        tx_m.groupby("product")["signed_amount"]
        .sum()
        .reset_index()
        .sort_values("signed_amount", ascending=False)
    )
    
    # --- NNM by Segment (Client Type) ---
    # We use 'client_type' because data_loader standardizes it
    segment_col = "client_type" if "client_type" in tx_m.columns else "segment"
    
    nnm_segment = (
        tx_m.groupby(segment_col)["signed_amount"]
        .sum()
        .reset_index()
        .sort_values("signed_amount", ascending=False)
    )

    # --- New vs Existing Clients ---
    clients_cohort = clients.copy()
    clients_cohort["cohort_month"] = (
        clients_cohort["registration_date"].dt.to_period("M").astype(str)
    )

    tx_m = tx_m.merge(
        clients_cohort[["client_id", "cohort_month"]],
        on="client_id",
        how="left",
        suffixes=("", "_cohort"),
    )

    tx_m["client_type_cohort"] = np.where(
        tx_m["cohort_month"] == peak_month,
        "Nuevo en ese mes",
        "Existente",
    )
    
    nnm_new_existing = (
        tx_m.groupby("client_type_cohort")["signed_amount"]
        .sum()
        .reset_index()
        .sort_values("signed_amount", ascending=False)
    )

    # --- Pareto Calculation ---
    nnm_clients = (
        tx_m.groupby("client_id")["signed_amount"]
        .sum()
        .reset_index()
        .sort_values("signed_amount", ascending=False)
    )
    nnm_clients["cum_nnm"] = nnm_clients["signed_amount"].cumsum()
    nnm_clients["cum_prop"] = nnm_clients["cum_nnm"] / total_nnm
    nnm_clients["rank_pct"] = (np.arange(1, len(nnm_clients) + 1) / len(nnm_clients)) * 100
    
    # Check 80% threshold
    try:
        idx_80 = (nnm_clients["cum_prop"] >= 0.80).idxmax()
        pct_clients_80 = nnm_clients.loc[idx_80, "rank_pct"]
    except:
        pct_clients_80 = 100 # Fallback

    # --- UI RENDERING ---
    
    # KPIs Row
    st.markdown("#### Resumen del Mes")
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("NNM Total Mes", format_currency(total_nnm))
    kpi2.metric("Concentración (Pareto 80%)", f"{pct_clients_80:.1f}% Clientes")
    kpi3.metric("Transacciones Totales", f"{len(tx_m):,}")
    
    st.markdown("---")
    
    # Charts Row
    # Charts Row
    st.subheader("Desglose de Drivers")
    chart_col1, chart_col2, chart_col3 = st.columns(3)
    
    with chart_col1:
        fig_prod = plot_horizontal_bar(
            nnm_product.head(10), # Top 10
            y_col="product", 
            x_col="signed_amount", 
            title=f"Top Productos ({peak_month})"
        )
        st.plotly_chart(fig_prod, use_container_width=True)
        
    with chart_col2:
        fig_seg = plot_horizontal_bar(
            nnm_segment, 
            y_col=segment_col, 
            x_col="signed_amount", 
            title=f"Por Segmento ({peak_month})"
        )
        st.plotly_chart(fig_seg, use_container_width=True)

    with chart_col3:
        fig_new = plot_vertical_bar(
            nnm_new_existing, 
            x_col="client_type_cohort", 
            y_col="signed_amount", 
            title="Nuevos vs Existentes"
        )
        st.plotly_chart(fig_new, use_container_width=True)

    # Top Clients Table
    st.subheader("Top Clientes (Pareto Drivers)")
    st.markdown("Estos clientes generaron el mayor volumen de NNM en el mes seleccionado.")
    
    top_clients_display = nnm_clients.head(20).copy()
    top_clients_display["signed_amount"] = top_clients_display["signed_amount"].apply(lambda x: f"${x:,.0f}")
    top_clients_display["cum_prop"] = top_clients_display["cum_prop"].apply(lambda x: f"{x:.1%}")
    top_clients_display.rename(columns={
        "client_id": "Cliente ID",
        "signed_amount": "NNM Total ($)",
        "cum_prop": "% Acumulado del Total"
    }, inplace=True)
    
    st.dataframe(
        top_clients_display[["Cliente ID", "NNM Total ($)", "% Acumulado del Total"]],
        use_container_width=True,
        hide_index=True
    )
    
    # Footer
    st.markdown("---")
    st.caption("Construido por Juan David Rincón")
