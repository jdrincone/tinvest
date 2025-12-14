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
from dashboard.charts import plot_heatmap, plot_stacked_bar_time_series, plot_horizontal_bar, plot_donut, plot_retention_efficiency

@st.cache_data
def calculate_advanced_metrics(transactions, clients, portfolio):
    # --- ANÁLISIS 1: PARETO & VIPs ---
    # Merge basic client info
    # Use standardized col 'client_type' if avail, else 'segment'
    segment_col = "client_type" if "client_type" in clients.columns else "segment"

    nnm_per_client = transactions.groupby('client_id').agg(
        total_deposits=('amount', lambda x: x[transactions['type']=='deposit'].sum()),
        total_withdrawals=('amount', lambda x: x[transactions['type']=='withdrawal'].sum()),
        net_new_money=('signed_amount', 'sum'),
        transaction_count=('signed_amount', 'count')
    ).reset_index()

    df_master = pd.merge(nnm_per_client, clients, on='client_id', how='left')

    # Sort & Cumsum
    df_master = df_master.sort_values(by='net_new_money', ascending=False).reset_index(drop=True)
    df_master['cumulative_nnm'] = df_master['net_new_money'].cumsum()
    total_nnm_system = df_master['net_new_money'].sum()
    df_master['percentage_contribution'] = df_master['cumulative_nnm'] / total_nnm_system

    # Top contributors (0.8 of NNM)
    top_contributors = df_master[df_master['percentage_contribution'] <= 0.8]
    vip_count = len(top_contributors)
    vip_pct = vip_count / len(df_master) * 100

    # --- ANÁLISIS 2: POTENCIAL GROWTH ---
    df_master['investment_intensity'] = df_master['net_new_money'] / df_master['income_monthly']
    # Filter infinite or NaN
    df_master.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    income_75 = df_master['income_monthly'].quantile(0.75)
    high_potential = df_master[
        (df_master['income_monthly'] > income_75) &
        (df_master['investment_intensity'] < 0.1)
    ]
    growth_count = len(high_potential)
    
    # --- ANÁLISIS 3: PRODUCTO ESTRELLA ---
    # Simplified NNM by product
    nnm_by_product = transactions.groupby('product')["signed_amount"].sum().sort_values(ascending=False).reset_index()
    star_product = nnm_by_product.iloc[0]['product']
    star_product_val = nnm_by_product.iloc[0]['signed_amount']

    # --- ANÁLISIS 4: RETENCIÓN POR SEGMENTO ---
    if segment_col in df_master.columns:
        df_segment_retention = df_master.groupby(segment_col).agg({
            'total_deposits': 'sum',
            'total_withdrawals': 'sum'
        }).reset_index()
        
        df_segment_retention['capital_retention_rate'] = (
            (df_segment_retention['total_deposits'] - df_segment_retention['total_withdrawals']) /
            df_segment_retention['total_deposits']
        ).fillna(0)
    else:
        df_segment_retention = pd.DataFrame()

    return {
        "vip_count": vip_count,
        "vip_pct": vip_pct,
        "growth_count": growth_count,
        "star_product": star_product,
        "star_product_val": star_product_val,
        "nnm_by_product": nnm_by_product, # NEW: Full DF for bar chart
        "retention_df": df_segment_retention,
        "pareto_stats": {
            "top_clients": vip_count,
            "top_pct": vip_pct,
            "rest_clients": len(df_master) - vip_count,
            "rest_pct": 100 - vip_pct
        }
    }

@st.cache_data
def compute_vintage_metrics(transactions, portfolio):
    tx = transactions.copy()
    pf = portfolio.copy()
    
    # Dates
    tx["date"] = pd.to_datetime(tx["date"])
    pf["date"] = pd.to_datetime(pf["date"])

    # First deposit date
    client_product_start = (
        tx[tx["type"] == "deposit"]
        .groupby(["client_id", "product"])["date"]
        .min()
        .reset_index()
        .rename(columns={"date": "start_date"})
    )
    
    if client_product_start.empty:
        return pd.DataFrame(), pd.DataFrame()

    cutoff_date = tx["date"].max()
    client_product_start["tenure_days"] = (cutoff_date - client_product_start["start_date"]).dt.days

    # Last balance
    last_balances = (
        pf.sort_values("date")
        .groupby(["client_id", "product"])
        .tail(1)
        .copy()
    )

    vintage_df = last_balances.merge(
        client_product_start,
        on=["client_id", "product"],
        how="inner",
    )
    vintage_df["vintage_year"] = vintage_df["start_date"].dt.year
    
    # Pivot for Heatmap
    vintage_pivot = (
        vintage_df.pivot_table(
            index="product",
            columns="vintage_year",
            values="balance",
            aggfunc="sum",
        )
        .fillna(0)
        .sort_index()
    )
    
    # Stacked Time Series: New vs Existing Count
    # Start date per client global
    client_start = tx.groupby("client_id")["date"].min().reset_index(name="registration_date")
    client_start["cohort_month"] = client_start["registration_date"].dt.to_period("M")
    
    tx_merged = tx.merge(client_start, on="client_id", how="left")
    tx_merged["tx_month"] = tx_merged["date"].dt.to_period("M")
    
    tx_merged["status"] = np.where(
        tx_merged["cohort_month"] == tx_merged["tx_month"],
        "Nuevo", "Existente"
    )
    
    # Count unique clients per month per status
    evolution_df = (
        tx_merged.groupby(["tx_month", "status"])["client_id"]
        .nunique()
        .reset_index(name="client_count")
    )
    evolution_df["tx_month"] = evolution_df["tx_month"].astype(str)
    
    return vintage_pivot, evolution_df

def render_detailed_analytics(clients: pd.DataFrame, transactions: pd.DataFrame, portfolio: pd.DataFrame):
    st.header("Analítica Detallada")
    st.markdown("Visión profunda de Clientes, Vintages y Oportunidades.")

    # 1. Advanced Metrics
    metrics = calculate_advanced_metrics(transactions, clients, portfolio)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Clientes VIP (Drivers)", f"{metrics['vip_count']} ({metrics['vip_pct']:.1f}%)", help="Generan el 80% del NNM")
    kpi2.metric("Oportunidad Growth", f"{metrics['growth_count']}", help="Altos ingresos, baja inversión")
    kpi3.metric("Producto Estrella", metrics['star_product'])
    
    # Retention info as metric or small chart? Let's use metric avg for now
    avg_retention = metrics['retention_df']['capital_retention_rate'].mean() if not metrics['retention_df'].empty else 0
    kpi4.metric("Retención Promedio", f"{avg_retention:.1%}")

    st.markdown("---")

    # --- ROW 1: Vintages (Bars) & Evolution ---
    st.subheader("Análisis de Vintages y Evolución")
    
    vintage_pivot, evolution_df = compute_vintage_metrics(transactions, portfolio)
    
    col_v1, col_v2 = st.columns(2)
    
    with col_v1:
        if not vintage_pivot.empty:
            # Prepare data for Stacked Bar: Index (Product) to Column, Melt Years
            vp_reset = vintage_pivot.reset_index().melt(id_vars="product", var_name="vintage_year", value_name="balance")
            
            fig_vintage_bar = plot_stacked_bar_time_series(
                vp_reset,
                x_col="vintage_year",
                y_col="balance",
                color_col="product",
                title="Balance Total ($) por Año de Inicio y Producto",
                y_label="Balance Total ($)"
            )
            st.plotly_chart(fig_vintage_bar, use_container_width=True)
        else:
            st.info("No hay datos suficientes para Vintages.")
            
    with col_v2:
         # Time Series Evolution
        if not evolution_df.empty:
            sorted_ev = evolution_df.sort_values("tx_month")
            fig_ev = plot_stacked_bar_time_series(
                sorted_ev,
                x_col="tx_month",
                y_col="client_count",
                color_col="status",
                title="Evolución Mensual: Clientes Nuevos vs Existentes",
                y_label="Clientes Únicos"
            )
            st.plotly_chart(fig_ev, use_container_width=True)

    st.markdown("---")

    # --- ROW 2: Motor & Retention Efficiency ---
    st.markdown("### Estrategia y Concentración")
    r2_col1, r2_col2 = st.columns(2)

    with r2_col1:
        # Motor de Crecimiento (Horizontal Bar for NNM by Product)
        nnm_prod_df = metrics["nnm_by_product"].head(5) # Top 5
        fig_motor = plot_horizontal_bar(
            nnm_prod_df,
            y_col="product",
            x_col="signed_amount",
            title="Motor de Crecimiento (NNM Neto)",
            color="#1A494C"
        )
        st.plotly_chart(fig_motor, use_container_width=True)
        
    with r2_col2:
        # Efficiency of Retention (New Chart)
        if not metrics['retention_df'].empty:
             # Use the new plot_retention_efficiency chart
             # Expected columns: segment_col (x), capital_retention_rate (y)
             # Identify segment col
             seg_col = metrics['retention_df'].columns[0]
             
             fig_eff = plot_retention_efficiency(
                 metrics['retention_df'],
                 x_col=seg_col,
                 y_col="capital_retention_rate",
                 title="Eficiencia de retención",
                 target=0.5 # As per image "Meta: 50.0%"
             )
             st.plotly_chart(fig_eff, use_container_width=True)

    # --- ROW 3: Pareto Donut & Growth Card ---
    r3_col1, r3_col2 = st.columns(2)
    
    with r3_col1:
        # Donut Chart for Pareto
        p_stats = metrics["pareto_stats"]
        fig_donut = plot_donut(
            values=[p_stats["top_clients"], p_stats["rest_clients"]],
            labels=[f"Top {p_stats['top_pct']:.1f}% (Aportan 80%)", f"Resto {p_stats['rest_pct']:.1f}%"],
            title="Concentración (Pareto)",
            colors=["#1A494C", "#34495E"]
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        
    with r3_col2:
        # Growth Opportunity Big Number
        st.markdown(
            f"""
            <div style="text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 10px; margin-top: 40px;">
                <p style="font-size: 16px; color: #1A494C; font-weight: bold;">OPORTUNIDAD<br>"Ballenas sin invertir"</p>
                <p style="font-size: 64px; color: #E74C3C; font-weight: bold; margin: 0;">{metrics['growth_count']}</p>
                <p style="color: grey;">Clientes ricos sub-invertidos</p>
                <p style="font-size: 14px; font-style: italic;">Acción: Contactar</p>
            </div>
            """, unsafe_allow_html=True
        )
