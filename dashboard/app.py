import streamlit as st
import pandas as pd
import os
import sys

# Add the project root directory to sys.path so we can import utils
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import local modules directly (since they are in the same directory as this script)
from data_loader import load_data, filter_data
from utils.utils import plot_nnm_vs_aum

st.set_page_config(layout="wide", page_title="Tinvest Dashboard")

# Load data
clients, transactions, portfolio = load_data()

# --- SIDEBAR ---
with st.sidebar:
    st.header("Filtros")
    
    # 1. Product Filter
    available_products = sorted(list(set(transactions["product"].unique()) | set(portfolio["product"].unique())))
    selected_products = st.multiselect("Producto", available_products, default=available_products)
    
    # 2. Client Type Filter
    available_types = sorted(list(clients["client_type"].unique()))
    selected_types = st.multiselect("Tipo de Cliente", available_types, default=available_types)

# Filter data
filtered_clients, filtered_transactions, filtered_portfolio = filter_data(
    clients, transactions, portfolio, 
    products=selected_products, 
    client_types=selected_types
)

# --- KPI CALCULATIONS ---

# 1. Total Clients (Unique filtered clients)
total_clients = filtered_clients["client_id"].nunique()

# 2. AUM Actual (Latest month sum for filtered portfolio)
# Group by year_month to get total AUM chain
aum_monthly = (
    filtered_portfolio
    .groupby("year_month")["balance"]
    .sum()
    .reset_index(name="aum")
    .sort_values("year_month")
)

if not aum_monthly.empty:
    current_aum = aum_monthly.iloc[-1]["aum"]
    prev_aum = aum_monthly.iloc[-2]["aum"] if len(aum_monthly) > 1 else 0
    aum_delta = current_aum - prev_aum
    aum_delta_pct = (aum_delta / prev_aum) if prev_aum != 0 else 0
else:
    current_aum = 0
    prev_aum = 0
    aum_delta = 0
    aum_delta_pct = 0

# 3. NNM Total (Sum of filtered transactions signed_amount)
# We can sum all 'signed_amount' from filtered_transactions
total_nnm = filtered_transactions["signed_amount"].sum()

# Helper function for formatting large numbers
def format_currency(value):
    if abs(value) >= 1e9:
        return f"${value / 1e9:,.1f}B"
    elif abs(value) >= 1e6:
        return f"${value / 1e6:,.1f}M"
    else:
        return f"${value:,.0f}"

# --- CUSTOM CSS FOR METRIC CARDS ---
st.markdown(
    """
    <style>
    /* Style the metric container */
    div[data-testid="stMetric"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
        /* Force same height and visual consistency */
        min-height: 140px; 
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    
    /* Center text in metrics and make labels bold */
    div[data-testid="stMetricLabel"] {
        text-align: center;
        font-weight: 900 !important; /* Extra bold */
        color: #6c757d;
        font-size: 1.1rem !important;
        margin-bottom: 5px;
        width: 100%;
        display: block; /* Ensure it takes full width to center properly */
    }
    
    div[data-testid="stMetricValue"] {
        text-align: center;
        color: #1A494C; /* Global corporate color */
        font-weight: bold;
        width: 100%;
    }
    
    div[data-testid="stMetricDelta"] {
        text-align: center;
        width: 100%;
        justify-content: center; /* Flex alignment for delta arrow */
        display: flex;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- KPI DISPLAY ---
st.title("Tinvest Executive Dashboard")

col1, col2, col3 = st.columns(3)

# Requested Order: NNM, AUM, Clients

with col1:
    st.metric(label="NNM Total Histórico", value=format_currency(total_nnm))

with col2:
    st.metric(
        label="AUM Actual", 
        value=format_currency(current_aum), 
        delta=f"{format_currency(aum_delta)} ({aum_delta_pct:.1%})"
    )

with col3:
    st.metric(label="Clientes Totales", value=f"{total_clients:,}")

st.markdown("---")

# --- NNM vs AUM PLOT ---

# Prepare data for plot
# NNM Monthly
nnm_monthly = (
    filtered_transactions.groupby("year_month")["signed_amount"]
    .sum()
    .reset_index(name="nnm")
)

# Merge NNM and AUM
if not nnm_monthly.empty and not aum_monthly.empty:
    nnm_aum = pd.merge(nnm_monthly, aum_monthly, on="year_month", how="outer").fillna(0)
    nnm_aum = nnm_aum.sort_values("year_month")
    nnm_aum["year_month_dt"] = pd.to_datetime(nnm_aum["year_month"])
    
    # Generate Plot
    fig = plot_nnm_vs_aum(nnm_aum, html_filename=None)
    
    # Check if 'plot_nnm_vs_aum' returns a figure (it handles the title issue internally now or we fix it here)
    # The user issue is that subplot titles overlap with the plot frame.
    # We can adjust the top margin or move the subplot titles.
    # However, since 'plot_nnm_vs_aum' in utils.py sets fixed margins, we should try to override them here
    # or rely on a better layout update.
    
    fig.update_layout(
        margin=dict(t=120, l=60, r=40, b=40), # Increase top margin
    )
    
    # Adjust annotation positions to prevent overlap
    # We iterate through annotations to identify subplot titles and adjust them individually
    fig.layout.annotations = [
        ann.update(yshift=30) if "NNM mensual" in ann.text else 
        ann.update(yshift=-30) if "Evolución del AUM" in ann.text else 
        ann 
        for ann in fig.layout.annotations
    ]

    # Add Top 2 NNM Annotations
    top_nnm = nnm_aum.nlargest(2, "nnm")
    
    for _, row in top_nnm.iterrows():
        fig.add_annotation(
            x=row["year_month_dt"],
            y=row["nnm"],
            xref="x", # Use 'x' and 'y' for the first subplot (NNM)
            yref="y",
            text=f"Peak NNM<br>{row['year_month']}",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-40,
            bgcolor="white",
            bordercolor="black",
            font=dict(size=10),
        )
        
    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("No hay datos suficientes para mostrar el gráfico con los filtros seleccionados.")

