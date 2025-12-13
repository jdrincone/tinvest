import streamlit as st
import pandas as pd
import sys
import os

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../..")) # dashboard/views -> root
if project_root not in sys.path:
    sys.path.append(project_root)

# Import local modules
from dashboard.styling import format_currency
from dashboard.charts import get_nnm_aum_plot

def render_overview(clients: pd.DataFrame, transactions: pd.DataFrame, portfolio: pd.DataFrame):
    """
    Renders the Executive Overview page.
    This function expects already filtered data.
    """
    
    # --- KPI CALCULATIONS ---
    # 1. Total Clients
    total_clients = len(clients)

    # 2. Current AUM
    latest_date = portfolio["date"].max()
    current_aum = portfolio[portfolio["date"] == latest_date]["balance"].sum()

    # Previous Month AUM for Delta
    previous_month_date = latest_date - pd.DateOffset(months=1)
    # Handle month-end alignment if needed, but for now simple offset
    # A cleaner way is to use 'year_month'
    current_ym = latest_date.strftime("%Y-%m")
    
    # Calculate AUM by month (globally for this view)
    aum_monthly = (
        portfolio
        .assign(year_month=lambda d: d["date"].dt.to_period("M").astype(str))
        .groupby("year_month")["balance"]
        .sum()
        .reset_index(name="aum")
    )
    
    # Get previous month AUM from the aggregated df
    # Sort just in case
    aum_monthly = aum_monthly.sort_values("year_month")
    
    # Find current index
    try:
        current_idx = aum_monthly[aum_monthly["year_month"] == current_ym].index[0]
        if current_idx > 0:
            prev_aum = aum_monthly.iloc[current_idx - 1]["aum"]
        else:
            prev_aum = 0
    except IndexError:
        # Fallback Logic if date matching fails
        prev_aum = 0

    aum_delta = current_aum - prev_aum
    aum_delta_pct = (aum_delta / prev_aum) if prev_aum != 0 else 0

    # 3. Total NNM (Signed Amount Sum)
    total_nnm = transactions["signed_amount"].sum()

    # --- KPI DISPLAY ---
    col1, col2, col3 = st.columns(3)

    # Order: NNM, AUM, Clients
    
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

    # --- PLOTTING ---
    # Prepare data for plotting (NNM monthly and AUM monthly)
    # NNM Monthly
    nnm_monthly = (
        transactions
        .assign(year_month=lambda d: d["date"].dt.to_period("M").astype(str))
        .groupby("year_month")["signed_amount"]
        .sum()
        .reset_index(name="nnm")
    )
    
    # AUM Monthly is already calculated above as 'aum_monthly'
    
    # Merge for the plot
    nnm_aum = (
        nnm_monthly
        .merge(aum_monthly, on="year_month", how="inner")
        .sort_values("year_month")
        .reset_index(drop=True)
    )
    nnm_aum["year_month_dt"] = pd.to_datetime(nnm_aum["year_month"])

    # Generate Chart using the Controller
    fig = get_nnm_aum_plot(nnm_aum)

    st.plotly_chart(fig, use_container_width=True)
