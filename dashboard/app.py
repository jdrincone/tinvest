import streamlit as st
import pandas as pd
import sys
import os

# Add the project root directory to sys.path so we can import utils
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import modular components
from dashboard.data_loader import load_data, filter_data
from dashboard.styling import apply_custom_css
# Import view modules
from dashboard.views.overview import render_overview
from dashboard.views.peak_analysis import render_peak_analysis
from dashboard.views.detailed_analytics import render_detailed_analytics
from dashboard.views.churn_analysis import render_churn_analysis
from dashboard.views.retention_strategy import render_retention_strategy
from dashboard.views.actionable_conclusions import render_actionable_conclusions

# --- APP CONFIGURATION ---
st.set_page_config(
    page_title="Tinvest Dashboard",
    page_icon="💸",
    layout="wide",
)

# Apply global styling
apply_custom_css()

# --- DATA LOADING ---
clients, transactions, portfolio = load_data()

# --- SIDEBAR & NAVIGATION ---
st.sidebar.title("Navegación")
current_view = st.sidebar.radio(
    "Ir a:",
    options=["Resumen Ejecutivo", "Análisis de Meses Pico", "Analítica Detallada", "Análisis de Churn", "Estrategia de Retención", "Conclusiones Accionables"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.title("Filtros")

# Render Filters ONLY for Overview (Peak Analysis has its own internal filter)
if current_view == "Resumen Ejecutivo":
    # Product Filter
    try:
        available_products = sorted(portfolio["product"].unique())
    except AttributeError:
        available_products = [] # Fallback if data loading fails

    selected_products = st.sidebar.multiselect(
        "Producto",
        options=available_products,
        default=available_products
    )

    # Client Type Filter
    try:
        available_client_types = sorted(clients["client_type"].dropna().unique())
    except AttributeError:
        available_client_types = []

    selected_client_types = st.sidebar.multiselect(
        "Tipo de Cliente",
        options=available_client_types,
        default=available_client_types
    )
    
    # Filter Data
    filtered_clients, filtered_transactions, filtered_portfolio = filter_data(
        clients, transactions, portfolio, 
        products=selected_products, 
        client_types=selected_client_types
    )

# --- MAIN CONTENT ROUTING ---
st.title("Tinvest Executive Dashboard")

if current_view == "Resumen Ejecutivo":
    render_overview(filtered_clients, filtered_transactions, filtered_portfolio)
elif current_view == "Análisis de Meses Pico":
    # Peak Analysis uses raw data and filters internally by month
    render_peak_analysis(clients, transactions, portfolio)
elif current_view == "Analítica Detallada":
    render_detailed_analytics(clients, transactions, portfolio)
elif current_view == "Análisis de Churn":
    render_churn_analysis(clients, transactions, portfolio)
elif current_view == "Estrategia de Retención":
    render_retention_strategy(clients, transactions, portfolio)
elif current_view == "Conclusiones Accionables":
    render_actionable_conclusions(clients, transactions, portfolio)
