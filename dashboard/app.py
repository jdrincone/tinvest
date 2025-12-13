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
from dashboard.views.overview import render_overview

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
st.sidebar.title("Filtros")

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

# --- FILTERING LOGIC ---
filtered_clients, filtered_transactions, filtered_portfolio = filter_data(
    clients, transactions, portfolio, 
    products=selected_products, 
    client_types=selected_client_types
)

# --- MAIN CONTENT ---
st.title("Tinvest Executive Dashboard")

# Router Logic (Scalable Design)
# Currently only one view, but structure allows easy addition:
# view_selection = st.sidebar.radio("Navegación", ["Resumen Ejecutivo", "Detalle Clientes"])
# if view_selection == "Resumen Ejecutivo": ...

render_overview(filtered_clients, filtered_transactions, filtered_portfolio)
