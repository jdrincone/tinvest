import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st

DATA_DIR = Path("data")

@st.cache_data
def load_data():
    """
    Loads clients, transactions, and portfolio data.
    Merges client information into transactions and portfolio for filtering.
    """
    clients = pd.read_csv(DATA_DIR / "clients.csv", parse_dates=["registration_date"])
    transactions = pd.read_csv(DATA_DIR / "transactions.csv", parse_dates=["date"])
    portfolio = pd.read_csv(DATA_DIR / "portfolio_balance.csv", parse_dates=["date"])

    # Clean object columns
    for df in [clients, transactions, portfolio]:
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].str.strip()

    # Merge client info (segment) into transactions and portfolio
    # Assuming 'segment' is what we want to filter by "Client Type"
    transactions = transactions.merge(clients[["client_id", "segment", "client_type"] if "client_type" in clients.columns else ["client_id", "segment"]], on="client_id", how="left")
    portfolio = portfolio.merge(clients[["client_id", "segment", "client_type"] if "client_type" in clients.columns else ["client_id", "segment"]], on="client_id", how="left")

    # Rename segment to client_type if client_type doesn't exist, for consistency
    if "client_type" not in clients.columns and "segment" in clients.columns:
        transactions.rename(columns={"segment": "client_type"}, inplace=True)
        portfolio.rename(columns={"segment": "client_type"}, inplace=True)
        clients.rename(columns={"segment": "client_type"}, inplace=True)

    # Pre-calculate signed amount for NNM
    transactions["signed_amount"] = np.where(
        transactions["type"] == "deposit",
        transactions["amount"],
        -transactions["amount"],
    )

    # Add year_month for aggregations
    transactions["year_month"] = transactions["date"].dt.to_period("M").astype(str)
    portfolio["year_month"] = portfolio["date"].dt.to_period("M").astype(str)
    
    # Ensure year_month_dt is available for plotting (first day of month)
    transactions["year_month_dt"] = pd.to_datetime(transactions["year_month"])
    
    return clients, transactions, portfolio

def filter_data(clients, transactions, portfolio, products=None, client_types=None):
    """
    Filters datasets based on selected products and client types.
    """
    filtered_clients = clients.copy()
    filtered_transactions = transactions.copy()
    filtered_portfolio = portfolio.copy()

    if client_types:
        filtered_clients = filtered_clients[filtered_clients["client_type"].isin(client_types)]
        filtered_transactions = filtered_transactions[filtered_transactions["client_type"].isin(client_types)]
        filtered_portfolio = filtered_portfolio[filtered_portfolio["client_type"].isin(client_types)]

    if products:
        # Clients don't have products directly, so we filter transactions and portfolio
        filtered_transactions = filtered_transactions[filtered_transactions["product"].isin(products)]
        filtered_portfolio = filtered_portfolio[filtered_portfolio["product"].isin(products)]
        
        # Optionally filter clients to only those who have transactions/balance in the selected products?
        # For now, let's keep clients filtered only by client_type to show "Total Clients" of that segment.
        pass

    return filtered_clients, filtered_transactions, filtered_portfolio
