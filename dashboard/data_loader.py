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


def build_client_profile(
    clients: pd.DataFrame,
    transactions: pd.DataFrame,
    portfolio: pd.DataFrame,
    ref_date: pd.Timestamp,
    *,
    day_threshold: int | None = None,
    balance_threshold: float | None = None,
) -> pd.DataFrame:
    """
    Construye el perfil de clientes a nivel fila (1 fila = 1 cliente).

    Incluye:
      - Tenure (días / meses)
      - Depósitos, retiros, NNM
      - Número de productos
      - Días desde la última transacción, ticket, frecuencia
      - Trayectoria de balance (primer / último balance)
      - Flags de churn:
            is_churn_90d        -> 90 días sin operar + saldo ≈ 0
            is_churn_operativo  -> regla basada en datos (si se pasan thresholds)
    """

    # Trabajamos siempre indexando por client_id para que todo alinee bien
    clients_fe = clients.set_index("client_id").copy()

    # ==========================
    # 1. Tenure
    # ==========================
    clients_fe["tenure_days"] = (ref_date - clients_fe["registration_date"]).dt.days
    clients_fe["tenure_months"] = (
        (clients_fe["tenure_days"] / 30)
        .clip(lower=1)
        .astype(int)
    )

    # ==========================
    # 2. Depósitos, retiros, NNM
    # ==========================
    dep = (
        transactions.query("type == 'deposit'")
        .groupby("client_id")["amount"]
        .sum()
    )
    wd = (
        transactions.query("type == 'withdrawal'")
        .groupby("client_id")["amount"]
        .sum()
    )

    clients_fe["total_deposits"] = dep
    clients_fe["total_withdrawals"] = wd
    clients_fe[["total_deposits", "total_withdrawals"]] = (
        clients_fe[["total_deposits", "total_withdrawals"]].fillna(0)
    )

    # NNM acumulado por cliente
    clients_fe["nnm_by_client"] = (
        clients_fe["total_deposits"] - clients_fe["total_withdrawals"]
    )

    # ==========================
    # 3. Número de productos
    # ==========================
    all_products = pd.concat(
        [
            transactions[["client_id", "product"]],
            portfolio[["client_id", "product"]],
        ],
        ignore_index=True,
    ).drop_duplicates()

    num_products = (
        all_products.groupby("client_id")["product"]
        .nunique()
    )

    clients_fe["num_products"] = num_products
    clients_fe["num_products"] = (
        clients_fe["num_products"].fillna(0).astype(int)
    )

    # ==========================
    # 4. Actividad: días desde última txn, ticket, frecuencia
    # ==========================
    last_txn_date = (
        transactions.groupby("client_id")["date"]
        .max()
    )

    clients_fe["last_txn_date"] = last_txn_date
    clients_fe["days_since_last_transaction"] = (
        ref_date - clients_fe["last_txn_date"]
    ).dt.days

    # Si nunca ha transaccionado, usamos días desde registro
    clients_fe["days_since_last_transaction"] = clients_fe[
        "days_since_last_transaction"
    ].fillna(clients_fe["tenure_days"])

    # Ticket promedio
    avg_txn_size = (
        transactions.groupby("client_id")["amount"]
        .mean()
    )
    clients_fe["avg_transaction_size"] = avg_txn_size.fillna(0)

    # Conteo y frecuencia
    txn_counts = (
        transactions.groupby("client_id")["date"]
        .count()
    )

    clients_fe["transaction_count"] = txn_counts.fillna(0).astype(int)
    clients_fe["transaction_frequency"] = (
        clients_fe["transaction_count"]
        / clients_fe["tenure_months"].replace(0, np.nan)
    ).fillna(0)

    # ==========================
    # 5. Trayectoria de balance
    # ==========================
    # AUM por cliente-fecha
    aum_client_date = (
        portfolio.groupby(["client_id", "date"])["balance"]
        .sum()
        .reset_index()
    )

    # Primer y último balance en la historia del cliente
    first_balance = (
        aum_client_date.sort_values("date")
        .groupby("client_id")["balance"]
        .first()
    )
    last_balance_hist = (
        aum_client_date.sort_values("date")
        .groupby("client_id")["balance"]
        .last()
    )

    # Balance a la fecha de referencia (foto actual)
    last_balance_snapshot = (
        aum_client_date[aum_client_date["date"] == ref_date]
        .groupby("client_id")["balance"]
        .sum()
    )

    clients_fe["first_balance"] = first_balance
    clients_fe["last_balance_hist"] = last_balance_hist
    clients_fe["last_balance"] = last_balance_snapshot

    clients_fe[["first_balance", "last_balance_hist", "last_balance"]] = (
        clients_fe[["first_balance", "last_balance_hist", "last_balance"]]
        .fillna(0)
    )

    # Tendencia de balance (crece o cae el saldo en el tiempo)
    clients_fe["balance_trend"] = (
        clients_fe["last_balance_hist"] - clients_fe["first_balance"]
    )

    # ==========================
    # 6. Flags de churn
    # ==========================

    # Regla clásica: 90 días + saldo ~ 0
    clients_fe["is_churn_90d"] = (
        (clients_fe["days_since_last_transaction"] >= 90)
        & (clients_fe["last_balance"].abs() < 1e-3)
    )

    # Regla data-driven (si pasas thresholds calculados antes)
    if (day_threshold is not None) and (balance_threshold is not None):
        clients_fe["is_churn_operativo"] = (
            (clients_fe["days_since_last_transaction"] >= day_threshold)
            & (clients_fe["last_balance"] <= balance_threshold)
        )
    else:
        clients_fe["is_churn_operativo"] = np.nan

    # Volvemos a tener client_id como columna normal
    clients_fe = clients_fe.reset_index()

    return clients_fe

def prepare_churn_profile_data(
    clients_fe: pd.DataFrame,
    churn_col: str = "is_churn_operativo",
) -> dict:
    """
    Prepara la información necesaria para perfilar clientes churn vs retenidos.

    Devuelve un diccionario con:
      - df: tabla a nivel cliente con columna 'churn_label' ('Churn' / 'Retenido')
      - churn_by_segment: tasa de churn por segmento
      - summary_numeric: stats de ingreso, edad, riesgo por churn
    """
    df = clients_fe.copy()

    # Nos aseguramos de que la columna de churn exista
    if churn_col not in df.columns:
        raise ValueError(f"Columna de churn '{churn_col}' no existe en clients_fe")

    # Etiqueta amigable para los gráficos
    df["churn_label"] = np.where(df[churn_col].astype(bool), "Churn", "Retenido")

    # Tasa de churn por segmento
    # Detect segment column (could be 'segment' or 'client_type')
    seg_col = "segment" if "segment" in df.columns else "client_type"
    if seg_col not in df.columns:
        # Fallback if neither exists, though unlikely given load_data logic
        df["segment_fallback"] = "All"
        seg_col = "segment_fallback"

    churn_by_segment = (
        df.groupby(seg_col)[churn_col]
        .agg(churn_rate="mean", n_clients="size")
        .reset_index()
    )

    # Resumen numérico básico para storytelling
    summary_numeric = (
        df.groupby("churn_label")[["income_monthly", "age", "risk_score"]]
        .agg(["mean", "median", "std", "min", "max"])
    )

    return {
        "df": df,
        "churn_by_segment": churn_by_segment,
        "summary_numeric": summary_numeric,
    }
