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

from dashboard.charts import plot_return_curve, plot_balance_sensitivity, plot_churn_profile_subplots
from dashboard.data_loader import build_client_profile, prepare_churn_profile_data

# --- USER PROVIDED LOGIC ---
def build_return_curve(
    transactions: pd.DataFrame,
    max_days: int = 120,
    candidate_days: tuple = (30, 45, 55, 60, 70, 90),
    target_return: float = 0.90,
) -> tuple[pd.DataFrame, pd.DataFrame, int, tuple, float]:
    """
    Construye la curva de retorno (latencia entre transacciones).
    """
    tx = transactions.copy()
    # Ensure date is datetime
    tx["date"] = pd.to_datetime(tx["date"])
    tx = tx.sort_values(["client_id", "date"])

    tx["prev_txn_date"] = tx.groupby("client_id")["date"].shift(1)
    tx["days_gap"] = (tx["date"] - tx["prev_txn_date"]).dt.days

    gaps = tx.dropna(subset=["days_gap"])
    total_gaps = len(gaps)
    
    # Defensive check
    if total_gaps == 0:
        return pd.DataFrame({"days": [], "return_prob": []}), pd.DataFrame(), 0, candidate_days, target_return

    days_range = range(1, max_days + 1)
    # Optimized calculation using numpy searchsorted could be faster, but keeping user logic
    return_rates = [
        (gaps["days_gap"] <= d).sum() / total_gaps for d in days_range
    ]

    curve_data = pd.DataFrame(
        {"days": list(days_range), "return_prob": return_rates}
    )

    # Tabla resumen para días candidatos
    resumen = []
    for d in candidate_days:
        # Avoid index error if d is out of range
        matched = curve_data.loc[curve_data["days"] == d, "return_prob"]
        if not matched.empty:
            prob = matched.values[0]
            resumen.append(
                {
                    "days_threshold": d,
                    "return_prob": prob,
                    "residual_prob": 1 - prob,
                }
            )
    resumen_df = pd.DataFrame(resumen)
    
    # Calculate threshold
    hits = curve_data.loc[curve_data["return_prob"] >= target_return, "days"]
    day_threshold = hits.iloc[0] if not hits.empty else max_days

    return curve_data, resumen_df, day_threshold, candidate_days, target_return

def build_balance_sensitivity(
    portfolio: pd.DataFrame,
    transactions: pd.DataFrame,
    day_threshold: int,
    saldo_target_inactivity: float = 0.90,
    bins=None,
    labels=None,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Calcula sensibilidad de inactividad por rango de saldo.
    """
    # Ensure dates
    portfolio = portfolio.copy()
    portfolio["date"] = pd.to_datetime(portfolio["date"])
    transactions = transactions.copy()
    transactions["date"] = pd.to_datetime(transactions["date"])

    latest_date = portfolio["date"].max()
    
    # Foto actual
    current_portfolio = portfolio[portfolio["date"] == latest_date]
    client_totals = (
        current_portfolio.groupby("client_id")["balance"]
        .sum()
        .reset_index(name="balance")
    )

    last_txn = (
        transactions.groupby("client_id")["date"]
        .max()
        .reset_index()
        .rename(columns={"date": "last_txn_date"})
    )

    client_status = client_totals.merge(last_txn, on="client_id", how="left")
    
    # Fill nan last_txn with start of time or something reasonable for calculation
    # Or just let days_inactive be NaN then fill
    if "last_txn_date" in client_status.columns:
        client_status["days_inactive"] = (latest_date - client_status["last_txn_date"]).dt.days
        client_status["days_inactive"] = client_status["days_inactive"].fillna(999) # Never transacted or data missing
    else:
        client_status["days_inactive"] = 999

    # Buckets de saldo
    if bins is None:
        bins = [0, 10_000, 20_000, 30_000, 40_000, 50_000, 100_000, 500_000, float("inf")]
    if labels is None:
        labels = [
            "0-10k", "10k-20k", "20k-30k", "30k-40k", 
            "40k-50k", "50k-100k", "100k-500k", ">500k"
        ]

    client_status["balance_bucket"] = pd.cut(
        client_status["balance"], bins=bins, labels=labels, right=False
    )
    
    # Convert category to object to avoid merge issues later if needed, but ordered cat is good for plotting
    # Leaving as category for sort order in charts

    bucket_counts = (
        client_status.groupby("balance_bucket", observed=False)
        .size()
        .reset_index(name="num_users")
    )

    client_status["is_inactive_at_cut"] = (
        client_status["days_inactive"] >= day_threshold
    ).astype(int)

    inactivity_rate = (
        client_status.groupby("balance_bucket", observed=False)["is_inactive_at_cut"]
        .mean()
        .reset_index(name="inactivity_rate")
    )

    sensitivity_report = bucket_counts.merge(inactivity_rate, on="balance_bucket")

    # Meta for thresholds
    bucket_meta = pd.DataFrame(
        {
            "balance_bucket": labels,
            "lower_bound": bins[:-1],
            "upper_bound": bins[1:],
        }
    )

    sensitivity_with_bounds = sensitivity_report.merge(
        bucket_meta, on="balance_bucket", how="left"
    )

    # Elegimos el primer bucket donde >= saldo_target_inactivity están inactivos
    candidates = sensitivity_with_bounds[
        sensitivity_with_bounds["inactivity_rate"] >= saldo_target_inactivity
    ].sort_values("upper_bound")

    if not candidates.empty:
        balance_threshold = candidates.iloc[0]["upper_bound"]
    else:
        balance_threshold = sensitivity_with_bounds["upper_bound"].min()

    return client_status, sensitivity_with_bounds, balance_threshold

# --- RENDER FUNCTION ---
def render_churn_analysis(clients: pd.DataFrame, transactions: pd.DataFrame, portfolio: pd.DataFrame):
    st.header("Análisis de Churn Operativo")
    st.markdown("""
        Definición dinámica de criterios de abandono basada en comportamiento histórico.
        1. **Curva de Retorno**: Define cuántos días esperar antes de considerar inactiva a una cuenta.
        2. **Sensibilidad de Saldo**: Define qué tan bajo debe ser el saldo para confirmar el abandono.
    """)
    st.markdown("---")
    
    # --- SECTION 0: Gap Analysis ---
    st.subheader("Análisis de Pausas (Gaps)")
    
    # Calculate Gaps on the fly
    tx_gap = transactions.copy().sort_values(["client_id", "date"])
    tx_gap["prev_date"] = tx_gap.groupby("client_id")["date"].shift(1)
    tx_gap["gap_days"] = (tx_gap["date"] - tx_gap["prev_date"]).dt.days
    
    # Filter valid gaps (> 0)
    valid_gaps = tx_gap["gap_days"].dropna()
    valid_gaps = valid_gaps[valid_gaps > 0]
    
    # Gap KPIs
    col_gap1, col_gap2 = st.columns(2)
    col_gap1.metric("Total de pausas (gaps) analizadas", f"{len(valid_gaps):,.0f}")
    col_gap2.metric("Tiempo promedio de silencio", f"{valid_gaps.mean():.1f} días")
    
    st.markdown("---")

    # 1. Curve Calculation
    curve_data, resumen_df, day_threshold, candidate_days, target_return = build_return_curve(transactions)
    
    # --- SECTION 1: Return Curve ---
    st.subheader(f"1. Umbral de Inactividad: {day_threshold} días")
    st.caption(f"El {target_return:.0%} de los clientes re-transacciona dentro de este periodo.")
    
    col_curve, col_stats = st.columns([2, 1])
    
    with col_curve:
        fig_curve = plot_return_curve(curve_data, candidate_days, day_threshold, target_return)
        st.plotly_chart(fig_curve, use_container_width=True)
        
    with col_stats:
        st.markdown("**Probabilidad de Retorno por Días**")
        st.dataframe(
            resumen_df.style.format({
                "return_prob": "{:.1%}",
                "residual_prob": "{:.1%}"
            }), 
            hide_index=True,
            use_container_width=True
        )

    st.markdown("---")

    # 2. Balance Sensitivity Calculation
    client_status, sensitivity_saldo, balance_threshold = build_balance_sensitivity(
        portfolio, transactions, day_threshold=day_threshold
    )

    # --- SECTION 2: Balance Sensitivity ---
    st.subheader(f"2. Umbral de Saldo: < ${balance_threshold:,.0f}")
    st.caption("Nivel de saldo donde la inactividad supera el 90% (Churn casi irreversible).")
    
    col_bal_chart, col_bal_data = st.columns([2, 1])
    
    with col_bal_chart:
        fig_bal = plot_balance_sensitivity(sensitivity_saldo)
        st.plotly_chart(fig_bal, use_container_width=True)
        
    with col_bal_data:
        st.markdown("**Inactividad por Rango de Saldo**")
        st.dataframe(
            sensitivity_saldo[["balance_bucket", "num_users", "inactivity_rate"]]
            .style.format({
                "inactivity_rate": "{:.1%}"
            }),
            hide_index=True,
            use_container_width=True
        )

    st.markdown("---")
    
    # 3. Final Churn Metrics
    st.subheader("3. Resumen de Churn Operativo")
    
    # Calculate Operational Churn
    # Flag clients who meet BOTH criteria
    # client_status already has 'days_inactive' and 'balance'
    
    # Filter only active clients? No, operational churn definition applies to the full base that has transacted
    # But usually we look at 'current clients'
    
    churn_flag_col = "is_churn_operativo"
    client_status[churn_flag_col] = (
        (client_status["days_inactive"] >= day_threshold) & 
        (client_status["balance"] <= balance_threshold)
    )
    
    total_tracked = len(client_status)
    total_churned = client_status[churn_flag_col].sum()
    churn_rate = total_churned / total_tracked if total_tracked > 0 else 0
    
    kpi1, kpi2, kpi3 = st.columns(3)
    
    kpi1.metric("Clientes Analizados", f"{total_tracked:,.0f}")
    kpi2.metric("Churn Operativo Detectado", f"{total_churned:,.0f}")
    kpi3.metric("Tasa de Churn Operativo", f"{churn_rate:.1%}")
    
    st.info(f"**Criterio Final:** Cliente inactivo por más de **{day_threshold} días** Y con saldo menor a **${balance_threshold:,.0f}**.")
    
    st.markdown("---")

    # --- ROW 4: Churn Profile ---
    st.subheader("Perfilamiento de Clientes")
    
    current_date = transactions["date"].max()
    
    # Churn Profile
    st.markdown("#### Perfilamiento de Clientes (Churn vs Retenido)")
    
    # Build profile
    # Use thresholds as per user request (or defaults)
    client_profile = build_client_profile(
        clients, transactions, portfolio, ref_date=current_date,
        day_threshold=90, balance_threshold=20000 # Example thresholds or derivation?
    )
    
    # Prepare chart data
    churn_data = prepare_churn_profile_data(client_profile, churn_col="is_churn_operativo")
    
    # Render chart
    # Render chart
    fig_churn = plot_churn_profile_subplots(churn_data, filename="churn_01_perfil_clientes.html")
    st.plotly_chart(fig_churn, use_container_width=True)
