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

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from utils.utils import plot_feature_importance_plotly, plot_risk_value_matrix_plotly


def build_retention_dataset(
    clients: pd.DataFrame,
    transactions: pd.DataFrame,
    portfolio: pd.DataFrame,
    factor_rentabilidad: float = 0.015,
    churn_balance_threshold: float = 50_000,
) -> tuple[pd.DataFrame, pd.Timestamp, list[str]]:
    """
    Construye el dataset de clientes con:
      - avg_aum, balance actual
      - days_since_txn, txn_count, tenure_months
      - estimated_annual_value (proxy CLV)
      - is_churn (target sencillo: saldo < threshold)
    Devuelve:
      df_features, ref_date, feature_cols
    """

    df_clients = clients.copy()
    tx = transactions.copy()
    port = portfolio.copy()

    # Asegurar tipos de fecha
    port["date"] = pd.to_datetime(port["date"])
    tx["date"] = pd.to_datetime(tx["date"])
    df_clients["registration_date"] = pd.to_datetime(df_clients["registration_date"])

    # Fecha de corte (última observación en portfolio)
    ref_date = port["date"].max()

    # 1) Saldo promedio administrado (AUM) por cliente
    avg_balance = (
        port.groupby("client_id")["balance"]
        .mean()
        .reset_index(name="avg_aum")
    )

    # 2) Recencia y actividad
    last_txn = (
        tx.groupby("client_id")["date"]
        .max()
        .reset_index(name="last_txn_date")
    )
    txn_count = (
        tx.groupby("client_id")
        .size()
        .reset_index(name="txn_count")
    )

    # 3) Saldo actual a la fecha de corte (sumar por cliente si hay múltiples productos)
    last_balance_df = (
        port[port["date"] == ref_date]
        .groupby("client_id")["balance"]
        .sum()
        .reset_index(name="balance")
    )

    # 4) Merge maestro
    df = df_clients.merge(avg_balance, on="client_id", how="left")
    df = df.merge(last_balance_df, on="client_id", how="left")
    df = df.merge(last_txn, on="client_id", how="left")
    df = df.merge(txn_count, on="client_id", how="left")
    
    # Eliminar duplicados por si acaso (mantener la primera ocurrencia)
    df = df.drop_duplicates(subset=["client_id"], keep="first")

    # Rellenos numéricos
    df["avg_aum"] = df["avg_aum"].fillna(0)
    df["balance"] = df["balance"].fillna(0)
    df["txn_count"] = df["txn_count"].fillna(0)

    # 5) Features derivados
    df["days_since_txn"] = (ref_date - df["last_txn_date"]).dt.days
    # Si nunca transaccionó, lo tratamos como muy inactivo
    df["days_since_txn"] = df["days_since_txn"].fillna(365)

    df["tenure_months"] = (
        (ref_date - df["registration_date"]).dt.days / 30
    ).astype(int)

    # Proxy de valor anual del cliente
    df["estimated_annual_value"] = df["avg_aum"] * factor_rentabilidad

    # Target de churn (definición sencilla: saldo actual bajo)
    df["is_churn"] = (df["balance"] < churn_balance_threshold).astype(int)

    # Codificar segmento (manejar tanto "segment" como "client_type")
    # Primero normalizar: si existe "client_type" pero no "segment", crear "segment"
    if "client_type" in df.columns and "segment" not in df.columns:
        df["segment"] = df["client_type"]
    
    if "segment" in df.columns:
        df["segment"] = (
            df["segment"]
            .replace({"retail": 0, "premium": 1, "Retail": 0, "Premium": 1})
            .fillna(0)
            .astype(int)
        )
    else:
        # Si no existe segment, crear columna con 0 (retail por defecto)
        df["segment"] = 0

    # Asegurar que "age" existe, si no, crear con 0
    if "age" not in df.columns:
        df["age"] = 0

    # Rellenar NaNs restantes
    df = df.fillna(0)

    # Columnas de features (sin duplicados)
    feature_cols = [
        "income_monthly",
        "risk_score",
        "avg_aum",
        "days_since_txn",
        "txn_count",
        "tenure_months",
        "age",
        "segment",
    ]

    # (opcional) chequeo de columnas que falten
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en df para features: {missing}")

    return df, ref_date, feature_cols


def train_churn_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "is_churn",
) -> RandomForestClassifier:
    """
    Entrena un RandomForest sencillo para predecir churn.
    """
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    y_prob = clf.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)
    
    # Imprimir para logging (aunque en Streamlit no se verá en consola, lo mantenemos)
    # En Streamlit usamos st.write o lo mostramos en la UI
    
    return clf, roc_auc


def score_and_define_strategy(
    df: pd.DataFrame,
    clf: RandomForestClassifier,
    feature_cols: list[str],
    risk_threshold: float = 0.60,
    value_quantile: float = 0.75,
) -> tuple[pd.DataFrame, float, float]:
    """
    - Calcula churn_probability para cada cliente.
    - Crea la columna strategy_segment (4 cuadrantes).
    Devuelve df_scored, risk_threshold, value_threshold.
    """
    df_scored = df.copy()

    df_scored["churn_probability"] = clf.predict_proba(df_scored[feature_cols])[:, 1]
    value_threshold = df_scored["estimated_annual_value"].quantile(value_quantile)

    def define_action(row):
        high_risk = row["churn_probability"] > risk_threshold
        high_value = row["estimated_annual_value"] > value_threshold

        if high_risk and high_value:
            return "1. PRIORIDAD CRÍTICA (Salvar)"
        elif (not high_risk) and high_value:
            return "2. FIDELIZAR (Cuidar)"
        elif high_risk and (not high_value):
            return "3. DEJAR IR (No rentable)"
        else:
            return "4. BAJA PRIORIDAD"

    df_scored["strategy_segment"] = df_scored.apply(define_action, axis=1)

    # Resumen de acción (para logging, en Streamlit lo mostramos en la UI)
    # print("\n--- RESUMEN DE ACCIÓN ---")
    # print(df_scored["strategy_segment"].value_counts())

    return df_scored, risk_threshold, value_threshold


def render_retention_strategy(clients: pd.DataFrame, transactions: pd.DataFrame, portfolio: pd.DataFrame):
    """
    Renderiza la vista de estrategia de retención con modelo de churn.
    """
    st.header("Estrategia de Retención - Modelo Predictivo de Churn")
    st.markdown("""
        Modelo de machine learning para predecir la probabilidad de churn de clientes
        y definir estrategias de retención basadas en riesgo y valor del cliente.
    """)
    st.markdown("---")
    
    # --- SIDEBAR FILTERS ---
    # Parámetros fijos del modelo (no configurables)
    factor_rentabilidad = 0.015
    churn_balance_threshold = 50_000
    
    st.sidebar.subheader("Parámetros de Segmentación")
    
    # Umbral de riesgo
    risk_threshold = st.sidebar.slider(
        "Umbral de Riesgo (Probabilidad de Churn)",
        min_value=0.0,
        max_value=1.0,
        value=0.60,
        step=0.05,
        help="Probabilidad de churn mínima para considerar un cliente de alto riesgo"
    )
    
    # Cuantil de valor
    value_quantile = st.sidebar.slider(
        "Cuantil de Valor",
        min_value=0.0,
        max_value=1.0,
        value=0.75,
        step=0.05,
        help="Cuantil del valor anual estimado para definir clientes de alto valor (ej: 0.75 = top 25%)"
    )
    
    # --- MAIN CONTENT ---
    
    # Mostrar parámetros actuales con explicaciones
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Umbral Riesgo",
            f"{risk_threshold:.1%}",
            help="Probabilidad mínima de churn para considerar a un cliente como de alto riesgo. Clientes con probabilidad de churn superior a este umbral requieren atención prioritaria."
        )
    
    with col2:
        st.metric(
            "Cuantil Valor",
            f"{value_quantile:.0%}",
            help="Percentil del valor anual estimado del cliente. Define qué clientes se consideran de alto valor. Por ejemplo, 75% significa que los clientes en el top 25% de valor son considerados de alto valor."
        )
    
    with col3:
        st.metric(
            "Factor Rentabilidad",
            "1.5%",
            help="Factor utilizado para calcular el valor anual estimado del cliente. Se calcula como: Valor Anual = AUM Promedio × Factor de Rentabilidad (1.5%). Representa el retorno anual esperado por cliente."
        )
    
    st.markdown("---")
    
    # Procesar modelo
    with st.spinner("Construyendo dataset y entrenando modelo..."):
        # 1) Dataset de features + target
        df, ref_date, feature_cols = build_retention_dataset(
            clients=clients,
            transactions=transactions,
            portfolio=portfolio,
            factor_rentabilidad=factor_rentabilidad,
            churn_balance_threshold=churn_balance_threshold,
        )
        
        # 2) Modelo
        clf, roc_auc = train_churn_model(df, feature_cols, target_col="is_churn")
        
        # 3) Scoring + estrategia
        df_scored, risk_threshold_used, value_threshold = score_and_define_strategy(
            df,
            clf,
            feature_cols,
            risk_threshold=risk_threshold,
            value_quantile=value_quantile,
        )
    
    # Mostrar métricas del modelo
    st.subheader("Métricas del Modelo")
    
    col_roc, col_date, col_total = st.columns(3)
    with col_roc:
        st.metric("ROC-AUC Score", f"{roc_auc:.3f}")
        with st.expander("❓ ¿Qué es el ROC-AUC Score?"):
            st.caption("""
            Medida de la capacidad del modelo para distinguir entre clientes que harán churn y los que no. 
            Un valor cercano a 1.0 indica un modelo muy preciso. Valores por encima de 0.7 se consideran buenos.
            """)
    with col_date:
        st.metric("Fecha de Corte", ref_date.strftime("%Y-%m-%d"))
        with st.expander("❓ ¿Qué es la Fecha de Corte?"):
            st.caption("""
            Última fecha con información disponible en el portfolio. 
            Todas las métricas y predicciones se calculan hasta esta fecha.
            """)
    with col_total:
        total_clients_original = clients["client_id"].nunique()
        st.metric("Total Clientes (Base)", f"{total_clients_original:,}")
        with st.expander("❓ ¿Qué es el Total de Clientes?"):
            st.caption("""
            Número total de clientes únicos en la base de datos. 
            Representa el universo completo de clientes disponibles para análisis.
            """)
    
    st.markdown("---")
    
    # Gráficos
    st.subheader("Análisis Visual")
    
    # Gráfico de importancia de features
    st.markdown("#### Importancia de Variables del Modelo")
    st.caption("¿Qué variables son más importantes para predecir el churn?")
    fig_importance = plot_feature_importance_plotly(clf, feature_cols)
    st.plotly_chart(fig_importance, use_container_width=True)
    
    st.markdown("---")
    
    # Matriz riesgo vs valor
    st.markdown("#### Matriz de Estrategia: Riesgo vs Valor")
    st.caption(f"Clasificación de clientes según probabilidad de churn y valor anual estimado. Umbral de riesgo: {risk_threshold:.1%}, Umbral de valor: ${value_threshold:,.0f}")
    fig_matrix = plot_risk_value_matrix_plotly(df_scored, risk_threshold_used, value_threshold)
    st.plotly_chart(fig_matrix, use_container_width=True)
    
    st.markdown("---")
    
    # Tabla detallada (opcional, con filtros)
    st.subheader("Detalle de Clientes por Estrategia")
    
    # Obtener strategy_counts para el selectbox
    strategy_counts = df_scored["strategy_segment"].value_counts()
    
    selected_strategy = st.selectbox(
        "Filtrar por Estrategia",
        options=["Todas"] + list(strategy_counts.index),
        index=0
    )
    
    # Columnas a mostrar: todas las features + información relevante
    columns_to_show = [
        "client_id",
        "strategy_segment",
        "churn_probability",
        "estimated_annual_value",
        "income_monthly",
        "risk_score",
        "age",
        "segment",
        "avg_aum",
        "balance",
        "days_since_txn",
        "txn_count",
        "tenure_months",
        "is_churn"
    ]
    
    # Filtrar solo las columnas que existen en el DataFrame
    available_columns = [col for col in columns_to_show if col in df_scored.columns]
    
    if selected_strategy == "Todas":
        display_df = df_scored[available_columns].copy()
    else:
        display_df = df_scored[df_scored["strategy_segment"] == selected_strategy][available_columns].copy()
    
    # Formatear columnas para visualización
    if "churn_probability" in display_df.columns:
        display_df["churn_probability"] = display_df["churn_probability"].apply(lambda x: f"{x:.1%}")
    if "estimated_annual_value" in display_df.columns:
        display_df["estimated_annual_value"] = display_df["estimated_annual_value"].apply(lambda x: f"${x:,.0f}")
    if "avg_aum" in display_df.columns:
        display_df["avg_aum"] = display_df["avg_aum"].apply(lambda x: f"${x:,.0f}")
    if "balance" in display_df.columns:
        display_df["balance"] = display_df["balance"].apply(lambda x: f"${x:,.0f}")
    if "income_monthly" in display_df.columns:
        display_df["income_monthly"] = display_df["income_monthly"].apply(lambda x: f"${x:,.0f}")
    if "segment" in display_df.columns:
        display_df["segment"] = display_df["segment"].replace({0: "Retail", 1: "Premium"})
    
    st.dataframe(
        display_df,
        hide_index=True,
        use_container_width=True,
        height=400
    )
    
    # Footer
    st.markdown("---")
    st.caption("Construido por Juan David Rincón")
