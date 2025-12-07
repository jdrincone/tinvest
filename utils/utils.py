
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from plotly.subplots import make_subplots

import shap



CORPORATE_COLORS = [
    "#1A494C",  # principal
    "#17877D",  # secundario
    "#343A40",  # verde suave
    "#F6B27A",  # naranja suave
    "#F18F01",  # naranja intenso
    "#E4572E",  # rojo/naranja
    "#6C757D",  # gris medio
    "#343A40",  # gris oscuro
    "#A3CED0",  # azul verdoso suave
]
OUTPUT_DIR = Path("plots_fintech_html")
OUTPUT_DIR.mkdir(exist_ok=True)
DATA_DIR = Path(".")


def apply_corporate_layout(fig: go.Figure, title: str) -> go.Figure:
    """Estilo corporativo para gráficos de Plotly."""
    fig.update_layout(
        title=title,
        title_x=0.5,
        template="plotly_white",
        font=dict(family="Arial", size=12, color="black"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="black"),
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(
        showline=True,
        linewidth=1.5,
        linecolor="black",
        mirror=True,
        tickfont=dict(color="black"),
        titlefont=dict(color="black"),
    )
    fig.update_yaxes(
        showline=True,
        linewidth=1.5,
        linecolor="black",
        mirror=True,
        tickfont=dict(color="black"),
        titlefont=dict(color="black"),
    )
    return fig


def save_html(fig: go.Figure, filename: str) -> None:
    path = OUTPUT_DIR / filename
    fig.write_html(str(path))
    print(f"[OK] Guardado: {path}")


from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import pandas as pd


def plot_nnm_vs_aum(
        nnm_aum: pd.DataFrame,
        *,
        date_col: str = "year_month_dt",
        nnm_col: str = "nnm",
        aum_col: str = "aum",
        html_filename: str | None = "00_nnm_aum_subplots.html",
) -> go.Figure:
    """
    Grafica NNM mensual (barras) y AUM (línea) en subplots.

    Parámetros
    ----------
    nnm_aum : DataFrame
        DataFrame ya transformado con columnas:
        - date_col (por defecto 'year_month_dt')
        - nnm_col  (por defecto 'nnm')
        - aum_col  (por defecto 'aum')
    date_col : str
        Nombre de la columna de fechas.
    nnm_col : str
        Nombre de la columna con NNM mensual.
    aum_col : str
        Nombre de la columna con AUM mensual.
    html_filename : str | None
        Si no es None, guarda el gráfico como HTML con ese nombre.

    Devuelve
    --------
    fig : go.Figure
        Figura de Plotly con los subplots.
    """

    df = nnm_aum.copy().sort_values(date_col)

    # Colores: verde si NNM ≥ 0, rojo si NNM < 0
    colors_nnm = np.where(
        df[nnm_col] >= 0,
        CORPORATE_COLORS[1],  # verde
        CORPORATE_COLORS[5],  # rojo
    )

    # ---------------------------------------
    # Subplots: NNM arriba, AUM abajo
    # ---------------------------------------
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.4, 0.6],
        subplot_titles=(
            "NNM mensual (entradas - salidas de dinero)",
            "Evolución del AUM (saldo gestionado)",
        ),
    )

    # --- Fila 1: NNM mensual ---
    fig.add_trace(
        go.Bar(
            x=df[date_col],
            y=df[nnm_col],
            marker_color=colors_nnm,
            name="NNM",
        ),
        row=1,
        col=1,
    )

    # Línea horizontal en 0 para ver rápido meses negativos
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="black",
        row=1,
        col=1,
    )

    # --- Fila 2: AUM mensual ---
    fig.add_trace(
        go.Scatter(
            x=df[date_col],
            y=df[aum_col],
            mode="lines+markers",
            name="AUM",
            line=dict(color=CORPORATE_COLORS[0], width=3),
        ),
        row=2,
        col=1,
    )

    # Ejes
    fig.update_xaxes(
        title_text="Mes",
        row=2,
        col=1,
    )
    fig.update_yaxes(
        title_text="NNM",
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="AUM",
        row=2,
        col=1,
    )

    # Layout corporativo
    fig = apply_corporate_layout(
        fig,
        "<b>NNM mensual vs evolución de AUM</b>",
    )

    fig.update_layout(
        margin=dict(t=90, l=60, r=40, b=40),
    )

    # Guardar opcionalmente
    if html_filename is not None:
        save_html(fig, html_filename)

    return fig



import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_nnm_peak_summary(
    summary: dict,
    *,
    html_filename: str | None = None,
) -> go.Figure:
    """
    Recibe el diccionario devuelto por build_nnm_peak_summary(...)
    y genera un gráfico 2x2 con:
      - NNM por producto
      - NNM por segmento
      - NNM nuevos vs existentes
      - Pareto de clientes (solo top 15)
    """
    peak_month        = summary["peak_month"]
    nnm_product       = summary["nnm_product"]
    nnm_segment       = summary["nnm_segment"]
    nnm_new_existing  = summary["nnm_new_existing"]
    nnm_clients_full  = summary["nnm_clients"]

    # 👇 SOLO aquí recortamos a top 15 clientes por NNM
    nnm_clients_top = (
        nnm_clients_full
        .sort_values("signed_amount", ascending=False)
        .head(15)
        .copy()
    )

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"secondary_y": True}],  # (2,2) con doble eje
        ],
        subplot_titles=(
            "NNM por producto",
            "NNM por segmento",
            "NNM: nuevos vs existentes",
            "Pareto de clientes (Top 15)",
        ),
    )

    # 1) NNM por producto
    fig.add_trace(
        go.Bar(
            x=nnm_product["product"],
            y=nnm_product["signed_amount"],
            name="Productos",
            marker_color=CORPORATE_COLORS[0],
        ),
        row=1, col=1,
    )

    # 2) NNM por segmento
    fig.add_trace(
        go.Bar(
            x=nnm_segment["segment"],
            y=nnm_segment["signed_amount"],
            name="Segmentos",
            marker_color=CORPORATE_COLORS[1],
        ),
        row=1, col=2,
    )

    # 3) NNM nuevos vs existentes
    fig.add_trace(
        go.Bar(
            x=nnm_new_existing["client_type"],
            y=nnm_new_existing["signed_amount"],
            name="Tipo de cliente",
            marker_color=CORPORATE_COLORS[4],
        ),
        row=2, col=1,
    )

    # 4) Pareto de clientes (solo top 15)
    # Barras: NNM por cliente
    fig.add_trace(
        go.Bar(
            x=nnm_clients_top["client_label"],
            y=nnm_clients_top["signed_amount"],
            name="NNM por cliente",
            marker_color=CORPORATE_COLORS[2],
        ),
        row=2, col=2,
        secondary_y=False,
    )

    # Línea: % NNM acumulado (usa cum_prop que ya viene en el summary)
    fig.add_trace(
        go.Scatter(
            x=nnm_clients_top["client_label"],
            y=nnm_clients_top["cum_prop"],  # 0–1
            name="% NNM acumulado (Top 15)",
            mode="lines+markers",
            line=dict(color=CORPORATE_COLORS[5], width=2),
        ),
        row=2, col=2,
        secondary_y=True,
    )

    # Ejes
    fig.update_xaxes(title_text="Producto", row=1, col=1)
    fig.update_yaxes(title_text="NNM",     row=1, col=1)

    fig.update_xaxes(title_text="Segmento", row=1, col=2)
    fig.update_yaxes(title_text="NNM",      row=1, col=2)

    fig.update_xaxes(title_text="Tipo de cliente", row=2, col=1)
    fig.update_yaxes(title_text="NNM",            row=2, col=1)

    fig.update_xaxes(
        title_text="Clientes (Top 15 por NNM)",
        row=2, col=2,
    )
    fig.update_yaxes(
        title_text="NNM por cliente",
        row=2, col=2,
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="% NNM acumulado",
        row=2, col=2,
        secondary_y=True,
        range=[0, 1.05],
        tickformat=".0%",
    )

    fig = apply_corporate_layout(
        fig,
        f"<b>Descomposición del NNM en {peak_month}</b>",
    )

    fig.update_layout(
        height=700,
        showlegend=False,
    )

    if html_filename is not None:
        save_html(fig, html_filename)

    return fig


def plot_return_curve(
    curve_data: pd.DataFrame,
    candidate_days: tuple,
    day_threshold: int,
    target_return: float,
    html_filename: str = "churn_01_curva_retornos.html",
) -> go.Figure:
    """
    Dibuja la curva de retorno + líneas de días candidatos y el umbral sugerido.
    """
    fig = go.Figure()

    # Línea principal
    fig.add_trace(
        go.Scatter(
            x=curve_data["days"],
            y=curve_data["return_prob"],
            mode="lines",
            name="Tasa de retorno acumulada",
            line=dict(color=CORPORATE_COLORS[0], width=3),
        )
    )

    # Líneas y anotaciones para días candidatos
    for d in candidate_days:
        prob = curve_data.loc[curve_data["days"] == d, "return_prob"].values[0]
        fig.add_vline(
            x=d,
            line_dash="dash",
            line_color=CORPORATE_COLORS[5] if d == 90 else "gray",
            opacity=0.6,
        )
        fig.add_annotation(
            x=d,
            y=prob,
            text=f"{d} días<br>{prob:.0%}",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-40,
            bgcolor="white",
            bordercolor="black",
            font=dict(size=10),
        )

    # Resaltar el umbral sugerido
    prob_thr = curve_data.loc[curve_data["days"] == day_threshold, "return_prob"].values[0]
    fig.add_vline(
        x=day_threshold,
        line_dash="solid",
        line_color=CORPORATE_COLORS[4],
        opacity=0.9,
    )
    fig.add_annotation(
        x=day_threshold,
        y=prob_thr,
        text=f"Umbral sugerido<br>{day_threshold} días\n(≈{target_return:.0%})",
        showarrow=True,
        arrowhead=2,
        ax=40,
        ay=-40,
        bgcolor="white",
        bordercolor=CORPORATE_COLORS[4],
        font=dict(size=11),
    )

    fig.update_xaxes(
        title_text="Días de silencio entre transacciones",
        range=[0, curve_data["days"].max()],
    )
    fig.update_yaxes(
        title_text="Probabilidad acumulada de retorno",
        range=[0, 1.05],
        tickformat=".0%",
    )

    fig = apply_corporate_layout(
        fig,
        "<b>Curva de latencia: ¿cuánto tardan en volver los clientes?</b>",
    )

    save_html(fig, html_filename)
    fig.show()
    return fig


# ------------------------------------------------------------------
# 2.2 Plot: sensibilidad por rango de saldo
# ------------------------------------------------------------------
def plot_balance_sensitivity(
    sensitivity_with_bounds: pd.DataFrame,
    html_filename: str = "churn_02_sensibilidad_saldo.html",
) -> go.Figure:
    """
    Dibuja barras de número de clientes por rango de saldo,
    coloreadas por % de inactivos.
    Espera columnas:
      balance_bucket, num_users, inactivity_rate
    en sensitivity_with_bounds.
    """
    df = sensitivity_with_bounds.copy()
    df["balance_bucket"] = df["balance_bucket"].astype(str)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df["balance_bucket"],
            y=df["num_users"],
            name="Clientes",
            marker=dict(
                color=df["inactivity_rate"],
                colorscale="Viridis",
                colorbar=dict(title="% inactivos", tickformat=".0%"),
            ),
            text=[
                f"{u} clientes<br>Inactivos: {r:.0%}"
                for u, r in zip(df["num_users"], df["inactivity_rate"])
            ],
            textposition="outside",
        )
    )

    fig.update_xaxes(title_text="Rango de saldo (COP)")
    fig.update_yaxes(title_text="Número de clientes")

    fig = apply_corporate_layout(
        fig,
        "<b>Churn operativo: inactividad por rango de saldo</b>",
    )

    save_html(fig, html_filename)
    fig.show()
    return fig




def plot_churn_profile_subplots(
    churn_profile: dict,
    filename: str = "churn_01_perfil_clientes.html",
) -> None:
    """
    Recibe el diccionario generado por `prepare_churn_profile_data`
    y genera un layout 2x2:

        [1,1] Boxplot de ingresos (en millones de COP), churn vs retenido
        [1,2] Densidad (KDE) de edad por churn
        [2,1] Violin de risk_score por churn
        [2,2] Barra de churn por segmento

    Guarda el resultado en HTML con `save_html`.
    """
    df = churn_profile["df"].copy()
    churn_by_segment = churn_profile["churn_by_segment"].copy()

    # Cast por seguridad
    df["income_monthly"] = df["income_monthly"].astype(float)
    df["age"] = df["age"].astype(float)
    df["risk_score"] = df["risk_score"].astype(float)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Poder adquisitivo: ¿quién se va?",
            "Edad: ¿estamos perdiendo a los jóvenes?",
            "Apetito de riesgo: ¿quién huye?",
            "Fuga por segmento de negocio",
        ),
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    # ---------------------------------------------------
    # [1,1] Boxplot de ingresos (en millones de COP)
    # ---------------------------------------------------
    for label, color in zip(["Retenido", "Churn"], [CORPORATE_COLORS[0], CORPORATE_COLORS[5]]):
        sub = df[df["churn_label"] == label]
        if len(sub) == 0:
            continue
        fig.add_trace(
            go.Box(
                x=[label] * len(sub),
                y=sub["income_monthly"] / 1e6,   # escala a millones
                name=label,
                marker_color=color,
                boxmean=False,                   # <-- sin rombo de mean+sd
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    fig.update_yaxes(
        title_text="Ingreso mensual (millones de COP)",
        tickformat=".1f",   # 5.0, 10.0 …
        tickprefix="$",     # $5.0
        ticksuffix="M",     # $5.0M
        row=1,
        col=1,
    )

    # ---------------------------------------------------
    # [1,2] Densidad de edad (KDE) tipo Seaborn
    # ---------------------------------------------------
    ages_ret = df[df["churn_label"] == "Retenido"]["age"].dropna().values
    ages_churn = df[df["churn_label"] == "Churn"]["age"].dropna().values

    hist_data = [ages_ret, ages_churn]
    group_labels = ["Retenido", "Churn"]
    colors = [CORPORATE_COLORS[2], CORPORATE_COLORS[6]]

    # Creamos fig de densidades y le pasamos los traces al subplot [1,2]
    dist_fig = ff.create_distplot(
        hist_data,
        group_labels,
        colors=colors,
        show_hist=False,   # solo curvas
        show_rug=False,
    )

    for trace in dist_fig.data:
        trace.showlegend = False  # el layout general no necesita leyenda aquí
        fig.add_trace(trace, row=1, col=2)

    fig.update_xaxes(title_text="Edad", row=1, col=2)
    fig.update_yaxes(title_text="Densidad", row=1, col=2)

    # ---------------------------------------------------
    # [2,1] Violin de risk_score
    # ---------------------------------------------------
    for label, color in zip(["Retenido", "Churn"], [CORPORATE_COLORS[3], CORPORATE_COLORS[5]]):
        sub = df[df["churn_label"] == label]
        if len(sub) == 0:
            continue
        fig.add_trace(
            go.Violin(
                x=[label] * len(sub),
                y=sub["risk_score"],
                name=label,
                line_color=color,
                fillcolor=color,
                opacity=0.5,
                showlegend=True,
                box_visible=True,
                meanline_visible=True,
            ),
            row=2,
            col=1,
        )

    fig.update_yaxes(
        title_text="Risk score",
        row=2,
        col=1,
    )

    # ---------------------------------------------------
    # [2,2] Churn por segmento
    # ---------------------------------------------------
    churn_by_segment["churn_pct"] = churn_by_segment["churn_rate"] * 100

    fig.add_trace(
        go.Bar(
            x=churn_by_segment["segment"],
            y=churn_by_segment["churn_pct"],
            marker_color=CORPORATE_COLORS[1],
            text=churn_by_segment["churn_pct"].round(1).astype(str) + "%",
            textposition="outside",
            name="Churn %",
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    fig.update_yaxes(
        title_text="Tasa de churn (%)",
        row=2,
        col=2,
    )
    fig.update_xaxes(
        title_text="Segmento",
        row=2,
        col=2,
    )

    # Layout general corporativo
    fig = apply_corporate_layout(
        fig,
        "<b>Perfilamiento de clientes: churn vs retenidos</b>",
    )

    fig.update_layout(
        height=800,
        margin=dict(t=90, l=60, r=40, b=40),
    )

    save_html(fig, filename)
    fig.show()


def plot_avg_tenure_by_product_plotly(
    avg_tenure_by_product: pd.DataFrame,
) -> go.Figure:
    """
    Barra vertical: antigüedad promedio (en días) por producto.
    """
    df = avg_tenure_by_product.copy()

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["product"],
            y=df["tenure_days"],
            marker=dict(color=CORPORATE_COLORS[0]),
            text=[f"{v:.0f} días" for v in df["tenure_days"]],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Antigüedad promedio: %{y:.0f} días<extra></extra>",
            name="Tenure promedio",
        )
    )

    fig.update_xaxes(title_text="Producto")
    fig.update_yaxes(title_text="Días promedio de relación")

    fig = apply_corporate_layout(
        fig,
        "<b>Antigüedad promedio de la relación cliente–producto</b>",
    )
    return fig

def plot_vintage_composition_plotly(
    vintage_pivot: pd.DataFrame,
) -> go.Figure:
    """
    Barras apiladas: para cada producto, cómo se compone el saldo
    según el año de inicio (vintage_year).
    """
    df = vintage_pivot.copy()
    df = df.sort_index()

    products = df.index.astype(str)
    years = list(df.columns)

    fig = go.Figure()
    palette = CORPORATE_COLORS

    for i, year in enumerate(years):
        fig.add_trace(
            go.Bar(
                x=products,
                y=df[year],
                name=str(year),
                marker=dict(color=palette[i % len(palette)]),
                hovertemplate=(
                    "<b>Producto:</b> %{x}<br>"
                    f"Año inicio: {year}<br>"
                    "Balance: %{y:,.0f} COP<extra></extra>"
                ),
            )
        )

    fig.update_layout(barmode="stack")

    fig.update_xaxes(title_text="Producto")
    fig.update_yaxes(
        title_text="Balance total (COP)",
        tickprefix="$",
        tickformat=",.0s",  # k, M, G
    )

    fig = apply_corporate_layout(
        fig,
        "<b>Calidad del saldo: composición por vintage (año de inicio)</b>",
    )
    return fig


def plot_product_vintage_dashboard_plotly(
    vintage_results: dict,
    save_path: str | None = None,
) -> go.Figure:
    """
    Dashboard 1×2:
      1. Antigüedad promedio cliente–producto.
      2. Composición de saldos por vintage (stacked).
    """
    avg_tenure = vintage_results["avg_tenure_by_product"]
    vintage_pivot = vintage_results["vintage_pivot"]

    df_ten = avg_tenure.copy()
    df_vin = vintage_pivot.copy().sort_index()

    products_ten = df_ten["product"]
    # Para antigüedad
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Antigüedad promedio de la relación (días)",
            "Composición del saldo por vintage",
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}]],
        column_widths=[0.45, 0.55],
    )

    # ---- Columna 1: antigüedad promedio ----
    fig.add_trace(
        go.Bar(
            x=products_ten,
            y=df_ten["tenure_days"],
            marker=dict(color=CORPORATE_COLORS[0]),
            text=[f"{v:.0f} días" for v in df_ten["tenure_days"]],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Antigüedad promedio: %{y:.0f} días<extra></extra>",
            name="Tenure promedio",
        ),
        row=1,
        col=1,
    )

    # ---- Columna 2: stacked de vintages ----
    products_vin = df_vin.index.astype(str)
    years = list(df_vin.columns)
    extra_colors = ["#A3CED0", "#D9E6D1", "#B0BEC5"]
    palette = CORPORATE_COLORS + extra_colors

    for i, year in enumerate(years):
        fig.add_trace(
            go.Bar(
                x=products_vin,
                y=df_vin[year],
                name=str(year),
                marker=dict(color=palette[i % len(palette)]),
                hovertemplate=(
                    "<b>Producto:</b> %{x}<br>"
                    f"Año inicio: {year}<br>"
                    "Balance: %{y:,.0f} COP<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )

    fig.update_layout(barmode="stack", legend_title_text="Año de inicio (vintage)")

    # Ejes
    fig.update_xaxes(title_text="Producto", row=1, col=1)
    fig.update_yaxes(title_text="Días promedio de relación", row=1, col=1)

    fig.update_xaxes(title_text="Producto", row=1, col=2)
    fig.update_yaxes(
        title_text="Balance total (COP)",
        tickprefix="$",
        tickformat=",.0s",
        row=1,
        col=2,
    )

    fig = apply_corporate_layout(
        fig,
        "<b>Historia y calidad del saldo por producto</b>",
    )

    if save_path is not None:
        fig.write_html(save_path)

    return fig






def plot_nnm_by_product(nnm_by_product: pd.DataFrame, ax=None):
    """Barplot horizontal de NNM neto por producto."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    df_plot = nnm_by_product.copy()
    colors_to_use = CORPORATE_COLORS * (len(df_plot) // len(CORPORATE_COLORS) + 1)

    sns.barplot(
        data=df_plot,
        x="NNM",
        y="product",
        palette=colors_to_use[:len(df_plot)], # Asignamos colores
        ax=ax,
    )

    ax.set_title("Motor de crecimiento (NNM neto)", fontsize=14, weight="bold")
    ax.set_xlabel("Monto neto (COP)", fontsize=10)
    ax.set_ylabel("", fontsize=10)

    for i, v in enumerate(df_plot["NNM"]):
        label = f"${v/1e9:.2f} B" if abs(v) >= 1e9 else f"${v/1e6:.1f} M"
        ax.text(
            v, i, f" {label}",
            va="center",
            ha="left" if v >= 0 else "right",
            weight="bold",
            fontsize=9,
            color=CORPORATE_COLORS[0]
        )
    return ax


def plot_capital_retention_by_segment(df_segment_retention: pd.DataFrame, threshold=0.50, ax=None):
    """Barplot de tasa de retención."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    df_plot = df_segment_retention.copy()
    colores = [
        CORPORATE_COLORS[1] if x > threshold else CORPORATE_COLORS[5]
        for x in df_plot["capital_retention_rate"]
    ]

    bars = ax.bar(
        df_plot["segment"],
        df_plot["capital_retention_rate"] * 100,
        color=colores,
        width=0.7,
    )

    ax.set_title("Eficiencia de retención", fontsize=14, weight="bold")
    ax.set_ylabel("Tasa de retención (%)", fontsize=10)
    ax.set_ylim(0, 100)

    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=15, weight="bold")

    # Línea de meta
    ax.axhline(threshold * 100, color="gray", linestyle="--", alpha=0.5)
    ax.text(0.5, (threshold*100)+2, f'Meta: {threshold*100}%', color='gray', ha='center', fontsize=12)

    return ax


def plot_pareto_clients(pareto_meta: dict, ax=None):
    """Pie chart de concentración."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    num_top = pareto_meta["num_top"]
    num_total = pareto_meta["num_total"]
    pct_top = pareto_meta["pct_top"] * 100 if num_total > 0 else 0
    pct_contribution = pareto_meta["pct_contribution"] * 100
    num_resto = max(num_total - num_top, 0)

    labels = [
        f"Top {pct_top:.1f}% clientes\n(Aportan ~{pct_contribution:.0f}% Dinero)",
        f"Resto {100-pct_top:.1f}% clientes",
    ]
    sizes = [num_top, num_resto]

    # Usamos colores corporativos: Principal vs Gris neutral
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=[CORPORATE_COLORS[1], CORPORATE_COLORS[2]],
        wedgeprops=dict(width=0.6, edgecolor='w'), # Estilo Donut
    )

    plt.setp(autotexts, size=11, weight="bold", color="white")
    plt.setp(texts, size=10)

    ax.set_title("Concentración (Pareto)", fontsize=14, weight="bold")

    # Texto central
    ax.text(0, 0, f"{num_total}\nClientes", ha='center', va='center', weight='bold')

    return ax


def plot_sleeping_giants_card(high_potential_segment: pd.DataFrame, ax=None):
    """KPI card."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    ax.axis("off")
    kpi_value = len(high_potential_segment)

    # Fondo suave
    rect = plt.Rectangle((0.05, 0.05), 0.9, 0.9, color="#ecf0f1", alpha=0.5, transform=ax.transAxes, zorder=0)
    ax.add_patch(rect)

    ax.text(0.5, 0.8, 'OPORTUNIDAD\n"Ballenas sin invertir"', ha="center", va='center', fontsize=14, weight="bold", color=CORPORATE_COLORS[0], transform=ax.transAxes)

    # Número gigante en Rojo (Alerta de oportunidad perdida)
    ax.text(0.5, 0.55, str(kpi_value), ha="center", va='center', fontsize=70, weight="bold", color=CORPORATE_COLORS[5], transform=ax.transAxes)

    ax.text(0.5, 0.35, "Clientes ricos sub-invertidos", ha="center", va='center', fontsize=12, color=CORPORATE_COLORS[6], transform=ax.transAxes)

    # Call to action en Azul
    ax.text(0.5, 0.15, "Acción: Contactar", ha="center", va='center', fontsize=11, style="italic", weight='bold', color=CORPORATE_COLORS[2], transform=ax.transAxes)

    return ax


def plot_nnm_strategy_dashboard(nnm_results: dict):
    """Dashboard 2x2 unificado."""
    nnm_by_product = nnm_results["nnm_by_product"]
    df_segment_retention = nnm_results["df_segment_retention"]
    pareto_meta = nnm_results["pareto_meta"]
    high_potential_segment = nnm_results["high_potential_segment"]

    fig = plt.figure(figsize=(16, 10))
    plt.suptitle(
        "Análisis de Flujos de Dinero (NNM)",
        fontsize=20,
        weight="bold",
        #color=CORPORATE_COLORS[1],
        y=0.95
    )

    ax1 = plt.subplot(2, 2, 1)
    plot_nnm_by_product(nnm_by_product, ax=ax1)

    ax2 = plt.subplot(2, 2, 2)
    plot_capital_retention_by_segment(df_segment_retention, ax=ax2)

    ax3 = plt.subplot(2, 2, 3)
    plot_pareto_clients(pareto_meta, ax=ax3)

    ax4 = plt.subplot(2, 2, 4)
    plot_sleeping_giants_card(high_potential_segment, ax=ax4)

    plt.tight_layout(pad=3.0, rect=[0, 0, 1, 0.92])
    return fig


def plot_feature_importance_plotly(
    clf: RandomForestClassifier,
    feature_cols: list[str],
) -> go.Figure:
    """
    Gráfico de importancia de variables (horizontal) en Plotly.
    """
    importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values()

    fig = go.Figure(
        go.Bar(
            x=importances.values,
            y=importances.index,
            orientation="h",
            marker_color=CORPORATE_COLORS[1],
        )
    )

    fig.update_xaxes(title_text="Peso en la decisión del modelo")
    fig.update_yaxes(title_text="")

    fig = apply_corporate_layout(
        fig,
        "<b>¿Qué define que un cliente se vaya? (Importancia de features)</b>",
    )

    return fig


def plot_risk_value_matrix_plotly(
    df_scored: pd.DataFrame,
    risk_threshold: float,
    value_threshold: float,
) -> go.Figure:
    """
    Matriz Riesgo vs Valor con Plotly:
      x = estimated_annual_value (en millones)
      y = churn_probability
      color = strategy_segment
    """
    df = df_scored.copy()
    # Trabajamos en millones para que el eje sea legible
    df["estimated_annual_value_M"] = df["estimated_annual_value"] / 1e6
    value_thr_M = value_threshold / 1e6

    color_map = {
        "1. PRIORIDAD CRÍTICA (Salvar)": CORPORATE_COLORS[5],  # rojo
        "2. FIDELIZAR (Cuidar)": CORPORATE_COLORS[1],         # verde medio
        "3. DEJAR IR (No rentable)": CORPORATE_COLORS[6],     # gris
        "4. BAJA PRIORIDAD": CORPORATE_COLORS[2],             # verde suave
    }

    fig = go.Figure()

    for seg, seg_df in df.groupby("strategy_segment"):
        fig.add_trace(
            go.Scatter(
                x=seg_df["estimated_annual_value_M"],
                y=seg_df["churn_probability"],
                mode="markers",
                name=seg,
                marker=dict(
                    color=color_map.get(seg, CORPORATE_COLORS[6]),
                    size=8,
                    opacity=0.7,
                ),
                hovertemplate=(
                    "Valor anual: $%{x:.2f}M<br>"
                    "Prob. churn: %{y:.1%}<br>"
                    "Segmento: " + seg + "<extra></extra>"
                ),
            )
        )

    # Líneas de umbral
    fig.add_hline(
        y=risk_threshold,
        line_dash="dash",
        line_color="black",
        annotation_text=f"Umbral riesgo ({risk_threshold:.2f})",
        annotation_position="top left",
    )
    fig.add_vline(
        x=value_thr_M,
        line_dash="dash",
        line_color="black",
        annotation_text="Umbral valor (Top 25%)",
        annotation_position="top right",
    )

    fig.update_xaxes(
        title_text="Valor anual estimado (millones COP)",
        rangemode="tozero",
    )
    fig.update_yaxes(
        title_text="Probabilidad de fuga (churn)",
        rangemode="tozero",
        tickformat=".0%",
    )

    fig = apply_corporate_layout(
        fig,
        "<b>Matriz de retención rentable: Riesgo vs Valor</b>",
    )

    # Un poco de margen en X
    max_x = df["estimated_annual_value_M"].quantile(0.98)
    fig.update_xaxes(range=[0, max_x * 1.05 if max_x > 0 else 1])

    return fig
