import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add project root to sys.path to import utils if needed
# (Assuming this file is in dashboard/charts.py and utils is in utils/utils.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.utils import plot_nnm_vs_aum

def get_nnm_aum_plot(nnm_aum_data: pd.DataFrame) -> go.Figure:
    """
    Generates the NNM vs AUM plot with specific dashboard customizations.
    """
    fig = plot_nnm_vs_aum(nnm_aum_data, html_filename=None)
    
    # 1. Remove the main title
    fig.update_layout(title_text="")
    
    # 2. Adjust margins for better layout
    fig.update_layout(
        margin=dict(t=120, l=60, r=40, b=40),
    )
    
    # 3. Adjust annotation positions to prevent overlap
    # We iterate through annotations to identify subplot titles and adjust them individually
    fig.layout.annotations = [
        ann.update(yshift=30) if "NNM mensual" in ann.text else 
        ann.update(yshift=-30) if "Evolución del AUM" in ann.text else 
        ann 
        for ann in fig.layout.annotations
    ]
    
    # 4. Add Top 2 NNM Annotations specific to this view
    if not nnm_aum_data.empty:
        top_nnm = nnm_aum_data.nlargest(2, "nnm")
        for _, row in top_nnm.iterrows():
            fig.add_annotation(
                x=row["year_month_dt"],
                y=row["nnm"],
                text=f"Peak NNM<br>{row['year_month']}",
                showarrow=True,
                arrowhead=1,
                yshift=10,
                align="center",
                bordercolor="black",
                borderwidth=1,
                bgcolor="white",
                opacity=0.8,
                row=1, col=1 # Specify subplot for annotation
            )
            
    return fig

def plot_horizontal_bar(
    df: pd.DataFrame, 
    y_col: str, 
    x_col: str, 
    title: str,
    color: str = "#1A494C",
    x_label: str = "NNM ($)"
) -> go.Figure:
    """
    Creates a horizontal bar chart for breakdowns (e.g., Product, Segment).
    """
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df[y_col],
        x=df[x_col],
        orientation='h',
        marker=dict(color=color),
        text=df[x_col].apply(lambda x: f"${x:,.0f}"),
        textposition='auto'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis=dict(autorange="reversed"), # Top values at top
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20),
        height=400
    )
    
    return fig

def plot_vertical_bar(
    df: pd.DataFrame, 
    x_col: str, 
    y_col: str, 
    title: str,
    color: str = "#F39C12", # Using orange/gold tone from user example
    y_label: str = "NNM ($)"
) -> go.Figure:
    """
    Creates a vertical bar chart (e.g., for New vs Existing).
    """
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df[x_col],
        y=df[y_col],
        marker=dict(color=color),
        # Smart formatting: B, M, or full
        text=df[y_col].apply(lambda x: f"${x/1e9:,.1f}B" if abs(x)>=1e9 else (f"${x/1e6:,.1f}M" if abs(x)>=1e6 else f"${x:,.0f}")),
        textposition='auto'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="", # Self explanatory categories
        yaxis_title=y_label,
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20),
        height=400
    )
    
    return fig

def plot_heatmap(
    df_pivot: pd.DataFrame,
    title: str,
    x_label: str = "Vintage Year",
    y_label: str = "Product"
) -> go.Figure:
    """
    Creates a heatmap for Vintage Analysis.
    Expects df_pivot to have Product index and Vintage Year columns.
    """
    fig = go.Figure(data=go.Heatmap(
        z=df_pivot.values,
        x=df_pivot.columns,
        y=df_pivot.index,
        colorscale='Teal', # Corporate-ish green/teal
        texttemplate="%{z:.2s}", # Smart number formatting (1.2k, 1M)
        showscale=True
    ))

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        margin=dict(l=20, r=20, t=40, b=20),
        height=400
    )
    return fig

def plot_stacked_bar_time_series(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str,
    title: str,
    y_label: str = "Count"
) -> go.Figure:
    """
    Creates a stacked bar chart for time series (e.g. New vs Existing counts over time).
    """
    fig = go.Figure()
    
    # Get unique categories for stacking
    categories = df[color_col].unique()
    colors = ["#1A494C", "#F39C12", "#6c757d", "#E74C3C"] # Corporate Palette + others
    
    for i, cat in enumerate(categories):
        subset = df[df[color_col] == cat]
        fig.add_trace(go.Bar(
            name=cat,
            x=subset[x_col],
            y=subset[y_col],
            marker_color=colors[i % len(colors)]
        ))

    fig.update_layout(
        title=title,
        barmode='stack',
        xaxis_title="",
        yaxis_title=y_label,
        legend_title=color_col,
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20),
        height=400
    )
    
    return fig

def plot_donut(
    values: list,
    labels: list,
    title: str,
    colors: list = ["#1A494C", "#34495E", "#95A5A6"], # Teal, Dark Blue, Grey
    hole: float = 0.4
) -> go.Figure:
    """
    Creates a donut chart for proportions (e.g. Pareto).
    """
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=hole,
        marker=dict(colors=colors),
        textinfo='label+percent',
        hoverinfo='label+value+percent'
    )])
    
    fig.update_layout(
        title=title,
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20),
        height=400,
        showlegend=True
    )
    
    return fig

def plot_retention_efficiency(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = "Eficiencia de retención",
    target: float = 0.5
) -> go.Figure:
    """
    Creates a vertical bar chart for retention with a target line.
    Colors bars based on whether they meet the target.
    """
    # Determine colors: Teal if >= target, OrangeRed if < target
    colors = [
        "#1A494C" if val >= target else "#E74C3C" 
        for val in df[y_col]
    ]

    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df[x_col],
        y=df[y_col],
        marker_color=colors,
        text=df[y_col].apply(lambda x: f"{x:.1%}"),
        textposition='outside',
        textfont=dict(size=14, color="black", family="Arial")
    ))
    
    # Add Target Line
    fig.add_shape(
        type="line",
        x0=-0.5, x1=len(df)-0.5,
        y0=target, y1=target,
        line=dict(color="gray", width=2, dash="dash"),
    )
    fig.add_annotation(
        x=len(df)-0.5, y=target,
        text=f"Meta: {target:.1%}",
        showarrow=False,
        yshift=10,
        xshift=-40,
        font=dict(color="gray")
    )

    fig.update_layout(
        title=title,
        xaxis_title="",
        yaxis_title="Tasa de retención (%)",
        yaxis=dict(range=[0, 1.1], tickformat=".0%"), # Ensure space for text
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20),
        height=400,
        showlegend=False
    )
    
    return fig

def plot_return_curve(
    curve_data: pd.DataFrame,
    candidate_days: tuple,
    day_threshold: int,
    target_return: float,
    title: str = "Curva de Retorno (Latencia entre Transacciones)"
) -> go.Figure:
    """
    Plots the probability of return within X days.
    """
    fig = go.Figure()
    
    # Main Curve
    fig.add_trace(go.Scatter(
        x=curve_data["days"],
        y=curve_data["return_prob"],
        mode='lines',
        name='Prob. Retorno',
        line=dict(color='#1A494C', width=3)
    ))
    
    # Target Line
    fig.add_hline(
        y=target_return, 
        line_dash="dash", 
        line_color="gray", 
        annotation_text=f"Target: {target_return:.0%}",
        annotation_position="bottom right"
    )
    
    # Threshold Line (Vertical)
    fig.add_vline(
        x=day_threshold,
        line_dash="dot",
        line_color="#E74C3C",
        annotation_text=f"Umbral: {day_threshold} días",
        annotation_position="top right"
    )

    fig.update_layout(
        title=title,
        xaxis_title="Días desde última transacción",
        yaxis_title="Probabilidad Acumulada",
        yaxis=dict(tickformat=".0%"),
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20),
        height=450,
        showlegend=False
    )
    
    return fig

def plot_balance_sensitivity(
    sensitivity_df: pd.DataFrame,
    title: str = "Sensibilidad de Saldo: Tasa de Inactividad por Bucket"
) -> go.Figure:
    """
    Plots Inactivity Rate per Balance Bucket using a bar chart.
    """
    fig = go.Figure()

    # Bar chart for Inactivity Rate
    fig.add_trace(go.Bar(
        x=sensitivity_df["balance_bucket"],
        y=sensitivity_df["inactivity_rate"],
        name="Tasa Inactividad",
        marker_color="#F39C12",
        text=sensitivity_df["inactivity_rate"].apply(lambda x: f"{x:.1%}"),
        textposition='auto'
    ))
    
    # Add count of users as line on secondary axis? Or just hover?
    # Let's keep it simple with hover for users
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Inactividad: %{y:.1%}<br>Usuarios: %{customdata}",
        customdata=sensitivity_df["num_users"]
    )

    fig.update_layout(
        title=title,
        xaxis_title="Rango de Saldo",
        yaxis_title="Tasa de Inactividad (%)",
        yaxis=dict(tickformat=".0%"),
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20),
        height=450
    )
    

    return fig

import scipy.stats as stats

def plot_churn_profile_subplots(churn_data: dict, filename: str = None) -> go.Figure:
    """
    Creates a 2x2 subplot comparing Churn vs Retained clients on key metrics,
    matching the exact layout from the reference notebook.
    
    Layout:
    1. Income (Box)
    2. Age (Density/KDE)
    3. Risk Score (Violin)
    4. Churn Rate by Segment (Bar)
    """
    df = churn_data["df"]
    churn_by_segment = churn_data["churn_by_segment"] # Pre-calculated in data_loader
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Poder adquisitivo: ¿quién se va?", 
            "Edad: ¿estamos perdiendo a los jóvenes?", 
            "Apetito de riesgo: ¿quién huye?", 
            "Fuga por segmento de negocio"
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )
    
    colors = {"Retenido": "#1A494C", "Churn": "#E4572E"} # Teal vs OrangeRed
    
    # --- 1. Income (Box Plot) ---
    # Top-Left
    for label, color in colors.items():
        subset = df[df["churn_label"] == label]
        fig.add_trace(go.Box(
            y=subset["income_monthly"],
            x=[label]*len(subset), # Group by label on X
            name=label,
            marker_color=color,
            boxpoints=False, 
            showlegend=True,
            legendgroup=label
        ), row=1, col=1)
        
    # --- 2. Age (Density Plot / KDE) ---
    # Top-Right
    x_range = np.linspace(df["age"].min(), df["age"].max(), 100)
    
    for label, color in colors.items():
        subset = df[df["churn_label"] == label]
        data = subset["age"].dropna()
        if len(data) > 1:
            kde = stats.gaussian_kde(data)
            y_kde = kde(x_range)
            
            # Specific colors for KDE lines as per JSON (darker)
            line_color = "#343A40" if label == "Retenido" else "#6C757D"
            
            fig.add_trace(go.Scatter(
                x=x_range,
                y=y_kde,
                mode='lines',
                name=label,
                line=dict(color=line_color, width=2),
                showlegend=False,
                legendgroup=label
            ), row=1, col=2)

    # --- 3. Risk Score (Violin Plot) ---
    # Bottom-Left
    for label, color in colors.items():
        subset = df[df["churn_label"] == label]
        # Color specific for Violin in JSON was lighter? 
        # "Retenido": "#F6B27A" (Orange-ish?), "Churn": "#E4572E"?
        # Let's stick to main colors but add transparency or use violin specific 
        fig.add_trace(go.Violin(
            y=subset["risk_score"],
            x=[label]*len(subset),
            name=label,
            # line_color=color,
            fillcolor=color,
            opacity=0.6,
            meanline_visible=True,
            showlegend=False,
            legendgroup=label
        ), row=2, col=1)

    # --- 4. Churn Rate by Segment (Bar Chart) ---
    # Bottom-Right
    # Use churn_by_segment dataframe
    # Expected cols: [segment_col, churn_rate, n_clients]
    seg_col = churn_by_segment.columns[0] # segment or client_type
    
    fig.add_trace(go.Bar(
        x=churn_by_segment[seg_col],
        y=churn_by_segment["churn_rate"] * 100, # as percentage
        name="Churn %",
        marker_color="#17877D", # Teal-ish but distinct
        text=churn_by_segment["churn_rate"].apply(lambda x: f"{x:.1%}"),
        textposition="outside",
        showlegend=False
    ), row=2, col=2)
    
    # --- Layout Updates ---
    fig.update_layout(
        title_text="<b>Perfilamiento de clientes: churn vs retenidos</b>",
        title_x=0.5,
        template="plotly_white",
        height=800,
        margin=dict(l=60, r=40, t=90, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Axes Titles
    fig.update_yaxes(title_text="Ingreso mensual (millones de COP)", row=1, col=1)
    
    fig.update_xaxes(title_text="Edad", row=1, col=2)
    fig.update_yaxes(title_text="Densidad", row=1, col=2)
    
    fig.update_yaxes(title_text="Risk score", row=2, col=1)
    
    fig.update_xaxes(title_text="Segmento", row=2, col=2)
    fig.update_yaxes(title_text="Tasa de churn (%)", row=2, col=2)
    
    return fig
