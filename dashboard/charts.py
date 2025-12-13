import pandas as pd
import plotly.graph_objects as go
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
