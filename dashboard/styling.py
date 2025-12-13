import streamlit as st

def apply_custom_css():
    """Injects custom CSS for metric cards and general styling."""
    st.markdown(
        """
        <style>
        /* Style the metric container */
        div[data-testid="stMetric"] {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
            /* Force same height and visual consistency */
            min-height: 140px; 
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        
        /* Center text in metrics and make labels bold */
        div[data-testid="stMetricLabel"] {
            text-align: center;
            font-weight: 900 !important; /* Extra bold */
            color: #6c757d;
            font-size: 1.1rem !important;
            margin-bottom: 5px;
            width: 100%;
            display: block; /* Ensure it takes full width to center properly */
        }
        
        div[data-testid="stMetricValue"] {
            text-align: center;
            color: #1A494C; /* Global corporate color */
            font-weight: bold;
            width: 100%;
        }
        
        div[data-testid="stMetricDelta"] {
            text-align: center;
            width: 100%;
            justify-content: center; /* Flex alignment for delta arrow */
            display: flex;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def format_currency(value):
    """Helper function for formatting large numbers (B/M/k)."""
    if abs(value) >= 1e9:
        return f"${value / 1e9:,.1f}B"
    elif abs(value) >= 1e6:
        return f"${value / 1e6:,.1f}M"
    else:
        return f"${value:,.0f}"
