import streamlit as st


def render_actionable_conclusions(clients, transactions, portfolio):
    """
    Renderiza la vista de Conclusiones Accionables.
    """
    st.header("Conclusiones Accionables")
    st.markdown("""
        Recomendaciones estratégicas basadas en el análisis de datos para mejorar 
        el crecimiento, retención y eficiencia operativa.
    """)
    st.markdown("---")
    
    # Container para las conclusiones con estilo
    conclusions = [
        {
            "title": "Growth Inmediato",
            "content": "Activar comercialmente a las 14 'Ballenas sin invertir'.",
            "icon": "🚀",
            "color": "#E74C3C"
        },
        {
            "title": "Urgencia de Producto",
            "content": "Intervenir urgentemente el FPV con incentivos de permanencia o de cruce de producto.",
            "icon": "⚠️",
            "color": "#F39C12"
        },
        {
            "title": "Fuga Premium",
            "content": "Crear un producto de inversión exclusivo para el segmento Premium que detenga su baja retención de capital.",
            "icon": "💎",
            "color": "#9B59B6"
        },
        {
            "title": "Eficiencia en reducir Churn",
            "content": "El modelo de churn permite priorizar a clientes con alta probabilidad de fuga y alto valor, que son los que realmente vale la pena salvar.",
            "icon": "🎯",
            "color": "#3498DB"
        },
        {
            "title": "Campañas",
            "content": "La curva de latencia permite conocer a tiempo los clientes que están demorados en realizar transacciones, si el cliente es VIP incentivarlo mover su dinero a tiempo.",
            "icon": "📊",
            "color": "#1ABC9C"
        },
        {
            "title": "Nuevos Clientes",
            "content": "No hay clientes nuevos del mediados del 2024. Se debe capturar nuevos clientes.",
            "icon": "👥",
            "color": "#16A085"
        }
    ]
    
    # Renderizar cada conclusión en un formato visual atractivo
    for i, conclusion in enumerate(conclusions, 1):
        st.markdown(
            f"""
            <div style="
                border-left: 4px solid {conclusion['color']};
                padding: 20px;
                margin: 15px 0;
                background-color: #f8f9fa;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            ">
                <h3 style="color: {conclusion['color']}; margin-top: 0;">
                    {conclusion['icon']} {conclusion['title']}
                </h3>
                <p style="font-size: 16px; color: #2c3e50; margin-bottom: 0;">
                    {conclusion['content']}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("---")
    
    # Footer
    st.caption("Construido por Juan David Rincón")
