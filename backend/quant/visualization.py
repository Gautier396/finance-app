import plotly.express as px
import pandas as pd

def create_risk_chart(metrics_dict):
    """
    Transforme le dictionnaire de métriques en un graphique à bulles interactif.
    """
    df_plot = pd.DataFrame(metrics_dict).T.reset_index()
    df_plot.columns = ['Ticker', 'Beta', 'Specific_Risk']

    df_plot['Total_Vol'] = df_plot['Beta'] + (df_plot['Specific_Risk'] / 10)

    fig = px.scatter(
        df_plot,
        x="Beta",
        y="Specific_Risk",
        size="Total_Vol",
        color="Ticker",
        hover_name="Ticker",
        text="Ticker",
        title="Cartographie du Risque : Systématique vs Spécifique",
        labels={
            "Beta": "Sensibilité au Marché (Bêta)",
            "Specific_Risk": "Risque Propre à l'Entreprise (%)"
        },
        template="plotly_dark"
    )

    fig.update_traces(textposition='top center')

    return fig
