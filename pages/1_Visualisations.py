import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import shap
import mlmarker
from mlmarker.model import MLMarker
import seaborn as sns
import matplotlib.pyplot as plt
scaler = MinMaxScaler()
from plotly.subplots import make_subplots
from gprofiler import GProfiler
import plotly.graph_objects as go
st.set_page_config(page_title="MLMarker", page_icon=":octopus:", layout='wide')
st.logo('octopus.png', size='large')
protein_df = pd.read_csv('MLMarker_features_bioservice_return.csv')
st.session_state["protein_df"] = protein_df
def visualise_custom_tissue_plot(df, tissue_name, top_n=10, show_others=False, threshold_others = 0.001):
    df = df.loc[[tissue_name]]

    # Separate positive and negative values for the tissue
    positive_contributions = df.clip(lower=0)  # Keep only positive values
    negative_contributions = df.clip(upper=0).abs()  # Keep absolute values of negatives

    # Filter significant contributions
    positive_main = positive_contributions.loc[:, (positive_contributions > threshold_others).any()]
    positive_others = positive_contributions.loc[:, (positive_contributions <= threshold_others).all()].sum(axis=1)

    negative_main = negative_contributions.loc[:, (negative_contributions > threshold_others).any()]
    negative_others = negative_contributions.loc[:, (negative_contributions <= threshold_others).all()].sum(axis=1)

    # Sort positive and negative contributions by total value
    sorted_positive = positive_main.sum(axis=0).sort_values(ascending=False)
    sorted_negative = negative_main.sum(axis=0).sort_values(ascending=False)

    # Select top N positive and negative proteins
    top_positive_contributions = sorted_positive.head(top_n).index.tolist()
    top_negative_contributions = sorted_negative.head(top_n).index.tolist()

    # Plotting
    fig = go.Figure()

    # Add all positive contributions (green bars)
    for protein in sorted_positive.index:
        # Check if the protein is one of the top N and add its label
        is_top = protein in top_positive_contributions
        fig.add_trace(
            go.Bar(
                x=positive_contributions.index,
                y=positive_main[protein],
                name=protein,
                marker_color="green" if is_top else "darkgreen",
                hoverinfo="name+y",
                hoverlabel=dict(namelength=-1),
                showlegend=False,
                text=protein if is_top else None,  # Show label for top proteins
                textposition="outside",  # Position the label inside the bar
                cliponaxis=False,  # Allow the label to be outside the bar
            )
        )
    # Add lines for top proteins to connect labels outside the bars
    for protein in top_positive_contributions:
        fig.add_trace(
            go.Scatter(
                x=[positive_contributions.index[0], positive_contributions.index[0]],
                y=[positive_contributions[protein].min(), positive_contributions[protein].max()],
                mode="lines+text",
                line=dict(color="green", width=2, dash="dot"),  # Line connecting label to bar
                text=[protein],
                textposition="middle right",
                showlegend=False,
                textfont=dict(color="green", size=12)
            )
        )
    # Add "Others" for positive contributions
    if show_others and positive_others.sum() > 0:
        fig.add_trace(
            go.Bar(
                x=positive_contributions.index,
                y=positive_others,
                name="Others (Positive)",
                marker_color="lightgreen",
                hoverinfo="name+y",
                hoverlabel=dict(namelength=-1),
                showlegend=False,
            )
        )

  # Add negative contributions (sorted by total contribution)
    for protein in sorted_negative.index:
        is_top = protein in top_negative_contributions
        fig.add_trace(
            go.Bar(
                x=negative_contributions.index,
                y=negative_main[protein],
                name=protein,
                marker_color="red" if is_top else "darkred",
                hoverinfo="name+y",
                hoverlabel=dict(namelength=-1),
                showlegend=False,
                text=protein if is_top else None,  # Show label for top proteins
                textposition="outside",  # Position the label outside the bar
                cliponaxis=False,  # Allow the label to be outside the bar
            )
        )

    # Add "Others" for negative contributions
    if show_others and negative_others.sum() > 0:
        fig.add_trace(
            go.Bar(
                x=negative_contributions.index,
                y=negative_others,
                name="Others (Negative)",
                marker_color="lightcoral",
                hoverinfo="name+y",
                hoverlabel=dict(namelength=-1),
                showlegend=False,
            )
        )

    # Customizing layout
    fig.update_layout(
        barmode="stack",  # Stack the bars
        title=f"""Protein Contributions for {tissue_name} (threshold={threshold_others})""",
        xaxis_title="Cluster",
        yaxis_title="Protein Contributions",
        xaxis={"categoryorder": "array", "categoryarray": sorted_positive.index.tolist() + sorted_negative.index.tolist()},
        hovermode="closest",
        template="plotly_white",
        width=600,
        height=800,
        margin=dict(l=100, r=100),  # Adjust margins
    )
    return fig


def visualise_custom_plot(df):
        
    # Aggregate positive and negative contributions per tissue
    positive_totals = df.clip(lower=0).sum(axis=1)
    negative_totals = df.clip(upper=0).abs().sum(axis=1)

    # Create the figure
    fig = go.Figure()

    # Add positive contributions (green bars)
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=positive_totals,
            name="Positive Contributions",
            marker_color='green',
            hoverinfo='x+y',
        )
    )

    # Add negative contributions (red bars)
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=negative_totals,
            name="Negative Contributions",
            marker_color='red',
            hoverinfo='x+y',
        )
    )

    # Customizing layout
    fig.update_layout(
        barmode='group',  # Group positive and negative bars side-by-side
        title='Grouped Barplot of Total Protein Contributions by Tissue',
        xaxis_title='Tissues',
        yaxis_title='Total Contributions',
        xaxis=dict(tickangle=-45),  # Tilt the x-axis labels for better readability
        template="plotly_white"
    )

    return fig


def scatterplot_of_proteins(df, selected_tissues):
    fig = px.scatter(x=df.loc[selected_tissues[0]], y =df.loc[selected_tissues[1]], hover_name = df.columns)    

    fig.add_shape(
        go.layout.Shape(type="line", x0=0, x1=0, y0=min(df.loc[selected_tissues[1]]), 
                        y1=max(df.loc[selected_tissues[1]]), line=dict(color="black", width=2))
    )
    fig.add_shape(
        go.layout.Shape(type="line", y0=0, y1=0, x0=min(df.loc[selected_tissues[0]]), 
                        x1=max(df.loc[selected_tissues[0]]), line=dict(color="black", width=2))
    )
    
    fig.update_layout(
        xaxis_title=selected_tissues[0],
        yaxis_title=selected_tissues[1],
        showlegend=False
    )
    
    return fig


st.title('Visualisations')

if "prediction" in st.session_state and "prediction_summed" in st.session_state:

    bigfig = visualise_custom_plot(df=st.session_state["prediction"])
    st.plotly_chart(bigfig)
    st.session_state['bigfig'] = bigfig



tissues_list = st.session_state["prediction_summed"].index.tolist()
selected_tissue = st.selectbox("Select tissue for custom forceplot", options=tissues_list)
if st.button("Generate Tissuespecific ForcePlot"):
    smallfig = visualise_custom_tissue_plot(st.session_state["prediction"], selected_tissue)
    st.plotly_chart(smallfig)
    st.session_state['smallfig'] = smallfig

selected_tissues = st.multiselect('Select two tissues for a protein level visualisation', options=tissues_list)
if st.button('Generate distribution'):
    subset = st.session_state["prediction"][st.session_state["prediction"].index.isin(selected_tissues)]
    fig = scatterplot_of_proteins(subset, selected_tissues)
    st.plotly_chart(fig)



with st.sidebar:
    st.download_button("Download Tissue level prediction", st.session_state["prediction_summed"].to_csv().encode(), "prediction_summed.csv", "text/csv")
    st.download_button("Download Protein level prediction", st.session_state["prediction"].to_csv().encode(), "prediction.csv", "text/csv")


