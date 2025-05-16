import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
scaler = MinMaxScaler()
from plotly.subplots import make_subplots
import mlmarker
import seaborn as sns
import matplotlib.pyplot as plt
from mlmarker.explainability import get_hpa_info
import numpy as np# GO enrichment
from mlmarker import get_go_enrichment
from plotly.subplots import make_subplots
from gprofiler import GProfiler
import plotly.graph_objects as go

st.set_page_config(page_title="MLMarker", page_icon=":octopus:", layout='wide')
st.logo('octopus.png')
st.header("GO Enrichment Analysis of the protein set selected in the previous step")
st.write("This section provides a GO enrichment analysis of the selected proteins. The analysis includes the following sources: GO:BP (Biological Process), GO:MF (Molecular Function), GO:CC (Cellular Component), HPA (Human Protein Atlas), and KEGG (Kyoto Encyclopedia of Genes and Genomes).")
st.write("The results are filtered based on the selected p-value threshold. The top 20 GO terms for each source are visualized in a bar plot, with the -log10(p-value) on the x-axis and the GO terms on the y-axis. The plot is divided into subplots for each source, allowing for easy comparison of enrichment across different biological contexts.")

st.write(f"**You selected {len(list(st.session_state['selected_proteins']))} proteins for GO enrichment analysis.**")


def visualise_go_enrichment(df, title, proteins, max_bars=20):
    # Add a -log10(p-value) column for better visualization of significance
    df['-log10(p_value)'] = -np.log10(df['p_value'])

    # Sort the data by -log10(p_value) for ranking from high to low
    df_sorted = df.sort_values(by="-log10(p_value)", ascending=False)

    # Get the unique source types (e.g., 'GO:CC', 'GO:BP', 'HPA')
    sources = sorted(df_sorted['source'].unique())
    max_log_p_value = df_sorted['-log10(p_value)'].max()

    # Create subplots: 3 rows and 2 columns
    fig = make_subplots(
        rows=3, cols=2,  # 3 rows, 2 columns
        subplot_titles=sources,  # Titles for each subplot (based on source)
        shared_yaxes=False,
        shared_xaxes=False,
        vertical_spacing=0.18,
        horizontal_spacing=0.12,
    )


    # Loop over each source and add a bar trace
    for i, source in enumerate(sources, start=1):
        source_df = df_sorted[df_sorted['source'] == source]
        source_df = source_df.head(max_bars)
        # Create the bar trace for this source
        trace = go.Bar(
            x=source_df['-log10(p_value)'],
            y=source_df['name'],
            orientation='h',
            name=source,
            hoverinfo='x+y',
            marker=dict(opacity=0.8, line=dict(width=1, color='DarkSlateGrey'))
        )

        # Calculate the row and column indices based on the loop counter
        row = (i - 1) // 2 + 1
        col = (i - 1) % 2 + 1

        # Add the trace to the corresponding subplot
        fig.add_trace(trace, row=row, col=col)

    # Update layout
    fig.update_layout(
        template="plotly_white",
        title=f"{title} GO Enrichment Term Rank (Max 20 bars) for {proteins} proteins",
        title_x=0.5,
        xaxis_title="-log10(p-value)",
        yaxis_title="GO Terms",
        height=2500,
        width=1800,  # Adjust the width to fit three columns
        showlegend=False
    )

    # Show the plot
    return fig




from gprofiler import GProfiler
import plotly.graph_objects as go
# Initialize g:Profiler
gp = GProfiler(return_dataframe=True)

# Dictionary to store GO terms and p-values for each tissue
go_dict = {}


# Perform GO enrichment
results = gp.profile(organism='hsapiens', query=list(st.session_state['selected_proteins']), sources=['GO:BP', 'GO:MF', 'GO:CC', 'HPA', 'KEGG'])

#p-value filter selection
p_value_filter = st.selectbox("Select p-value filter", options=["0.001", "0.01", "0.05"])
results = results[results['p_value'] <= float(p_value_filter)]

st.write(results)

bigfig = visualise_go_enrichment(results, title="GO Enrichment", proteins=len(list(st.session_state['selected_proteins'])))
# Store results in the dictionary: {tissue: {GO_term: p-value}}

st.plotly_chart(bigfig)
