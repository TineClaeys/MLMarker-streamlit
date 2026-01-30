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
from mlmarker.utils import get_hpa_info
import numpy as np# GO enrichment
from mlmarker.utils import get_go_enrichment
from plotly.subplots import make_subplots
from gprofiler import GProfiler
import plotly.graph_objects as go
import io
import streamlit.components.v1 as components
import base64
from custom_functions import mark_says







st.set_page_config(page_title="MLMarker", page_icon=":octopus:", layout='wide')
st.logo('octopus.png')

st.header("Over-Representation Analysis")

st.write("""
Analyze the functional context of your selected proteins using over-representation analysis (ORA) from multiple sources:
- **GO:BP** (Biological Process)
- **GO:MF** (Molecular Function)
- **GO:CC** (Cellular Component)
- **HPA** (Human Protein Atlas)
- **KEGG** (Pathways)

Results are filtered by p-value and visualized as grouped bar plots (top 20 terms per source), with **-log10(p-value)** on the x-axis and GO terms on the y-axis.

""")

#check if proteins were selected in the previous step
if 'selected_proteins' not in st.session_state:
    st.error("No proteins selected yet. Please select proteins from the previous step to perform over-representation analysis.")
    #make sure the rest of the page doesn't load
    st.stop()


def visualise_go_enrichment(df, title, proteins, max_bars=20):
    # Add a -log10(p-value) column for better visualization of significance
    df['-log10(p_value)'] = -np.log10(df['p_value'])

    # Sort the data by -log10(p_value) for ranking from high to low
    df_sorted = df.sort_values(by="-log10(p_value)", ascending=False)

    # Get the unique source types (e.g., 'GO:CC', 'GO:BP', 'HPA')
    sources = sorted(df_sorted['source'].unique())
    source_labels = {
        "GO:BP": "GO: Biological Process",
        "GO:MF": "GO: Molecular Function",
        "GO:CC": "GO: Cellular Component",
        "HPA": "Human Protein Atlas",
        "KEGG": "KEGG Pathways"
    }
    max_log_p_value = df_sorted['-log10(p_value)'].max()
    subplot_titles = [source_labels.get(src, src) for src in sources] 
    # Create subplots: 3 rows and 2 columns
    fig = make_subplots(
        rows=3, cols=2,  # 3 rows, 2 columns
        subplot_titles=subplot_titles ,  # Titles for each subplot (based on source)
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
        title=f"{title} Over-Representation Analysis Term Rank (Max 20 bars) for {proteins} proteins",
        title_x=0.5,
        xaxis_title="-log10(p-value)",
        yaxis_title="GO Terms",
        height=2500,
        width=1800,  # Adjust the width to fit three columns
        showlegend=False
    )

    # Show the plot
    return fig


with st.sidebar:
    mark_says("Markverse/cropped_images/Coding Mark.png", "Let's explore the functional context of your selected proteins!")

from gprofiler import GProfiler
import plotly.graph_objects as go
# Initialize g:Profiler
gp = GProfiler(return_dataframe=True)

# Dictionary to store GO terms and p-values for each tissue
go_dict = {}

# Perform over-representation analysis with spinner
with st.spinner("Running over-representation analysis... This may take a moment."):
    results = gp.profile(organism='hsapiens', query=list(st.session_state['selected_proteins']), sources=['GO:BP', 'GO:MF', 'GO:CC', 'HPA', 'KEGG'])

#p-value filter selection
p_value_filter = st.selectbox("Select p-value filter", options=["0.001", "0.01", "0.05"])
results = results[results['p_value'] <= float(p_value_filter)]

st.write(results)

with st.spinner("Generating visualization..."):
    bigfig = visualise_go_enrichment(results, title="ORA", proteins=len(list(st.session_state['selected_proteins'])))

st.plotly_chart(bigfig)
