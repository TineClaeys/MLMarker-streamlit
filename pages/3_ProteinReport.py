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
import numpy as np
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

# --- Sample Selection Logic ---
has_batch_results = "batch_results" in st.session_state and st.session_state.batch_results
has_single_result = "prediction" in st.session_state and "prediction_summed" in st.session_state

if not has_batch_results and not has_single_result:
    mark_says("Markverse/cropped_images/Bald Mark reading a book.png", "I need prediction data first! Run MLMarker on the Home page.")
    st.warning("No prediction data found. Please go to the **Home** page, upload your data, and run MLMarker first.")
    st.stop()

# Sample selector for batch results
if has_batch_results:
    mark_says("Markverse/cropped_images/Coding Mark.png", "Multiple samples ready for ORA! Pick one to analyze.")
    sample_options = list(st.session_state.batch_results.keys())
    
    default_idx = 0
    if "ora_selected_sample" in st.session_state and st.session_state.ora_selected_sample in sample_options:
        default_idx = sample_options.index(st.session_state.ora_selected_sample)
    
    selected_sample = st.selectbox(
        "Select sample for ORA",
        sample_options,
        index=default_idx,
        key="ora_sample_selector"
    )
    st.session_state.ora_selected_sample = selected_sample
    
    # Get prediction data for selected sample
    result = st.session_state.batch_results[selected_sample]
    prediction_df = result['prediction_df']
    prediction_summed = result['summed_pred']
    current_sample = selected_sample
else:
    # Use single sample results
    prediction_df = st.session_state["prediction"]
    prediction_summed = st.session_state["prediction_summed"]
    current_sample = st.session_state.get("sel_sample", "Sample")

st.markdown(f"### ORA for: **{current_sample}**")

# Select tissue and filters for protein selection
st.markdown("#### Select proteins for ORA")
tissues_list = prediction_summed.index.tolist()
selected_tissue = st.selectbox("Select tissue", options=tissues_list, key="ora_tissue",
    help="Choose tissue for protein selection.")
shap_filter = st.selectbox("Select model impact", options=["All", "Positive", "Negative"], key="ora_shap_filter",
    help="Positive: proteins supporting this tissue. Negative: proteins opposing it.")

# Get proteins based on selection
shap_df = prediction_df.loc[selected_tissue]
shap_df = shap_df[(shap_df.values != 0)]

if shap_filter == "Positive":
    shap_df = shap_df[shap_df > 0]
elif shap_filter == "Negative":
    shap_df = shap_df[shap_df < 0]

selected_proteins = list(shap_df.index)
st.session_state['selected_proteins'] = selected_proteins

st.write(f"**{len(selected_proteins)}** proteins selected for {selected_tissue} ({shap_filter} impact)")

if len(selected_proteins) == 0:
    st.error("No proteins match the current filter criteria. Please adjust your filters.")
    st.stop()


def visualise_go_enrichment(df, title, proteins, max_bars=20):
    df['-log10(p_value)'] = -np.log10(df['p_value'])
    df_sorted = df.sort_values(by="-log10(p_value)", ascending=False)

    sources = sorted(df_sorted['source'].unique())
    source_labels = {
        "GO:BP": "GO: Biological Process",
        "GO:MF": "GO: Molecular Function",
        "GO:CC": "GO: Cellular Component",
        "HPA": "Human Protein Atlas",
        "KEGG": "KEGG Pathways"
    }
    
    subplot_titles = [source_labels.get(src, src) for src in sources]
    
    # Calculate rows needed (2 columns)
    n_rows = (len(sources) + 1) // 2
    
    fig = make_subplots(
        rows=max(1, n_rows), cols=2,
        subplot_titles=subplot_titles,
        shared_yaxes=False,
        shared_xaxes=False,
        vertical_spacing=0.18,
        horizontal_spacing=0.12,
    )

    for i, source in enumerate(sources, start=1):
        source_df = df_sorted[df_sorted['source'] == source]
        source_df = source_df.head(max_bars)
        
        trace = go.Bar(
            x=source_df['-log10(p_value)'],
            y=source_df['name'],
            orientation='h',
            name=source,
            hoverinfo='x+y',
            marker=dict(opacity=0.8, line=dict(width=1, color='DarkSlateGrey'))
        )

        row = (i - 1) // 2 + 1
        col = (i - 1) % 2 + 1

        fig.add_trace(trace, row=row, col=col)

    fig.update_layout(
        template="plotly_white",
        title=f"{title} Over-Representation Analysis Term Rank (Max 20 bars) for {proteins} proteins",
        title_x=0.5,
        xaxis_title="-log10(p-value)",
        yaxis_title="GO Terms",
        height=max(800, 500 * n_rows),
        width=1800,
        showlegend=False
    )

    return fig


with st.sidebar:
    mark_says("Markverse/cropped_images/Coding Mark.png", "Let's explore the functional context of your selected proteins!")
    st.markdown(f"**Current Sample:** {current_sample}")

# Run ORA button
if st.button("Run Over-Representation Analysis", type="primary"):
    gp = GProfiler(return_dataframe=True)

    with st.spinner("Running over-representation analysis... This may take a moment."):
        results = gp.profile(
            organism='hsapiens', 
            query=selected_proteins, 
            sources=['GO:BP', 'GO:MF', 'GO:CC', 'HPA', 'KEGG']
        )
    
    if results.empty:
        st.warning("No significant enrichment found for the selected proteins.")
    else:
        st.session_state['ora_results'] = results
        st.success(f"Found {len(results)} enriched terms!")

# Display results if available
if 'ora_results' in st.session_state and not st.session_state['ora_results'].empty:
    results = st.session_state['ora_results']
    
    # p-value filter selection
    p_value_filter = st.selectbox("Select p-value filter", options=["0.001", "0.01", "0.05"], key="pval_filter",
        help="Filter results by significance level.")
    filtered_results = results[results['p_value'] <= float(p_value_filter)]

    st.write(f"**{len(filtered_results)}** terms with p-value ≤ {p_value_filter}")
    
    with st.expander("View raw results table"):
        st.dataframe(filtered_results)

    if not filtered_results.empty:
        with st.spinner("Generating visualization..."):
            bigfig = visualise_go_enrichment(
                filtered_results, 
                title=f"ORA - {current_sample}", 
                proteins=len(selected_proteins)
            )
        st.plotly_chart(bigfig, use_container_width=True)
        
        # Download button
        csv_results = filtered_results.to_csv(index=False)
        st.download_button(
            label="Download ORA Results",
            data=csv_results,
            file_name=f"ora_results_{current_sample}_{selected_tissue}.csv",
            mime="text/csv"
        )
    else:
        st.warning(f"No terms pass the p-value filter of {p_value_filter}. Try a less stringent threshold.")
