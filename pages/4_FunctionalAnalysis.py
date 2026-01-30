import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from gprofiler import GProfiler
from custom_functions import mark_says, render_sample_selector, get_sample_data

st.set_page_config(page_title="Functional Analysis - MLMarker", page_icon=":octopus:", layout='wide')
st.logo('octopus.png')

# --- Header ---
st.title("Functional Analysis")
st.markdown("""
Perform **Over-Representation Analysis (ORA)** on proteins driving tissue predictions. 
This analysis uses g:Profiler to identify enriched biological processes, molecular functions, 
cellular components, and pathways.

Select proteins that **support** (pro) or **oppose** (con) a tissue prediction, 
then run ORA to understand their biological functions.
""")

# --- Check data ---
prediction_df, prediction_summed, current_sample, is_batch = get_sample_data()

if prediction_df is None:
    mark_says("Markverse/cropped_images/Bald Mark reading a book.png", "No predictions yet! Run MLMarker first.")
    st.warning("No prediction data. Go to **Home** and run MLMarker first.")
    st.stop()

# --- Sidebar ---
with st.sidebar:
    # Sample selector
    if is_batch:
        st.markdown("### Sample")
        current_sample = render_sample_selector("ora")
        result = st.session_state.batch_results[current_sample]
        prediction_df = result['prediction_df']
        prediction_summed = result['summed_pred']
    
    st.markdown("### Analysis Options")
    show_selection = st.checkbox("Protein Selection", value=True)
    show_results = st.checkbox("Results Table", value=False)
    show_visual = st.checkbox("Visualization", value=False)
    
    st.markdown("---")
    mark_says("Markverse/cropped_images/Coding Mark.png", "Let's find what functions your proteins have!")

st.markdown(f"**Current Sample:** {current_sample}")

# ==============================================================================
# SECTION: Protein Selection
# ==============================================================================
if show_selection:
    st.markdown("---")
    st.markdown("## Protein Selection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        tissues = prediction_summed.index.tolist()
        # Try to use tissue from ProteinExplorer if available
        default_tissue = st.session_state.get('selected_tissue_for_ora', tissues[0])
        if default_tissue not in tissues:
            default_tissue = tissues[0]
        
        selected_tissue = st.selectbox(
            "Tissue", tissues,
            index=tissues.index(default_tissue),
            key="ora_tissue"
        )
    
    with col2:
        shap_filter = st.selectbox(
            "Impact", ["All", "Pro (positive)", "Con (negative)"],
            key="ora_shap"
        )
    
    # Get proteins
    shap_data = prediction_df.loc[selected_tissue]
    shap_data = shap_data[shap_data != 0]
    
    if "Pro" in shap_filter:
        shap_data = shap_data[shap_data > 0]
    elif "Con" in shap_filter:
        shap_data = shap_data[shap_data < 0]
    
    selected_proteins = list(shap_data.index)
    st.session_state['ora_proteins'] = selected_proteins
    
    # Check if proteins came from ProteinExplorer
    if 'selected_proteins' in st.session_state:
        prev_proteins = st.session_state['selected_proteins']
        if set(prev_proteins) != set(selected_proteins):
            st.info(f"Using {len(selected_proteins)} proteins for {selected_tissue}. "
                   f"(Previous selection from Protein Explorer: {len(prev_proteins)} proteins)")
    
    st.markdown(f"**{len(selected_proteins)} proteins** selected for ORA")
    
    if len(selected_proteins) == 0:
        st.error("No proteins match filters. Adjust selection.")
        st.stop()

# --- Run ORA ---
st.markdown("---")

col_run, col_pval = st.columns([2, 1])
with col_pval:
    p_threshold = st.selectbox("P-value threshold", [0.001, 0.01, 0.05], index=1, key="pval")

with col_run:
    run_button = st.button("Run Over-Representation Analysis", type="primary", use_container_width=True)

if run_button:
    if len(selected_proteins) < 3:
        st.error("Need at least 3 proteins for ORA.")
        st.stop()
    
    gp = GProfiler(return_dataframe=True)
    
    with st.spinner("Running ORA..."):
        results = gp.profile(
            organism='hsapiens',
            query=selected_proteins,
            sources=['GO:BP', 'GO:MF', 'GO:CC', 'HPA', 'KEGG']
        )
    
    if results.empty:
        st.warning("No significant enrichment found.")
    else:
        st.session_state['ora_results'] = results
        st.session_state['ora_sample'] = current_sample
        st.session_state['ora_tissue_used'] = selected_tissue
        st.success(f"Found {len(results)} enriched terms!")
        mark_says("Markverse/cropped_images/Mark digging for gold.png", 
                  f"Found {len(results)} enriched terms! Check out the biological functions below.")

# ==============================================================================
# SECTION: Results
# ==============================================================================
if 'ora_results' in st.session_state and not st.session_state['ora_results'].empty:
    results = st.session_state['ora_results']
    filtered = results[results['p_value'] <= p_threshold]
    
    st.markdown(f"**{len(filtered)} terms** with p-value ≤ {p_threshold}")
    
    if show_results and len(filtered) > 0:
        st.markdown("---")
        st.markdown("## Results Table")
        
        # Summary by source
        source_counts = filtered['source'].value_counts()
        cols = st.columns(len(source_counts))
        for i, (source, count) in enumerate(source_counts.items()):
            cols[i].metric(source, count)
        
        # Table
        display_cols = ['source', 'name', 'p_value', 'intersection_size', 'term_size']
        display_df = filtered[display_cols].copy()
        display_df.columns = ['Source', 'Term', 'P-value', 'Hits', 'Term Size']
        display_df['P-value'] = display_df['P-value'].apply(lambda x: f"{x:.2e}")
        
        st.dataframe(display_df, use_container_width=True, height=400)
        
        # Download
        st.download_button(
            "Download Results",
            filtered.to_csv(index=False),
            f"ora_{current_sample}_{selected_tissue}.csv",
            "text/csv"
        )
    
    if show_visual and len(filtered) > 0:
        st.markdown("---")
        st.markdown("## Visualization")
        
        # Add -log10 p-value
        plot_df = filtered.copy()
        plot_df['-log10(p)'] = -np.log10(plot_df['p_value'])
        
        sources = sorted(plot_df['source'].unique())
        
        # Create subplots - one per source
        n_sources = len(sources)
        
        if n_sources > 0:
            tabs = st.tabs(sources)
            
            for i, source in enumerate(sources):
                with tabs[i]:
                    source_df = plot_df[plot_df['source'] == source].nlargest(15, '-log10(p)')
                    
                    fig = px.bar(
                        source_df,
                        x='-log10(p)',
                        y='name',
                        orientation='h',
                        title=f"Top {source} Terms",
                        hover_data=['p_value', 'intersection_size']
                    )
                    fig.update_layout(
                        height=max(300, 25 * len(source_df)),
                        margin=dict(t=40, b=20),
                        yaxis={'categoryorder': 'total ascending'},
                        xaxis_title='-log10(p-value)',
                        yaxis_title=''
                    )
                    st.plotly_chart(fig, use_container_width=True)

elif 'ora_results' not in st.session_state:
    st.info("Click **Run Over-Representation Analysis** to analyze your proteins.")
