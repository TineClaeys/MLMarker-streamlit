import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from gprofiler import GProfiler
import streamlit.components.v1 as components

# Import custom functions with fallbacks
from custom_functions import mark_says, render_sample_selector, get_sample_data
try:
    from custom_functions import copy_to_clipboard_button, show_help, HELP_CONTENT
except ImportError:
    def copy_to_clipboard_button(text, label="Copy", key=None): pass
    def show_help(topic, title=None): pass
    HELP_CONTENT = {}

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
    mark_says("Markverse/mark pointing.png", "No predictions yet! Run MLMarker first.")
    st.warning("No prediction data. Go to **Home** and run MLMarker first.")
    st.stop()

# Check for abundance data
if "df" not in st.session_state:
    df = None
else:
    df = st.session_state["df"]

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
    mark_says("Markverse/Markwithamassspec.png", "Let's find what functions your proteins have!")

st.markdown(f"**Current Sample:** {current_sample}")

# ==============================================================================
# SECTION: Protein Selection
# ==============================================================================
if show_selection:
    st.markdown("---")
    st.markdown("## Protein Selection")
    
    # Check if we have proteins from ProteinExplorer
    has_explorer_proteins = 'selected_proteins' in st.session_state and len(st.session_state.get('selected_proteins', [])) > 0
    explorer_tissue = st.session_state.get('selected_tissue_for_ora', None)
    
    # Option to use proteins from ProteinExplorer
    if has_explorer_proteins:
        st.info(f"**{len(st.session_state['selected_proteins'])} proteins** transferred from Protein Explorer (Tissue: {explorer_tissue})")
        use_explorer = st.checkbox("Use proteins from Protein Explorer", value=True, key="use_explorer_proteins")
    else:
        use_explorer = False
        st.caption("Tip: Select proteins in **Protein Explorer** first, then come here to run ORA on them.")
    
    if use_explorer:
        # Use proteins directly from ProteinExplorer
        selected_proteins = st.session_state['selected_proteins']
        selected_tissue = explorer_tissue
        st.success(f"Using **{len(selected_proteins)} proteins** from Protein Explorer for **{selected_tissue}**")
    else:
        # Manual selection with same filters as ProteinExplorer
        st.markdown("### Select Filters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            tissues = prediction_summed.index.tolist()
            default_tissue = explorer_tissue if explorer_tissue in tissues else tissues[0]
            
            selected_tissue = st.selectbox(
                "Tissue", tissues,
                index=tissues.index(default_tissue),
                key="ora_tissue"
            )
        
        with col2:
            abundance_filter = st.selectbox(
                "Abundance", ["All", "Present", "Absent"],
                key="ora_abundance",
                help="Filter by protein presence in your sample"
            )
        
        with col3:
            shap_filter = st.selectbox(
                "Impact", ["All", "Pro (positive)", "Con (negative)"],
                key="ora_shap"
            )
        
        # Get proteins based on filters
        shap_data = prediction_df.loc[selected_tissue]
        shap_data = shap_data[shap_data != 0]
        
        if "Pro" in shap_filter:
            shap_data = shap_data[shap_data > 0]
        elif "Con" in shap_filter:
            shap_data = shap_data[shap_data < 0]
        
        # Apply abundance filter if we have abundance data
        if df is not None and abundance_filter != "All":
            abundance_data = df.loc[current_sample].fillna(0)
            if abundance_filter == "Present":
                valid_proteins = abundance_data[abundance_data > 0].index
                shap_data = shap_data[shap_data.index.isin(valid_proteins)]
            elif abundance_filter == "Absent":
                valid_proteins = abundance_data[abundance_data == 0].index
                shap_data = shap_data[shap_data.index.isin(valid_proteins)]
        elif df is None and abundance_filter != "All":
            st.warning("No abundance data available. Showing all proteins regardless of abundance filter.")
        
        selected_proteins = list(shap_data.index)
        st.markdown(f"**{len(selected_proteins)} proteins** match filters")
    
    st.session_state['ora_proteins'] = selected_proteins
    
    if len(selected_proteins) == 0:
        st.error("No proteins match filters. Adjust selection.")
        st.stop()
    
    # Copy protein IDs button
    col_count, col_copy = st.columns([2, 1])
    with col_count:
        st.caption(f"{len(selected_proteins)} proteins ready for ORA")
    with col_copy:
        copy_to_clipboard_button(selected_proteins, "📋 Copy IDs", key=f"copy_ora_{selected_tissue}")

# --- Run ORA ---
st.markdown("---")

col_run, col_pval = st.columns([2, 1])
with col_pval:
    p_threshold = st.selectbox("P-value threshold", [0.001, 0.01, 0.05], index=1, key="pval")

with col_run:
    run_button = st.button("Run Over-Representation Analysis", type="primary", width='content')

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
        mark_says("Markverse/markgraduation.png", 
                  f"Found {len(results)} enriched terms! You're learning so much about your proteins!")

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
        
        st.dataframe(display_df, width='content', height=400)
        
        # Download and copy
        col_dl, col_cp = st.columns(2)
        with col_dl:
            st.download_button(
                "📥 Download Results",
                filtered.to_csv(index=False),
                f"ora_{current_sample}_{selected_tissue}.csv",
                "text/csv"
            )
        with col_cp:
            # Copy term names to clipboard
            term_names = '\n'.join(filtered['name'].tolist())
            copy_to_clipboard_button(term_names, "📋 Copy Terms", key=f"copy_terms_{selected_tissue}")
    
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
                    st.plotly_chart(fig, width='content')

elif 'ora_results' not in st.session_state:
    st.info("Click **Run Over-Representation Analysis** to analyze your proteins.")
