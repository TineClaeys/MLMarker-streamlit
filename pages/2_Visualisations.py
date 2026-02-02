import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Import custom functions with fallbacks
from custom_functions import mark_says, render_sample_selector, get_sample_data
try:
    from custom_functions import show_help, HELP_CONTENT
except ImportError:
    def show_help(topic, title=None): pass
    HELP_CONTENT = {}

st.set_page_config(page_title="Visualisations - MLMarker", page_icon=":octopus:", layout='wide')
st.logo('octopus.png')

# Load protein info for name lookup
protein_df = pd.read_csv('MLMarker_features_bioservice_return.csv')
protein_name_map = dict(zip(protein_df['id'], protein_df['protein_names']))

# --- Header ---
st.title("Visualisations")
st.markdown("""
Explore how MLMarker interprets your sample through **SHAP values** (SHapley Additive exPlanations). 
Each protein contributes positively (**pro**) or negatively (**con**) to each tissue prediction.

- **Tissue Overview**: See total positive vs negative contributions per tissue
- **Tissue forceplot**: Identify which proteins drive predictions for a specific tissue
- **Protein Comparison**: Compare how the same proteins contribute to different tissues
""")

# --- Check data ---
prediction_df, prediction_summed, current_sample, is_batch = get_sample_data()

if prediction_df is None:
    mark_says("Markverse/mark pointing.png", "No predictions yet! Run MLMarker on the Home page first.")
    st.warning("No prediction data. Go to **Home** and run MLMarker first.")
    st.stop()

# --- Sidebar ---
with st.sidebar:
    # Sample selector
    if is_batch:
        st.markdown("### Sample")
        current_sample = render_sample_selector("vis")
        result = st.session_state.batch_results[current_sample]
        prediction_df = result['prediction_df']
        prediction_summed = result['summed_pred']
    
    st.markdown("### Analysis Options")
    show_overview = st.checkbox("Tissue Overview", value=True)
    show_tissue_detail = st.checkbox("Tissue forceplot", value=False)
    show_scatter = st.checkbox("Protein Comparison", value=False)
    
    st.markdown("---")
    st.download_button(
        "Download Predictions",
        prediction_df.to_csv(),
        f"prediction_{current_sample}.csv",
        "text/csv",
        width='content'
    )
    
    st.markdown("---")
    mark_says("Markverse/mark pointing.png", f"Viewing: {current_sample}")

st.markdown(f"**Current Sample:** {current_sample}")

# --- Helper functions ---
def visualise_tissue_overview(df):
    """Bar chart of positive/negative contributions per tissue."""
    positive = df.clip(lower=0).sum(axis=1)
    negative = df.clip(upper=0).abs().sum(axis=1)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df.index, y=positive,
        name="Pro", marker_color='#27ae60'
    ))
    fig.add_trace(go.Bar(
        x=df.index, y=negative,
        name="Con", marker_color='#c0392b'
    ))
    fig.update_layout(
        barmode='group',
        title='Tissue Contributions Overview',
        xaxis_title='Tissue',
        yaxis_title='Total SHAP Contribution',
        xaxis=dict(tickangle=-45),
        template="plotly_white",
        height=450,
        margin=dict(t=50, b=100)
    )
    return fig


def visualise_tissue_forceplot(df, tissue_name, top_n=10):
    """Stacked bar showing top pro/con proteins for a tissue."""
    tissue_data = df.loc[tissue_name]
    
    # Get top positive and negative
    positive = tissue_data[tissue_data > 0].nlargest(top_n)
    negative = tissue_data[tissue_data < 0].nsmallest(top_n).abs()
    
    fig = go.Figure()
    
    # Positive proteins
    for protein in positive.index:
        fig.add_trace(go.Bar(
            x=[tissue_name], y=[positive[protein]],
            name=protein, marker_color='#27ae60',
            hovertemplate=f"{protein}: %{{y:.4f}}<extra></extra>",
            showlegend=False
        ))
    
    # Negative proteins
    for protein in negative.index:
        fig.add_trace(go.Bar(
            x=[tissue_name], y=[negative[protein]],
            name=protein, marker_color='#c0392b',
            hovertemplate=f"{protein}: -%{{y:.4f}}<extra></extra>",
            showlegend=False
        ))
    
    fig.update_layout(
        barmode='stack',
        title=f"Top {top_n} Pro/Con Proteins for {tissue_name}",
        yaxis_title="SHAP Value",
        template="plotly_white",
        height=500
    )
    
    return fig, positive.index.tolist(), negative.index.tolist()


def scatterplot_tissues(df, tissue_a, tissue_b):
    """Scatter plot comparing SHAP values between two tissues."""
    fig = px.scatter(
        x=df.loc[tissue_a], y=df.loc[tissue_b],
        hover_name=df.columns,
        labels={'x': tissue_a, 'y': tissue_b},
        title=f"{tissue_a} vs {tissue_b}"
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(height=500, template="plotly_white")
    return fig


# ==============================================================================
# SECTION: Tissue Overview
# ==============================================================================
if show_overview:
    st.markdown("---")
    st.markdown("## Tissue Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = visualise_tissue_overview(prediction_df)
        st.plotly_chart(fig, width='content')
    
    with col2:
        st.markdown("### Top Predictions")
        top_5 = prediction_summed.nlargest(5)
        for tissue, prob in top_5.items():
            st.markdown(f"**{tissue}**: {prob:.1%}")
        
        mark_says("Markverse/Mark_on_a_rocket.png", 
                  f"Your top prediction is {top_5.index[0]} at {top_5.iloc[0]:.1%}!")

# ==============================================================================
# SECTION: Tissue forceplot
# ==============================================================================
if show_tissue_detail:
    st.markdown("---")
    st.markdown("## Tissue forceplot")
    
    tissues = prediction_summed.index.tolist()
    
    col_sel, col_opt = st.columns([2, 1])
    with col_sel:
        selected_tissue = st.selectbox("Select tissue", tissues, key="tissue_detail")
    with col_opt:
        top_n = st.slider("Top proteins to show", 5, 20, 10, key="top_n_proteins")
    
    if st.button("Generate Forceplot", key="btn_forceplot"):
        fig, pro_proteins, con_proteins = visualise_tissue_forceplot(
            prediction_df, selected_tissue, top_n
        )
        st.plotly_chart(fig, width='content')
        
        # Helper function to format protein with name
        def format_protein_list(proteins):
            formatted = []
            for p in proteins:
                name = protein_name_map.get(p, "Unknown")
                formatted.append(f"**{p}** ({name})")
            return formatted
        
        col_pro, col_con = st.columns(2)
        with col_pro:
            st.markdown("**Pro Proteins:**")
            for item in format_protein_list(pro_proteins):
                st.markdown(f"- {item}")
        with col_con:
            st.markdown("**Con Proteins:**")
            for item in format_protein_list(con_proteins):
                st.markdown(f"- {item}")
        
        mark_says("Markverse/cropped_images/Mark digging for gold.png", 
                  f"Found gold! {len(pro_proteins)} supporting and {len(con_proteins)} opposing proteins!")

# ==============================================================================
# SECTION: Protein Comparison
# ==============================================================================
if show_scatter:
    st.markdown("---")
    st.markdown("## Protein Comparison")
    st.caption("Compare how proteins contribute to two different tissues")
    
    tissues = prediction_summed.index.tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        tissue_a = st.selectbox("Tissue A", tissues, key="scatter_tissue_a")
    with col2:
        tissue_b = st.selectbox("Tissue B", tissues, 
                                index=min(1, len(tissues)-1), key="scatter_tissue_b")
    
    if tissue_a != tissue_b:
        if st.button("Compare Tissues", key="btn_scatter"):
            fig = scatterplot_tissues(prediction_df, tissue_a, tissue_b)
            st.plotly_chart(fig, width='content')
            mark_says("Markverse/Mark_touching_human_like_davincis.png", 
                      "Proteins in opposite quadrants have tissue-specific roles!")
    else:
        st.info("Select two different tissues to compare.")
