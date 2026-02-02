import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit.components.v1 as components

# Import custom functions with fallbacks
from custom_functions import mark_says, render_sample_selector, get_sample_data
try:
    from custom_functions import copy_to_clipboard_button, show_help
except ImportError:
    def copy_to_clipboard_button(text, label="Copy", key=None): pass
    def show_help(topic, title=None): pass

st.set_page_config(page_title="Protein Explorer - MLMarker", page_icon=":octopus:", layout='wide')
st.logo('octopus.png')

# --- Header ---
st.title("Protein Explorer")
st.markdown("""
Dig into individual proteins driving tissue predictions. Filter by **tissue**, **abundance** (present/absent in your sample), 
and **impact** (pro or con) to find the most relevant proteins.

The table links to UniProt for detailed protein information. Selected proteins can be analyzed 
for functional enrichment on the **Functional Analysis** page.
""")

# --- Check data ---
prediction_df, prediction_summed, current_sample, is_batch = get_sample_data()

if prediction_df is None:
    mark_says("Markverse/mark pointing.png", "No predictions yet! Run MLMarker first.")
    st.warning("No prediction data. Go to **Home** and run MLMarker first.")
    st.stop()

if "df" not in st.session_state:
    mark_says("Markverse/mark pointing.png", "No abundance data loaded! Upload on the Home page.")
    st.warning("No data found. Please upload data on **Home** page.")
    st.stop()

df = st.session_state["df"]
protein_df = pd.read_csv('MLMarker_features_bioservice_return.csv')

# --- Sidebar ---
with st.sidebar:
    # Sample selector
    if is_batch:
        st.markdown("### Sample")
        current_sample = render_sample_selector("protein")
        result = st.session_state.batch_results[current_sample]
        prediction_df = result['prediction_df']
        prediction_summed = result['summed_pred']
    
    st.markdown("### Analysis Options")
    show_table = st.checkbox("Protein Table", value=True)
    show_stats = st.checkbox("Statistics", value=False)
    
    st.markdown("---")
    mark_says("Markverse/Markwithamassspec.png", f"Exploring proteins for {current_sample}")

st.markdown(f"**Current Sample:** {current_sample}")

# --- Search & Custom Protein List ---
st.markdown("---")
col_search, col_upload = st.columns([2, 1])

with col_search:
    st.markdown("### 🔍 Search Proteins")
    search_query = st.text_input(
        "Search by UniProt ID or protein name",
        placeholder="e.g., P04406 or GAPDH or albumin",
        key="protein_search"
    )

with col_upload:
    st.markdown("### 📤 Custom Protein List")
    custom_proteins_input = st.text_area(
        "Paste protein IDs (one per line)",
        placeholder="P04406\nP68871\nP02768",
        height=100,
        key="custom_proteins"
    )

# Process custom protein list
custom_protein_set = set()
if custom_proteins_input:
    custom_protein_set = set(line.strip() for line in custom_proteins_input.split('\n') if line.strip())
    st.caption(f"📋 {len(custom_protein_set)} proteins in custom list")

# --- Tissue & Filter Selection ---
st.markdown("### Select Filters")
col1, col2, col3 = st.columns(3)

with col1:
    tissues = prediction_summed.index.tolist()
    selected_tissue = st.selectbox("Tissue", tissues, key="protein_tissue")

with col2:
    abundance_filter = st.selectbox("Abundance", ["All", "Present", "Absent"], key="abundance")

with col3:
    shap_filter = st.selectbox("Impact", ["All", "Pro (positive)", "Con (negative)"], key="shap")

# --- Get filtered data ---
shap_data = prediction_df.loc[selected_tissue]
shap_data = shap_data[shap_data != 0]

if "Pro" in shap_filter:
    shap_data = shap_data[shap_data > 0]
elif "Con" in shap_filter:
    shap_data = shap_data[shap_data < 0]

# Get abundance data
abundance_data = df.loc[current_sample].fillna(0)

if abundance_filter == "Present":
    valid_proteins = abundance_data[abundance_data > 0].index
    shap_data = shap_data[shap_data.index.isin(valid_proteins)]
elif abundance_filter == "Absent":
    valid_proteins = abundance_data[abundance_data == 0].index
    shap_data = shap_data[shap_data.index.isin(valid_proteins)]

# Apply search filter
if search_query:
    search_lower = search_query.lower()
    # Search in protein IDs and names
    matching_ids = protein_df[
        (protein_df['id'].str.lower().str.contains(search_lower, na=False)) |
        (protein_df['entry name'].str.lower().str.contains(search_lower, na=False)) |
        (protein_df['protein_names'].str.lower().str.contains(search_lower, na=False))
    ]['id'].tolist()
    shap_data = shap_data[shap_data.index.isin(matching_ids)]
    if len(shap_data) == 0:
        st.warning(f"No proteins found matching '{search_query}'")

# Apply custom protein list filter
if custom_protein_set:
    # Find proteins that are in the custom list AND in SHAP data
    matching_custom = shap_data.index.intersection(custom_protein_set)
    if len(matching_custom) > 0:
        shap_data = shap_data[shap_data.index.isin(custom_protein_set)]
        st.info(f"Showing {len(matching_custom)} of {len(custom_protein_set)} custom proteins (others not in MLMarker features)")
    else:
        st.warning("None of the custom proteins were found in MLMarker features")

# Store for ORA
st.session_state['selected_proteins'] = list(shap_data.index)
st.session_state['selected_tissue_for_ora'] = selected_tissue

st.markdown(f"**{len(shap_data)} proteins** match filters")

# ==============================================================================
# SECTION: Protein Table
# ==============================================================================
if show_table and len(shap_data) > 0:
    st.markdown("---")
    st.markdown("## Protein Table")
    
    # Build display dataframe
    display_proteins = shap_data.index.tolist()
    subset = protein_df[protein_df['id'].isin(display_proteins)].copy()
    subset['SHAP Value'] = subset['id'].map(shap_data)
    subset['Abundance'] = subset['id'].map(abundance_data)
    subset = subset.sort_values('SHAP Value', ascending=False, key=abs)
    
    # Pagination
    page_size = 25
    total_pages = max(1, (len(subset) + page_size - 1) // page_size)
    
    if "protein_page" not in st.session_state:
        st.session_state.protein_page = 1
    
    col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
    with col_nav1:
        if st.button("◀ Previous") and st.session_state.protein_page > 1:
            st.session_state.protein_page -= 1
    with col_nav2:
        st.markdown(f"Page **{st.session_state.protein_page}** of **{total_pages}**")
    with col_nav3:
        if st.button("Next ▶") and st.session_state.protein_page < total_pages:
            st.session_state.protein_page += 1
    
    start = (st.session_state.protein_page - 1) * page_size
    end = min(start + page_size, len(subset))
    page_df = subset.iloc[start:end].copy()
    
    # Format table
    page_df['Protein'] = page_df['id'].apply(
        lambda x: f'<a href="https://www.uniprot.org/uniprotkb/{x}/entry" target="_blank">{x}</a>'
    )
    
    display_cols = ['Protein', 'entry name', 'SHAP Value', 'Abundance', 'protein_names']
    if 'tissue_specificity' in page_df.columns:
        display_cols.append('tissue_specificity')
    
    page_df = page_df[['id'] + [c for c in display_cols if c in page_df.columns or c == 'Protein']]
    page_df = page_df.drop(columns=['id'], errors='ignore')
    
    # Rename columns
    rename_map = {
        'entry name': 'Entry',
        'protein_names': 'Description',
        'tissue_specificity': 'UniProt Tissue'
    }
    page_df = page_df.rename(columns=rename_map)
    
    st.markdown(
        page_df.to_html(escape=False, index=False),
        unsafe_allow_html=True
    )
    
    # Download
    col_dl, col_cp = st.columns(2)
    with col_dl:
        st.download_button(
            "📥 Download Full Table",
            subset.to_csv(index=False),
            f"proteins_{current_sample}_{selected_tissue}.csv",
            "text/csv"
        )
    with col_cp:
        # Copy protein IDs to clipboard
        protein_ids = '\n'.join(subset['id'].tolist())
        copy_to_clipboard_button(protein_ids, "📋 Copy Protein IDs", key=f"copy_proteins_{current_sample}_{selected_tissue}")

elif show_table:
    st.info("No proteins match the current filters.")

# ==============================================================================
# SECTION: Statistics
# ==============================================================================
if show_stats and len(shap_data) > 0:
    st.markdown("---")
    st.markdown("## Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    pro_count = (shap_data > 0).sum()
    con_count = (shap_data < 0).sum()
    
    col1.metric("Total Proteins", len(shap_data))
    col2.metric("Pro Proteins", pro_count)
    col3.metric("Con Proteins", con_count)
    col4.metric("Net Score", f"{shap_data.sum():.3f}")
    
    # Distribution
    fig = px.histogram(
        x=shap_data.values, nbins=30,
        title=f"SHAP Distribution for {selected_tissue}",
        labels={'x': 'SHAP Value', 'y': 'Count'}
    )
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(height=300, margin=dict(t=40, b=20))
    st.plotly_chart(fig, width='content')


# --- Navigation hint ---
if len(shap_data) > 0:
    st.markdown("---")
    st.info(f"**{len(shap_data)} proteins** selected. Go to **Functional Analysis** to run ORA on these proteins.")
    mark_says("Markverse/Mark digging for gold.png", 
              "I'm keeping track of your protein selection for Functional Analysis!")
