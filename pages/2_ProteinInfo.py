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
import numpy as np
import io
import streamlit.components.v1 as components
import base64

from custom_functions import mark_says

st.set_page_config(page_title="MLMarker", page_icon=":octopus:", layout='wide')
st.logo('octopus.png', size='large')

protein_df = pd.read_csv('MLMarker_features_bioservice_return.csv')
st.session_state["protein_df"] = protein_df

st.title('Protein-level Insights')
st.write("""
Explore key proteins driving the prediction for a selected tissue.

- Filter by tissue, abundance, and MLMarker impact (SHAP).
- Results include: Protein ID, entry name, MLMarker value, sample abundance, UniProt description, and UniProt tissue specificity.
- Proteins are sorted by SHAP value (most impactful first).
- Click a protein ID to view its UniProt page.
- Use selected proteins for GO enrichment in the next step.
""")

df = st.session_state["df"]
sel_sample = st.session_state['sel_sample']

tissues_list = st.session_state["prediction_summed"].index.tolist()
selected_tissue = st.selectbox("Select tissue for protein contributions", options=tissues_list)
abundance_filter = st.selectbox("Select abundance filter", options=["All", "Present", "Absent"])
shap_filter = st.selectbox("Select model impact", options=["All", "Positive", "Negative"])
# letsgo = st.button("Let's go!")
# Function to display proteins with pagination
def display_paginated_proteins_slider(protein_df, selected_tissue, abundance_df, shap_df, page_size=15):
    # Filter protein data based on the selected tissue
    selected_proteins = set(shap_df.index).intersection(set(abundance_df['id'].values.tolist()))
    st.session_state["selected_proteins"] = selected_proteins
    subset_proteins = protein_df[protein_df['id'].isin(selected_proteins)]
    subset_proteins['MLMarker value'] = subset_proteins['id'].map(shap_df)
    subset_proteins['Abundance'] = subset_proteins['id'].map(abundance_df.set_index('id')['Abundance'])
    subset_proteins = subset_proteins.sort_values(by='MLMarker value', ascending=False)   

    # Calculate the total number of pages
    total_proteins = len(subset_proteins)
    total_pages = (total_proteins // page_size) + (1 if total_proteins % page_size != 0 else 0)
    
    # Select the current page number using a slider
    current_page = st.slider(
        "Select Page",
        min_value=1,
        max_value=total_pages,
        value=1,
        step=1,
    )

    # Calculate the start and end indices for the current page
    start_idx = (current_page - 1) * page_size
    end_idx = min(current_page * page_size, total_proteins)

    # Slice the DataFrame to get the relevant page
    page_df = subset_proteins.iloc[start_idx:end_idx]

    # Add UniProt link dynamically using Protein ID
    page_df['Protein'] = page_df['id'].apply(lambda x: f'<a href="https://www.uniprot.org/uniprotkb/{x}/entry" target="_blank">{x}</a>')

    # Drop unnecessary columns and rename others
    page_df.drop(columns=['Unnamed: 0', 'id'], inplace=True)
    page_df.rename(columns={'entry name': 'Entry', 'protein_names':'UniProt Description', 'tissue_specificity': 'UniProt Tissue specificity'}, inplace=True)
    page_df['UniProt Tissue specificity'].fillna('No tissue specificity information available in UniProt', inplace=True)
    # Display the dataframe with clickable UniProt links
    st.write(f"Protein level values for {selected_tissue}, Page {current_page} of {total_pages}")
    st.markdown(page_df[['Protein', 'Entry', 'MLMarker value', 'Abundance', 'UniProt Description', 'UniProt Tissue specificity']].to_html(escape=False, index=False), unsafe_allow_html=True)

def display_paginated_proteins(protein_df, selected_tissue, abundance_df, shap_df, page_size=15):
    # Filter protein data based on the selected tissue
    selected_proteins = set(shap_df.index).intersection(set(abundance_df['id'].values.tolist()))
    st.session_state["selected_proteins"] = selected_proteins
    subset_proteins = protein_df[protein_df['id'].isin(selected_proteins)]
    subset_proteins['MLMarker value'] = subset_proteins['id'].map(shap_df)
    subset_proteins['Abundance'] = subset_proteins['id'].map(abundance_df.set_index('id')['Abundance'])
    subset_proteins = subset_proteins.sort_values(by='MLMarker value', ascending=False)   

    # Calculate the total number of pages
    total_proteins = len(subset_proteins)
    total_pages = (total_proteins // page_size) + (1 if total_proteins % page_size != 0 else 0)
    if "current_page" not in st.session_state:
        st.session_state.current_page = 1
    options = ['◀', '▶']

    selection = st.pills("Navigate pages", options, selection_mode = "single")

    if selection == "◀" and st.session_state.current_page > 1:
        st.session_state.current_page -= 1
    elif selection == "▶" and st.session_state.current_page < total_pages:
        st.session_state.current_page += 1

    current_page = st.session_state.current_page
    # Calculate the start and end indices for the current page
    start_idx = (current_page - 1) * page_size
    end_idx = min(current_page * page_size, total_proteins)

    # Slice the DataFrame to get the relevant page
    page_df = subset_proteins.iloc[start_idx:end_idx]

    # Add UniProt link dynamically using Protein ID
    page_df['Protein'] = page_df['id'].apply(lambda x: f'<a href="https://www.uniprot.org/uniprotkb/{x}/entry" target="_blank">{x}</a>')

    # Drop unnecessary columns and rename others
    page_df.drop(columns=['Unnamed: 0', 'id'], inplace=True)
    page_df.rename(columns={'entry name': 'Entry', 'protein_names':'UniProt Description', 'tissue_specificity': 'UniProt Tissue specificity'}, inplace=True)
    page_df['UniProt Tissue specificity'].fillna('No tissue specificity information available in UniProt', inplace=True)
    # Display the dataframe with clickable UniProt links
    st.write(f"Protein level values for {selected_tissue}, Page {current_page} of {total_pages}")
    st.markdown(page_df[['Protein', 'Entry', 'MLMarker value', 'Abundance', 'UniProt Description', 'UniProt Tissue specificity']].to_html(escape=False, index=False), unsafe_allow_html=True)


if "show_proteins" not in st.session_state:
    st.session_state.show_proteins = False

if st.button("Let's go!"):
    st.session_state.show_proteins = True
    mark_says("Markverse/cropped_images/Bald Mark reading a book.png", "What is going on here?")

if st.session_state.show_proteins:
    st.write(f"Protein level values for {selected_tissue} with {abundance_filter} abundance and {shap_filter} model contributions")
    shap_df = st.session_state["prediction"].loc[selected_tissue]
    shap_df = shap_df[(shap_df.values != 0)]
    if shap_filter ==  "Positive":
        shap_df = shap_df[shap_df > 0]
    elif shap_filter =="Negative":
        shap_df = shap_df[shap_df < 0]    
    abundance_df = df[df.index==sel_sample].T.reset_index().rename(columns={'index':'id', sel_sample:'Abundance'})
    #replace nan with zero
    abundance_df['Abundance'] = abundance_df['Abundance'].fillna(0)
    if abundance_filter == "Present":
        abundance_df = abundance_df[(abundance_df['Abundance']>0) ]
    elif abundance_filter == "Absent":
        abundance_df = abundance_df[abundance_df['Abundance'] == 0]

    # Display the proteins with pagination
    display_paginated_proteins(protein_df, selected_tissue, abundance_df, shap_df)




with st.sidebar:
    st.download_button("Download Tissue level prediction", st.session_state["prediction_summed"].to_csv().encode(), "prediction_summed.csv", "text/csv")
    st.download_button("Download Protein level prediction", st.session_state["prediction"].to_csv().encode(), "prediction.csv", "text/csv")

