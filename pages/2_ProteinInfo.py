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
st.set_page_config(page_title="MLMarker", page_icon=":octopus:", layout='wide')
st.logo('octopus.png', size='large')

protein_df = pd.read_csv('MLMarker_features_bioservice_return.csv')
st.session_state["protein_df"] = protein_df

st.title('Protein level information')
st.write("Here you can gain more protein level insights into the prediction. Select the tissue of interest and filter for abundance and model impact. The result will be a table with the top proteins contributing to the prediction for the selected tissue. The table includes the protein ID, entry name, value (SHAP value), abundance, description, and tissue specificity.")
st.write("The table is paginated to allow for easy navigation through the results. You can select the page number using the slider. The proteins are sorted by their SHAP value, with the most significant contributors appearing first.")
st.write('When clicking on a protein identifier, you will be redirected to the UniProt page for that protein, where you can find more detailed information about its function, structure, and interactions.')
st.write("The tissue specificty column is directly from UniProt")
st.write("The selection of proteins here can be used in the next step for GO enrichment analysis.")
df = st.session_state["df"]
sel_sample = st.session_state['sel_sample']

tissues_list = st.session_state["prediction_summed"].index.tolist()
selected_tissue = st.selectbox("Select tissue for protein contributions", options=tissues_list)
abundance_filter = st.selectbox("Select abundance filter", options=["All", "Present", "Absent"])
shap_filter = st.selectbox("Select model impact", options=["All", "Positive", "Negative"])
letsgo = st.button("Let's go!")
# Function to display proteins with pagination
def display_paginated_proteins(protein_df, selected_tissue, abundance_df, shap_df, page_size=12):
    # Filter protein data based on the selected tissue
    selected_proteins = set(shap_df.index).intersection(set(abundance_df['id'].values.tolist()))
    st.session_state["selected_proteins"] = selected_proteins
    subset_proteins = protein_df[protein_df['id'].isin(selected_proteins)]
    subset_proteins['Value'] = subset_proteins['id'].map(shap_df)
    subset_proteins['Abundance'] = subset_proteins['id'].map(abundance_df.set_index('id')['Abundance'])
    subset_proteins = subset_proteins.sort_values(by='Value', ascending=False)   

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
    page_df.rename(columns={'entry name': 'Entry', 'protein_names':'Description', 'tissue_specificity': 'Tissue specificity'}, inplace=True)

    # Display the dataframe with clickable UniProt links
    st.write(f"Protein level values for {selected_tissue}, Page {current_page} of {total_pages}")
    st.markdown(page_df[['Protein', 'Entry', 'Value', 'Abundance', 'Description', 'Tissue specificity']].to_html(escape=False, index=False), unsafe_allow_html=True)

if letsgo:
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

