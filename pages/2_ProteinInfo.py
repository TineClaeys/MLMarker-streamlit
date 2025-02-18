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
st.set_page_config(page_title="MLMarker", page_icon=":octopus:", layout='wide')
st.logo('octopus.png', size='large')
protein_df = pd.read_csv('MLMarker_features_bioservice_return.csv')
st.session_state["protein_df"] = protein_df

st.title('Protein level information')

tissues_list = st.session_state["prediction_summed"].index.tolist()
selected_tissue = st.selectbox("Select tissue for custom forceplot", options=tissues_list)

# Function to display proteins with pagination
def display_paginated_proteins(protein_df, selected_tissue, subset, page_size=12):
    # Filter protein data based on the selected tissue
    subset_proteins = protein_df[protein_df['id'].isin(subset.index)]
    subset_proteins['Value'] = subset_proteins['id'].map(subset)
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
    st.markdown(page_df[['Protein', 'Entry', 'Value', 'Description', 'Tissue specificity']].to_html(escape=False, index=False), unsafe_allow_html=True)

if selected_tissue:
    st.write(f"Protein level values for {selected_tissue}")
    subset = st.session_state["prediction"].loc[selected_tissue]
    subset = subset[(subset.values != 0)]

    # Display the proteins with pagination
    display_paginated_proteins(protein_df, selected_tissue, subset)




with st.sidebar:
    st.download_button("Download Tissue level prediction", st.session_state["prediction_summed"].to_csv().encode(), "prediction_summed.csv", "text/csv")
    st.download_button("Download Protein level prediction", st.session_state["prediction"].to_csv().encode(), "prediction.csv", "text/csv")