
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import shap
import seaborn as sns
import matplotlib.pyplot as plt
scaler = MinMaxScaler()
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import mlmarker
import io
import streamlit.components.v1 as components
import base64
from custom_functions import mark_says


st.set_page_config(page_title="MLMarker", page_icon=":octopus:", layout='wide')
st.logo('octopus.png')

def extract_uniprot_ids(df, column, separators=[';', '|', ',']):
    """
    Splits values in the specified column of a DataFrame based on multiple separators,
    and expands the DataFrame with duplicate rows for each split value.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - column (str): Column containing the strings to split.
    - separators (list): List of separator characters.

    Returns:
    - pd.DataFrame: Expanded DataFrame with split values.
    """
    # Create a regex pattern to match any of the separators
    pattern = f"[{''.join(map(re.escape, separators))}]"
    
    # Split the column by the pattern and explode the resulting lists into rows
    expanded_df = df.assign(**{column: df[column].str.split(pattern)}).explode(column)
    
    # Strip whitespace or clean the split values if needed
    expanded_df[column] = expanded_df[column].str.strip()
    
    return expanded_df

@st.cache_data
def read_file(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    elif file.name.endswith(".tsv"):
        return pd.read_csv(file, sep="\t")
    elif file.name.endswith(".xlsx"):
        return pd.read_excel(file)
    else:
        st.error("Unsupported file format. Please upload a CSV, TSV, or XLSX file.")
        return None
@st.cache_data    
def expand_and_extract_ids(df, column, separators=[';']):
    df[column] = df[column].str.split(separators[0])
    df = df.explode(column).reset_index(drop=True)
    return df

@st.cache_data
def transform_data(df, row_type):
    return df.T if row_type == "Samples" else df


# --- Cache model ---
@st.cache_resource
def load_model(penalty, analysis_type):
    pen = 0
    if penalty == "No":
        pen = 0
    else:
        pen = 1
    if analysis_type == "Quantified proteins":
        return mlmarker.MLMarker(dev=True, penalty_factor=pen, binary=False)
    else:
        return mlmarker.MLMarker(dev=True, penalty_factor=pen, binary=True)

# --- Cache file reading ---
@st.cache_data
def read_file(file):
    if isinstance(file, str):
        name = file
    else:
        name = file.name
    if name.endswith(".csv"):
        return pd.read_csv(file)
    elif name.endswith(".tsv"):
        return pd.read_csv(file, sep="\t")
    elif name.endswith(".xlsx"):
        return pd.read_excel(file)
    else:
        st.error("Unsupported file format. Please upload CSV, TSV, or XLSX.")
        return None

# --- Clean input ---
def clean_input(df):
    df.columns = df.columns.str.strip()
    df = df.set_index(df.columns[0])
    return df.apply(pd.to_numeric, errors='coerce')

# --- Preprocess one sample only ---
def preprocess_sample(sample_df, method):
    if method == "Quant":
        scaler = MinMaxScaler()
        return pd.DataFrame(scaler.fit_transform(sample_df.T).T,
                            index=sample_df.index, columns=sample_df.columns)
    else:
        return sample_df.applymap(lambda x: 1 if x > 0 else 0)

# --- Run MLMarker prediction ---
def run_mlmarker(model, sample_df):
    model.load_sample(sample_df)
    return model.explainability.adjusted_absent_shap_values_df(n_preds=50)

# -- documentation --
eft_co, cent_co,last_co = st.columns(3)
with cent_co:
    st.image('logo.png')
st.write("MLMarker is a machine learning-based tool for predicting tissue-specific protein expression patterns. It uses a pre-trained model to analyze protein data and provide insights into the tissue distribution of proteins.")

with st.expander("ℹ️ What is MLMarker? Click to learn more!", expanded=False):
    st.markdown("""
    **MLMarker** is a machine learning-based tool for predicting tissue-specific protein expression patterns.

    - Uses a pre-trained model to analyze protein data.
    - Supports **quantitative** and **binary** analysis.
    - Ideal for inferring tissue origin of proteomics samples.
    - ⚠️ For sparse samples (e.g. fluids or cell lines), enable the penalty option to reduce bias from absent proteins.

    **Upload data:** Rows = samples, Columns = proteins. First column = sample IDs.
    """)
with st.sidebar:
    mark_says("Markverse/cropped_images/octopus.png", "Hi there! I'm Mark and I'll help you out! Let's predict what tissue is in your sample.")

# --- Load protein data ---
protein_df = pd.read_csv('MLMarker_features_bioservice_return.csv')
st.session_state["protein_df"] = protein_df

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

# --- Upload & display ---
col1, col2 = st.columns(2)
with col1: 
    st.markdown("#####")
    st.write("""Upload your data in the format of columns as proteins and rows as samples. 
    The first column should contain the sample IDs.""")
with col2:
    file = st.file_uploader("Upload your file", type=["csv", "tsv", "xlsx"],  help="Upload proteomics data with samples as rows and proteins as columns.")
    # Simulate uploaded file when test button is pressed
    test_button = st.button("Test with example file", use_container_width=True)
    if test_button:
        mark_says("Markverse/cropped_images/octopus.png", "Exciting, let me show you around!")
        file = "testsample2.tsv"
        st.session_state.uploaded_file = file    
if file is not None:
    st.session_state.uploaded_file = file
st.markdown("##")
uploaded_file = st.session_state.uploaded_file
if uploaded_file is not None:
    df = read_file(uploaded_file)
    df = clean_input(df)
    st.session_state.df = df
    st.write("Uploaded data preview:")
    st.dataframe(df, use_container_width=True)
    if "sample_id" not in st.session_state:
        st.session_state.sample_id = df.index[0]
    # Select sample
    sample_id = st.selectbox("Select sample to analyze", df.index.tolist(), key="sample_id", help="This application allows you to run one sample at a time which you should select here. If you want to analyze at higher throughputs, use the python package")

    # Choose analysis type and penalty
    analysis_type = st.selectbox("Use quantificatied or binary data", ["Quantified proteins", "Binary quantification"], key="analysis_type", help="Quantified proteins will minmax normalize the quantification of your sample. When you have no little quantitative information or are working with e.g. Olink data, you can use binary classification, this will result in decreased performance and should be used with caution")
    penalty = st.selectbox("Penalize absent proteins", ["No", "Yes"], key="penalty", help="Setting this to Yes will decrease the impact of missing proteins and can be used when working with cell lines, fluids, organoids or single cells. For normal tissue samples this will result in decreased performance")
    if penalty == "Yes":
        st.warning("🐙 Mark says: Penalty is ON. I’ll down-weight missing proteins — perfect for cell lines, fluids, or organoids!")
    else:
        st.info("🐙 Mark says: Penalty is OFF. Great for solid tissue samples — I won’t tweak missing values.")

    if st.button("Run MLMarker", use_container_width=True):
        mark_says("Markverse/cropped_images/Mark knitting.png", "Seeing some cool tissues there?")

        model = load_model(st.session_state.penalty, analysis_type)
        sample_df = st.session_state.df.loc[[st.session_state.sample_id]]
        st.session_state.sel_sample= sample_id
        processed_sample = preprocess_sample(sample_df, analysis_type)
        prediction_df = run_mlmarker(model, processed_sample)

        summed_pred = prediction_df.sum(axis=1)
        summed_pred[summed_pred < 0] = 0
        #rename columns Tissue and Probability
        summed_pred /= summed_pred.sum()
        st.session_state.prediction_summed = summed_pred
        st.session_state.prediction = prediction_df
        #make a barplot of prediction_summed

        summed_pred= summed_pred.reset_index().rename(columns={"tissue": "Tissue", 0:"Probability"})

        fig = px.bar(summed_pred.sort_values(by="Probability", ascending=True), 
                        x="Probability", y="Tissue", title="Tissue Probability Prediction", 
                        orientation="h", labels={'value': 'Probability', 'index': 'Tissue'})

        fig.update_traces(textposition='auto', insidetextanchor='start')

        bar_count = len(summed_pred)
        fig.update_layout(
            height=30 * bar_count,
            margin=dict(l=120, r=40, t=60, b=60),
            yaxis=dict(automargin=True)
        )

        st.plotly_chart(fig, use_container_width=True)
