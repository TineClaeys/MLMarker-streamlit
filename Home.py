
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
    if analysis_type == "Quant":
        return mlmarker.MLMarker(dev=True, penalty_factor=penalty, binary=False)
    else:
        return mlmarker.MLMarker(dev=True, penalty_factor=penalty, binary=True)

# --- Cache file reading ---
@st.cache_data
def read_file(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    elif file.name.endswith(".tsv"):
        return pd.read_csv(file, sep="\t")
    elif file.name.endswith(".xlsx"):
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
st.write("This app allows you to upload your protein data, select a sample for analysis, and choose the type of analysis (Quantitative or Binary). The results will show the predicted tissue probabilities based on the selected sample.")
st.write("Upload your data in the format of columns as proteins and rows as samples. The first column should contain the sample IDs.")
st.write("When working with a sparse sample or not solid tissue such as cell line, biofluid, etc. you can set the penalty factor to 1 which will reduce the impact of absent proteins on the predictions")

# --- Load protein data ---
protein_df = pd.read_csv('MLMarker_features_bioservice_return.csv')
st.session_state["protein_df"] = protein_df

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

# --- Upload & display ---
file = st.file_uploader("Upload your file", type=["csv", "tsv", "xlsx"])
if file is not None:
    st.session_state.uploaded_file = file

# Simulate uploaded file when test button is pressed
test_button = st.button("Test with example file", use_container_width=True)
if test_button:
    with open("testsample2.tsv", "rb") as f:
        fake_upload = io.BytesIO(f.read())
        fake_upload.name = "testsample2.tsv"
        st.session_state.uploaded_file = fake_upload

uploaded_file = st.session_state.uploaded_file
if uploaded_file:
    df = read_file(uploaded_file)
    df = clean_input(df)
    st.session_state.df = df
    st.write("Uploaded data preview:")
    st.dataframe(df.head(5), use_container_width=True)
    if "sample_id" not in st.session_state:
        st.session_state.sample_id = df.index[0]
    # Select sample
    sample_id = st.selectbox("Select sample to analyze", df.index.tolist(), key="sample_id", help="This application allows you to run one sample at a time which you should select here. If you want to analyze at higher throughputs, use the python package")

    # Choose analysis type and penalty
    analysis_type = st.selectbox("Analysis Type", ["Quant", "Binary"], key="analysis_type", help="Quant will minmax normalize the quantification of your sample. When you have no quantitative information to your availability you can use binary classification, this will result in decreased performance and should be used with caution")
    penalty = st.selectbox("Penalty Factor", [0, 1], key="penalty", help="Penalty factor set to 1 will decrease the impact of missing proteins and can be used when working with cell lines, fluids, organoids or single cells. For normal tissue samples this will result in decreased performance")

    # Run
    if st.button("Run MLMarker", use_container_width=True):
        with st.spinner("Running MLMarker..."):
            
            model = load_model(st.session_state.penalty, analysis_type)
            sample_df = st.session_state.df.loc[[st.session_state.sample_id]]
            st.session_state.sel_sample= sample_id
            processed_sample = preprocess_sample(sample_df, analysis_type)
            prediction_df = run_mlmarker(model, processed_sample)

            summed_pred = prediction_df.sum(axis=1)
            summed_pred[summed_pred < 0] = 0
            summed_pred /= summed_pred.sum()

            st.subheader("Tissue Probability Prediction")
            st.dataframe(summed_pred.sort_values(ascending=False), use_container_width=True)
            st.session_state.prediction_summed = summed_pred
            st.session_state.prediction = prediction_df