
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import shap
import mlmarker
from mlmarker.model import MLMarker
import seaborn as sns
import matplotlib.pyplot as plt
scaler = MinMaxScaler()
from plotly.subplots import make_subplots
from gprofiler import GProfiler
import plotly.graph_objects as go

st.set_page_config(page_title="MLMarker", page_icon=":octopus:", layout='wide')
st.logo('octopus.png', size='large')

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


protein_df = pd.read_csv('MLMarker_features_bioservice_return.csv')
st.session_state["protein_df"] = protein_df

# --- Streamlit App ---
left_co, cent_co,last_co = st.columns(3)
with cent_co:
    st.image('logo.png', use_container_width=True)

st.write("""Upload your data according to the following guidelines. Rows are samples. Columns are proteins. Cells are protein quant values. 
Note that the first column should contain the sample IDs""")


# File Upload
uploaded_file = st.file_uploader("Upload a file (CSV, TSV, or XLSX)", type=["csv", "tsv", "xlsx"])
# Initialize session state
if "df" not in st.session_state:
    st.session_state["df"] = None

if uploaded_file:
    st.session_state["df"] = read_file(uploaded_file)
    
    df = st.session_state["df"]
#set everything as a string just to be sure

if st.session_state["df"] is not None:
    df = df.astype(str)
    st.write("Uploaded Data:")
    st.dataframe(df.head(5))
        
    cola, colb = st.columns(2)
    if "df" in st.session_state:
        df = st.session_state["df"].copy()
        with cola:
            analysis_selection = st.selectbox("Choose Analysis Type", ["", "Binary", "Quant"])

            if analysis_selection:
                df.set_index(df.columns[0], inplace=True)
                df = df.astype(float)
                
                with colb:
                    if st.button("Process Data"):
                        if analysis_selection == "Quant":
                            df_normalized = pd.DataFrame(scaler.fit_transform(df.T).T, index=df.index, columns=df.columns)
                        else:  # Binary
                            df_normalized = df.applymap(lambda x: 1 if x > 0 else 0)
                        
                        st.session_state["df_normalized"] = df_normalized
                        st.write(f"Processed Data ({analysis_selection}):")
                        st.dataframe(df_normalized.head(5))

if "df_normalized" in st.session_state:
    row_selection = st.selectbox("Select a sample to analyze", st.session_state["df_normalized"].index.tolist())
    
    if st.button("Run MLMarker Analysis", icon="🐙", use_container_width=True):
        test = MLMarker(st.session_state["df_normalized"].loc[[row_selection]], binary=False, dev=True)
        prediction = test.explainability.adjusted_absent_shap_values_df(n_preds=50)
        prediction_summed = prediction.sum(axis=1).sort_values(ascending=False)
        prediction_summed[prediction_summed < 0] = 0
        prediction_summed /= prediction_summed.sum()
        st.session_state["prediction"] = prediction
        st.session_state["prediction_summed"] = prediction_summed
        
        st.write("Tissue level predictions:")
        st.dataframe(prediction_summed)
