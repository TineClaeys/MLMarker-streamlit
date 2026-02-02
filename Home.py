# Fix for pkg_resources issue in deployment
import sys
import os

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
from mlmarker.model import MLMarker
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
        return mlmarker.MLMarker(penalty_factor=pen, binary=False)
    else:
        return mlmarker.MLMarker(penalty_factor=pen, binary=True)

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
    # Fill NaN values with 0 before processing for MLMarker
    sample_df = sample_df.fillna(0)
    if method == "Quantified proteins":
        scaler = MinMaxScaler()
        return pd.DataFrame(scaler.fit_transform(sample_df.T).T,
                            index=sample_df.index, columns=sample_df.columns)
    else:
        return sample_df.map(lambda x: 1 if x > 0 else 0)

# --- Run MLMarker prediction ---
def run_mlmarker(model, sample_df):
    model.load_sample(sample_df)
    return model.explainability.get_shap_values(n_preds=34)


# --- Functions for multi-sample support ---
@st.cache_data
def get_mlmarker_features():
    """Get the set of MLMarker model features."""
    model = MLMarker()
    return set(model.get_model_features())


def calculate_coverage(sample_df, mlmarker_features):
    """Calculate the coverage of MLMarker features in a sample."""
    sample_row = sample_df.iloc[0]
    detected_proteins = sample_row[sample_row.notna() & (sample_row != 0)]
    detected_set = set(detected_proteins.index)
    mlmarker_overlap = detected_set.intersection(mlmarker_features)
    
    return {
        'total_proteins': len(detected_set),
        'mlmarker_detected': len(mlmarker_overlap),
        'coverage_pct': 100 * len(mlmarker_overlap) / len(mlmarker_features) if mlmarker_features else 0
    }


def run_mlmarker_batch(df, sample_settings, progress_callback=None):
    """Run MLMarker on multiple samples with individual settings."""
    results = {}
    total_samples = len(sample_settings)
    
    for idx, (sample_id, settings) in enumerate(sample_settings.items()):
        if progress_callback:
            progress_callback((idx + 1) / total_samples, f"Processing {sample_id}...")
        
        model = load_model(settings['penalty'], settings['analysis_type'])
        sample_df = df.loc[[sample_id]]
        processed_sample = preprocess_sample(sample_df, settings['analysis_type'])
        model.load_sample(processed_sample)
        prediction_df = model.explainability.get_shap_values(n_preds=34)
        
        summed_pred = prediction_df.sum(axis=1)
        summed_pred[summed_pred < 0] = 0
        summed_pred /= summed_pred.sum()
        
        results[sample_id] = {
            'prediction_df': prediction_df,
            'summed_pred': summed_pred,
            'settings': settings
        }
    
    return results

    
all_possible_tissues = sorted(['Nasal Polyps', 'Duodenum', 'Small intestine', 'Parotid gland', 'Colon', 'Liver', 'Ovary', 'Testis', 'B-cells', 'Prostate', 'Esophagus', 'Skeletal muscle', 'Stomach', 'Adrenal gland', 'Appendix', 'Salivary gland', 'Urinary bladder', 'Smooth muscle', 'Oviduct', 'Lung', 'Pituitary gland', 'Brain', 'Placenta', 'Tonsil', 'Endometrium', 'Rectum', 'Lymph node', 'Thyroid', 'Bone marrow', 'Kidney', 'Adipose tissue', 'Heart', 'Monocytes', 'Spleen'])

# --- Sidebar ---
with st.sidebar:
    mark_says("Markverse/mark pointing.png", "Hi! I'm Mark. Let's predict what tissue is in your sample!")

# --- Header ---
col_logo1, col_logo2, col_logo3 = st.columns([1, 2, 1])
with col_logo2:
    st.image('logo.png')

st.caption("Predict tissue origin from proteomics data using machine learning")

with st.expander("About MLMarker", expanded=False):
    st.markdown(f"""
    **MLMarker** predicts tissue-specific protein expression patterns using machine learning.

    - Supports **quantitative** and **binary** analysis
    - Ideal for inferring tissue origin of proteomics samples
    - Enable penalty for sparse samples (fluids, cell lines, organoids)

    **Input:** Rows = samples, Columns = proteins, First column = sample IDs

    **34 tissue classes:** {', '.join(all_possible_tissues[:10])}...
    """)

# --- Load protein data ---
protein_df = pd.read_csv('MLMarker_features_bioservice_return.csv')
st.session_state["protein_df"] = protein_df

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

# --- Upload section ---
st.markdown("---")
st.markdown("### Upload Data")

col1, col2 = st.columns([1, 2])
with col1:
    st.markdown("**Format:** Proteins as columns, samples as rows")
    test_button = st.button("Try Example Data")
    if test_button:
        mark_says("Markverse/mark_binoculars.png", "Test file loaded! Let me show you around!")
        st.session_state.uploaded_file = "testsample.tsv"

with col2:
    file = st.file_uploader("Upload file", type=["csv", "tsv", "xlsx"], label_visibility="collapsed")
    if file is not None:
        st.session_state.uploaded_file = file

uploaded_file = st.session_state.uploaded_file
if uploaded_file is not None:
    df = read_file(uploaded_file)
    df = clean_input(df)
    st.session_state.df = df
    st.write("Uploaded data preview:")
    st.dataframe(df)
    st.write(df.shape)
    
    # Get MLMarker features for coverage calculation
    mlmarker_features = get_mlmarker_features()
    
    # --- Sample Mode Selection ---
    st.markdown("---")
    sample_mode = st.radio(
        "How many samples do you want to analyze?",
        ["Single sample", "Multiple samples"],
        horizontal=True,
        help="Single sample: detailed analysis of one sample. Multiple samples: batch processing with comparative analysis."
    )
    
    if sample_mode == "Single sample":
        # ============================================
        # SINGLE SAMPLE MODE (original behavior)
        # ============================================
        if "sample_id" not in st.session_state:
            st.session_state.sample_id = df.index[0]
        # Select sample
        sample_id = st.selectbox("Select sample to analyze", df.index.tolist(), key="sample_id", help="This application allows you to run one sample at a time which you should select here. If you want to analyze at higher throughputs, use the python package")

        # Choose analysis type and penalty
        analysis_type = st.selectbox("Use quantified or binary data", ["Quantified proteins", "Binary quantification"], key="analysis_type", help="Quantified proteins will minmax normalize the quantification of your sample. When you have no little quantitative information or are working with e.g. Olink data, you can use binary classification, this will result in decreased performance and should be used with caution")
        penalty = st.selectbox("Penalize absent proteins", ["No", "Yes"], key="penalty", help="Setting this to Yes will decrease the impact of missing proteins and can be used when working with cell lines, fluids, organoids or single cells. For normal tissue samples this will result in decreased performance")
        
        if penalty == "Yes":
            mark_says("Markverse/cropped_images/Coding Mark.png", "Penalty is ON! I'll down-weight missing proteins - perfect for cell lines, fluids, or organoids!")
        else:
            mark_says("Markverse/cropped_images/Mark on a book.png", "Penalty is OFF. Great for solid tissue samples - I won't tweak missing values.")

        if st.button("Run MLMarker"):
            mark_says("Markverse/cropped_images/Coding Mark.png", "Running the analysis... let me crunch those numbers!")

            model = load_model(st.session_state.penalty, analysis_type)
            sample_df = st.session_state.df.loc[[st.session_state.sample_id]]
            st.session_state.sel_sample= sample_id
            processed_sample = preprocess_sample(sample_df, analysis_type)
            model.load_sample(processed_sample)
            prediction_df = model.explainability.get_shap_values(n_preds=34)

            summed_pred = prediction_df.sum(axis=1)
            summed_pred[summed_pred < 0] = 0
            summed_pred /= summed_pred.sum()
            st.session_state.prediction_summed = summed_pred
            st.session_state.prediction = prediction_df

            summed_pred= summed_pred.reset_index().rename(columns={"tissue": "Tissue", 0:"Similarity"})

            fig = px.bar(summed_pred.sort_values(by="Similarity", ascending=True), 
                            x="Similarity", y="Tissue", title="Tissue Similarity Prediction", 
                            orientation="h", labels={'value': 'Similarity', 'index': 'Tissue'})

            fig.update_traces(textposition='auto', insidetextanchor='start')

            bar_count = len(summed_pred)
            fig.update_layout(
                height=30 * bar_count,
                margin=dict(l=120, r=40, t=60, b=60),
                yaxis=dict(automargin=True)
            )

            st.plotly_chart(fig)
    
    else:
        # ============================================
        # MULTIPLE SAMPLES MODE
        # ============================================
        st.markdown("### Multi-Sample Analysis")
        
        # Calculate coverage for all samples
        coverage_data = []
        for sample_id in df.index:
            sample_df = df.loc[[sample_id]]
            cov = calculate_coverage(sample_df, mlmarker_features)
            cov['sample_id'] = sample_id
            coverage_data.append(cov)
        
        coverage_df = pd.DataFrame(coverage_data)
        low_coverage_samples = set(coverage_df[coverage_df['coverage_pct'] < 5]['sample_id'])
        
        # Initialize editable table data in session state
        if "sample_table" not in st.session_state or len(st.session_state.sample_table) != len(df.index):
            table_data = []
            for _, row in coverage_df.iterrows():
                sample_id = row['sample_id']
                is_low_cov = sample_id in low_coverage_samples
                table_data.append({
                    'Select': True,
                    'Sample': sample_id,
                    'Coverage': f"{row['coverage_pct']:.1f}%",
                    'Features': int(row['mlmarker_detected']),
                    'Low Cov': '🔴' if is_low_cov else '',
                    'Penalty': False,
                    'Analysis': 'Quantified proteins',
                    '_low_cov': is_low_cov  # hidden flag
                })
            st.session_state.sample_table = table_data
        
        # Show warning and button for low coverage samples
        if len(low_coverage_samples) > 0:
            col_warn, col_btn = st.columns([3, 1])
            with col_warn:
                st.warning(f"**{len(low_coverage_samples)} sample(s)** have <5% coverage (highlighted in red)")
            with col_btn:
                if st.button("Enable Penalty for Low Coverage", width='content'):
                    for i, row in enumerate(st.session_state.sample_table):
                        if row['_low_cov']:
                            st.session_state.sample_table[i]['Penalty'] = True
                    st.rerun()
        
        # Create the editable dataframe
        table_df = pd.DataFrame(st.session_state.sample_table)
        
        # Configure columns for data editor
        column_config = {
            'Select': st.column_config.CheckboxColumn(
                'Select',
                help='Include sample in analysis',
                default=True,
                width='small'
            ),
            'Sample': st.column_config.TextColumn(
                'Sample',
                disabled=True,
                width='medium'
            ),
            'Coverage': st.column_config.TextColumn(
                'Coverage',
                disabled=True,
                width='small'
            ),
            'Features': st.column_config.NumberColumn(
                'Features',
                disabled=True,
                width='small',
                help='MLMarker features detected'
            ),
            'Low Cov': st.column_config.TextColumn(
                '⚠️',
                disabled=True,
                width='small',
                help='Low coverage indicator (<5%)'
            ),
            'Penalty': st.column_config.CheckboxColumn(
                'Penalty',
                help='Enable for sparse samples (fluids, cell lines)',
                default=False,
                width='small'
            ),
            'Analysis': st.column_config.SelectboxColumn(
                'Analysis',
                options=['Quantified proteins', 'Binary quantification'],
                default='Quantified proteins',
                width='medium'
            ),
            '_low_cov': None  # Hide this column
        }
        
        # Display editable table
        display_cols = ['Select', 'Sample', 'Coverage', 'Features', 'Low Cov', 'Penalty', 'Analysis']
        
        # Use a form to batch edits and prevent reruns on every change
        with st.form("sample_table_form", clear_on_submit=False):
            edited_df = st.data_editor(
                table_df[display_cols],
                column_config=column_config,
                width='content',
                hide_index=True,
                num_rows='fixed',
                key='sample_editor'
            )
            
            # Show selected count inside form
            n_selected = edited_df['Select'].sum()
            st.caption(f"**{n_selected} samples** selected")
            
            # Quick action buttons and Run inside form
            col_sel1, col_sel2, col_sel3, col_run = st.columns([1, 1, 1, 2])
            with col_sel1:
                select_all = st.form_submit_button("Select All", width='content')
            with col_sel2:
                deselect_all = st.form_submit_button("Deselect All", width='content')
            with col_sel3:
                reset_settings = st.form_submit_button("Reset", width='content')
            with col_run:
                run_clicked = st.form_submit_button("Run MLMarker", type="primary", width='content')
        
        # Handle form submissions
        if select_all:
            for i in range(len(st.session_state.sample_table)):
                st.session_state.sample_table[i]['Select'] = True
            st.rerun()
        elif deselect_all:
            for i in range(len(st.session_state.sample_table)):
                st.session_state.sample_table[i]['Select'] = False
            st.rerun()
        elif reset_settings:
            del st.session_state.sample_table
            st.rerun()
        elif run_clicked:
            # Save current edits to session state
            for i, row in edited_df.iterrows():
                st.session_state.sample_table[i]['Select'] = row['Select']
                st.session_state.sample_table[i]['Penalty'] = row['Penalty']
                st.session_state.sample_table[i]['Analysis'] = row['Analysis']
            
            # Build selected samples dict from edited_df
            selected_samples = {}
            for i, row in edited_df.iterrows():
                if row['Select']:
                    selected_samples[st.session_state.sample_table[i]['Sample']] = {
                        'analysis_type': row['Analysis'],
                        'penalty': 'Yes' if row['Penalty'] else 'No',
                        'selected': True
                    }
            
            if len(selected_samples) > 0:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(progress, text):
                    progress_bar.progress(progress)
                    status_text.text(text)
                
                results = run_mlmarker_batch(df, selected_samples, update_progress)
                
                st.session_state.batch_results = results
                st.session_state.batch_coverage = coverage_df[coverage_df['sample_id'].isin(selected_samples.keys())]
                
                status_text.text("Done!")
                mark_says("Markverse/Mark_on_a_rocket.png", f"Processed {len(results)} samples! To the results!")
                st.success(f"Processed {len(results)} samples!")
            else:
                st.warning("No samples selected. Please select at least one sample.")
        
        # Display results if available
        if "batch_results" in st.session_state and st.session_state.batch_results:
            st.markdown("---")
            st.markdown("### Results")
            
            results = st.session_state.batch_results
            
            # Summary table
            summary_data = []
            for sample_id, result in results.items():
                top_tissue = result['summed_pred'].idxmax()
                top_prob = result['summed_pred'].max()
                summary_data.append({
                    'Sample': sample_id,
                    'Top Tissue': top_tissue,
                    'Probability': f"{top_prob:.1%}"
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, width='content', hide_index=True)
            
            # Heatmap
            with st.expander("View Heatmap"):
                heatmap_data = pd.DataFrame({
                    sample_id: result['summed_pred'] 
                    for sample_id, result in results.items()
                }).T
                
                # Calculate dynamic sizing based on data dimensions
                n_samples = len(heatmap_data)
                n_tissues = len(heatmap_data.columns)
                
                # Height: minimum 400px, scale with samples (40px per sample)
                fig_height = max(400, 40 * n_samples + 150)  # +150 for x-axis labels
                # Width: scale with tissues (30px per tissue), but use container width
                fig_width = max(800, 30 * n_tissues + 150)  # +150 for y-axis labels
                
                fig = px.imshow(
                    heatmap_data,
                    labels=dict(x="Tissue", y="Sample", color="Probability"),
                    aspect="auto",
                    color_continuous_scale="RdYlBu_r"
                )
                fig.update_layout(
                    height=fig_height,
                    width=fig_width,
                    xaxis=dict(
                        tickangle=45,
                        tickfont=dict(size=10),
                        side='bottom'
                    ),
                    yaxis=dict(
                        tickfont=dict(size=10)
                    ),
                    margin=dict(l=120, r=50, t=50, b=150)  # Extra margins for labels
                )
                st.plotly_chart(fig)
            
            st.info("Go to **Comparison** page for detailed analysis.")

