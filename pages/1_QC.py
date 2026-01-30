import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import mlmarker
from custom_functions import mark_says

st.set_page_config(page_title="Quality Control - MLMarker", page_icon=":octopus:", layout='wide')
st.logo('octopus.png')

# --- Header ---
st.title("Quality Control")
st.markdown("""
MLMarker relies on a set of **5,979 predefined proteins** seen during training. Any missing proteins 
are assigned a value of zero, which can affect prediction accuracy. This page helps you assess whether 
technical factors (intensity, coverage) might influence your predictions.

**Key question**: Are predictions driven by biology or by data quality artifacts?
""")

# --- Sidebar ---
with st.sidebar:
    st.markdown("### Analysis Options")
    show_pre_pred = st.checkbox("Pre-Prediction QC", value=True)
    show_intensity_analysis = st.checkbox("Intensity Analysis", value=False)
    show_feature_correlation = st.checkbox("Feature Correlation", value=False)
    show_post_pred = st.checkbox("Post-Prediction QC", value=False)
    st.markdown("---")
    mark_says("Markverse/cropped_images/octopus.png", "Let's check your data quality!")

# --- Get MLMarker features ---
@st.cache_data
def get_mlmarker_features():
    model = mlmarker.MLMarker()
    return set(model.get_model_features())

mlmarker_features = get_mlmarker_features()

# --- Check data availability ---
if "df" not in st.session_state or st.session_state.df is None:
    mark_says("Markverse/cropped_images/Bald Mark reading a book.png", "No data yet! Upload on the Home page first.")
    st.warning("No data loaded. Please go to **Home** and upload your data.")
    st.stop()

df = st.session_state.df
sample_ids = df.index.tolist()

# --- Calculate comprehensive QC metrics for all samples ---
coverage_data = []
for sample_id in sample_ids:
    sample_row = df.loc[sample_id]
    detected = sample_row[sample_row.notna() & (sample_row != 0)]
    detected_proteins = set(detected.index)
    mlmarker_overlap = detected_proteins.intersection(mlmarker_features)
    non_mlmarker = detected_proteins - mlmarker_features
    
    mlmarker_intensity = detected[list(mlmarker_overlap)].sum() if mlmarker_overlap else 0
    non_mlmarker_intensity = detected[list(non_mlmarker)].sum() if non_mlmarker else 0
    total_intensity = detected.sum()
    
    coverage_data.append({
        'Sample': sample_id,
        'Total Proteins': len(detected_proteins),
        'MLMarker Features': len(mlmarker_overlap),
        'Coverage (%)': 100 * len(mlmarker_overlap) / len(mlmarker_features),
        'MLMarker Intensity': mlmarker_intensity,
        'Non-MLMarker Intensity': non_mlmarker_intensity,
        'Total Intensity': total_intensity,
        'MLMarker Proportion (%)': 100 * mlmarker_intensity / total_intensity if total_intensity > 0 else 0
    })

coverage_df = pd.DataFrame(coverage_data)

# --- Quick Stats ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Samples", len(sample_ids))
col2.metric("Avg Coverage", f"{coverage_df['Coverage (%)'].mean():.1f}%")
col3.metric("Low Coverage (<5%)", f"{(coverage_df['Coverage (%)'] < 5).sum()}")
col4.metric("Avg MLMarker Signal", f"{coverage_df['MLMarker Proportion (%)'].mean():.1f}%")

# ==============================================================================
# SECTION: Pre-Prediction QC (Feature Coverage)
# ==============================================================================
if show_pre_pred:
    st.markdown("---")
    st.markdown("## Pre-Prediction Quality Assessment")
    st.markdown("""
    Before running MLMarker, assess your data quality. **Low coverage (<5%)** typically indicates 
    sparse samples (e.g., plasma, urine) where the penalty factor should be enabled.
    """)
    
    col_cov1, col_cov2 = st.columns(2)
    
    with col_cov1:
        fig_hist = px.histogram(
            coverage_df, x='Coverage (%)', nbins=20,
            title="Feature Coverage Distribution",
            labels={'Coverage (%)': 'MLMarker Feature Coverage (%)'}
        )
        fig_hist.add_vline(x=5, line_dash="dash", line_color="red", 
                          annotation_text="5% threshold")
        fig_hist.update_layout(
            height=350, 
            margin=dict(t=40, b=20),
            xaxis_title="Coverage (%)",
            yaxis_title="Number of Samples"
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col_cov2:
        fig_bar = px.bar(
            coverage_df.sort_values('Coverage (%)'),
            x='Coverage (%)', y='Sample', orientation='h',
            title="Coverage by Sample",
            color='Coverage (%)', color_continuous_scale='RdYlGn'
        )
        fig_bar.update_layout(
            height=max(300, 22 * len(sample_ids)),
            margin=dict(t=40, b=20),
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Warn about low coverage
    low_cov = coverage_df[coverage_df['Coverage (%)'] < 5]
    if len(low_cov) > 0:
        st.warning(f"**{len(low_cov)} sample(s)** have <5% coverage. Consider enabling penalty factor for these.")
        mark_says("Markverse/cropped_images/Bald Mark reading a book.png", 
                  f"Looks like {len(low_cov)} sample(s) have low coverage. Don't worry - just enable the penalty factor!")

# ==============================================================================
# SECTION: Intensity Analysis
# ==============================================================================
if show_intensity_analysis:
    st.markdown("---")
    st.markdown("## Intensity Distribution Analysis")
    st.markdown("""
    Compare the intensity distribution between MLMarker features and non-MLMarker proteins. 
    If MLMarker predictions are driven by biology (not artifacts), the **MLMarker signal proportion** 
    should be similar across samples regardless of their predicted tissue.
    """)
    
    col_int1, col_int2 = st.columns(2)
    
    with col_int1:
        # Total intensity distribution
        fig_total = px.box(
            coverage_df, y='Total Intensity',
            points='all', title="Total Protein Intensity",
            hover_data=['Sample']
        )
        fig_total.update_layout(height=350, margin=dict(t=40, b=20))
        st.plotly_chart(fig_total, use_container_width=True)
    
    with col_int2:
        # MLMarker proportion
        fig_prop = px.box(
            coverage_df, y='MLMarker Proportion (%)',
            points='all', title="MLMarker Signal Proportion",
            hover_data=['Sample']
        )
        fig_prop.update_layout(height=350, margin=dict(t=40, b=20))
        st.plotly_chart(fig_prop, use_container_width=True)
    
    # Stats
    st.markdown(f"""
    **Summary Statistics:**
    - Mean MLMarker proportion: **{coverage_df['MLMarker Proportion (%)'].mean():.1f}%**
    - Std deviation: **{coverage_df['MLMarker Proportion (%)'].std():.1f}%**
    
    A consistent MLMarker proportion across samples suggests intensity variation is a **global sample-level 
    effect** unrelated to MLMarker's feature space.
    """)

# ==============================================================================
# SECTION: Feature Correlation
# ==============================================================================
if show_feature_correlation:
    st.markdown("---")
    st.markdown("## MLMarker vs Non-MLMarker Intensity Correlation")
    st.markdown("""
    A near-perfect correlation between MLMarker and non-MLMarker feature intensities indicates that 
    intensity variation is a **global effect** (e.g., sample loading, ionization efficiency) rather 
    than being specific to MLMarker's feature space.
    """)
    
    # Scatter plot
    fig_scatter = px.scatter(
        coverage_df,
        x='Non-MLMarker Intensity', y='MLMarker Intensity',
        hover_data=['Sample', 'Coverage (%)'],
        title="MLMarker vs Non-MLMarker Intensity (per sample)"
    )
    
    # Add regression line and correlation
    if len(coverage_df) > 2:
        x_vals = coverage_df['Non-MLMarker Intensity']
        y_vals = coverage_df['MLMarker Intensity']
        if x_vals.nunique() > 1 and y_vals.nunique() > 1:
            try:
                slope, intercept, r, p, _ = stats.linregress(x_vals, y_vals)
                x_line = np.linspace(x_vals.min(), x_vals.max(), 50)
                fig_scatter.add_trace(go.Scatter(
                    x=x_line, y=slope * x_line + intercept,
                    mode='lines', name=f'r={r:.2f}, p={p:.2e}',
                    line=dict(dash='dash', color='gray')
                ))
                
                fig_scatter.update_layout(
                    height=450, 
                    margin=dict(t=40, b=20),
                    xaxis_title="Non-MLMarker Intensity (sum)",
                    yaxis_title="MLMarker Intensity (sum)"
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Interpretation
                if r > 0.9:
                    st.success(f"""
                    **Strong correlation (r = {r:.2f})**: Intensity variation is a global sample-level effect. 
                    This suggests MLMarker predictions are **not driven by intensity artifacts**.
                    """)
                    mark_says("Markverse/cropped_images/Mark digging for gold.png", 
                              f"Excellent! r = {r:.2f} shows intensity is a global effect, not MLMarker-specific.")
                elif r > 0.7:
                    st.info(f"""
                    **Good correlation (r = {r:.2f})**: Most intensity variation appears to be global. 
                    Predictions are likely reliable.
                    """)
                else:
                    st.warning(f"""
                    **Moderate correlation (r = {r:.2f})**: Some intensity variation may be feature-specific. 
                    Interpret predictions with caution.
                    """)
            except:
                st.plotly_chart(fig_scatter, use_container_width=True)

# ==============================================================================
# SECTION: Post-Prediction QC
# ==============================================================================
if show_post_pred:
    st.markdown("---")
    st.markdown("## Post-Prediction Quality Assessment")
    st.markdown("""
    After running MLMarker, check if technical metrics correlate with predictions. 
    If predictions are driven by **biology** (not artifacts):
    - Coverage should **not** significantly differ between predicted groups
    - MLMarker signal proportion should be **similar** across predicted groups
    - Intensity correlation with prediction should be a **global effect**
    """)
    
    if "batch_results" not in st.session_state or not st.session_state.batch_results:
        st.info("Run MLMarker predictions on the **Home** page to see Post-Prediction QC.")
    else:
        results = st.session_state.batch_results
        
        # Add prediction results to QC data
        qc_data = coverage_df.copy()
        qc_data['Top Tissue'] = qc_data['Sample'].apply(
            lambda x: results[x]['summed_pred'].idxmax() if x in results else None
        )
        qc_data['Top Prob'] = qc_data['Sample'].apply(
            lambda x: results[x]['summed_pred'].max() if x in results else None
        )
        qc_data = qc_data.dropna(subset=['Top Tissue'])
        
        if len(qc_data) == 0:
            st.warning("No matching predictions found for samples.")
        else:
            # Get unique tissues for comparison
            unique_tissues = qc_data['Top Tissue'].unique()
            
            if len(unique_tissues) < 2:
                st.info(f"All samples predicted as **{unique_tissues[0]}**. Need multiple groups for comparison.")
            else:
                st.markdown("### Comparing Technical Metrics by Predicted Tissue")
                st.caption("Based on the analysis approach from Claeys et al. - checking if predictions are driven by data quality")
                
                # Create 2x2 grid of boxplots
                col_post1, col_post2 = st.columns(2)
                
                with col_post1:
                    fig_int = px.box(
                        qc_data, x='Top Tissue', y='Total Intensity',
                        points='all', title="A) Total Intensity by Prediction",
                        color='Top Tissue'
                    )
                    fig_int.update_layout(height=350, margin=dict(t=40, b=20), showlegend=False)
                    st.plotly_chart(fig_int, use_container_width=True)
                
                with col_post2:
                    fig_mlm = px.box(
                        qc_data, x='Top Tissue', y='MLMarker Intensity',
                        points='all', title="B) MLMarker Features Intensity by Prediction",
                        color='Top Tissue'
                    )
                    fig_mlm.update_layout(height=350, margin=dict(t=40, b=20), showlegend=False)
                    st.plotly_chart(fig_mlm, use_container_width=True)
                
                col_post3, col_post4 = st.columns(2)
                
                with col_post3:
                    fig_cov = px.box(
                        qc_data, x='Top Tissue', y='Coverage (%)',
                        points='all', title="C) Feature Coverage by Prediction",
                        color='Top Tissue'
                    )
                    fig_cov.update_layout(height=350, margin=dict(t=40, b=20), showlegend=False)
                    st.plotly_chart(fig_cov, use_container_width=True)
                
                with col_post4:
                    fig_prop = px.box(
                        qc_data, x='Top Tissue', y='MLMarker Proportion (%)',
                        points='all', title="D) MLMarker Signal Proportion by Prediction",
                        color='Top Tissue'
                    )
                    fig_prop.update_layout(height=350, margin=dict(t=40, b=20), showlegend=False)
                    st.plotly_chart(fig_prop, use_container_width=True)
                
                # Statistical tests if we have enough samples
                if len(unique_tissues) == 2 and all(qc_data.groupby('Top Tissue').size() >= 3):
                    group1, group2 = unique_tissues
                    g1_data = qc_data[qc_data['Top Tissue'] == group1]
                    g2_data = qc_data[qc_data['Top Tissue'] == group2]
                    
                    st.markdown("### Statistical Tests (Mann-Whitney U)")
                    
                    test_results = []
                    for metric in ['Total Intensity', 'MLMarker Intensity', 'Coverage (%)', 'MLMarker Proportion (%)']:
                        stat, p = stats.mannwhitneyu(g1_data[metric], g2_data[metric], alternative='two-sided')
                        test_results.append({
                            'Metric': metric,
                            f'{group1} (median)': f"{g1_data[metric].median():.2f}",
                            f'{group2} (median)': f"{g2_data[metric].median():.2f}",
                            'p-value': f"{p:.4f}",
                            'Significant': 'Yes' if p < 0.05 else 'No'
                        })
                    
                    test_df = pd.DataFrame(test_results)
                    st.dataframe(test_df, use_container_width=True, hide_index=True)
                    
                    # Interpretation
                    sig_count = sum(1 for r in test_results if r['Significant'] == 'Yes')
                    if sig_count == 0:
                        st.success("""
                        **No significant differences** in technical metrics between predicted groups.
                        This suggests predictions are driven by **biology, not artifacts**.
                        """)
                        mark_says("Markverse/cropped_images/Mark digging for gold.png", 
                                  "Great news! Technical metrics don't differ between groups - your predictions look solid!")
                    elif test_results[2]['Significant'] == 'No' and test_results[3]['Significant'] == 'No':
                        st.info("""
                        **Coverage and MLMarker proportion** don't differ significantly between groups.
                        Intensity differences are likely a **global sample effect**, not MLMarker-specific.
                        """)
                    else:
                        st.warning("""
                        Some technical metrics differ between groups. Interpret predictions with caution 
                        and consider whether these differences might influence results.
                        """)
                
                # E) Global intensity correlation scatter
                st.markdown("### E) Global Intensity Effect")
                st.caption("Each dot represents one sample - showing MLMarker vs Non-MLMarker intensity colored by prediction")
                
                fig_global = px.scatter(
                    qc_data,
                    x='Non-MLMarker Intensity', y='MLMarker Intensity',
                    color='Top Tissue', hover_data=['Sample', 'Coverage (%)'],
                    title="MLMarker vs Non-MLMarker Intensity by Predicted Tissue"
                )
                
                # Add overall regression line
                if len(qc_data) > 2:
                    x_vals = qc_data['Non-MLMarker Intensity']
                    y_vals = qc_data['MLMarker Intensity']
                    if x_vals.nunique() > 1 and y_vals.nunique() > 1:
                        try:
                            slope, intercept, r, p, _ = stats.linregress(x_vals, y_vals)
                            x_line = np.linspace(x_vals.min(), x_vals.max(), 50)
                            fig_global.add_trace(go.Scatter(
                                x=x_line, y=slope * x_line + intercept,
                                mode='lines', name=f'Overall r={r:.2f}',
                                line=dict(dash='dash', color='gray')
                            ))
                        except:
                            pass
                
                fig_global.update_layout(height=400, margin=dict(t=40, b=20))
                st.plotly_chart(fig_global, use_container_width=True)
                
                st.markdown("""
                **Interpretation**: If both predicted groups fall along the same regression line, 
                intensity differences are a **global sample-level effect** unrelated to MLMarker's feature space.
                This demonstrates that predictions are **not driven by data quality artifacts**.
                """)

# --- Download ---
st.markdown("---")
st.download_button(
    "Download QC Metrics",
    coverage_df.to_csv(index=False),
    "mlmarker_qc_metrics.csv",
    "text/csv",
    use_container_width=True
)

mark_says("Markverse/cropped_images/octopus.png", 
          "QC looks good? Head over to Visualisations to explore your predictions!")
