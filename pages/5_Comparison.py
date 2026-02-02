import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import mlmarker
from custom_functions import mark_says

st.set_page_config(page_title="Comparative Analysis - MLMarker", page_icon=":octopus:", layout='wide')
st.logo('octopus.png')

# --- Header ---
st.title("Comparative Analysis")
st.markdown("""
Compare MLMarker predictions across **multiple samples**. This page provides:

- **Heatmap**: Visual overview of tissue probabilities across all samples
- **SHAP Profile PCA**: Cluster samples by which proteins drive their predictions
- **Tissue-Specific Analysis**: Deep dive into how samples differ for a specific tissue
- **Sample Comparison**: Directly compare two samples' tissue profiles
""")

# Check if batch results are available
if "batch_results" not in st.session_state or not st.session_state.batch_results:
    mark_says("Markverse/cropped_images/Bald Mark reading a book.png", 
              "No batch results yet! Go to Home and run MLMarker on multiple samples first.")
    st.warning("""
    **No batch results available.**
    
    Please go to the **Home** page and run MLMarker on multiple samples first.
    
    1. Upload your data
    2. Select "Multiple samples" mode
    3. Configure and run predictions
    4. Return here for comparative analysis
    """)
    st.stop()

results = st.session_state.batch_results
sample_ids = list(results.keys())

# --- Build data structures ---
# Probability matrix: samples x tissues
prob_matrix = pd.DataFrame({
    sample_id: result['summed_pred'] 
    for sample_id, result in results.items()
}).T

all_tissues = sorted(list(results[sample_ids[0]]['summed_pred'].index))

# ==============================================================================
# SIDEBAR: Analysis Selection Panel
# ==============================================================================
with st.sidebar:
    st.markdown("### Analysis Options")
    
    show_heatmap = st.checkbox("Tissue Probability Heatmap", value=True)
    show_pca = st.checkbox("PCA Analysis", value=False)
    show_tissue_analysis = st.checkbox("Tissue-Specific Analysis", value=False)
    show_sample_comparison = st.checkbox("Sample Comparison", value=False)
    show_summary = st.checkbox("Top Tissue Summary", value=False)
    show_downloads = st.checkbox("Download Results", value=False)
    
    st.markdown("---")
    mark_says("Markverse/cropped_images/Mark digging for gold.png", 
              f"Analyzing {len(sample_ids)} samples!")

# ==============================================================================
# SECTION: Overview Heatmap
# ==============================================================================
if show_heatmap:
    st.markdown("---")
    with st.container():
        col_title, col_options = st.columns([2, 1])
        with col_title:
            st.markdown("## Tissue Probability Heatmap")
            st.caption("Higher values indicate stronger tissue similarity")
        
        with col_options:
            with st.expander("Display Options", expanded=False):
                sort_samples = st.selectbox(
                    "Sort samples by",
                    ["Original order", "Top tissue", "Hierarchical clustering"],
                    key="sort_samples"
                )
                color_scale = st.selectbox(
                    "Color scale",
                    ["RdYlBu_r", "Viridis", "Plasma", "Blues"],
                    key="color_scale"
                )
    
    # Sort samples if needed
    plot_matrix = prob_matrix.copy()
    if sort_samples == "Top tissue":
        plot_matrix['_top'] = plot_matrix.idxmax(axis=1)
        plot_matrix = plot_matrix.sort_values('_top').drop('_top', axis=1)
    elif sort_samples == "Hierarchical clustering":
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import pdist
        if len(plot_matrix) > 1:
            linkage_matrix = linkage(pdist(plot_matrix.values), method='ward')
            order = leaves_list(linkage_matrix)
            plot_matrix = plot_matrix.iloc[order]

    fig_heatmap = px.imshow(
        plot_matrix,
        labels=dict(x="Tissue", y="Sample", color="Probability"),
        aspect="auto",
        color_continuous_scale=color_scale
    )
    fig_heatmap.update_layout(
        height=max(350, 28 * len(sample_ids)),
        xaxis={'tickangle': -45},
        margin=dict(l=10, r=10, t=30, b=10)
    )
    st.plotly_chart(fig_heatmap, width='stretch')

# ==============================================================================
# SECTION: PCA Analysis
# ==============================================================================
if show_pca:
    st.markdown("---")
    st.markdown("## PCA Analysis")
    st.caption("Cluster samples based on different data types")
    
    # Get MLMarker features
    from mlmarker.model import MLMarker
    mlmarker_model = MLMarker()
    mlmarker_features = set(mlmarker_model.get_model_features())
    
    # Get raw abundance data
    if "df" not in st.session_state:
        st.warning("No abundance data available. Upload data on Home page first.")
        abundance_df = None
    else:
        abundance_df = st.session_state["df"]
    
    # PCA mode selection with clear descriptions
    pca_options = {
        "All Proteins (Abundances)": "PCA on all protein abundances in your dataset",
        "MLMarker Features (Abundances)": "PCA on abundances of the MLMarker model proteins",
        "Overall SHAP Profile": "PCA on mean |SHAP| values across all tissues",
        "Tissue-Specific SHAP": "PCA on SHAP values for a specific tissue"
    }
    
    pca_mode = st.selectbox(
        "Select PCA type",
        list(pca_options.keys()),
        key="pca_mode",
        help="Choose what data to use for clustering samples"
    )
    st.caption(pca_options[pca_mode])
    
    # Additional options based on mode
    pca_tissue = None
    if pca_mode == "Tissue-Specific SHAP":
        pca_tissue = st.selectbox("Select tissue", all_tissues, key="pca_tissue")
    
    # Build the data matrix based on selection
    pca_matrix = None
    pca_title = ""
    
    if pca_mode == "All Proteins (Abundances)":
        if abundance_df is not None:
            # Filter to samples in batch results
            pca_matrix = abundance_df.loc[sample_ids].fillna(0)
            pca_title = "All Protein Abundances"
        else:
            st.error("No abundance data available.")
    
    elif pca_mode == "MLMarker Features (Abundances)":
        if abundance_df is not None:
            # Get MLMarker features that exist in the data
            available_features = [f for f in mlmarker_features if f in abundance_df.columns]
            if len(available_features) > 0:
                pca_matrix = abundance_df.loc[sample_ids, available_features].fillna(0)
                pca_title = f"MLMarker Features ({len(available_features)} proteins)"
            else:
                st.error("No MLMarker features found in your data.")
        else:
            st.error("No abundance data available.")
    
    elif pca_mode == "Overall SHAP Profile":
        shap_data = {}
        for sample_id, result in results.items():
            pred_df = result['prediction_df']
            # Mean absolute SHAP across all tissues for each protein
            shap_data[sample_id] = pred_df.abs().mean(axis=0)
        pca_matrix = pd.DataFrame(shap_data).T.fillna(0)
        pca_title = "Overall SHAP Profile"
    
    elif pca_mode == "Tissue-Specific SHAP":
        shap_data = {}
        for sample_id, result in results.items():
            pred_df = result['prediction_df']
            if pca_tissue in pred_df.index:
                shap_data[sample_id] = pred_df.loc[pca_tissue]
            else:
                shap_data[sample_id] = pd.Series(0, index=pred_df.columns)
        pca_matrix = pd.DataFrame(shap_data).T.fillna(0)
        pca_title = f"{pca_tissue} SHAP Profile"
    
    # Run PCA if we have data
    if pca_matrix is not None and len(pca_matrix) >= 3:
        # Remove zero-variance columns
        pca_matrix = pca_matrix.loc[:, pca_matrix.var() > 0]
        
        if len(pca_matrix.columns) < 2:
            st.warning("Not enough variable features for PCA.")
        else:
            # Standardize and run PCA
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(pca_matrix)
            
            n_components = min(3, len(pca_matrix) - 1, len(pca_matrix.columns))
            pca = PCA(n_components=n_components)
            pca_result = pca.fit_transform(data_scaled)
            
            # Create PCA dataframe
            pca_df = pd.DataFrame(
                pca_result[:, :min(2, n_components)],
                columns=['PC1', 'PC2'] if n_components >= 2 else ['PC1'],
                index=pca_matrix.index
            )
            pca_df['Sample'] = pca_df.index
            pca_df['Top Tissue'] = [prob_matrix.loc[s].idxmax() for s in pca_df.index]
            
            col_pca1, col_pca2 = st.columns([2, 1])
            
            with col_pca1:
                if n_components >= 2:
                    fig_pca = px.scatter(
                        pca_df,
                        x='PC1',
                        y='PC2',
                        color='Top Tissue',
                        text='Sample',
                        title=f"Sample Clustering: {pca_title}"
                    )
                    fig_pca.update_traces(textposition='top center', marker=dict(size=12))
                    fig_pca.update_layout(
                        height=450,
                        xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
                        yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)",
                        margin=dict(l=10, r=10, t=40, b=10)
                    )
                    st.plotly_chart(fig_pca, width='stretch')
                else:
                    st.info("Need more features for 2D PCA visualization.")
            
            with col_pca2:
                st.markdown("#### Variance Explained")
                var_df = pd.DataFrame({
                    'Component': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
                    'Variance': pca.explained_variance_ratio_
                })
                fig_var = px.bar(
                    var_df, x='Component', y='Variance',
                    title="PCA Variance"
                )
                fig_var.update_layout(height=200, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_var, width='stretch')
                
                # Show top contributing proteins/features
                st.markdown("#### Top Contributing Features")
                loadings = pd.DataFrame(
                    pca.components_.T,
                    columns=[f'PC{i+1}' for i in range(n_components)],
                    index=pca_matrix.columns
                )
                top_pc1 = loadings['PC1'].abs().nlargest(5)
                st.dataframe(
                    pd.DataFrame({'Feature': top_pc1.index, '|Loading|': top_pc1.values.round(4)}),
                    hide_index=True,
                    width='stretch'
                )
            
            # Info about the analysis
            st.info(f"**{pca_title}**: {len(pca_matrix.columns):,} features used for clustering {len(pca_matrix)} samples.")
            
            mark_says("Markverse/cropped_images/Coding Mark.png", 
                      "Samples close together have similar profiles!")
    elif pca_matrix is not None:
        st.info("Need at least 3 samples for PCA clustering.")

# ==============================================================================
# SECTION: Tissue-Specific Analysis
# ==============================================================================
if show_tissue_analysis:
    st.markdown("---")
    st.markdown("## Tissue-Specific Analysis")
    st.caption("Deep dive into a specific tissue's prediction patterns")
    
    selected_tissue = st.selectbox(
        "Select tissue to analyze",
        all_tissues,
        key="selected_tissue"
    )
    
    col_tissue1, col_tissue2 = st.columns(2)
    
    with col_tissue1:
        # Boxplot of probability distribution
        tissue_probs = prob_matrix[selected_tissue].reset_index()
        tissue_probs.columns = ['Sample', 'Probability']
        
        fig_box = px.box(
            tissue_probs, 
            y='Probability',
            points='all',
            title=f"{selected_tissue} Probability Distribution"
        )
        fig_box.update_traces(
            hovertemplate="Sample: %{text}<br>Probability: %{y:.3f}",
            text=tissue_probs['Sample']
        )
        fig_box.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_box, width='stretch')
    
    with col_tissue2:
        # Bar chart sorted by probability
        tissue_probs_sorted = tissue_probs.sort_values('Probability', ascending=True)
        
        fig_bar = px.bar(
            tissue_probs_sorted,
            x='Probability',
            y='Sample',
            orientation='h',
            title=f"Samples Ranked by {selected_tissue}",
            color='Probability',
            color_continuous_scale='RdYlBu_r'
        )
        fig_bar.update_layout(
            height=max(280, 22 * len(sample_ids)),
            showlegend=False,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_bar, width='stretch')
    
    # Pro/Con protein analysis
    with st.expander(f"Pro/Con Proteins for {selected_tissue}", expanded=False):
        st.caption("Positive SHAP = supports prediction, Negative = opposes")
        
        pro_con_data = []
        for sample_id, result in results.items():
            pred_df = result['prediction_df']
            if selected_tissue in pred_df.index:
                tissue_shap = pred_df.loc[selected_tissue]
                pro_count = (tissue_shap > 0).sum()
                con_count = (tissue_shap < 0).sum()
                pro_sum = tissue_shap[tissue_shap > 0].sum()
                con_sum = abs(tissue_shap[tissue_shap < 0].sum())
                pro_con_data.append({
                    'Sample': sample_id,
                    'Pro Proteins': pro_count,
                    'Con Proteins': con_count,
                    'Pro Score': pro_sum,
                    'Con Score': con_sum,
                    'Net Score': pro_sum - con_sum
                })
        
        pro_con_df = pd.DataFrame(pro_con_data)
        
        col_pc1, col_pc2 = st.columns(2)
        
        with col_pc1:
            fig_procon = go.Figure()
            fig_procon.add_trace(go.Bar(
                name='Pro', x=pro_con_df['Sample'], y=pro_con_df['Pro Proteins'],
                marker_color='#27ae60'
            ))
            fig_procon.add_trace(go.Bar(
                name='Con', x=pro_con_df['Sample'], y=-pro_con_df['Con Proteins'],
                marker_color='#e74c3c'
            ))
            fig_procon.update_layout(
                barmode='relative', title="Pro/Con Protein Counts",
                yaxis_title="Count", height=350, margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig_procon, width='stretch')
        
        with col_pc2:
            fig_net = px.bar(
                pro_con_df.sort_values('Net Score', ascending=True),
                x='Net Score', y='Sample', orientation='h',
                title="Net SHAP Score",
                color='Net Score', color_continuous_scale='RdYlGn',
                color_continuous_midpoint=0
            )
            fig_net.update_layout(height=max(280, 22 * len(sample_ids)), 
                                  margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_net, width='stretch')

# ==============================================================================
# SECTION: Sample Comparison
# ==============================================================================
if show_sample_comparison:
    st.markdown("---")
    st.markdown("## Sample Comparison")
    st.caption("Compare tissue profiles between two samples")
    
    col_comp1, col_comp2 = st.columns(2)
    with col_comp1:
        sample_a = st.selectbox("Sample A", sample_ids, key="sample_a")
    with col_comp2:
        sample_b = st.selectbox("Sample B", sample_ids, index=min(1, len(sample_ids)-1), key="sample_b")
    
    if sample_a != sample_b:
        col_radar, col_diff = st.columns(2)
        
        with col_radar:
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=prob_matrix.loc[sample_a].values,
                theta=prob_matrix.columns,
                fill='toself', name=sample_a, opacity=0.6
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=prob_matrix.loc[sample_b].values,
                theta=prob_matrix.columns,
                fill='toself', name=sample_b, opacity=0.6
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, max(prob_matrix.max().max(), 0.5)])),
                title=f"{sample_a} vs {sample_b}",
                height=400, margin=dict(l=40, r=40, t=60, b=40)
            )
            st.plotly_chart(fig_radar, width='stretch')
        
        with col_diff:
            diff = prob_matrix.loc[sample_a] - prob_matrix.loc[sample_b]
            diff_df = diff.reset_index()
            diff_df.columns = ['Tissue', 'Difference']
            diff_df = diff_df.sort_values('Difference')
            
            fig_diff = px.bar(
                diff_df, x='Difference', y='Tissue', orientation='h',
                title=f"Difference ({sample_a} - {sample_b})",
                color='Difference', color_continuous_scale='RdBu',
                color_continuous_midpoint=0
            )
            fig_diff.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_diff, width='stretch')
        
        mark_says("Markverse/cropped_images/Mark knitting.png", 
                  "Positive differences mean sample A has higher probability for that tissue!")
    else:
        st.info("Select two different samples to compare.")

# ==============================================================================
# SECTION: Top Tissue Summary
# ==============================================================================
if show_summary:
    st.markdown("---")
    st.markdown("## Prediction Summary")
    st.caption("Overview of tissue predictions across all samples")
    
    # Build summary data
    summary_data = []
    for sample_id in sample_ids:
        probs = prob_matrix.loc[sample_id]
        top_tissue = probs.idxmax()
        top_prob = probs.max()
        second_tissue = probs.drop(top_tissue).idxmax()
        second_prob = probs.drop(top_tissue).max()
        
        # Confidence assessment
        gap = top_prob - second_prob
        if gap > 0.3:
            confidence = "High"
            conf_color = "🟢"
        elif gap > 0.1:
            confidence = "Medium"
            conf_color = "🟡"
        else:
            confidence = "Low"
            conf_color = "🔴"
        
        summary_data.append({
            'Sample': sample_id,
            'Predicted Tissue': top_tissue,
            'Confidence': f"{conf_color} {confidence}",
            'Score': top_prob,
            'Runner-up': second_tissue,
            'Runner-up Score': second_prob,
            'Margin': gap
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Top row: Key metrics
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    n_unique_tissues = summary_df['Predicted Tissue'].nunique()
    n_high_conf = sum(1 for d in summary_data if "High" in d['Confidence'])
    avg_score = summary_df['Score'].mean()
    
    col_m1.metric("Samples Analyzed", len(sample_ids))
    col_m2.metric("Unique Tissues", n_unique_tissues)
    col_m3.metric("High Confidence", f"{n_high_conf}/{len(sample_ids)}")
    col_m4.metric("Avg Top Score", f"{avg_score:.1%}")
    
    st.markdown("")
    
    # Main content
    col_table, col_viz = st.columns([3, 2])
    
    with col_table:
        st.markdown("#### Sample Predictions")
        
        # Format for display
        display_df = summary_df.copy()
        display_df['Score'] = display_df['Score'].apply(lambda x: f"{x:.1%}")
        display_df['Runner-up Score'] = display_df['Runner-up Score'].apply(lambda x: f"{x:.1%}")
        display_df['Margin'] = display_df['Margin'].apply(lambda x: f"{x:.1%}")
        
        st.dataframe(
            display_df,
            width='stretch',
            height=min(400, 35 * len(sample_ids) + 40),
            hide_index=True
        )
    
    with col_viz:
        st.markdown("#### Tissue Distribution")
        
        # Pie chart of predicted tissues
        tissue_counts = summary_df['Predicted Tissue'].value_counts()
        fig_pie = px.pie(
            values=tissue_counts.values,
            names=tissue_counts.index,
            hole=0.4  # Donut chart
        )
        fig_pie.update_layout(
            height=250, 
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.3)
        )
        fig_pie.update_traces(textposition='inside', textinfo='value+label')
        st.plotly_chart(fig_pie, width='stretch')
        
        st.markdown("#### Confidence Distribution")
        
        # Confidence bar chart
        conf_counts = pd.Series([d['Confidence'].split()[1] for d in summary_data]).value_counts()
        conf_order = ['High', 'Medium', 'Low']
        conf_colors = {'High': '#27ae60', 'Medium': '#f39c12', 'Low': '#e74c3c'}
        
        fig_conf = go.Figure()
        for conf in conf_order:
            count = conf_counts.get(conf, 0)
            fig_conf.add_trace(go.Bar(
                x=[conf], y=[count],
                name=conf,
                marker_color=conf_colors[conf],
                text=[count],
                textposition='auto'
            ))
        fig_conf.update_layout(
            height=200,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
            xaxis_title="",
            yaxis_title="Samples"
        )
        st.plotly_chart(fig_conf, width='stretch')
    
# ==============================================================================
# SECTION: Download Results
# ==============================================================================
if show_downloads:
    st.markdown("---")
    st.markdown("## Download Results")
    
    col_dl1, col_dl2, col_dl3 = st.columns(3)
    
    with col_dl1:
        csv_prob = prob_matrix.to_csv()
        st.download_button(
            label="Probability Matrix",
            data=csv_prob,
            file_name="mlmarker_probabilities.csv",
            mime="text/csv",
            width='stretch'
        )
    
    with col_dl2:
        if 'summary_df' in dir():
            csv_summary = summary_df.to_csv(index=False)
            st.download_button(
                label="Summary Table",
                data=csv_summary,
                file_name="mlmarker_summary.csv",
                mime="text/csv",
                width='stretch'
            )
    
    with col_dl3:
        if 'pro_con_df' in dir() and 'selected_tissue' in dir():
            csv_procon = pro_con_df.to_csv(index=False)
            st.download_button(
                label=f"Pro/Con ({selected_tissue})",
                data=csv_procon,
                file_name=f"mlmarker_procon_{selected_tissue}.csv",
                mime="text/csv",
                width='stretch'
            )

mark_says("Markverse/cropped_images/octopus.png", 
          "Thanks for using MLMarker! Don't forget to cite us in your publications.")
