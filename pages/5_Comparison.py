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
    show_pca = st.checkbox("SHAP Profile PCA", value=False)
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
    st.plotly_chart(fig_heatmap, use_container_width=True)

# ==============================================================================
# SECTION: PCA based on SHAP Profiles
# ==============================================================================
if show_pca:
    st.markdown("---")
    st.markdown("## SHAP Profile Clustering")
    st.caption("Samples clustered by which proteins drive their tissue predictions")
    
    # Build SHAP feature matrix per tissue or overall
    pca_mode = st.radio(
        "Cluster samples by:",
        ["Overall SHAP profile", "Tissue-specific SHAP profile"],
        horizontal=True,
        key="pca_mode"
    )
    
    if pca_mode == "Tissue-specific SHAP profile":
        pca_tissue = st.selectbox("Select tissue for PCA", all_tissues, key="pca_tissue")
    
    # Build the SHAP matrix
    shap_data = {}
    for sample_id, result in results.items():
        pred_df = result['prediction_df']
        if pca_mode == "Overall SHAP profile":
            # Use mean absolute SHAP across all tissues for each protein
            shap_data[sample_id] = pred_df.abs().mean(axis=0)
        else:
            # Use SHAP values for specific tissue
            if pca_tissue in pred_df.index:
                shap_data[sample_id] = pred_df.loc[pca_tissue]
            else:
                shap_data[sample_id] = pd.Series(0, index=pred_df.columns)
    
    shap_matrix = pd.DataFrame(shap_data).T.fillna(0)
    
    # Only proceed if we have enough samples
    if len(shap_matrix) >= 3:
        # Standardize and run PCA
        scaler = StandardScaler()
        shap_scaled = scaler.fit_transform(shap_matrix)
        
        n_components = min(3, len(shap_matrix) - 1, len(shap_matrix.columns))
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(shap_scaled)
        
        # Create PCA dataframe
        pca_df = pd.DataFrame(
            pca_result[:, :min(2, n_components)],
            columns=['PC1', 'PC2'] if n_components >= 2 else ['PC1'],
            index=shap_matrix.index
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
                    title=f"Sample Clustering by {'Overall' if pca_mode == 'Overall SHAP profile' else pca_tissue} SHAP Profile"
                )
                fig_pca.update_traces(textposition='top center', marker=dict(size=12))
                fig_pca.update_layout(
                    height=450,
                    xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
                    yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)",
                    margin=dict(l=10, r=10, t=40, b=10)
                )
                st.plotly_chart(fig_pca, use_container_width=True)
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
            fig_var.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_var, use_container_width=True)
            
            # Show top contributing proteins
            st.markdown("#### Top Contributing Proteins")
            loadings = pd.DataFrame(
                pca.components_.T,
                columns=[f'PC{i+1}' for i in range(n_components)],
                index=shap_matrix.columns
            )
            top_pc1 = loadings['PC1'].abs().nlargest(5)
            st.dataframe(
                pd.DataFrame({'Protein': top_pc1.index, 'Loading': top_pc1.values}),
                hide_index=True,
                use_container_width=True
            )
        
        mark_says("Markverse/cropped_images/Coding Mark.png", 
                  "Samples close together in PCA have similar protein expression patterns!")
    else:
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
        st.plotly_chart(fig_box, use_container_width=True)
    
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
        st.plotly_chart(fig_bar, use_container_width=True)
    
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
            st.plotly_chart(fig_procon, use_container_width=True)
        
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
            st.plotly_chart(fig_net, use_container_width=True)

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
            st.plotly_chart(fig_radar, use_container_width=True)
        
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
            st.plotly_chart(fig_diff, use_container_width=True)
        
        mark_says("Markverse/cropped_images/Mark knitting.png", 
                  "Positive differences mean sample A has higher probability for that tissue!")
    else:
        st.info("Select two different samples to compare.")

# ==============================================================================
# SECTION: Top Tissue Summary
# ==============================================================================
if show_summary:
    st.markdown("---")
    st.markdown("## Top Tissue Summary")
    st.caption("Highest-probability tissue predictions per sample")
    
    summary_data = []
    for sample_id in sample_ids:
        probs = prob_matrix.loc[sample_id]
        top_tissue = probs.idxmax()
        top_prob = probs.max()
        second_tissue = probs.drop(top_tissue).idxmax()
        second_prob = probs.drop(top_tissue).max()
        summary_data.append({
            'Sample': sample_id,
            'Top Tissue': top_tissue,
            'Probability': top_prob,
            '2nd Tissue': second_tissue,
            '2nd Prob': second_prob,
            'Gap': top_prob - second_prob
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    col_table, col_pie = st.columns([2, 1])
    
    with col_table:
        st.dataframe(
            summary_df.style.format({
                'Probability': '{:.1%}',
                '2nd Prob': '{:.1%}',
                'Gap': '{:.1%}'
            }).background_gradient(subset=['Probability', 'Gap'], cmap='RdYlGn'),
            use_container_width=True,
            height=min(400, 35 * len(sample_ids) + 40)
        )
    
    with col_pie:
        tissue_counts = summary_df['Top Tissue'].value_counts()
        fig_pie = px.pie(
            values=tissue_counts.values,
            names=tissue_counts.index,
            title="Tissue Distribution"
        )
        fig_pie.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)

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
            use_container_width=True
        )
    
    with col_dl2:
        if 'summary_df' in dir():
            csv_summary = summary_df.to_csv(index=False)
            st.download_button(
                label="Summary Table",
                data=csv_summary,
                file_name="mlmarker_summary.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col_dl3:
        if 'pro_con_df' in dir() and 'selected_tissue' in dir():
            csv_procon = pro_con_df.to_csv(index=False)
            st.download_button(
                label=f"Pro/Con ({selected_tissue})",
                data=csv_procon,
                file_name=f"mlmarker_procon_{selected_tissue}.csv",
                mime="text/csv",
                use_container_width=True
            )

mark_says("Markverse/cropped_images/octopus.png", 
          "Thanks for using MLMarker! Don't forget to cite us in your publications.")
