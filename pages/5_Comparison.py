import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
import mlmarker

# Import custom functions with fallbacks
from custom_functions import mark_says
try:
    from custom_functions import show_help, HELP_CONTENT
except ImportError:
    def show_help(topic, title=None): pass
    HELP_CONTENT = {}

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
    mark_says("Markverse/mark pointing.png", 
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
    show_group_comparison = st.checkbox("Group Comparison", value=False, 
                                        help="Compare two groups of samples with statistical tests")
    show_tissue_analysis = st.checkbox("Tissue-Specific Analysis", value=False)
    show_sample_comparison = st.checkbox("Sample Comparison", value=False)
    show_summary = st.checkbox("Top Tissue Summary", value=False)
    show_downloads = st.checkbox("Download Results", value=False)
    
    st.markdown("---")
    mark_says("Markverse/mark_binoculars.png", 
              f"Scanning {len(sample_ids)} samples! Let's see what we find!")

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
    st.plotly_chart(fig_heatmap, width='content')

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
                    st.plotly_chart(fig_pca, width='content')
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
                st.plotly_chart(fig_var, width='content')
                
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
                    width='content'
                )
            
            # Info about the analysis
            st.info(f"**{pca_title}**: {len(pca_matrix.columns):,} features used for clustering {len(pca_matrix)} samples.")
            
            mark_says("Markverse/Mark_touching_human_like_davincis.png", 
                      "Samples close together have similar profiles! Look at those patterns!")
    elif pca_matrix is not None:
        st.info("Need at least 3 samples for PCA clustering.")

# ==============================================================================
# SECTION: Group Comparison with Statistical Tests
# ==============================================================================
if show_group_comparison:
    st.markdown("---")
    st.markdown("## Group Comparison")
    st.caption("Compare tissue probability distributions between two groups using Mann-Whitney U test")
    
    if len(sample_ids) < 4:
        st.warning("Need at least 4 samples (2 per group) for group comparison.")
    else:
        # Group assignment interface
        st.markdown("### Define Groups")
        st.info("Assign samples to **Group A** or **Group B** for comparison. Leave unchecked samples out of the analysis.")
        
        # Create columns for group assignment
        col_assign1, col_assign2 = st.columns(2)
        
        group_a_samples = []
        group_b_samples = []
        
        with col_assign1:
            st.markdown("**Group A** (e.g., Control)")
            for sample in sample_ids:
                if st.checkbox(sample, key=f"grp_a_{sample}"):
                    group_a_samples.append(sample)
        
        with col_assign2:
            st.markdown("**Group B** (e.g., Disease)")
            for sample in sample_ids:
                # Only show if not in group A
                if sample not in group_a_samples:
                    if st.checkbox(sample, key=f"grp_b_{sample}"):
                        group_b_samples.append(sample)
        
        # Show group summary
        col_sum1, col_sum2 = st.columns(2)
        with col_sum1:
            st.metric("Group A", f"{len(group_a_samples)} samples")
        with col_sum2:
            st.metric("Group B", f"{len(group_b_samples)} samples")
        
        # Run analysis if both groups have samples
        if len(group_a_samples) >= 2 and len(group_b_samples) >= 2:
            st.markdown("---")
            st.markdown("### Statistical Comparison (Mann-Whitney U)")
            
            # Get probabilities for each group
            group_a_probs = prob_matrix.loc[group_a_samples]
            group_b_probs = prob_matrix.loc[group_b_samples]
            
            # Calculate Mann-Whitney U test for each tissue
            stats_results = []
            for tissue in all_tissues:
                a_vals = group_a_probs[tissue].values
                b_vals = group_b_probs[tissue].values
                
                # Mann-Whitney U test (non-parametric)
                try:
                    u_stat, u_pval = stats.mannwhitneyu(a_vals, b_vals, alternative='two-sided')
                except ValueError:
                    u_stat, u_pval = np.nan, np.nan
                
                stats_results.append({
                    'Tissue': tissue,
                    'Median A': np.median(a_vals),
                    'Median B': np.median(b_vals),
                    'Diff (A-B)': np.median(a_vals) - np.median(b_vals),
                    'U-statistic': u_stat,
                    'p-value': u_pval
                })
            
            stats_df = pd.DataFrame(stats_results)
            
            # Sort by significance
            stats_df = stats_df.sort_values('p-value')
            
            # Summary of significant findings
            sig_tissues = stats_df[stats_df['p-value'] < 0.05]
            
            col_res1, col_res2 = st.columns([1, 1])
            
            with col_res1:
                st.markdown("#### Significant Differences (p < 0.05)")
                if len(sig_tissues) > 0:
                    for _, row in sig_tissues.iterrows():
                        direction = "higher in A" if row['Diff (A-B)'] > 0 else "higher in B"
                        st.write(f"**{row['Tissue']}**: {direction} (p={row['p-value']:.4f})")
                else:
                    st.info("No tissues show significant differences at p < 0.05")
            
            with col_res2:
                st.markdown("#### All Tissues Summary")
                st.caption(f"{len(sig_tissues)} of {len(all_tissues)} tissues differ significantly")
            
            # Results table
            with st.expander("Full Statistical Results", expanded=False):
                display_df = stats_df.copy()
                for col in ['Median A', 'Median B', 'Diff (A-B)', 'U-statistic']:
                    display_df[col] = display_df[col].round(4)
                display_df['p-value'] = display_df['p-value'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
                
                st.dataframe(display_df, hide_index=True)
                
                # Download button
                csv_stats = stats_df.to_csv(index=False)
                st.download_button(
                    "Download Statistics CSV",
                    csv_stats,
                    "group_comparison_stats.csv",
                    "text/csv"
                )
            
            # Box plots for top 4 tissues by significance
            st.markdown("### Tissue Probability Distributions by Group")
            
            top_tissues = stats_df.head(4)['Tissue'].tolist()
            
            # Prepare data for box plots
            box_data = []
            for tissue in top_tissues:
                for sample in group_a_samples:
                    box_data.append({'Tissue': tissue, 'Group': 'A', 'Probability': prob_matrix.loc[sample, tissue]})
                for sample in group_b_samples:
                    box_data.append({'Tissue': tissue, 'Group': 'B', 'Probability': prob_matrix.loc[sample, tissue]})
            box_df = pd.DataFrame(box_data)
            
            fig_boxes = px.box(
                box_df,
                x='Tissue',
                y='Probability',
                color='Group',
                points='all',
                title="Top 4 Tissues by Significance",
                color_discrete_map={'A': '#3498db', 'B': '#e74c3c'}
            )
            fig_boxes.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_boxes, width='content')
            
            mark_says("Markverse/Markwithamassspec.png", 
                      "Group comparison complete! Check the distributions above.")
        
        elif len(group_a_samples) > 0 or len(group_b_samples) > 0:
            st.info("Select at least 2 samples in each group to run statistical tests.")
        else:
            st.info("Assign samples to groups above to begin comparison.")

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
        st.plotly_chart(fig_box, width='content')
    
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
        st.plotly_chart(fig_bar, width='content')
    
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
            st.plotly_chart(fig_procon, width='content')
        
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
            st.plotly_chart(fig_net, width='content')

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
            st.plotly_chart(fig_radar, width='content')
        
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
            st.plotly_chart(fig_diff, width='content')
        
        mark_says("Markverse/cropped_images/Coding Mark.png", 
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
            width='content',
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
        st.plotly_chart(fig_pie, width='content')
        
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
        st.plotly_chart(fig_conf, width='content')
    
# ==============================================================================
# SECTION: Download Results
# ==============================================================================
if show_downloads:
    st.markdown("---")
    st.markdown("## Download Results")
    
    # Build summary dataframe for downloads
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
            'Top Probability': top_prob,
            'Second Tissue': second_tissue,
            'Second Probability': second_prob
        })
    download_summary_df = pd.DataFrame(summary_data)
    
    col_dl1, col_dl2, col_dl3 = st.columns(3)
    
    with col_dl1:
        csv_prob = prob_matrix.to_csv()
        st.download_button(
            label="📊 Probability Matrix",
            data=csv_prob,
            file_name="mlmarker_probabilities.csv",
            mime="text/csv",
            width='content'
        )
    
    with col_dl2:
        csv_summary = download_summary_df.to_csv(index=False)
        st.download_button(
            label="📋 Summary Table",
            data=csv_summary,
            file_name="mlmarker_summary.csv",
            mime="text/csv",
            width='content'
        )
    
    with col_dl3:
        # Generate comprehensive report as HTML
        html_report = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>MLMarker Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .metric {{ display: inline-block; margin: 10px 20px; padding: 15px; background: #ecf0f1; border-radius: 8px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
                .metric-label {{ font-size: 14px; color: #7f8c8d; }}
            </style>
        </head>
        <body>
            <h1>🐙 MLMarker Analysis Report</h1>
            <p>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Summary Statistics</h2>
            <div class="metric">
                <div class="metric-value">{len(sample_ids)}</div>
                <div class="metric-label">Samples Analyzed</div>
            </div>
            <div class="metric">
                <div class="metric-value">{download_summary_df['Top Tissue'].nunique()}</div>
                <div class="metric-label">Unique Tissues Predicted</div>
            </div>
            <div class="metric">
                <div class="metric-value">{download_summary_df['Top Probability'].mean():.1%}</div>
                <div class="metric-label">Average Top Probability</div>
            </div>
            
            <h2>Sample Predictions</h2>
            {download_summary_df.to_html(index=False, classes='dataframe')}
            
            <h2>Tissue Distribution</h2>
            <table>
                <tr><th>Tissue</th><th>Sample Count</th><th>Percentage</th></tr>
                {''.join(f"<tr><td>{tissue}</td><td>{count}</td><td>{count/len(sample_ids)*100:.1f}%</td></tr>" for tissue, count in download_summary_df['Top Tissue'].value_counts().items())}
            </table>
            
            <h2>Full Probability Matrix</h2>
            {prob_matrix.round(4).to_html(classes='dataframe')}
            
            <hr>
            <p><i>Report generated by MLMarker Streamlit App</i></p>
        </body>
        </html>
        """
        st.download_button(
            label="📄 HTML Report",
            data=html_report,
            file_name="mlmarker_report.html",
            mime="text/html",
            width='content'
        )
    
    # ZIP export with all data
    st.markdown("### Complete Export")
    
    import io
    import zipfile
    
    # Create ZIP file in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add probability matrix
        zf.writestr('probabilities.csv', prob_matrix.to_csv())
        
        # Add summary
        zf.writestr('summary.csv', download_summary_df.to_csv(index=False))
        
        # Add HTML report
        zf.writestr('report.html', html_report)
        
        # Add individual sample SHAP values
        for sample_id, result in results.items():
            safe_name = sample_id.replace('/', '_').replace('\\', '_')
            zf.writestr(f'shap_values/{safe_name}_shap.csv', result['prediction_df'].to_csv())
    
    zip_buffer.seek(0)
    
    st.download_button(
        label="📦 Download All Results (ZIP)",
        data=zip_buffer.getvalue(),
        file_name="mlmarker_complete_results.zip",
        mime="application/zip",
        type="primary"
    )

mark_says("Markverse/markgraduation.png", 
          "Thanks for using MLMarker! Don't forget to cite us in your publications.")
