import plotly.graph_objects as go
import bioservices
from gprofiler import GProfiler

import plotly.graph_objects as go

def get_protein_info(protein_id):
    """
    Get protein information from UniProt.
    
    Parameters:
        protein_id (str): UniProt protein ID.
    
    Returns:
        dict: Protein information.
    """
    import bioservices
    u = bioservices.UniProt()
    try:
        protein_info = u.search(protein_id, columns="accession, id, protein_name, cc_tissue_specificity")
        protein_info = protein_info.split('\n')[1].split('\t')
        protein_dict = {
            'id': protein_info[0],
            'entry name': protein_info[1],
            'protein_names': protein_info[2]
        }
        if len(protein_info) == 4:
            protein_dict['tissue_specificity'] = protein_info[3]
        return protein_dict
    except:
        print(f"Error retrieving information for protein {protein_id}")
        return None

def get_go_enrichment(protein_list):
    from gprofiler import GProfiler
    import plotly.graph_objects as go
    # Initialize g:Profiler
    gp = GProfiler(return_dataframe=True)

    # Dictionary to store GO terms and p-values for each tissue
    go_dict = {}


    # Perform over-representation analysis
    results = gp.profile(organism='hsapiens', query=protein_list, sources=['GO:BP', 'GO:MF', 'GO:CC', 'HPA'], combined=True)
    results = results[results['p_value']< 0.05]
    # Store results in the dictionary: {tissue: {GO_term: p-value}}
    return results

def visualise_custom_plot(df):
        
    # Aggregate positive and negative contributions per tissue
    positive_totals = df.clip(lower=0).sum(axis=1)
    negative_totals = df.clip(upper=0).abs().sum(axis=1)

    # Create the figure
    fig = go.Figure()

    # Add positive contributions (green bars)
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=positive_totals,
            name="Positive Contributions",
            marker_color='green',
            hoverinfo='x+y',
        )
    )

    # Add negative contributions (red bars)
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=negative_totals,
            name="Negative Contributions",
            marker_color='red',
            hoverinfo='x+y',
        )
    )

    # Customizing layout
    fig.update_layout(
        barmode='group',  # Group positive and negative bars side-by-side
        title='Grouped Barplot of Total Protein Contributions by Tissue',
        xaxis_title='Tissues',
        yaxis_title='Total Contributions',
        xaxis=dict(tickangle=-45),  # Tilt the x-axis labels for better readability
        template="plotly_white"
    )

    fig.show()


def visualise_custom_tissue_plot(df, tissue_name, top_n=10, show_others=False, threshold_others = 0.001):
    df = df.loc[[tissue_name]]

    # Separate positive and negative values for the tissue
    positive_contributions = df.clip(lower=0)  # Keep only positive values
    negative_contributions = df.clip(upper=0).abs()  # Keep absolute values of negatives

    # Filter significant contributions
    positive_main = positive_contributions.loc[:, (positive_contributions > threshold_others).any()]
    positive_others = positive_contributions.loc[:, (positive_contributions <= threshold_others).all()].sum(axis=1)

    negative_main = negative_contributions.loc[:, (negative_contributions > threshold_others).any()]
    negative_others = negative_contributions.loc[:, (negative_contributions <= threshold_others).all()].sum(axis=1)

    # Sort positive and negative contributions by total value
    sorted_positive = positive_main.sum(axis=0).sort_values(ascending=False)
    sorted_negative = negative_main.sum(axis=0).sort_values(ascending=False)

    # Select top N positive and negative proteins
    top_positive_contributions = sorted_positive.head(top_n).index.tolist()
    top_negative_contributions = sorted_negative.head(top_n).index.tolist()

    # Plotting
    fig = go.Figure()

    # Add all positive contributions (green bars)
    for protein in sorted_positive.index:
        # Check if the protein is one of the top N and add its label
        is_top = protein in top_positive_contributions
        fig.add_trace(
            go.Bar(
                x=positive_contributions.index,
                y=positive_main[protein],
                name=protein,
                marker_color="green" if is_top else "darkgreen",
                hoverinfo="name+y",
                hoverlabel=dict(namelength=-1),
                showlegend=False,
                text=protein if is_top else None,  # Show label for top proteins
                textposition="outside",  # Position the label inside the bar
                cliponaxis=False,  # Allow the label to be outside the bar
            )
        )
    # Add lines for top proteins to connect labels outside the bars
    for protein in top_positive_contributions:
        fig.add_trace(
            go.Scatter(
                x=[positive_contributions.index[0], positive_contributions.index[0]],
                y=[positive_contributions[protein].min(), positive_contributions[protein].max()],
                mode="lines+text",
                line=dict(color="green", width=2, dash="dot"),  # Line connecting label to bar
                text=[protein],
                textposition="middle right",
                showlegend=False,
                textfont=dict(color="green", size=12)
            )
        )
    # Add "Others" for positive contributions
    if show_others and positive_others.sum() > 0:
        fig.add_trace(
            go.Bar(
                x=positive_contributions.index,
                y=positive_others,
                name="Others (Positive)",
                marker_color="lightgreen",
                hoverinfo="name+y",
                hoverlabel=dict(namelength=-1),
                showlegend=False,
            )
        )

  # Add negative contributions (sorted by total contribution)
    for protein in sorted_negative.index:
        is_top = protein in top_negative_contributions
        fig.add_trace(
            go.Bar(
                x=negative_contributions.index,
                y=negative_main[protein],
                name=protein,
                marker_color="red" if is_top else "darkred",
                hoverinfo="name+y",
                hoverlabel=dict(namelength=-1),
                showlegend=False,
                text=protein if is_top else None,  # Show label for top proteins
                textposition="outside",  # Position the label outside the bar
                cliponaxis=False,  # Allow the label to be outside the bar
            )
        )

    # Add "Others" for negative contributions
    if show_others and negative_others.sum() > 0:
        fig.add_trace(
            go.Bar(
                x=negative_contributions.index,
                y=negative_others,
                name="Others (Negative)",
                marker_color="lightcoral",
                hoverinfo="name+y",
                hoverlabel=dict(namelength=-1),
                showlegend=False,
            )
        )

    # Customizing layout
    fig.update_layout(
        barmode="stack",  # Stack the bars
        title=f"""Protein Contributions for {tissue_name} (threshold={threshold_others})""",
        xaxis_title="Cluster",
        yaxis_title="Protein Contributions",
        xaxis={"categoryorder": "array", "categoryarray": sorted_positive.index.tolist() + sorted_negative.index.tolist()},
        hovermode="closest",
        template="plotly_white",
        width=600,
        height=800,
        margin=dict(l=100, r=100),  # Adjust margins
    )
    return fig

def prediction_df_2tissues_scatterplot(df, tissues=list):
    tissueA = tissues[0]
    tissueB = tissues[1]
    df_vis = df.T
    fig = go.Figure(data=go.Scatter(
        x=df_vis[tissueA],
        y=df_vis[tissueB],
        mode='markers',
        marker=dict(
            size=8,
            color='blue',  # You can change the color here
            opacity=0.7
        ),
        text=[f"Protein: {protein}<br>{tissueA} SHAP: {pg_shap}<br>{tissueB} value: {brain_value}" 
            for protein, pg_shap, brain_value in zip(df_vis.index, df_vis[tissueA], df_vis[tissueB])],
        hoverinfo='text'
    ))

    fig.update_layout(
        title=f'Scatterplot of {tissueA} SHAP values vs {tissueB} values',
        xaxis_title=f'{tissueA} SHAP values',
        yaxis_title=f'{tissueB} SHAP values',
        xaxis=dict(color='black', zeroline=True, zerolinecolor='darkgrey'),
        yaxis=dict(color='black', zeroline=True, zerolinecolor='darkgrey')
    )

    fig.show()
import io
import streamlit.components.v1 as components
import base64
import streamlit as st

# ==============================================================================
# HELP CONTENT - Consistent explanations across the app
# ==============================================================================

HELP_CONTENT = {
    'shap': """
**SHAP (SHapley Additive exPlanations)** values explain how each protein contributes to a prediction.

- **Positive SHAP** = Protein supports the tissue prediction (pro)
- **Negative SHAP** = Protein opposes the tissue prediction (con)
- **Magnitude** = Strength of the contribution

SHAP values sum up to the final prediction score for each tissue.
""",
    'coverage': """
**Coverage** measures what percentage of MLMarker's 5,979 known proteins were detected in your sample.

- **>20%** = Good coverage, reliable predictions
- **5-20%** = Moderate coverage, predictions should be reliable
- **<5%** = Low coverage (typical for fluids, cell lines) - enable penalty factor

Higher coverage generally means more confident predictions.
""",
    'penalty': """
**Penalty Factor** adjusts how MLMarker handles missing proteins.

- **OFF** = Best for solid tissue samples where most proteins should be present
- **ON** = Best for sparse samples (plasma, urine, cell lines, organoids) where many proteins are naturally absent

When enabled, missing proteins have less negative impact on predictions.
""",
    'ora': """
**Over-Representation Analysis (ORA)** identifies biological functions enriched in your protein list.

Uses g:Profiler to test if your proteins are statistically over-represented in:
- **GO:BP** = Biological Process
- **GO:MF** = Molecular Function  
- **GO:CC** = Cellular Component
- **HPA** = Human Protein Atlas tissue expression
- **KEGG** = Pathway annotations

P-value indicates statistical significance of enrichment.
""",
    'tissue_probability': """
**Tissue Probability** represents how similar your sample's protein expression pattern is to each tissue.

- Values range from 0 to 1 (shown as percentages)
- Higher = More similar to that tissue's expression profile
- Top prediction = Most likely tissue of origin

Probabilities are normalized to sum to 100% across all tissues.
""",
    'binary_vs_quantified': """
**Analysis Type** determines how protein intensities are used:

- **Quantified**: Uses actual intensity values (MinMax normalized). Best for quantitative proteomics.
- **Binary**: Only considers presence/absence (1 or 0). Use for semi-quantitative data like Olink.

Quantified analysis generally provides better predictions when intensity values are reliable.
""",
    'pca': """
**PCA (Principal Component Analysis)** reduces high-dimensional data for visualization.

- Samples close together have similar profiles
- PC1 captures the most variance, PC2 the second most
- Clustering indicates similar expression patterns

Useful for identifying sample groups and outliers.
""",
    'pro_con': """
**Pro/Con Proteins** indicate how each protein influences the prediction for a specific tissue.

- **Pro (positive SHAP)**: Protein expression pattern supports this tissue prediction
- **Con (negative SHAP)**: Protein expression pattern opposes this tissue prediction

The sum of all pro/con contributions determines the final tissue probability.
""",
    'abundance': """
**Abundance** refers to the protein intensity/expression level in your sample.

- **Present**: Protein was detected (value > 0)
- **Absent**: Protein was not detected or below detection limit (value = 0)

Filtering by abundance helps identify which detected proteins drive predictions.
""",
    'mlmarker_features': """
**MLMarker Features** are the 5,979 proteins used by the MLMarker model.

These proteins were selected during model training as informative for distinguishing between 34 tissue types.
Only proteins in this feature set contribute to predictions.
""",
    'mann_whitney': """
**Mann-Whitney U Test** is a non-parametric statistical test.

It compares whether two groups have different distributions without assuming normality.
A p-value < 0.05 suggests the groups have significantly different tissue probability distributions.
""",
    'heatmap': """
**Tissue Probability Heatmap** shows predictions for all samples at once.

- Rows = Samples
- Columns = Tissues
- Color intensity = Probability (higher = more similar)

Useful for identifying patterns across multiple samples.
"""
}


def show_help(topic, title=None):
    """Display a help popover for a given topic."""
    if topic in HELP_CONTENT:
        with st.popover("?" if title is None else f"? {title}"):
            st.markdown(HELP_CONTENT[topic])


# ==============================================================================
# DARK MODE SUPPORT
# ==============================================================================

def get_theme_colors():
    """
    Get color scheme based on Streamlit's theme.
    Returns a dict with plot-friendly colors.
    """
    # Try to detect Streamlit theme from query params or use defaults
    # Streamlit doesn't expose theme directly, so we use a workaround
    try:
        import streamlit as st
        # Check if user set a preference in session state
        if "dark_mode" in st.session_state:
            is_dark = st.session_state.dark_mode
        else:
            # Default to light mode (Streamlit's default)
            is_dark = False
    except:
        is_dark = False
    
    if is_dark:
        return {
            'background': '#0e1117',
            'paper_bg': '#262730',
            'text': '#fafafa',
            'grid': '#4a4a5a',
            'primary': '#ff4b4b',
            'secondary': '#1f77b4',
            'positive': '#2ecc71',
            'negative': '#e74c3c',
            'neutral': '#95a5a6',
            'template': 'plotly_dark'
        }
    else:
        return {
            'background': '#ffffff',
            'paper_bg': '#ffffff', 
            'text': '#31333f',
            'grid': '#e6e9ef',
            'primary': '#ff4b4b',
            'secondary': '#1f77b4',
            'positive': '#27ae60',
            'negative': '#c0392b',
            'neutral': '#7f8c8d',
            'template': 'plotly_white'
        }


def apply_theme_to_figure(fig, theme_colors=None):
    """
    Apply theme colors to a Plotly figure.
    
    Parameters:
        fig: Plotly figure object
        theme_colors: Dict from get_theme_colors() or None to auto-detect
    
    Returns:
        Modified figure
    """
    if theme_colors is None:
        theme_colors = get_theme_colors()
    
    fig.update_layout(
        template=theme_colors['template'],
        paper_bgcolor=theme_colors['paper_bg'],
        plot_bgcolor=theme_colors['background'],
        font=dict(color=theme_colors['text'])
    )
    
    return fig


def render_theme_toggle():
    """
    Render a theme toggle in the sidebar.
    Call this at the start of any page to enable theme switching.
    """
    import streamlit as st
    
    with st.sidebar:
        if "dark_mode" not in st.session_state:
            st.session_state.dark_mode = False
        
        dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode, key="dark_mode_toggle")
        st.session_state.dark_mode = dark_mode
        
        return dark_mode


# ==============================================================================
# KEYBOARD SHORTCUTS
# ==============================================================================

def render_keyboard_shortcuts_help():
    """
    Display keyboard shortcuts help in an expander.
    """
    import streamlit as st
    
    with st.expander("⌨️ Keyboard Shortcuts", expanded=False):
        st.markdown("""
        | Shortcut | Action |
        |----------|--------|
        | `Ctrl + Enter` | Submit form / Run analysis |
        | `Ctrl + S` | Download results (when available) |
        | `Ctrl + /` | Show this help |
        | `Esc` | Close dialogs |
        
        *Note: Shortcuts work when the form/page is focused.*
        """)


def inject_keyboard_shortcuts(download_callback_key=None):
    """
    Inject keyboard shortcut JavaScript into the page.
    
    This enables:
    - Ctrl+Enter to submit forms
    - Ctrl+S to trigger downloads
    
    Parameters:
        download_callback_key: Session state key for download button to trigger
    """
    import streamlit.components.v1 as components
    
    # JavaScript for keyboard shortcuts
    js_code = """
    <script>
    document.addEventListener('keydown', function(e) {
        // Ctrl+Enter - Submit forms
        if (e.ctrlKey && e.key === 'Enter') {
            // Find and click the primary submit button
            const buttons = document.querySelectorAll('button[kind="primary"]');
            if (buttons.length > 0) {
                buttons[0].click();
                e.preventDefault();
            }
        }
        
        // Ctrl+S - Trigger download
        if (e.ctrlKey && e.key === 's') {
            // Find download buttons
            const downloadBtns = document.querySelectorAll('button[data-testid="stDownloadButton"]');
            if (downloadBtns.length > 0) {
                downloadBtns[0].click();
                e.preventDefault();
            }
        }
        
        // Escape - Close any open dialogs
        if (e.key === 'Escape') {
            const closeButtons = document.querySelectorAll('[aria-label="Close"]');
            closeButtons.forEach(btn => btn.click());
        }
    });
    </script>
    """
    
    components.html(js_code, height=0)


def copy_to_clipboard_button(text, button_label="Copy", key=None):
    """Create a button that copies text to clipboard."""
    # Use a unique key for each button
    if key is None:
        import hashlib
        key = f"copy_{hashlib.md5(str(text).encode()).hexdigest()[:8]}"
    
    # JavaScript to copy to clipboard
    if isinstance(text, list):
        text = '\n'.join(text)
    
    components.html(f"""
    <button onclick="navigator.clipboard.writeText(`{text}`).then(() => {{
        this.innerHTML = 'Copied!';
        setTimeout(() => this.innerHTML = '{button_label}', 2000);
    }})" style="
        padding: 5px 10px;
        border: 1px solid #ccc;
        border-radius: 5px;
        background: #f0f2f6;
        cursor: pointer;
        font-size: 14px;
    ">{button_label}</button>
    """, height=40)


def mark_says(image_path, message):
        with open(image_path, "rb") as img_file:
            octo_base64 = base64.b64encode(img_file.read()).decode()
        components.html(f"""
        <div id="mark-box" style="
            position: fixed;
            bottom: 20px;
            right: 20px;
            background-color: #f0f2f6;
            padding: 10px 15px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.2);
            z-index: 1000;
            display: flex;
            align-items: center;
            font-family: 'Source Sans Pro', sans-serif;
        ">
            <img src="data:image/png;base64,{octo_base64}" style="height:60px;width:auto;margin-right:10px;">
            <div>{message}</div>
             <span onclick="document.getElementById('mark-box').style.display='none';"
                style="cursor:pointer; margin-left:10px; font-weight:bold; font-size:18px;">×</span>
        </div>
        """, height=110)


def get_sample_data():
    """
    Global helper to get the current sample data.
    Returns (prediction_df, prediction_summed, sample_name, is_batch) or (None, None, None, False) if no data.
    """
    has_batch = "batch_results" in st.session_state and st.session_state.batch_results
    has_single = "prediction" in st.session_state and "prediction_summed" in st.session_state
    
    if not has_batch and not has_single:
        return None, None, None, False
    
    if has_batch:
        # Use global selected sample if set, otherwise first sample
        sample_ids = list(st.session_state.batch_results.keys())
        if "global_selected_sample" not in st.session_state:
            st.session_state.global_selected_sample = sample_ids[0]
        elif st.session_state.global_selected_sample not in sample_ids:
            st.session_state.global_selected_sample = sample_ids[0]
        
        sample = st.session_state.global_selected_sample
        result = st.session_state.batch_results[sample]
        return result['prediction_df'], result['summed_pred'], sample, True
    else:
        return (
            st.session_state["prediction"],
            st.session_state["prediction_summed"],
            st.session_state.get("sel_sample", "Sample"),
            False
        )


def render_sample_selector(page_key="default"):
    """
    Renders a global sample selector in the sidebar. 
    Call this at the top of each analysis page.
    Returns the current sample name.
    """
    has_batch = "batch_results" in st.session_state and st.session_state.batch_results
    
    if has_batch:
        sample_ids = list(st.session_state.batch_results.keys())
        
        # Get current selection
        if "global_selected_sample" not in st.session_state:
            st.session_state.global_selected_sample = sample_ids[0]
        
        current_idx = 0
        if st.session_state.global_selected_sample in sample_ids:
            current_idx = sample_ids.index(st.session_state.global_selected_sample)
        
        selected = st.selectbox(
            "Select Sample",
            sample_ids,
            index=current_idx,
            key=f"sample_selector_{page_key}"
        )
        
        # Update global state
        st.session_state.global_selected_sample = selected
        return selected
    else:
        return st.session_state.get("sel_sample", "Sample")


def render_workflow_progress(current_step):
    """
    Renders a workflow progress indicator in the sidebar.
    Steps: 1=Home, 2=QC, 3=Visualisations, 4=Proteins, 5=Functional, 6=Comparison
    """
    steps = [
        ("Home", "Upload & Run"),
        ("QC", "Quality Check"),
        ("Visualisations", "View Results"),
        ("Proteins", "Explore Proteins"),
        ("Functional", "ORA Analysis"),
        ("Comparison", "Compare Samples")
    ]
    
    st.markdown("### Workflow")
    for i, (name, desc) in enumerate(steps, 1):
        if i < current_step:
            st.markdown(f"~~{i}. {name}~~")
        elif i == current_step:
            st.markdown(f"**→ {i}. {name}**")
        else:
            st.markdown(f"{i}. {name}", help=desc)