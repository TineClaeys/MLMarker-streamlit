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


    # Perform GO enrichment
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