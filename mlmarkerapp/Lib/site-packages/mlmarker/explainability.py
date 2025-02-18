import shap
import pandas as pd
import numpy as np
import plotly.express as px

class Explainability:
    def __init__(self, model, features, sample, penalty_factor, explainer=None):
        self.model = model
        self.features = features
        self.sample = sample  # Store sample
        # Initialize the explainer (default is TreeExplainer if not passed)
        self.explainer = explainer or shap.TreeExplainer(model)  
        self.zero_shaps = self.zero_sample()
        self.penalty_factor = penalty_factor
    
    def zero_sample(self):
        zero_sample = pd.DataFrame(np.zeros((1, len(self.features))), columns=self.features)
        zero_shaps = self.shap_values_df(sample=zero_sample, n_preds=100)
        return zero_shaps


    def predict_top_tissues(self, n_preds=5):
        if not isinstance(n_preds, int):
            raise ValueError(f"n_preds should be an integer, got {type(n_preds)}.")
        probabilities = self.model.predict_proba(self.sample).flatten()
        classes = self.model.classes_
        result = sorted(zip(classes, probabilities), key=lambda x: x[1], reverse=True)[:n_preds]
        formatted_result = [(pred_tissue, round(float(prob), 4)) for pred_tissue, prob in result]
        return formatted_result

    def calculate_shap(self, sample=None):
        """Calculate SHAP values for a given sample, or use the class sample by default."""
        if sample is None:
            sample = self.sample
        shap_values = self.explainer.shap_values(self.sample, check_additivity=False)
        original_order = np.array(shap_values).shape
        
        classes = self.model.classes_
        desired_order = (original_order.index(1), original_order.index(len(classes)), original_order.index(len(self.features)))
        shap_values = np.transpose(shap_values, desired_order)
        shap_values = shap_values[0]  # remove the first dimension
        #round to 4 decimal 
        shap_values = shap_values.round(5)
        return shap_values


    def shap_values_df(self, sample=None, n_preds=5):
        """Get a dataframe with the SHAP values for each feature for the top n_preds tissues"""
        if sample is None:
            sample = self.sample
        shap_values = self.calculate_shap(sample)
        classes = self.model.classes_
        predictions = self.predict_top_tissues(n_preds)
        
        shap_df = pd.DataFrame(shap_values)
        shap_df.columns = self.features
        shap_df['tissue'] = classes
        shap_df = shap_df.set_index('tissue')
        shap_df = shap_df.loc[[item[0] for item in predictions]]
        return shap_df

    def adjusted_absent_shap_values_df(self, n_preds=5):
        """
        Adjust SHAP values by penalizing absent features based on a penalty factor.
        Keeps SHAP values for present features unchanged and handles contributing absent features separately.
        
        Args:
            n_preds (int): Number of top predicted tissues to include.
            penalty_factor (float): Factor to penalize SHAP values for absent features that contribute.

        Returns:
            pd.DataFrame: Adjusted SHAP values for the top predicted tissues.
        """
        # Get original SHAP values dataframe
        penalty_factor = self.penalty_factor
        shap_df = self.shap_values_df(n_preds=n_preds)
        
        # Identify proteins that are absent (value == 0) in the sample
        absent_proteins = self.sample.columns[self.sample.iloc[0] == 0]
        present_proteins = [col for col in shap_df.columns if col not in absent_proteins]
        
        # Separate SHAP values for present and absent features
        present_shap = shap_df[present_proteins]  # SHAP values for present features remain unchanged
        absent_shap = shap_df[absent_proteins]
        
        # Handle absent features:
        # - Identify absent features that contribute (non-zero SHAP values)
        # - Penalize them using the penalty factor and pre-stored zero SHAP values
        contributing_absent_proteins = absent_shap.columns[absent_shap.sum() != 0]
        non_contributing_absent_proteins = absent_shap.columns[absent_shap.sum() == 0]
        
        # Penalize contributing absent features
        if len(contributing_absent_proteins) > 0:
            zero_absent_shap = self.zero_shaps[contributing_absent_proteins]  # Reference zero SHAP values
            penalized_absent_shap = absent_shap[contributing_absent_proteins] - (penalty_factor * zero_absent_shap)
        else:
            penalized_absent_shap = pd.DataFrame(columns=contributing_absent_proteins)  # Empty if no contributing absent features
        
        # Combine present SHAP values, penalized absent SHAPs, and non-contributing SHAPs
        combined_df = pd.concat(
            [
                present_shap,
                absent_shap[non_contributing_absent_proteins],  # Non-contributing SHAP values remain as is
                penalized_absent_shap,  # Adjusted SHAP values for contributing absent features
            ],
            axis=1
        )
        
        # Reorder to match original column order
        combined_df = combined_df[shap_df.columns]
        
        return combined_df

    def visualize_shap_force_plot(self, shap_values, sample, tissue_idx=None, n_preds=5, tissue_name=None):
        """
        Visualizes SHAP force plots for top predicted tissues or for a specific tissue.
        """
        shap_values = shap_values.astype(float)  # Convert shap_values to float type
        shap_values = np.round(shap_values, 5)  # Round shap_values to 5 decimal places
        predictions = self.predict_top_tissues(n_preds=n_preds)
        i = 0
        shap.initjs()
        # If tissue_name is provided, check if it's valid
        if tissue_name:
            if tissue_name not in self.model.classes_:
                raise ValueError(f"Tissue '{tissue_name}' is not a valid class in the model.")
            tissue_loc = list(self.model.classes_).index(tissue_name)
            print(f"Visualizing force plot for tissue: {tissue_name}")
            # Visualize force plot for the specified tissue
            display(shap.force_plot(self.explainer.expected_value[1], np.round(shap_values[tissue_loc], 5), sample, matplotlib=True))
        else:
            # If no tissue_name is provided, visualize for top n predicted tissues
            print("No specific tissue provided, visualizing force plots for top predicted tissues:")
            for tissue, _ in predictions:
                tissue_loc = list(self.model.classes_).index(tissue)
                print(f"Tissue: {tissue}")

                i += 1
                # Display force plot for each top tissue
                display(shap.force_plot(self.explainer.expected_value[1], np.round(shap_values[tissue_loc], 5), sample, matplotlib=True))
                if i == n_preds:
                    break
        
    def visualize_radar_chart(self, sample=None):
        if sample is None:
            sample = self.sample
        shap_df = self.adjusted_absent_shap_values_df(n_preds=100, penalty_factor=0.5)
        predictions = shap_df.sum(axis=1).sort_values(ascending=False)
        prediction_df = pd.DataFrame(predictions)
        prediction_df.reset_index(inplace=True)
        prediction_df.columns = ['tissue', 'prob']
        # if prob negative, set to 0
        prediction_df.loc[prediction_df['prob'] < 0, 'prob'] = 0
        prediction_df['prob'] = prediction_df['prob'] *100
        prediction_df = prediction_df.sort_values(by='tissue')
        fig = px.line_polar(prediction_df, r='prob', theta='tissue', line_close=True)
        fig.show()

    def calculate_NSAF(self, df, lengths):
        """Calculate NSAF scores for proteins"""
        df['count'] = df['count'].astype(float)
        df['Length'] = df['Length'].astype(float)
        df['SAF'] = df['count'] / df['Length']
        total_SAF = df['SAF'].sum()
        df['NSAF'] = df['SAF'] / total_SAF
        return df

    def visualize_shap_waterfall(self, shap_values, sample, tissue_idx=None, n_preds=5, tissue_name=None):
        """
        Visualizes SHAP waterfall plots for top predicted tissues or for a specific tissue.
        """

        shap_values = shap_values.astype(float)  # Convert shap_values to float type
        shap_values = np.round(shap_values, 5)  # Round shap_values to 5 decimal places
        predictions = self.predict_top_tissues(n_preds=n_preds)

        # If tissue_name is provided, check if it's valid
        if tissue_name:
            if tissue_name not in self.model.classes_:
                raise ValueError(f"Tissue '{tissue_name}' is not a valid class in the model.")
            tissue_loc = list(self.model.classes_).index(tissue_name)
            print(f"Visualizing waterfall plot for tissue: {tissue_name}")
            # Visualize waterfall plot for the specified tissue
            shap.plots.waterfall(
                shap.Explanation(values=shap_values[tissue_loc], 
                                base_values=self.explainer.expected_value[1], 
                                data=sample, 
                                feature_names=self.features),
                max_display=15,
                show=True
            )
            plt.show()
        else:
            # If no tissue_name is provided, visualize for top n predicted tissues
            print("No specific tissue provided, visualizing waterfall plots for top predicted tissues:")
            for tissue, _ in predictions:
                tissue_loc = list(self.model.classes_).index(tissue)
                print(f"Tissue: {tissue}")

                # Display waterfall plot for each top tissue
                shap.plots.waterfall(
                    shap.Explanation(values=shap_values[tissue_loc], 
                                    base_values=self.explainer.expected_value[1], 
                                    data=sample, 
                                    feature_names=self.features),
                    max_display=15,
                    show=True
                )
                plt.show()

#some independent functions for visualisation purposes
import plotly.graph_objects as go
import bioservices
from gprofiler import GProfiler

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

import plotly.graph_objects as go

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
    fig.show()

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