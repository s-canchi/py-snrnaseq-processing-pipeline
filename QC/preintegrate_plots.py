"""
Author: Saranya Canchi
Date Created: 2025-04
Description:
- Generates PCA-based visualizations and summary plots for integrated snRNA-seq data.
- Produces interactive UMAP plots colored by key QC metrics, batch, and sample-level variables.
- Supports flexible visualization of UMAP embeddings by experimental metadata for comprehensive exploratory analysis.
- Organizes and exports interactive reports to facilitate downstream data interpretation and sharing.
"""

# Load libraries
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import plotly.subplots as sp
import os
import scanpy as sc
import gc
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import tempfile

pd.set_option('display.max_columns', None)

def plot_additional_metrics(pca_loadings_file, pca_variance_ratio_file, hvg_file, output_html):
    # Read data 
    pca_loadings = pd.read_csv(pca_loadings_file, index_col=0)  
    pca_variance_ratio = pd.read_csv(pca_variance_ratio_file, index_col=0) 
    hvg_data = pd.read_csv(hvg_file, index_col=0)  

    # Create a subplot figure
    fig = sp.make_subplots(
        rows=3, cols=1,
        subplot_titles=("PCA Loadings (PC1 vs PC2 and PC3 vs PC4)", "PCA Variance Ratio", "Highly Variable Genes"),
        vertical_spacing=0.1
    )

    # Add PCA Loadings for PC1 vs PC2
    fig.add_trace(
        go.Scatter(x=pca_loadings['PC1'], y=pca_loadings['PC2'], mode='markers', name='PC1 vs PC2'),
        row=1, col=1
    )

    # Add PCA Loadings for PC3 vs PC4
    fig.add_trace(
        go.Scatter(x=pca_loadings['PC3'], y=pca_loadings['PC4'], mode='markers', name='PC3 vs PC4'),
        row=1, col=1
    )

    # Add PCA variance ratio
    fig.add_trace(
        go.Bar(x=pca_variance_ratio.index, y=pca_variance_ratio['Variance_Ratio']),
        row=2, col=1
    )

    # Add hvgs
    true_hvg_data = hvg_data[hvg_data['highly_variable']]  
    fig.add_trace(
        go.Bar(x=true_hvg_data.index, y=true_hvg_data['highly_variable'].astype(int)),  
        row=3, col=1
    )

    # Update layout
    fig.update_layout(
        title_text="Additional Metrics Overview",
        height=900,
        width=800,
        showlegend=False,
        template='plotly_white'
    )

    # Save plot 
    pio.write_html(fig, file=output_html, auto_open=False, include_plotlyjs='cdn')
    print(f"Combined HTML report saved to {output_html}")

def resolve_column_names(obs):
    column_mapping = {
        'batch': ['batch'],
        'total_counts': ['total_counts', 'n_counts'],
        'n_genes_by_counts': ['n_genes_by_counts', 'n_genes']
    }
    
    resolved_names = {}
    for key, variations in column_mapping.items():
        # Look for the first matching column name in the df
        resolved_names[key] = next((col for col in variations if col in obs.columns), None)
    
    return resolved_names

def generate_umap_report_for_metric(umap_coords, obs, output_dir):
    # Resolve actual column names
    column_names = resolve_column_names(obs)
    continuous_metrics = [column_names['total_counts'], column_names['n_genes_by_counts']]

    # Load UMAP coords and obs data into df
    umap_df = pd.DataFrame(umap_coords, index=obs.index, columns=['UMAP1', 'UMAP2'])
    data_combined = obs.join(umap_df)
    print(data_combined.head())

    for metric in continuous_metrics:
        print(f"Generating UMAP plot for {metric}...")
        if metric in obs.columns:
            fig = px.scatter(
                data_combined, x='UMAP1', y='UMAP2',
                color=metric,
                color_continuous_scale='Viridis',
                title=f"UMAP by {metric}",
                template='plotly_white',
                labels={'color': metric}
            )
            fig.update_traces(marker=dict(size=4, opacity=0.7))

            # Update layout
            fig.update_layout(
                autosize=True,
                width=None,
                height=None,
                plot_bgcolor='white',
                showlegend=True
            )
            fig.update_xaxes(
                title_text='UMAP 1',
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor='black',
                gridcolor='lightgrey'
            )
            fig.update_yaxes(
                title_text='UMAP 2',
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor='black',
                gridcolor='lightgrey'
            )

            # Save each metric as a separate file
            html_output_file = os.path.join(output_dir, f"umap_{metric}.html")
            pio.write_html(fig, file=html_output_file)
            print(f"Saved UMAP report for {metric} to {html_output_file}")

def plot_umap_all_batches(umap_coords, obs, html_output_file):
    column_names = resolve_column_names(obs)

    umap_df = pd.DataFrame(umap_coords, index=obs.index, columns=['UMAP1', 'UMAP2'])
    data_combined = obs.join(umap_df)
    print(data_combined.head())
    unique_batches = data_combined[column_names['batch']].unique()
    print(f"Unique batches in data: {unique_batches}")

    # Assign colors to batches
    custom_colors = [
        '#aaffc3', '#808000', '#eda258', '#48dbb2', '#fabebe',
        '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#000075',
        '#008080', '#e6beff', '#9a6324', '#dbd33d', '#800000', 
        '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231'
    ]
    batch_colors = {batch: custom_colors[i % len(custom_colors)] for i, batch in enumerate(unique_batches)}

    # Generate batch UMAP plot separately
    batch_metric = column_names['batch']
    if batch_metric in obs.columns:
        batch_scatter = px.scatter(
            data_combined, x='UMAP1', y='UMAP2',
            color=batch_metric, title=f"UMAP by Batch",
            template='plotly_white',
            labels={'color': 'Batch'},
            opacity=0.3,
            color_discrete_map=batch_colors
        )
        batch_scatter.update_traces(marker=dict(size=2, opacity=0.3), showlegend=True)  
        for batch, color in batch_colors.items():
            batch_scatter.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color=color),
                showlegend=True,
                name=batch
            ))

        batch_scatter.update_layout(
            legend=dict(
                title="Batch",
                x=1, y=1,
                orientation='v',
            )
        )
        batch_scatter.update_xaxes(
            title_text='UMAP 1',
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='black',
            gridcolor='lightgrey'
        )
        batch_scatter.update_yaxes(
            title_text='UMAP 2',
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='black',
            gridcolor='lightgrey'
        )
        pio.write_html(batch_scatter, file=html_output_file)
        print(f"Saved UMAP batch report to {html_output_file}")

def generate_umap_by_batch_report(umap_coords, obs, output_dir):
    unique_batches = obs['batch'].unique()
    umap_df = pd.DataFrame(umap_coords, index=obs.index, columns=['UMAP1', 'UMAP2'])

    for batch in unique_batches:
        batch_mask = obs['batch'] == batch
        batch_obs = obs.loc[batch_mask]
        print(f"Generating UMAP plot for batch {batch}...")
        print(f"Batch data head for {batch}:\n{batch_obs.head()}")

        # Create color mapping
        custom_colors = [
        '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
        '#911eb4', '#46f0f0', '#bcf60c', '#dbd33d', '#e6beff' 
        ]
        sample_prefixes_in_batch = batch_obs['sample_prefix'].unique()
        sample_colors = {sample: custom_colors[i % len(custom_colors)] for i, sample in enumerate(sample_prefixes_in_batch)}
        neutral_gray = '#d3d3d3'
        umap_df['plot_color'] = umap_df.index.map(lambda idx: sample_colors.get(obs.loc[idx, 'sample_prefix'], neutral_gray))
        umap_df['plot_label'] = umap_df.index.map(lambda idx: obs.loc[idx, 'sample_prefix'] if idx in batch_obs.index else "Background")
        main_batch_df = umap_df.loc[batch_obs.index]
        background_df = umap_df.loc[~umap_df.index.isin(batch_obs.index)]

        fig_batch = px.scatter(
            background_df, x='UMAP1', y='UMAP2',
            color='plot_label',
            title=f"UMAP by Sample Prefix for {batch}",
            labels={'color': 'Sample Prefix'},
            color_discrete_map={"Background": neutral_gray},
            opacity=0.3  
        )

        # Add the main batch data on top
        main_scatter = px.scatter(
            main_batch_df, x='UMAP1', y='UMAP2',
            color='plot_label',
            color_discrete_map=sample_colors,
            opacity=0.7  
        )

        # Add traces from the main scatter plot
        for trace in main_scatter.update_traces(marker=dict(size=4), showlegend=True).data:
            fig_batch.add_trace(trace)

        # Legend configuration
        fig_batch.update_layout(
            legend=dict(
                title="Sample Prefix",
                x=1, y=1,
                orientation='v',
            ),
            plot_bgcolor='white'  
        )

        fig_batch.update_xaxes(
            title_text='UMAP 1',
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='black',
            gridcolor='lightgrey'
        )
        fig_batch.update_yaxes(
            title_text='UMAP 2',
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='black',
            gridcolor='lightgrey'
        )

        html_output_file = os.path.join(output_dir, f"umap_{batch}.html")
        pio.write_html(fig_batch, file=html_output_file)
        print(f"Saved UMAP batch report for batch {batch} to {html_output_file}")

def generate_umap_by_experimental_variable(umap_coords, obs, metadata_file, output_dir):
    # Read the metadata file
    metadata = pd.read_csv(metadata_file, index_col=0).drop(columns=["...1", "...2"], errors='ignore')
    metadata.rename(columns={'Genotype.x': 'Genotype'}, inplace=True)
    metadata['sample_prefix'] = metadata['sampleID'].str.split('_').str[0]
    print(metadata.head())
    
    # Merge obs with metadata on sample_prefix 
    obs['cell_barcode'] = obs.index
    print(f"Cell barcode in obs:\n{obs['cell_barcode'].head()}")
    combined_df = obs.merge(metadata, left_on='sample_prefix', right_on='sample_prefix', how='left')
    print(combined_df.head())

    # Load UMAP coordinates into df
    umap_df = pd.DataFrame(umap_coords, index=obs['cell_barcode'], columns=['UMAP1', 'UMAP2'])
    data_combined = combined_df.set_index('cell_barcode').join(umap_df, how='inner')
    data_combined['AgeGrp'] = data_combined['AgeGrp'].astype(str)
    print(data_combined.head())

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Variables to plot
    variables = ['Sex', 'Genotype', 'AgeGrp', 'assigned_strain']

    for variable in variables:
        if variable not in data_combined.columns:
            print(f"Variable {variable} not found in combined data.")
            continue

        print(f"Generating UMAP plot for {variable}...")

        unique_values = data_combined[variable].nunique()
        # Check if the number of unique values exceeds the default color palette size
        if unique_values > 20:
            # Create a larger color map
            color_map = px.colors.qualitative.Light24 + px.colors.qualitative.Dark24
            color_palette = color_map[:unique_values]  
        else:
            color_palette = px.colors.qualitative.Plotly

        fig = px.scatter(
            data_combined, x='UMAP1', y='UMAP2',
            color=variable,
            title=f"UMAP by {variable}",
            template='plotly_white',
            labels={'color': variable},
            opacity=0.4,
            color_discrete_sequence=color_palette
        )

        fig.update_traces(marker=dict(size=4, opacity=0.5, line=dict(width=0.5, color='DarkSlateGrey')))

        # Update layout
        fig.update_layout(
            autosize=True,
            plot_bgcolor='white',
            showlegend=True
        )
        fig.update_xaxes(
            title_text='UMAP 1',
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='black',
            gridcolor='lightgrey'
        )
        fig.update_yaxes(
            title_text='UMAP 2',
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='black',
            gridcolor='lightgrey'
        )

        html_output_file = os.path.join(output_dir, f"umap_{variable}.html")
        pio.write_html(fig, file=html_output_file)
        print(f"Saved UMAP report for {variable} to {html_output_file}")

# Base dir containing batch files
base_dir = "/path/to/folder/"
plot_dir = os.path.join(base_dir, "merged_batch")
pca_loadings = os.path.join(plot_dir, "pca_loadings.csv")
pca_variance = os.path.join(plot_dir, "pca_variance_ratio.csv")
pca_hvg = os.path.join(plot_dir, "highly_variable_genes.csv")
umap_file = os.path.join(plot_dir, "preintegrate_umap_coords.csv")
obs_file = os.path.join(plot_dir, "preintegrate_obs_data.csv")
metadata_file = "/path/to/folder/"
pca_plot = os.path.join(plot_dir, "preintegrate_pca_plots.html")
batch_output = os.path.join(plot_dir, "preintegrate_umap_all_batches.html")

# Generate PCA-based plots
plot_additional_metrics(pca_loadings, pca_variance, pca_hvg, pca_plot)

# Read data
umap_coords = pd.read_csv(umap_file, index_col=0).values
obs = pd.read_csv(obs_file, index_col=0)

# Generate UMAP reports
generate_umap_report_for_metric(umap_coords, obs, plot_dir)
plot_umap_all_batches(umap_coords, obs, batch_output)
generate_umap_by_batch_report(umap_coords, obs, plot_dir)
generate_umap_by_experimental_variable(umap_coords, obs, metadata_file, plot_dir)