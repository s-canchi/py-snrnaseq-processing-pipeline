"""
Author: Saranya Canchi
Date Created: 2025-04
Description:
- Merges integrated cell annotations, observation metadata, and UMAP coordinates.
- Performs stratified sampling to ensure balanced representation of all classes.
- Generates interactive UMAP visualizations and an accompanying summary table of class counts for data exploration and reporting. 
"""

# Load libraries
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import io

pd.set_option('display.max_columns', None)

# Read a csv file and skip comments
def read_csv_skip_comments(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    # Filter out comment lines starting with #
    filtered_lines = [line for line in lines if not line.lstrip().startswith('#')]
    filtered_content = "\n".join(filtered_lines)
    
    return pd.read_csv(io.StringIO(filtered_content))

# Perform stratified sampling to ensure all classes are represented
def stratified_sampling(df, class_column, frac=0.25, min_samples=1):
    # Group by class and apply sampling
    sampled_df = df.groupby(class_column, group_keys=False).apply(
        lambda x: x.sample(max(min_samples, int(len(x) * frac)), random_state=42)
    ).reset_index(drop=True)

    # Print the number of classes
    num_classes = sampled_df[class_column].nunique()
    print(f"Number of unique classes in the sampled data: {num_classes}")

    return sampled_df

# Plot annotated UMAP and add table with counts
def plot_umap_and_table(data_combined, class_counts, metric, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Generating UMAP plot for {metric}...")

    unique_values = data_combined[metric].unique()
    custom_colors = [
        '#aaffc3', '#808000', '#eda258', '#48dbb2', '#fabebe',
        '#911eb4', '#dbd33d', '#f032e6', '#f58231', '#000075',
        '#008080', '#e6beff', '#9a6324', '#46f0f0', '#800000',
        '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#bcf60c'
    ]
    color_map = {val: custom_colors[i % len(custom_colors)] for i, val in enumerate(unique_values)}
    opacity = 0.3

    # Define the subplot figure with side-by-side layout
    fig = make_subplots(
    rows=2, cols=1,  
    row_heights=[0.5, 0.5],
    specs=[[{"type": "scatter"}], [{"type": "table"}]],  # Specify plot types
    vertical_spacing=0.1,
    )

    # Add UMAP scatter with legend
    for class_value in sampled_df['class_name'].unique():
        class_data = sampled_df[sampled_df['class_name'] == class_value]
        scatter = go.Scatter(
            x=class_data['UMAP1'], y=class_data['UMAP2'],
            mode='markers',
            marker=dict(color=color_map[class_value], size=4, opacity=opacity),
            name=class_value,
            showlegend=False
        )
        fig.add_trace(scatter, row=1, col=1)

    # Add table with styled class names
    fig.add_trace(
        go.Table(
            header=dict(
                values=["Class", "Count"],
                fill_color='white',
                line_color='white',
                align='left'
            ),
            cells=dict(
                values=[
                    [
                        f'<span style="color:{color_map.get(class_name, "#000")}">{class_name}</span>'
                        for class_name in class_counts['class_name']
                    ],
                    class_counts['count']
                ],
                fill_color='white',
                line_color='white',
                align='left',
                font=dict(size=14)
            )
        ),
        row=2, col=1
    )

    # Update layout
    fig.update_layout(
        template='plotly_white',
        height=2000,  # Total figure height
        width=1200,   # Total figure width
        title={'text': "UMAP Plot by class with counts", 'x': 0.5},
        margin=dict(l=20, r=20, t=40, b=20)
    )

    fig.update_xaxes(
        title_text='UMAP 1',
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey',
        scaleanchor="x",  
        scaleratio=1.0,
        row=1, col=1
    )
    fig.update_yaxes(
        title_text='UMAP 2',
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey',
        row=1, col=1
    )

    html_output_file = os.path.join(output_dir, f"umap_{metric}.html")
    fig.write_html(html_output_file)
    print(f"Saved UMAP plot with class counts to {html_output_file}")


# Input paths
base_dir = "/path/to/folder/"
annotation_file = os.path.join(base_dir, "wmb_precompute_mapping_output.csv") 
obs_file = os.path.join(base_dir, "integrated_scvi_obs.csv")
umap_coords_file = os.path.join(base_dir, "integrated_scvi_umap.csv")

# Load the data
annotations_df = read_csv_skip_comments(annotation_file)
obs_df = pd.read_csv(obs_file, index_col=0)
umap_coords_df = pd.read_csv(umap_coords_file, index_col=0)

# Merge annotation with obs based on cell_id
combined_df = obs_df.join(annotations_df.set_index('cell_id'), how='inner').join(umap_coords_df)
sampled_df = stratified_sampling(combined_df, 'class_name', frac=0.5)
print(combined_df.columns)
print(combined_df.head())

# Count cells per class_name
class_counts = combined_df['class_name'].value_counts().reset_index()
class_counts.columns = ['class_name', 'count']
print(class_counts.head())

plot_umap_and_table(sampled_df, class_counts, 'class_name', base_dir)