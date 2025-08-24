"""
Author: Saranya Canchi
Date Created: 2025-04
Description: 
- Maps Ensembl gene identifiers to gene symbols for each sample.
- Aggregates and constructs AnnData objects from CellBender-filtered count matrices.
- Computes and saves key quality control (QC) metrics per sample.
- Generates integrated QC visualizations for all samples in a batch.
- Concatenates samples, performs PCA, and exports metrics and plots.
"""

# Load libraries
import scanpy as sc
import h5py
import pandas as pd
import numpy as np
import os
import scipy.sparse as sp
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import gzip
import gc
import sys
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

# Map Ensembl IDs to gene names
def map_ensembl_to_gene_names(marker_file):
    print("Mapping Ensembl IDs to gene names...")
    with gzip.open(marker_file, 'rt') as f:
        features_df = pd.read_csv(f, sep='\t', header=None)
    features_df.columns = ["ensembl_id", "gene_name", "feature_type"]
    return {row["ensembl_id"]: row["gene_name"] for _, row in features_df.iterrows()}

# Aggregate and create AnnData object
def aggregate_and_create_anndata(counts_data, gene_mapping, sample_prefix):
    print("Processing an HDF5 sample...")

    # Extract data from the h5 file
    counts = counts_data["matrix/data"][:]
    indices = counts_data["matrix/indices"][:]
    indptr = counts_data["matrix/indptr"][:]
    shape = counts_data["matrix/shape"][:]
    ensembl_ids = counts_data["matrix/features/id"][:]
    barcodes = counts_data["matrix/barcodes"][:]

    # Construct a sparse matrix
    sparse_matrix = sp.csc_matrix((counts, indices, indptr), shape=shape)
    barcodes = [f"{sample_prefix}_{bc.decode('utf-8')}" if isinstance(bc, bytes) else f"{sample_prefix}_{bc}" for bc in barcodes]

    # Create an AnnData object with transpose, assume ens_ids and barcodes are already strings
    adata = sc.AnnData(X=sparse_matrix.transpose(), var=pd.DataFrame(index=[ens_id.decode('utf-8') if isinstance(ens_id, bytes) else ens_id for ens_id in ensembl_ids]), obs=pd.DataFrame(index=[bc.decode('utf-8') if isinstance(bc, bytes) else bc for bc in barcodes]))

    # Map Ensembl IDs to gene names
    gene_names = [gene_mapping.get(ens_id, ens_id) for ens_id in adata.var_names]
    adata.var_names = gene_names

    # Remove genes with unwanted patterns
    print("Filtering unwanted gene patterns...")
    unwanted_patterns = ["^Gm", "^ENSM", "Rik$"]
    adata = adata[:, ~adata.var_names.str.contains("|".join(unwanted_patterns))]

    # Aggregate duplicated gene entries
    print("Aggregating duplicated gene entries...")
    unique_gene_names, index = np.unique(adata.var_names, return_index=True)
    aggregated_matrix = np.zeros((adata.n_obs, len(unique_gene_names)))

    for i, gene in enumerate(unique_gene_names):
        gene_indices = np.where(adata.var_names == gene)[0]
        aggregated_matrix[:, i] = adata[:, gene_indices].X.toarray().sum(axis=1)

    # Create the aggregated AnnData object
    adata_aggregated = sc.AnnData(X=sp.csr_matrix(aggregated_matrix), var=pd.DataFrame(index=unique_gene_names), obs=adata.obs.copy())

    return adata_aggregated

# Calculate QC metrics 
def calculate_qc_metrics(adata, sample_prefix, data_source, output_dir):
    print("Calculating QC metrics...")
    adata.var["mt"] = adata.var_names.str.startswith("mt-")
    rpS_genes = adata.var_names.str.startswith("Rps")
    rpL_genes = adata.var_names.str.startswith("Rpl")
    ps_genes = adata.var_names.str.contains("-ps")
    
    rpS_genes = np.setdiff1d(adata.var_names[rpS_genes], adata.var_names[ps_genes])
    rpL_genes = np.setdiff1d(adata.var_names[rpL_genes], adata.var_names[ps_genes])

    adata.obs['pct_counts_mt'] = np.sum(adata[:, adata.var['mt']].X.toarray(), axis=1) / np.sum(adata.X.toarray(), axis=1) * 100
    adata.obs['pct_counts_rpS'] = np.sum(adata[:, rpS_genes].X.toarray(), axis=1) / np.sum(adata.X.toarray(), axis=1) * 100
    adata.obs['pct_counts_rpL'] = np.sum(adata[:, rpL_genes].X.toarray(), axis=1) / np.sum(adata.X.toarray(), axis=1) * 100
    adata.obs['pct_counts_ps'] = np.sum(adata[:, ps_genes].X.toarray(), axis=1) / np.sum(adata.X.toarray(), axis=1) * 100

    sc.pp.calculate_qc_metrics(adata, inplace=True, log1p=False)
    print("QC Metrics:")
    print(adata.obs.head())

    # Save metrics 
    qc_report_path = os.path.join(output_dir, f"{sample_prefix}_{data_source}_qc_metrics.csv")
    adata.obs.to_csv(qc_report_path)
    print(f"Saved QC metrics to {qc_report_path}")

# Generate integrated density and scatter plots
def generate_integrated_plot(adata_list, batch_name, plot_dir, data_source):
    print(f"Generating integrated plots for {data_source} batch: {batch_name}...")
    metrics = ['total_counts', 'n_genes_by_counts', 'pct_counts_mt', 
               'pct_counts_rpS', 'pct_counts_rpL', 'pct_counts_ps']

    # Create a subplot figure with one row per metric, plus one row for the scatter plot
    fig = make_subplots(
        rows=len(metrics) + 1, cols=1,
        subplot_titles=[f"{metric}" for metric in metrics] + ["Counts vs Genes"]
    )

    colors = px.colors.qualitative.Plotly

    # Add plots for each adata
    for idx, adata in enumerate(adata_list):
        color = colors[idx % len(colors)]
        sample_name = adata.obs['sample_prefix'].unique()[0]
        
        # Add histograms for each metric
        for i, metric in enumerate(metrics):
            fig.add_trace(
                go.Histogram(
                    x=adata.obs[metric], histnorm='density', 
                    name=sample_name if i == 0 else None,  
                    marker_color=color, opacity=0.7,
                    showlegend=(i == 0)
                ),
                row=i + 1, col=1
            )

        # Add scatter plot for the last row
        fig.add_trace(
            go.Scatter(
                x=adata.obs['total_counts'], y=adata.obs['n_genes_by_counts'], 
                mode='markers', marker=dict(color=color, size=5),
                opacity=0.7,
                name=None,         
                showlegend=False
            ),
            row=len(metrics) + 1, col=1
        )
    
    fig.update_layout(
        height=400 * (len(metrics) + 1), width=1000,
        title_text=f"{batch_name} CellBender QC Metrics",
        showlegend=True,
        plot_bgcolor='white'
    )

    fig.update_xaxes(
    mirror=True,
    ticks='outside',
    showline=True,
    linecolor='black',
    gridcolor='lightgrey'
    )

    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )
    
    # Save the figure as an HTML file
    html_file = os.path.join(plot_dir, f"{batch_name}_{data_source}_qc_report.html")
    pio.write_html(fig, file=html_file)
    print(f"Saved QC report for {data_source} for batch {batch_name}: {html_file}")

# Perform PCA and generate plots
def perform_pca_and_save(adata, batch_output_dir, batch_name):
    adata.raw = adata.copy()  

    # PCA workflow without subsetting
    sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=2000)
    sc.tl.pca(adata)

    # Save the AnnData object with PCA results
    adata_file = os.path.join(batch_output_dir, f"{batch_name}_pca.h5ad")
    adata.write(adata_file)
    print(f"Saved AnnData object with PCA results to {adata_file}")

    # Define the method and description for plots
    plot_methods = [
        (sc.pl.highest_expr_genes, "Highest Expr Genes"),
        (sc.pl.highly_variable_genes, "Highly Variable Genes"),
        (sc.pl.pca_overview, "PCA Overview"),
        (sc.pl.pca_loadings, "PCA Loadings"),
        (sc.pl.pca, "PCA Dimensions"),
        (sc.pl.pca_variance_ratio, "PCA Variance Ratio")
    ]

    # Create output file
    output_pdf = os.path.join(batch_output_dir, f"{batch_name}_pca_plots.pdf")
    c = canvas.Canvas(output_pdf, pagesize=letter)
    width, height = letter

    # Loop through plotting methods
    for plot_func, title in plot_methods:
        plt.figure()  
        plot_func(adata, show=False)  
        plt.title(title) 

        # Save the figure into a temporary file
        tmp_file = os.path.join(batch_output_dir, f"{title}.png")
        plt.savefig(tmp_file)
        plt.close()

        # Add image
        c.drawImage(tmp_file, 0, 0, width=width, height=height)
        c.showPage()  #

        # Remove the temporary image file
        os.remove(tmp_file)

    c.save()
    print(f"Concatenated PCA plots into {output_pdf}")

# Loop over all batches
def main():
    cellbender_input_dir = "/path/to/folder/"
    plot_dir = "/path/to/folder/"
    marker_file = "/path/to/folder/"
    os.makedirs(plot_dir, exist_ok=True)

    batch_map = pd.read_csv("/path/to/folder/")
    gene_mapping = map_ensembl_to_gene_names(marker_file)
    data_source = "cb"  

    if len(sys.argv) < 2:
        print("Usage: python process_batch.py <batch_name>")
        sys.exit(1)

    batch_name = sys.argv[1]
    print(f"Processing batch {batch_name}")

    # Create a batch-specific output directory
    batch_output_dir = os.path.join(plot_dir, batch_name)
    if not os.path.exists(batch_output_dir):
        os.makedirs(batch_output_dir, exist_ok=True)

    # Check for output files
    expected_html = os.path.join(batch_output_dir, f"{batch_name}_{data_source}_qc_report.html")
    if os.path.exists(expected_html):
        print(f"Output files for batch {batch_name} already exist. Skipping...")
        return

    batch_info = batch_map[batch_map['batch'] == batch_name]
    cb_adata_list = []

    for _, row in batch_info.iterrows():
        sample_id = row['sampleID']
        sample_prefix = row['libraryID']
        print(f"Processing CellBender sample: {sample_prefix}...")
        cellbender_path = os.path.join(cellbender_input_dir, sample_id, f"{sample_id}_cellbender_filtered.h5")

        if os.path.exists(cellbender_path):
            with h5py.File(cellbender_path, 'r') as counts_data:
                adata = aggregate_and_create_anndata(counts_data, gene_mapping, sample_prefix)
            
            adata.obs['batch'] = batch_name
            adata.obs['sample_prefix'] = sample_prefix
            calculate_qc_metrics(adata, sample_prefix, data_source, batch_output_dir)
            cb_adata_list.append(adata)

    if cb_adata_list:
        generate_integrated_plot(cb_adata_list, batch_name, batch_output_dir, data_source)
        concatenated_adata = cb_adata_list[0].concatenate(*cb_adata_list[1:], batch_key='sample_prefix')
        perform_pca_and_save(concatenated_adata, batch_output_dir, batch_name)
        del cb_adata_list
        gc.collect()

    print(f"Batch {batch_name} processing completed.")

if __name__ == "__main__":
    main()




