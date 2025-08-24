"""
Author: Saranya Canchi
Date Created: 2025-04
Description:
- Merges batch-level filtered AnnData objects into a unified dataset for downstream analysis.
- Computes highly variable genes and performs principal component analysis (PCA) on the merged data.
- Calculates and saves UMAP embeddings and cell metadata for integrated visualization.
- Generates comprehensive reports of key gene expression and PCA metrics.
- Utilizes memory profiling to monitor and optimize resource usage throughout data integration and plotting.
"""

# Load libraries
import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import tempfile
from memory_profiler import profile

@profile
def merge_and_perform_pca(base_dir, output_file):
    # Check if PCA output file already exists
    if os.path.exists(output_file):
        print(f"{output_file} already exists. Loading it...")
        return sc.read_h5ad(output_file)

    # List to hold each batch's AnnData
    adata_list = []

    for batch_dir in os.listdir(base_dir):
        batch_path = os.path.join(base_dir, batch_dir)
        if not os.path.isdir(batch_path):
            continue

        for file in os.listdir(batch_path):
            if file.endswith("_filtered.h5ad"):
                file_path = os.path.join(batch_path, file)
                print(f"Reading {file_path} for merging...")
                adata = sc.read_h5ad(file_path)
                adata_list.append(adata)

    # Concatenate all the AnnData objects into one
    concatenated_adata = ad.concat(adata_list, join='outer', label='batch', keys=[f'Batch_{i+1}' for i in range(len(adata_list))])

    # Store raw counts into a separate layer
    concatenated_adata.layers["raw_counts"] = concatenated_adata.X.copy()

    # Compute highly variable genes and perform PCA
    sc.pp.highly_variable_genes(concatenated_adata, flavor='seurat_v3', n_top_genes=2000)
    sc.tl.pca(concatenated_adata)
    print(adata)

    # Save the AnnData object with PCA results
    concatenated_adata.write(output_file)
    print(f"Saved merged AnnData object with PCA results to {output_file}")
    
    return concatenated_adata

@profile
def calculate_and_save_umap(adata, umap_file, obs_file, output_file):
    sc.pp.neighbors(adata, n_neighbors=30, use_rep='X_pca')
    sc.tl.umap(adata)

    pd.DataFrame(adata.obsm['X_umap'], index=adata.obs_names).to_csv(umap_file)
    adata.obs.to_csv(obs_file)
    print(f"Saved UMAP coordinates to {umap_file}")
    print(f"Saved obs data to {obs_file}")

    # Save updated AnnData object with UMAP 
    adata.write(output_file)
    print(f"Updated AnnData object with UMAP results saved to {output_file}")

@profile
def plot_additional_metrics(adata, output_pdf, temp_dir):
    additional_plots = [
        (sc.pl.highest_expr_genes, "Top Expressed Genes", {"n_top": 35}),
        (sc.pl.highly_variable_genes, "Highly Variable Genes", {}),
        (sc.pl.pca_loadings, "PC Loadings", {"components": [1, 2]}),
        (sc.pl.pca_variance_ratio, "PCA Variance Ratio", {})
    ]

    custom_width, custom_height = 14 * 72, 10 * 72
    c = canvas.Canvas(output_pdf, pagesize=(custom_width, custom_height))

    for plot_func, plot_title, plot_params in additional_plots:
        try:
            plot_func(adata, show=False, **plot_params)
            plt.title(plot_title, pad=20)

            tmp_file = os.path.join(temp_dir, f"{plot_title.replace(' ', '_')}.png")
            plt.savefig(tmp_file, dpi=300, bbox_inches='tight')
            plt.close()

            c.drawImage(tmp_file, 0, 0, width=custom_width, height=custom_height)
            c.showPage()
        except Exception as e:
            print(f"Error during additional plotting for {plot_title}: {e}")
        gc.collect()

    c.save()
    print(f"Additional plots saved into {output_pdf}")

# Base dir containing batch sub-dir
base_dir = "/path/to/folder/"
plot_dir = os.path.join(base_dir, "merged_batch")
output_file = os.path.join(plot_dir, "preintegrate_umap.h5ad")
umap_file = os.path.join(plot_dir, "preintegrate_umap_coords.csv")
obs_file = os.path.join(plot_dir, "preintegrate_obs_data.csv")
pca_plot = os.path.join(plot_dir, "preintegrate_pca_plots.pdf")

# Run merge and PCA if necessary
adata = merge_and_perform_pca(base_dir, output_file)
calculate_and_save_umap(adata, umap_file, obs_file, output_file)
plot_additional_metrics(adata, pca_plot, tempfile.gettempdir())