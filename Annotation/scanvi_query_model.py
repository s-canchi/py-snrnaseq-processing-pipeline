"""
Author: Saranya Canchi
Date Created: 2025-04
Description:
- Loads a pretrained scANVI reference model and prepares a query snRNA-seq dataset for integration and annotation.
- Adapts and trains the scANVI model on the query data, extracting integrated latent representations and predicting cell labels.
- Exports the annotated and integrated AnnData object along with cell-level metadata for downstream analysis.
"""

# Load libraries
import scvi
import anndata as ad
import pandas as pd
import sys
import os
import datetime
from scvi.model import SCVI

# Base dir and file paths
query_dir = "/path/to/folder/"
query_data_path = os.path.join(query_dir, "query_data.h5ad")
query_output_path = os.path.join(query_dir, "integrated_scanvi.h5ad")
query_umap_file = os.path.join(query_dir, "integrated_scanvi_umap.csv")
query_obs_file = os.path.join(query_dir, "integrated_scanvi_obs.csv")
ref_dir = "/path/to/folder/"
SCANVI_LATENT_KEY = "X_scANVI"
SCANVI_PREDICTIONS_KEY = "predictions_scanvi"

# Load query AnnData
query_adata = ad.read_h5ad(query_data_path)
if 'chemistry' not in query_adata.obs.columns:
    query_adata.obs['chemistry'] = "10Xv3"

# Locate the latest SCANVI model directory
scanvi_model_dirs = [d for d in os.listdir(ref_dir) if os.path.isdir(os.path.join(ref_dir, d)) and d.startswith("scanvi-model")]

if not scanvi_model_dirs:
    raise Exception("No SCANVI model directories found.")

latest_model_dir = max(
    (os.path.join(ref_dir, d) for d in scanvi_model_dirs),
    key=os.path.getctime
)

print(f"Loading SCANVI model from {latest_model_dir}...")

# Prepare the query AnnData using the reference model setup
scvi.model.SCANVI.prepare_query_anndata(query_adata, latest_model_dir)

# Load the SCANVI model for predicting query data
scanvi_query = scvi.model.SCANVI.load_query_data(query_adata, latest_model_dir)

# Train SCANVI on the query data 
scanvi_query.train(
    max_epochs=100,
    plan_kwargs={"weight_decay": 0.0},
    check_val_every_n_epoch=10
)

# Extract the latent representations and predict labels for the query dataset
query_adata.obsm[SCANVI_LATENT_KEY] = scanvi_query.get_latent_representation()
query_adata.obs[SCANVI_PREDICTIONS_KEY] = scanvi_query.predict()

# Save the annotated query AnnData
query_adata.write_h5ad(query_output_path)

# Save observation metadata to CSV
umap_coords_df = pd.DataFrame(query_adata.obsm.get("X_umap", []), index=query_adata.obs.index, columns=['UMAP1', 'UMAP2'])
obs_df = query_adata.obs.copy()

umap_coords_df.to_csv(query_umap_file)
obs_df.to_csv(query_obs_file)

print("Query dataset annotated, and metadata saved.")
