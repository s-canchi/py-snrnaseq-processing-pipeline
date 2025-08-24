"""
Author: Saranya Canchi
Date Created: 2025-04
Description:
- Initializes and/or loads a reference-trained scANVI model for semi-supervised integration and annotation of snRNA-seq data.
- Extracts and saves the integrated latent representation, computes clustering and UMAP embeddings in the scANVI space.
- Exports the updated AnnData object, UMAP coordinates, and annotated cell metadata for downstream use.
"""

# Load libarries
import scvi
import scanpy as sc
import anndata as ad
import pandas as pd
import sys
import os
import datetime
from scvi.model import SCVI

# Base dir and file paths
base_dir =  "/path/to/folder/"
adata_file = os.path.join(base_dir, "ref_scvi_integrated.h5ad")
output_adata_file = os.path.join(base_dir, "ref_scanvi.h5ad")
output_umap_file = os.path.join(base_dir, "ref_scanvi_umap.csv")
output_obs_file = os.path.join(base_dir, "ref_scanvi_obs.csv")
SCANVI_LABELS_KEY = "labels_scanvi"
SCANVI_LATENT_KEY = "X_scANVI"

# Load latest model dir
model_dirs = [
    d for d in os.listdir(base_dir)
    if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("scvi-model-")
]

if not model_dirs:
    raise FileNotFoundError("No scVI model directories found in {}".format(base_dir))

# Get the dir with the latest creation/modification time
latest_model_dir = max(
    (os.path.join(base_dir, d) for d in model_dirs),
    key=os.path.getmtime  
)
scvi_model = latest_model_dir

# Load reference AnnData
reference_adata = ad.read_h5ad(adata_file)

# Ensure the labels key is correctly set 
if SCANVI_LABELS_KEY not in reference_adata.obs.columns:
    print(f"Setting {SCANVI_LABELS_KEY} in AnnData.obs...")
    reference_adata.obs[SCANVI_LABELS_KEY] = reference_adata.obs["class"].values

# Check for existing model
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
scanvi_model_path = os.path.join(base_dir, f"scanvi-model-{timestamp}")

# Check if the SCANVI model dir exists
scanvi_model_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("scanvi-model")]

if scanvi_model_dirs:
    # Load the latest model based on dir creation time
    latest_model_dir = max(
        (os.path.join(base_dir, d) for d in scanvi_model_dirs),
        key=os.path.getmtime
    )
    print(f"Loading existing SCANVI model from {latest_model_dir}...")
    scanvi_ref = scvi.model.SCANVI.load(latest_model_dir, adata=reference_adata)
else:
    print("Initializing and training SCANVI model...")
    
    # Set the labels key in the AnnData object
    reference_adata.obs[SCANVI_LABELS_KEY] = reference_adata.obs["class"].values

    # Initialize SCANVI from the trained SCVI model
    scvi_model = scvi.model.SCVI.load("path_to_saved_scvi_model", adata=reference_adata)
    scanvi_ref = scvi.model.SCANVI.from_scvi_model(
        scvi_model,
        unlabeled_category="Unknown",
        labels_key=SCANVI_LABELS_KEY
    )

    # Find the smallest class size in the reference dataset
    min_class_size = reference_adata.obs[SCANVI_LABELS_KEY].value_counts().min()

    # Train the SCANVI model
    scanvi_ref.train(max_epochs=40, n_samples_per_label=min_class_size)

    # Save the SCANVI model as a directory
    scanvi_ref.save(scanvi_model_path)

# Extract latent representation
reference_adata.obsm[SCANVI_LATENT_KEY] = scanvi_ref.get_latent_representation()

# Perform neighbors, clustering, and UMAP
sc.pp.neighbors(reference_adata, use_rep=SCANVI_LATENT_KEY)
sc.tl.umap(reference_adata)
sc.tl.leiden(reference_adata, key_added="leiden_scANVI", resolution=0.5, flavor='igraph', n_iterations=2 )

# Save the updated reference AnnData
reference_adata.write_h5ad(output_adata_file)

# Save UMAP coordinates and observation metadata to CSV
umap_coords_df = pd.DataFrame(reference_adata.obsm['X_umap'], index=reference_adata.obs.index, columns=['UMAP1', 'UMAP2'])
obs_df = reference_adata.obs.copy()

umap_coords_df.to_csv(output_umap_file)
obs_df.to_csv(output_obs_file)

print("Reference dataset processed, UMAP coordinates and metadata saved.")