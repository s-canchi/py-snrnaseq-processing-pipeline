"""
Author: Saranya Canchi
Date Created: 2025-04
Description:
- Prepares AnnData objects for batch-aware variational autoencoder modeling using scVI.
- Tunes the number of training epochs based on dataset size for efficient and robust convergence.
- Trains the scVI model with checkpointing and early stopping enabled for reproducible integration.
- Extracts integrated latent representations and updates the AnnData object for downstream analysis.
"""

# Load libraries
import scvi
import anndata as ad
import sys
import os
import datetime
from scvi.model import SCVI
from scvi.train import SaveCheckpoint
from scvi.train import LoudEarlyStopping

# Set base dir
base_dir = "/path/to/folder/"

# Construct paths using base dir
dataset_path = os.path.join(base_dir, "preintegrate_umap.h5ad")
checkpoints_dir = os.path.join(base_dir, "scvi-chkpt")
output_path = os.path.join(base_dir, "scvi_integrated_5xep.h5ad")

# Read data
adata = ad.read_h5ad(dataset_path)

# Setup AnnData for scVI
scvi.model.SCVI.setup_anndata(adata, layer='raw_counts', batch_key='sample_prefix', continuous_covariate_keys=['total_counts', 'n_genes_by_counts'])

n_obs = adata.n_obs  
epochs_cap = 1000   

# Use get_max_epochs_heuristic to determine the actual number of epochs
max_epochs = scvi.model.get_max_epochs_heuristic(n_obs, epochs_cap=epochs_cap)
multiplier = 5  
effective_epochs = max_epochs * multiplier

print(f"Calculated maximum epochs: {max_epochs} for {n_obs} observations.")
print(f"Calculated effective epochs: {effective_epochs} for {n_obs} observations.")

# Initialize and train the model 
model = scvi.model.SCVI(adata)  

# Train the model with both callbacks
model.train(
    max_epochs=effective_epochs,
    batch_size=512,
    datasplitter_kwargs={
        'pin_memory': True,  
        'num_workers': 12    
    },
    enable_checkpointing=True,
    early_stopping=True
)

# Save model
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model_dir = os.path.join(base_dir, f"scvi-model-{timestamp}")
model.save(model_dir)
latent = model.get_latent_representation()
adata.obsm["X_scVI"] = latent

# Save the updated AnnData object
adata.write(output_path)