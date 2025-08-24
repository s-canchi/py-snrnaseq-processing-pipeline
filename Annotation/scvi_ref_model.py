"""
Author: Saranya Canchi
Date Created: 2025-04
Description:
- Sets up and trains a batch-aware variational autoencoder model (scVI) on reference snRNA-seq data.
- Uses early stopping and checkpointing for robust, reproducible model training.
- Exports integrated latent representations and saves the updated AnnData object for downstream analysis.
"""

# Load libraries
import scvi
import anndata as ad
import sys
import os
import datetime
from scvi.model import SCVI

# Set base dir
base_dir = "/path/to/folder/"

# Construct paths using base dir
reference_path = os.path.join(base_dir, "reference_filtered.h5ad")
checkpoints_dir = os.path.join(base_dir, "scvi-chkpt")
output_path = os.path.join(base_dir, "ref_scvi_integrated.h5ad")

# Read data
reference_adata = ad.read_h5ad(reference_path)

# Set up the AnnData object for SCVI
scvi.model.SCVI.setup_anndata(reference_adata, batch_key='chemistry')

# Create the SCVI model
scvi_model = scvi.model.SCVI(
    reference_adata,
    n_layers=2,         
    dropout_rate=0.2    
)

# Train the model with early stopping 
scvi_model.train(
    max_epochs=400,
    batch_size=512,
    early_stopping=True,
    early_stopping_monitor='elbo_validation',
    datasplitter_kwargs={
        'pin_memory': True,
        'num_workers': 12
    },
    enable_checkpointing=True
)

# Save model
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model_dir = os.path.join(base_dir, f"scvi-model-{timestamp}")
scvi_model.save(model_dir)
reference_adata.obsm["X_scVI"] = scvi_model.get_latent_representation()

# Save the updated AnnData object
reference_adata.write_h5ad(output_path)

print("Reference model training complete and saved.")

