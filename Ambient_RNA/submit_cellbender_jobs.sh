#!/bin/bash

"""
Author: Saranya Canchi
Date Created: 2025-04
Description: 
- Iterates through a list of sample IDs to automate batch submission of CellBender background removal jobs for snRNA-seq data.
- For each sample:
    - Creates dedicated output and logging directories.
    - Submits a GPU-enabled SLURM job to run CellBender on the raw feature matrix.
    - Logs output and errors for each sample-specific job.
- Ensures reproducible and scalable preprocessing.
"""

# Base dir 
base_dir="/path/to/folder/"

# Data dir
data_dir="/path/to/folder/"

# Logs base dir
logs_base_dir="${base_dir}/slurm_logs"

# Ensure the base logs dir exists
mkdir -p ${logs_base_dir}

# File with the list ids
ids_file="${data_dir}/pfc_samples.txt"

# Ensure the id file exists
if [ ! -f ${ids_file} ]; then
  echo "File ${ids_file} not found!"
  exit 1
fi

# Find all sample dirs
while read -r id; do
  input_file="${data_dir}/${id}/cellranger/raw_feature_bc_matrix.h5"
  output_dir="${base_dir}/${id}"
  cbout="${output_dir}/${id}_cellbender.h5"
  log_dir="${logs_base_dir}/${id}"

  # Ensure output and log dirs exist
  mkdir -p ${output_dir}
  mkdir -p ${log_dir}

  # Define job name for each sample
  job_name="${id}_cellbender"

  # Submit the job for each sample
  sbatch --job-name=${job_name} --output=${log_dir}/cellbender_%j.out --error=${log_dir}/cellbender_%j.err <<EOF
#!/bin/bash
#SBATCH --account=kaczoro99
#SBATCH --time=8:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --mem=110g
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=gpu

# Change to output dir
cd ${output_dir}

# Activate conda env
source /path/to/conda/
conda activate [ENV_NAME]

# Run cellbender
cellbender remove-background \
  --cuda \
  --input ${input_file} \
  --output ${cbout} \
  --learning-rate 1e-5 \
  --debug

# Deactivate env
conda deactivate
EOF

done < ${ids_file}
