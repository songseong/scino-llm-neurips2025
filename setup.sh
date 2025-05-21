#!/bin/bash
# Upgrade requests package before creating the virtual environment
echo "Upgrading requests package..."
pip install --upgrade requests

Set the virtual environment name as a variable
ENV_NAME="SciNO"

# Initialize Conda by sourcing conda.sh
echo "Initializing Conda..."
source $(conda info --base)/etc/profile.d/conda.sh

# Step 1: Create the Conda virtual environment with Python 3.10.0
echo "Creating Conda environment '$ENV_NAME' with Python 3.10..."
conda create -n $ENV_NAME python=3.10 -y

# Step 2: Install Python dependencies inside the Conda environment
echo "Installing Python dependencies in the environment '$ENV_NAME'..."
conda run -n $ENV_NAME pip install --upgrade pip
conda run -n $ENV_NAME pip install -r requirements.txt

# Step 3: Set up R dependencies
echo "Installing R dependencies..."
conda install -n $ENV_NAME -c conda-forge r-base=4.4.0
conda run -n $ENV_NAME Rscript -e "install.packages('BiocManager', repos='https://cloud.r-project.org')"
conda run -n $ENV_NAME Rscript -e "BiocManager::install(c('graph', 'RBGL', 'pcalg'))"
conda run -n $ENV_NAME Rscript -e "install.packages('SID', repos='https://cloud.r-project.org')"
conda run -n $ENV_NAME Rscript -e "install.packages(c('mgcv', 'Matrix', 'MatrixModels', 'parallel'), repos='https://cloud.r-project.org')"