#!/bin/bash

# Check if environment name is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <ENV_NAME>"
    exit 1
fi

ENV_NAME=$1

echo "Creating conda environment '$ENV_NAME'..."
conda create -y -n "$ENV_NAME" python=3.11

echo "Activating environment '$ENV_NAME'..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "Installing required packages..."

# PyTorch with pip (auto-detects CUDA if available)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Other packages via conda
conda install -y opencv pyyaml numpy matplotlib tqdm 
# Other packages via pip
pip install hydra-core scikit-learn wandb 

echo "Setup complete for environment '$ENV_NAME'."
 