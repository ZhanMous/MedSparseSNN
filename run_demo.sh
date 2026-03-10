#!/usr/bin/env bash
set -e

# Quick demo setup for MedSparseSNN (small smoke test)
# Usage: bash run_demo.sh

if ! command -v conda &> /dev/null; then
  echo "conda not found; please install Anaconda/Miniconda or run steps manually"
  exit 1
fi

ENV_NAME=medsparsesnn

echo "Creating/updating conda env '$ENV_NAME'..."
conda env create -f environment.yml || conda env update -f environment.yml

echo "Activating env and installing requirements via pip..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
pip install -r requirements.txt

echo "Running a small data loader smoke test..."
python - <<'PY'
from data.dataloader import get_blood_mnist_loaders
train_loader, val_loader, test_loader, info = get_blood_mnist_loaders(batch_size=4, T=6, mode='ann', augment=False)
batch = next(iter(train_loader))
print('Smoke test passed: batch shapes:', [x.shape for x in batch])
PY

echo "Done."
