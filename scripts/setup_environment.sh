#!/usr/bin/env bash
# Setup environment for Xylia

set -e

echo "Setting up Xylia environment..."
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Creating necessary directories..."
mkdir -p data/raw data/processed logs

echo "Environment setup complete."
