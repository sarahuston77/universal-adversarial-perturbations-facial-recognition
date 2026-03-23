#!/bin/bash

mkdir -p data
cd data

# Install gdown if not present
if ! command -v gdown &> /dev/null
then
    echo "Installing gdown..."
    pip install gdown
fi

# Download subset
if [ ! -f "crop_part1.zip" ]; then
    echo "Downloading dataset subset..."
    gdown https://drive.google.com/uc?id=19r9gWG5MUSBhgLt0uPNjH3HxSDMwLPYs
fi

# Unzip
if [ ! -d "crop_part1" ]; then
    echo "Unzipping..."
    unzip crop_part1.zip
fi

echo "Dataset ready in data/crop_part1/"