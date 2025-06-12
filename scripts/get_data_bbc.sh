#!/bin/bash

# Define paths
DATA_FOLDER="data"
DATASET_ZIP="BBC.News.Summary.zip"
DATASET_URL="https://github.com/mmRoshani/bbc-news-classification/releases/download/published/$DATASET_ZIP"
EXTRACTED_PATH="$DATA_FOLDER/BBC.News.Summary"

# Make data folder if it doesn't exist
mkdir -p "$DATA_FOLDER"

# Skip if dataset is already in data folder
if [ -d "$EXTRACTED_PATH" ] && [ -n "$(ls -A "$EXTRACTED_PATH")" ]; then
    echo "Dataset already exists in $EXTRACTED_PATH. Skipping download and extraction."
else
    cd "$DATA_FOLDER" || exit 1

    # Download only if ZIP doesn't already exist
    if [ ! -f "$DATASET_ZIP" ]; then
        wget "$DATASET_URL"
    fi

    # Unzip only if ZIP exists
    if [ -f "$DATASET_ZIP" ]; then
        unzip -o "$DATASET_ZIP"
        rm "$DATASET_ZIP"
        # Handle possible folder names after unzip
        if [ -d "BBC.News.Summary" ]; then
            echo "Extracted to BBC.News.Summary"
        elif [ -d "BBC News Summary" ]; then
            mv "BBC News Summary" "BBC.News.Summary"
        else
            echo "Error: Could not find expected dataset folder after unzip."
            exit 1
        fi
    else
        echo "Error: Failed to download $DATASET_ZIP."
        exit 1
    fi

    cd ..
fi

# Run your data splitting script if needed
BBC_ARTICLES_PATH="$EXTRACTED_PATH/BBC News Summary/News Articles"
if [ -f "/datasets/text/bbc/data_spliting.py" ]; then
    python3 /datasets/text/bbc/data_spliting.py --directory "$BBC_ARTICLES_PATH" --output ../../../../data/bbc --max 3
else
    echo "Warning: data_spliting.py not found. Skipping data splitting."
fi
