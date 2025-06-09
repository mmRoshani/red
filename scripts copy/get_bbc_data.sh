#!/bin/bash

#todo Dataset already exists in /home/mehdi_ktb/data/BBC.News.Summary. Skipping download and extraction. it's not the correct path
#todo it should be /home/mehdi_ktb/Documents/coordinate-similarity/coordinate-similarity/data/BBC.News.Summary.

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Define paths
DATA_FOLDER="$HOME/data"
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
    if [ ! -f "$DATA_FOLDER/$DATASET_ZIP" ]; then
        wget "$DATASET_URL"
    fi

    # Unzip only if ZIP exists
    if [ -f "$DATA_FOLDER/$DATASET_ZIP" ]; then
        unzip -o "$DATA_FOLDER/$DATASET_ZIP"
        rm "$DATA_FOLDER/$DATASET_ZIP"
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
        echo "Error: Failed to download $DATA_FOLDER/$DATASET_ZIP."
        exit 1
    fi

    cd ..
fi

# Run your data splitting script if needed
BBC_ARTICLES_PATH="$EXTRACTED_PATH/BBC News Summary/News Articles"
if [ -f "$PROJECT_ROOT/src/datasets/text/bbc/data_spliting.py" ]; then
    python3 "$PROJECT_ROOT/src/datasets/text/bbc/data_spliting.py" --directory "$BBC_ARTICLES_PATH" --output "$HOME/data/bbc" --max 3
else
    echo "Warning: data_spliting.py not found at $PROJECT_ROOT/src/datasets/text/bbc/data_spliting.py. Skipping data splitting."
fi
