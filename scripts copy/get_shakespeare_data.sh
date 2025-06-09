#!/bin/bash
#todo we have to run shakespeare_factory.py after preprocess_shakespeare
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Define paths
DATA_FOLDER="$HOME/data"
RAW_DATA_FILE="$DATA_FOLDER/raw_data.txt"

mkdir -p "$DATA_FOLDER"

if [ ! -d "shakespear_data" ] || [ ! "$(ls -A shakespear_data)" ]; then
    if [ ! -d "$DATA_FOLDER" ]; then
        mkdir -p "$DATA_FOLDER"
    fi

    if [ ! -f "$DATA_FOLDER/raw_data.txt" ]; then
        echo "------------------------------"
        echo "retrieving raw data"
        cd "$DATA_FOLDER" || exit

        wget http://www.gutenberg.org/files/100/old/old/1994-01-100.zip
        unzip 1994-01-100.zip
        rm 1994-01-100.zip
        mv 100.txt raw_data.txt

        cd "$PROJECT_ROOT" || exit
    fi
fi

if [ ! -d "$DATA_FOLDER/by_play_and_character" ]; then
   echo "------------------------------"
   echo "dividing txt data between users"
   if [ -f "$PROJECT_ROOT/src/datasets/text/shakespeare/preprocess_shakespeare.py" ]; then
       python3 "$PROJECT_ROOT/src/datasets/text/shakespeare/preprocess_shakespeare.py" "$RAW_DATA_FILE" "$DATA_FOLDER/"
   else
       echo "Warning: preprocess_shakespeare.py not found at $PROJECT_ROOT/src/datasets/text/shakespeare/preprocess_shakespeare.py. Skipping data preprocessing."
   fi
fi