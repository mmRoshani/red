import os
import zipfile
import requests
import argparse
import subprocess

def download_and_prepare_bbc_dataset():
    # Constants
    HOME = os.path.expanduser("~")
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

    DATA_FOLDER = os.path.join(HOME, "Documents/coordinate-similarity/coordinate-similarity/data")
    DATASET_ZIP = "BBC.News.Summary.zip"
    DATASET_URL = f"https://github.com/mmRoshani/bbc-news-classification/releases/download/published/{DATASET_ZIP}"
    EXTRACTED_PATH = os.path.join(DATA_FOLDER, "BBC.News.Summary")
    BBC_ARTICLES_PATH = os.path.join(EXTRACTED_PATH, "BBC News Summary/News Articles")
    DATA_SPLITTING_SCRIPT = os.path.join(PROJECT_ROOT, "src/datasets/text/bbc/data_spliting.py")

    # Ensure data folder exists
    os.makedirs(DATA_FOLDER, exist_ok=True)

    # Check if dataset already exists
    if os.path.isdir(EXTRACTED_PATH) and os.listdir(EXTRACTED_PATH):
        print(f"Dataset already exists in {EXTRACTED_PATH}. Skipping download and extraction.")
    else:
        zip_path = os.path.join(DATA_FOLDER, DATASET_ZIP)

        # Download if ZIP not already present
        if not os.path.isfile(zip_path):
            print("Downloading dataset...")
            response = requests.get(DATASET_URL)
            if response.status_code == 200:
                with open(zip_path, "wb") as f:
                    f.write(response.content)
            else:
                raise Exception(f"Failed to download file: HTTP {response.status_code}")

        # Extract ZIP
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_FOLDER)

        os.remove(zip_path)

        # Handle possible folder name after unzip
        extracted_names = os.listdir(DATA_FOLDER)
        if "BBC News Summary" in extracted_names and "BBC.News.Summary" not in extracted_names:
            os.rename(os.path.join(DATA_FOLDER, "BBC News Summary"), EXTRACTED_PATH)
        elif "BBC.News.Summary" not in extracted_names:
            raise FileNotFoundError("Expected folder 'BBC News Summary' or 'BBC.News.Summary' not found after unzip.")

    # Run the data splitting script if available
    if os.path.isfile(DATA_SPLITTING_SCRIPT):
        print("Running data splitting script...")
        subprocess.run([
            "python3", DATA_SPLITTING_SCRIPT,
            "--directory", BBC_ARTICLES_PATH,
            "--output", os.path.join(HOME, "data/bbc"),
            "--max", "3"
        ])
    else:
        print(f"Warning: data_spliting.py not found at {DATA_SPLITTING_SCRIPT}. Skipping data splitting.")