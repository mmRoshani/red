import os
import zipfile
import requests
import subprocess

def download_and_prepare_yahoo_dataset():
    # Constants
    HOME = os.path.expanduser("~")
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

    DATA_FOLDER = os.path.join(HOME, "Documents/coordinate-similarity/coordinate-similarity/data")
    DATASET_ZIP = "archive.zip"
    DATASET_URL = "https://github.com/mmRoshani/yahoo-qanda-classification/releases/download/re-publish/archive.zip"
    EXTRACTED_PATH = os.path.join(DATA_FOLDER, "yahoo")
    DATA_SPLITTING_SCRIPT = os.path.join(PROJECT_ROOT, "src/datasets/text/yahoo/data_spliting.py")

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
            zip_ref.extractall(EXTRACTED_PATH)

        os.remove(zip_path)

    # Run the data splitting script if available
    if os.path.isfile(DATA_SPLITTING_SCRIPT):
        print("Running data splitting script...")
        subprocess.run([
            "python3", DATA_SPLITTING_SCRIPT,
            "--directory", EXTRACTED_PATH,
            "--output", os.path.join(HOME, "data/yahoo"),
            "--max", "3"
        ])
    else:
        print(f"Warning: data_spliting.py not found at {DATA_SPLITTING_SCRIPT}. Skipping data splitting.")
