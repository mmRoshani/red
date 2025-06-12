import os
import requests
import zipfile
import subprocess

def download_and_preprocess_shakespeare():
    HOME = os.path.expanduser("~")
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

    DATA_FOLDER = os.path.join(HOME, "data")
    RAW_DATA_FILE = os.path.join(DATA_FOLDER, "raw_data.txt")
    PROCESSED_FOLDER = os.path.join(DATA_FOLDER, "by_play_and_character")
    PREPROCESS_SCRIPT = os.path.join(PROJECT_ROOT, "src/datasets/text/shakespeare/preprocess_shakespeare.py")

    os.makedirs(DATA_FOLDER, exist_ok=True)

    # Check if shakespear_data directory is missing or empty
    if not os.path.isdir("shakespear_data") or not os.listdir("shakespear_data"):
        if not os.path.exists(RAW_DATA_FILE):
            print("------------------------------")
            print("Retrieving raw data")

            os.chdir(DATA_FOLDER)
            zip_url = "http://www.gutenberg.org/files/100/old/old/1994-01-100.zip"
            response = requests.get(zip_url)
            if response.status_code == 200:
                with open("1994-01-100.zip", "wb") as f:
                    f.write(response.content)
                with zipfile.ZipFile("1994-01-100.zip", 'r') as zip_ref:
                    zip_ref.extractall()
                os.remove("1994-01-100.zip")
                os.rename("100.txt", "raw_data.txt")
            else:
                raise Exception("Failed to download Shakespeare data.")

            os.chdir(PROJECT_ROOT)

    # Run preprocessing if output folder does not exist
    if not os.path.isdir(PROCESSED_FOLDER):
        print("------------------------------")
        print("Dividing txt data between users")
        if os.path.isfile(PREPROCESS_SCRIPT):
            subprocess.run([
                "python3", PREPROCESS_SCRIPT, RAW_DATA_FILE, f"{DATA_FOLDER}/"
            ])
        else:
            print(f"Warning: preprocess_shakespeare.py not found at {PREPROCESS_SCRIPT}. Skipping data preprocessing.")
