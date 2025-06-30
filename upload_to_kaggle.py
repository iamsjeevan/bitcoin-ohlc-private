# upload_to_kaggle.py

import os
import json
import logging
from datetime import datetime
from kaggle.api.kaggle_api_extended import KaggleApi

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_kaggle_dataset_dir():
    """
    Returns the path to the directory containing the data to be uploaded to Kaggle.
    This should be the directory that contains the Parquet files and the dataset-metadata.json.
    """
    # This path points to the directory containing all your monthly Parquet files
    return os.path.join("data", "raw", "binance", "BTCUSDT", "1m")

def generate_dataset_metadata(dataset_dir: str, username: str, dataset_slug: str):
    """
    Generates or updates the dataset-metadata.json file required by Kaggle.
    Args:
        dataset_dir (str): The local directory where the data to be uploaded resides.
        username (str): Your Kaggle username.
        dataset_slug (str): The unique identifier for your dataset (e.g., 'btcusdt-1m-ohlcv-ta-lib-raw').
    """
    metadata_path = os.path.join(dataset_dir, "dataset-metadata.json")
    dataset_id = f"{username}/{dataset_slug}"

    # --- IMPORTANT: Ensure title is between 6 and 50 characters ---
    # This title describes your full dataset
    dataset_title = "BTCUSDT 1-Minute OHLCV with TA-Lib Indicators" # 44 characters - perfect!

    metadata = {
        "title": dataset_title,
        "id": dataset_id,
        "licenses": [{"name": "CC0-1.0"}], # Public Domain Dedication
        "resources": [], # Kaggle infers files when uploading a directory, so this remains empty
        "description": (
            "Comprehensive historical 1-minute OHLCV data for BTC/USDT collected from Binance, "
            "enriched with a wide range of TA-Lib technical indicators. "
            "Data spans from September 2017 to the latest available date. "
            "This dataset is intended for training machine learning models for Bitcoin price prediction."
        )
    }

    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logging.info(f"Generated/Updated dataset-metadata.json at: {metadata_path}")
    except IOError as e:
        logging.error(f"Error writing dataset-metadata.json: {e}")
        raise

def upload_dataset_to_kaggle(dataset_dir: str, dataset_id: str):
    """
    Uploads or updates a dataset to Kaggle.
    Args:
        dataset_dir (str): The local directory containing the dataset files and dataset-metadata.json.
        dataset_id (str): The full dataset ID (e.g., 'your_username/your_dataset_slug').
    """
    api = KaggleApi()
    api.authenticate() # This looks for kaggle.json in ~/.kaggle/

    logging.info(f"Attempting to upload/update dataset from: {dataset_dir}")
    logging.info(f"Dataset ID: {dataset_id}")

    try:
        # First attempt to create a new dataset.
        # If it already exists (from a previous successful run), it will fail and fall back to create_version.
        try:
            logging.info(f"Trying to create new dataset: {dataset_id}")
            api.dataset_create_new(
                folder=dataset_dir,
                # public=True # Default is False (private) if not specified. Change to True if you want it public.
            )
            logging.info(f"Successfully created new dataset: {dataset_id}")
        except Exception as e:
            # If creation fails, assume it's because it already exists and try to create a new version
            logging.warning(f"Dataset creation failed (might already exist): {e}")
            logging.info(f"Trying to update existing dataset: {dataset_id}")
            api.dataset_create_version(
                folder=dataset_dir,
                version_notes=f"Full dataset update with latest 1-minute data and TA-Lib indicators - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            logging.info(f"Successfully updated dataset: {dataset_id}")

        logging.info(f"Dataset upload/update process completed for {dataset_id}.")
        logging.info(f"You can view your dataset at: https://www.kaggle.com/datasets/{dataset_id}")

    except Exception as e:
        logging.error(f"Error during Kaggle dataset upload/update: {e}")
        if "403" in str(e) or "authentication" in str(e).lower():
            logging.error("Authentication error. Please ensure your kaggle.json is correctly placed and permissions are set (chmod 600 ~/.kaggle/kaggle.json). Also, verify that your KAGGLE_USERNAME matches your Kaggle account username exactly.")
        elif "409" in str(e) and "dataset with the same id already exists" in str(e).lower():
             logging.error("A dataset with this ID already exists and it failed to update. Check your dataset-metadata.json 'id' field or permissions.")


if __name__ == "__main__":
    # --- Configuration ---
    # !! IMPORTANT: This MUST be your actual Kaggle username
    KAGGLE_USERNAME = "jeevans13"

    # !! IMPORTANT: This is a NEW, unique dataset SLUG for your full dataset.
    # Choose something clear. Example: "btcusdt-1m-ohlcv-indicators-full"
    # This ensures a fresh upload and avoids conflicts with previous attempts.
    DATASET_SLUG = "btcusdt-1m-ohlcv-indicators-full"
    # ---------------------

    DATA_UPLOAD_PATH = get_kaggle_dataset_dir()
    FULL_DATASET_ID = f"{KAGGLE_USERNAME}/{DATASET_SLUG}"

    # 1. Ensure the directory to be uploaded exists
    if not os.path.isdir(DATA_UPLOAD_PATH):
        logging.error(f"Data upload directory not found: {DATA_UPLOAD_PATH}. Please check your path.")
        exit(1)

    # 2. Generate/Update dataset-metadata.json
    generate_dataset_metadata(DATA_UPLOAD_PATH, KAGGLE_USERNAME, DATASET_SLUG)

    # 3. Upload the dataset
    upload_dataset_to_kaggle(DATA_UPLOAD_PATH, FULL_DATASET_ID)