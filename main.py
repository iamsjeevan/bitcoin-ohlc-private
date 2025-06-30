# main.py

import yaml
import os
import logging
from src.data_collector import MarketDataCollector

# --- Setup Global Logging ---
# This initial setup is for general console output.
# The MarketDataCollector will set up its own file logger.
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Loads configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    """Main function to run the data collection pipeline."""
    config_path = os.path.join('config', 'config.yaml')
    
    try:
        config = load_config(config_path)
        logger.info(f"Configuration loaded from {config_path}.")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return

    try:
        collector = MarketDataCollector(config)
        collector.collect_data()
    except Exception as e:
        logger.critical(f"An unrecoverable error occurred during data collection: {e}", exc_info=True)


if __name__ == "__main__":
    main()