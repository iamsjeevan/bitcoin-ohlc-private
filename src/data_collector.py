# src/data_collector.py

import os
import json
from datetime import datetime, timedelta, timezone
import pandas as pd
import logging
from tqdm.auto import tqdm # For progress bars

# Import our custom modules
from src.exchange_clients import BinanceClient, ExchangeClient # Add other clients here as they are implemented
from src.ta_lib_utils import calculate_ta_indicators

# Set up a basic logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class MarketDataCollector:
    """
    Orchestrates the historical market data collection, processing, and storage.
    Handles modularity for exchanges, symbols, intervals, TA-Lib indicators,
    and provides fault tolerance with checkpointing.
    """
    def __init__(self, config: dict):
        self.config = config
        self.exchange_name = self.config['exchange']
        self.symbol = self.config['symbol']
        self.interval = self.config['interval']
        self.start_date_str = self.config['start_date']
        
        # Construct dynamic output path: data/raw/{exchange}/{symbol}/{interval}/
        self.output_dir = self._get_data_output_path()
        self.checkpoint_file_path = self._get_checkpoint_path()
        self.log_file_path = os.path.join(self.output_dir, self.config['log_file']) # Log file specifically for this symbol/interval

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Configure file logger for this specific run
        self._setup_file_logger()

        self.exchange_client = self._get_exchange_client()
        if not self.exchange_client:
            raise ValueError(f"Unsupported or uninitialized exchange client: {self.exchange_name}")

        self.klines_per_request = self.config['klines_per_request']
        self.api_retries = self.config['api_retries']
        self.api_backoff_factor = self.config['api_backoff_factor']
        self.ta_indicator_config = self.config['ta_indicators']

        logger.info(f"MarketDataCollector initialized for {self.exchange_name} {self.symbol} {self.interval}")
        logger.info(f"Data will be stored in: {self.output_dir}")

    def _setup_file_logger(self):
        """Sets up a file handler for logging specific to this data collection run."""
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                logger.removeHandler(handler) # Remove any existing file handler

        file_handler = logging.FileHandler(self.log_file_path)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {self.log_file_path}")


    def _get_data_output_path(self) -> str:
        """Constructs the structured output directory path."""
        base = self.config['base_data_output_dir']
        return os.path.join(base, self.exchange_name, self.symbol, self.interval)

    def _get_checkpoint_path(self) -> str:
        """Constructs the checkpoint file path within the output directory."""
        return os.path.join(self.output_dir, self.config['checkpoint_filename'])

    def _load_checkpoint(self) -> dict:
        """Loads the last successful timestamps from the checkpoint file."""
        if os.path.exists(self.checkpoint_file_path):
            try:
                with open(self.checkpoint_file_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding checkpoint file {self.checkpoint_file_path}: {e}. Starting fresh.")
                return {}
        return {}

    def _save_checkpoint(self, checkpoint_data: dict):
        """Saves the current download progress to the checkpoint file."""
        os.makedirs(self.output_dir, exist_ok=True) # Ensure dir exists before writing
        with open(self.checkpoint_file_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=4)
        logger.debug(f"Checkpoint saved to {self.checkpoint_file_path}")

    def _get_exchange_client(self) -> ExchangeClient:
        """
        Returns the appropriate ExchangeClient instance based on configuration.
        Add more 'elif' blocks here for other exchanges.
        """
        if self.exchange_name.lower() == 'binance':
            return BinanceClient(
                api_key=self.config.get('binance_api_key', ''),
                api_secret=self.config.get('binance_api_secret', '')
            )
        # elif self.exchange_name.lower() == 'kraken':
        #     return KrakenClient(...)
        # elif self.exchange_name.lower() == 'coinbase':
        #     return CoinbaseClient(...)
        else:
            logger.error(f"Unsupported exchange specified in config: {self.exchange_name}")
            return None

    def _get_month_range_ms(self, year: int, month: int, current_time_utc: datetime) -> tuple[int, int]:
        """
        Calculates start and end timestamps (in milliseconds) for a given month.
        The end timestamp is either the end of the month or the current time, whichever is earlier.
        """
        start_of_month = datetime(year, month, 1, 0, 0, 0, tzinfo=timezone.utc)
        
        # Calculate end of month
        if month == 12:
            end_of_month = datetime(year + 1, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        else:
            end_of_month = datetime(year, month + 1, 1, 0, 0, 0, tzinfo=timezone.utc)

        # Ensure we don't fetch data past the current time
        actual_end_of_period = min(end_of_month, current_time_utc)

        return int(start_of_month.timestamp() * 1000), int(actual_end_of_period.timestamp() * 1000)

    def collect_data(self):
        """
        Main method to start/resume the data collection process.
        Iterates month by month, fetches data, calculates indicators, and saves.
        """
        logger.info("Starting data collection process...")

        checkpoint_data = self._load_checkpoint()
        
        # Define the overall date range for data collection
        # pyyaml now converts 'start_date' string to datetime.date object automatically
        # We convert it to a timezone-aware datetime object for consistent calculations
        start_date_obj = datetime(self.config['start_date'].year,
                                  self.config['start_date'].month,
                                  self.config['start_date'].day,
                                  0, 0, 0, 0, tzinfo=timezone.utc)
        
        current_time_utc = datetime.now(timezone.utc)
        
        # --- MODIFIED: Iterate from latest month to oldest month ---
        year_month_iterator = []
        # Start from the current month or the month of start_date, whichever is later
        # and iterate backwards to the month of start_date
        # Ensure we start from the first day of the current month
        current_iter_date = current_time_utc.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        while current_iter_date >= start_date_obj:
            year_month_iterator.append((current_iter_date.year, current_iter_date.month))
            # Move to the first day of the previous month
            # Subtract one day, then set day to 1 to go to start of previous month
            current_iter_date = (current_iter_date - timedelta(days=1)).replace(day=1)

        # The iterator is now latest to oldest because of the while loop's nature
        # No need to reverse if we built it in reverse directly.
        # Example: June, May, April...
        # The loop naturally produces from current_month down to start_month.

        for year, month in tqdm(year_month_iterator, desc="Processing Months"):
            month_str = f"{year}-{month:02d}"
            parquet_filename = os.path.join(self.output_dir, f'{self.symbol.lower()}_{month_str}.parquet')

            month_start_ts_ms, month_end_ts_ms = self._get_month_range_ms(year, month, current_time_utc)
            
            # Check checkpoint for this month
            last_downloaded_ts_ms = checkpoint_data.get(month_str)

            # Skip if month is already fully downloaded
            # We subtract 1 minute's worth of milliseconds to ensure we consider the *end* of the last minute
            # as the true "end of data for that month" for comparison.
            if last_downloaded_ts_ms is not None and last_downloaded_ts_ms >= month_end_ts_ms - (60 * 1000):
                logger.info(f"Month {month_str} already fully downloaded up to {datetime.fromtimestamp(last_downloaded_ts_ms / 1000, tz=timezone.utc)}. Skipping.")
                continue
            
            # Determine where to start fetching data for this month
            fetch_start_ts_ms = month_start_ts_ms
            existing_df = pd.DataFrame()

            # If a partial download exists, load it and adjust fetch_start_ts_ms
            if os.path.exists(parquet_filename):
                try:
                    existing_df = pd.read_parquet(parquet_filename)
                    if not existing_df.empty:
                        # Ensure the index is timezone-aware if it's not already
                        if existing_df.index.tz is None:
                            existing_df.index = existing_df.index.tz_localize('UTC')
                        
                        # Filter out potentially corrupt or future data
                        existing_df = existing_df[existing_df.index.astype(int) <= (current_time_utc.timestamp() * 10**9)] # Filter future data

                        latest_ts_in_file = int(existing_df.index.max().timestamp() * 1000)
                        
                        # If file has more recent data than checkpoint, update checkpoint's view
                        # This handles cases where checkpoint might be old but file has more data
                        if last_downloaded_ts_ms is None or latest_ts_in_file > last_downloaded_ts_ms:
                            checkpoint_data[month_str] = latest_ts_in_file
                            last_downloaded_ts_ms = latest_ts_in_file
                            logger.info(f"Checkpoint for {month_str} updated from file's latest timestamp: {datetime.fromtimestamp(latest_ts_in_file / 1000, tz=timezone.utc)}")

                        # Adjust fetch start time: next minute after the latest *saved* timestamp
                        fetch_start_ts_ms = last_downloaded_ts_ms + (60 * 1000) 
                        logger.info(f"Resuming {month_str} from {datetime.fromtimestamp(fetch_start_ts_ms / 1000, tz=timezone.utc)}")
                    else:
                        logger.info(f"Existing Parquet for {month_str} is empty. Starting from beginning of month.")
                        fetch_start_ts_ms = month_start_ts_ms
                except Exception as e:
                    logger.warning(f"Could not load existing Parquet for {month_str}: {e}. Will re-download/start fresh for this month.")
                    existing_df = pd.DataFrame() # Reset to empty if corrupted or unreadable
                    fetch_start_ts_ms = month_start_ts_ms # Start from beginning of month for safety


            # If after all checks, we are already at or past the end of the month's fetch range
            if fetch_start_ts_ms >= month_end_ts_ms - (60 * 1000): # -1 minute buffer
                logger.info(f"Month {month_str} is already up to date, no new data to fetch past {datetime.fromtimestamp(fetch_start_ts_ms / 1000, tz=timezone.utc)}. Skipping.")
                # Mark as fully downloaded in checkpoint, as no more data to fetch for this month
                checkpoint_data[month_str] = month_end_ts_ms - (60 * 1000) # Mark end of last minute fetched
                self._save_checkpoint(checkpoint_data)
                continue

            logger.info(f"Fetching data for {month_str} from {datetime.fromtimestamp(fetch_start_ts_ms / 1000, tz=timezone.utc)} to {datetime.fromtimestamp(month_end_ts_ms / 1000, tz=timezone.utc)}")
            
            # Fetch klines using the selected exchange client
            klines_data = self.exchange_client.fetch_klines(
                symbol=self.symbol,
                interval=self.interval,
                start_ts_ms=fetch_start_ts_ms,
                end_ts_ms=month_end_ts_ms,
                limit=self.klines_per_request,
                retries=self.api_retries,
                backoff_factor=self.api_backoff_factor
            )

            if klines_data is None: # Persistent error occurred in fetching
                logger.error(f"Failed to fetch klines for {month_str}. Aborting month download. Will retry from previous point on next run.")
                # Checkpoint is NOT updated, so it will retry this month from its last successful point
                continue # Move to next month, or stop if this is the last month

            if not klines_data:
                logger.info(f"No new data fetched for {month_str} in range {datetime.fromtimestamp(fetch_start_ts_ms / 1000, tz=timezone.utc)} to {datetime.fromtimestamp(month_end_ts_ms / 1000, tz=timezone.utc)}. This might mean no data is available yet or all data fetched.")
                # If fetch_start_ts_ms was at or past month_end_ts_ms, mark month as complete.
                if fetch_start_ts_ms >= month_end_ts_ms - (60 * 1000): # -1 minute buffer
                    checkpoint_data[month_str] = month_end_ts_ms - (60 * 1000)
                    self._save_checkpoint(checkpoint_data)
                continue # Move to next month

            new_df = self.exchange_client.transform_klines_to_dataframe(klines_data)
            
            # Combine existing data with newly fetched data
            if not existing_df.empty:
                # Use index for concatenation and drop duplicates based on index
                combined_df = pd.concat([existing_df, new_df]).sort_index()
                combined_df = combined_df[~combined_df.index.duplicated(keep='first')] # Keep first occurrence
            else:
                combined_df = new_df

            # Calculate TA-Lib indicators on the combined (or new) data
            combined_df = calculate_ta_indicators(combined_df, self.ta_indicator_config)
            
            # Save the DataFrame to Parquet
            try:
                # Parquet files are typically saved with the symbol name and month
                output_file = os.path.join(self.output_dir, f'{self.symbol.lower()}_{month_str}.parquet')
                combined_df.to_parquet(output_file, index=True) # Save index (Open time)
                
                # Update checkpoint with the last timestamp of successfully saved data
                # Use the last timestamp from the *saved* DataFrame to be precise
                if not combined_df.empty:
                    checkpoint_data[month_str] = int(combined_df.index.max().timestamp() * 1000)
                    self._save_checkpoint(checkpoint_data)
                    logger.info(f"Successfully saved {len(combined_df)} rows for {month_str} to {output_file}. Last timestamp: {datetime.fromtimestamp(checkpoint_data[month_str] / 1000, tz=timezone.utc)}")
                else:
                    logger.warning(f"No data to save for {month_str} after processing. Checkpoint NOT updated.")

            except Exception as e:
                logger.error(f"Failed to save Parquet file for {month_str}: {e}. Checkpoint NOT updated. This month will retry from previous point.")
                # Do NOT update checkpoint if save fails, ensuring retry from previous point
        
        logger.info("Data collection process finished.")