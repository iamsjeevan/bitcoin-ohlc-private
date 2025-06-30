# src/exchange_clients.py

import time
from datetime import datetime, timezone
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException
import logging

# Set up a basic logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class ExchangeClient:
    """
    Abstract base class for exchange clients.
    Defines the interface for fetching historical klines.
    """
    def __init__(self, api_key: str = "", api_secret: str = ""):
        self.api_key = api_key
        self.api_secret = api_secret
        self.exchange_name = "unknown" # To be set by subclasses

    def fetch_klines(self, symbol: str, interval: str, start_ts_ms: int, end_ts_ms: int,
                     limit: int, retries: int, backoff_factor: int) -> list:
        """
        Fetches historical klines for a given symbol and interval within a time range.
        Args:
            symbol (str): Trading pair (e.g., 'BTCUSDT').
            interval (str): Kline interval (e.g., '1m', '1h').
            start_ts_ms (int): Start timestamp in milliseconds (inclusive).
            end_ts_ms (int): End timestamp in milliseconds (exclusive).
            limit (int): Max number of klines per request.
            retries (int): Number of retry attempts for API calls.
            backoff_factor (int): Factor to increase wait time between retries.
        Returns:
            list: A list of klines data, or None if persistent failure.
        """
        raise NotImplementedError("Subclasses must implement fetch_klines method.")

    def transform_klines_to_dataframe(self, klines_data: list) -> pd.DataFrame:
        """
        Transforms raw klines data (list of lists) into a standardized Pandas DataFrame.
        This method should be implemented by subclasses to handle exchange-specific formats.
        """
        raise NotImplementedError("Subclasses must implement transform_klines_to_dataframe method.")


class BinanceClient(ExchangeClient):
    """
    Binance API client implementation.
    """
    def __init__(self, api_key: str = "", api_secret: str = ""):
        super().__init__(api_key, api_secret)
        self.client = Client(api_key, api_secret)
        self.exchange_name = "binance"
        logger.info(f"Initialized BinanceClient (public data access).")

    def fetch_klines(self, symbol: str, interval: str, start_ts_ms: int, end_ts_ms: int,
                     limit: int, retries: int, backoff_factor: int) -> list:
        """
        Fetches klines from Binance API with retry logic.
        Binance API expects start/end_str in milliseconds.
        """
        all_klines = []
        current_start_ts = start_ts_ms
        
        # Binance's get_historical_klines function often handles internal pagination
        # by itself when given a broad range and a high limit.
        # However, for very long periods, it might be better to make chunked calls
        # as a fail-safe against unexpected server limits or large response sizes.
        # The client.get_historical_klines will fetch up to 'limit' klines.
        
        while current_start_ts < end_ts_ms:
            attempt = 0
            while attempt <= retries:
                try:
                    # Binance API's 'get_historical_klines' is quite robust.
                    # It will fetch max 'limit' klines from 'start_str' up to 'end_str'.
                    # We manage 'current_start_ts' to ensure we get *new* data.
                    
                    klines_chunk = self.client.get_historical_klines(
                        symbol=symbol,
                        interval=interval,
                        start_str=str(current_start_ts),
                        end_str=str(end_ts_ms), # Binance can handle larger end_str
                        limit=limit # Request max klines per call
                    )

                    if not klines_chunk:
                        # No more data in this range, typically at the very end of the available data or month
                        logger.debug(f"No klines returned for {symbol} from {datetime.fromtimestamp(current_start_ts / 1000, tz=timezone.utc)} to {datetime.fromtimestamp(end_ts_ms / 1000, tz=timezone.utc)}. Breaking.")
                        break # Exit inner retry loop and outer while loop

                    # Filter out klines that might have been included in previous chunk due to API overlap
                    # This happens if a chunk returns < limit and the next call's start_str overlaps slightly.
                    unique_klines = [k for k in klines_chunk if k[0] >= current_start_ts]
                    
                    if not unique_klines:
                        # If all klines in the chunk were duplicates/already processed, move forward
                        # This can happen if Binance returns exactly the same last kline from previous chunk
                        logger.debug(f"Fetched chunk was empty after de-duplication from {datetime.fromtimestamp(current_start_ts / 1000, tz=timezone.utc)}. Moving to next possible timestamp.")
                        current_start_ts += (limit * self._interval_to_ms(interval)) # Jump forward by max possible klines
                        continue # Try fetching next chunk
                    
                    all_klines.extend(unique_klines)
                    
                    # Update current_start_ts to the minute *after* the last kline received
                    current_start_ts = unique_klines[-1][0] + self._interval_to_ms(interval)
                    
                    attempt = 0 # Reset retry attempt on success
                    
                    # If the last fetched kline's open time plus interval is >= end_ts_ms, we've got all data
                    if current_start_ts >= end_ts_ms:
                        break # Acquired all data up to end_ts_ms
                    
                    # Small delay to respect rate limits even if not explicitly hit
                    time.sleep(0.1) 

                except BinanceAPIException as e:
                    if e.code == -1003: # Too many requests (rate limit)
                        # Binance 'get_historical_klines' is smart, but if it hits a specific rate limit,
                        # we need to respect it. Binance API docs suggest some specific delays for klines.
                        # For simplicity, we use a general backoff.
                        wait_time = backoff_factor ** attempt
                        logger.warning(f"Binance API Rate Limit ({e.code}) hit. Retrying in {wait_time}s. ({attempt+1}/{retries})")
                        time.sleep(wait_time)
                        attempt += 1
                    elif e.code == -1021: # Timestamp for this request is outside of the recvWindow.
                        # This usually means your system clock is out of sync or there's a problem with start/end_str
                        logger.error(f"Binance API Error {e.code}: {e.message}. Check system clock or timestamp logic. Aborting fetch for this chunk.")
                        return None
                    else:
                        logger.warning(f"Binance API Error {e.code}: {e.message}. Retrying in {backoff_factor**attempt}s. ({attempt+1}/{retries})")
                        time.sleep(backoff_factor**attempt)
                        attempt += 1

                except Exception as e:
                    logger.error(f"An unexpected error occurred during Binance kline fetch: {e}. Retrying in {backoff_factor**attempt}s. ({attempt+1}/{retries})")
                    time.sleep(backoff_factor**attempt)
                    attempt += 1
            
            if attempt > retries:
                logger.error(f"Max retries reached for fetching {symbol} klines from {datetime.fromtimestamp(start_ts_ms / 1000, tz=timezone.utc)} to {datetime.fromtimestamp(end_ts_ms / 1000, tz=timezone.utc)}. Aborting this range.")
                return None # Indicate persistent failure for this chunk

        return all_klines

    def transform_klines_to_dataframe(self, klines_data: list) -> pd.DataFrame:
        """
        Transforms raw Binance klines data (list of lists) into a standardized Pandas DataFrame.
        """
        if not klines_data:
            return pd.DataFrame() # Return empty DataFrame if no data

        df = pd.DataFrame(klines_data, columns=[
            'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Close time', 'Quote asset volume', 'Number of trades',
            'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
        ])
        
        df['Open time'] = pd.to_datetime(df['Open time'], unit='ms', utc=True)
        df = df.astype({
            'Open': float, 'High': float, 'Low': float, 'Close': float, 'Volume': float,
            'Quote asset volume': float, 'Number of trades': int,
            'Taker buy base asset volume': float, 'Taker buy quote asset volume': float
        })
        df = df.set_index('Open time')
        df = df.drop(columns=['Close time', 'Ignore']) # Drop unnecessary columns
        
        # Remove duplicate index entries if any (can happen with API overlaps)
        df = df[~df.index.duplicated(keep='first')]
        
        return df

    def _interval_to_ms(self, interval: str) -> int:
        """Helper to convert Binance interval string to milliseconds."""
        if interval.endswith('m'):
            return int(interval[:-1]) * 60 * 1000
        elif interval.endswith('h'):
            return int(interval[:-1]) * 60 * 60 * 1000
        elif interval.endswith('d'):
            return int(interval[:-1]) * 24 * 60 * 60 * 1000
        elif interval.endswith('w'):
            return int(interval[:-1]) * 7 * 24 * 60 * 60 * 1000
        elif interval.endswith('M'):
            # Approximating a month, safer to calculate from datetime object
            return 30 * 24 * 60 * 60 * 1000 # Approx. 30 days for monthly
        else:
            raise ValueError(f"Unsupported interval: {interval}")

# You could add other exchange clients here:
# class KrakenClient(ExchangeClient):
#    ... implement Kraken specific API calls and data transformation ...
# class CoinbaseClient(ExchangeClient):
#    ...