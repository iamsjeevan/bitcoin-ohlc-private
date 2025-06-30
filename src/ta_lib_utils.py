# src/ta_lib_utils.py

import pandas as pd
import talib as TA
import logging
from tqdm.auto import tqdm
import numpy as np # Needed for NaN handling

# Set up a basic logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def calculate_ta_indicators(df: pd.DataFrame, indicator_config: dict) -> pd.DataFrame:
    """
    Calculates TA-Lib indicators based on the provided configuration and adds them to the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame with OHLCV data. Expected columns:
                           'Open', 'High', 'Low', 'Close', 'Volume'.
                           Index should be datetime.
        indicator_config (dict): Dictionary mapping TA-Lib function names to their parameters.
                                 Example: {'RSI': [14, 28], 'MACD': [(12, 26, 9)]}

    Returns:
        pd.DataFrame: DataFrame with original data and new indicator columns.
    """
    if df.empty:
        logger.warning("Input DataFrame is empty. No TA-Lib indicators will be calculated.")
        return df.copy()

    df_with_indicators = df.copy()

    # Ensure OHLCV columns exist and are numeric (float) for TA-Lib
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df_with_indicators.columns:
            logger.error(f"Missing required column for TA-Lib: '{col}'. Cannot calculate indicators.")
            return df_with_indicators # Return as is if core columns are missing
        df_with_indicators[col] = pd.to_numeric(df_with_indicators[col], errors='coerce')
        # Replace NaNs with actual NaN float for NumPy compatibility if needed, though pd.to_numeric(errors='coerce') usually handles this
        if df_with_indicators[col].isnull().any():
            logger.debug(f"Column '{col}' contains NaN values. TA-Lib functions will produce NaNs at the start.")

    logger.info(f"Calculating {len(indicator_config)} TA-Lib indicator types...")

    # Prepare NumPy arrays for TA-Lib input (TA-Lib works best with NumPy arrays)
    # Ensure 'values' property is used for performance
    talib_inputs = {
        'open': df_with_indicators['Open'].values,
        'high': df_with_indicators['High'].values,
        'low': df_with_indicators['Low'].values,
        'close': df_with_indicators['Close'].values,
        'volume': df_with_indicators['Volume'].values,
    }

    for func_name, periods in tqdm(indicator_config.items(), desc="TA-Lib Indicators"):
        func = getattr(TA, func_name, None)
        if not func:
            logger.warning(f"TA-Lib function '{func_name}' not found. Skipping.")
            continue

        # Initialize 'params' for the outer loop to prevent UnboundLocalError in except block
        current_params_str = "N/A" 

        try:
            # Iterate through the periods/parameter tuples defined in config
            for params in periods: # 'params' will be set here, accessible in inner loop
                
                # Update current_params_str for logging in case of error
                current_params_str = str(params) 

                # Dynamically determine the inputs required for the TA-Lib function
                # This relies on common knowledge of TA-Lib function signatures
                inputs_for_func = []
                if hasattr(func, 'input_names'): # Some TA-Lib functions expose this
                    for name_in_talib_input in func.input_names:
                        if name_in_talib_input in talib_inputs:
                            inputs_for_func.append(talib_inputs[name_in_talib_input])
                        else:
                            logger.warning(f"Missing required input '{name_in_talib_input}' for {func_name}. Skipping {params}.")
                            break # Break from inner loop if inputs missing for this parameter set
                    else: # This 'else' executes if the inner 'break' was not hit (i.e., all inputs found)
                        pass # Proceed to call the function
                elif func_name in ['AD', 'OBV']: # Specific handling for common ones not strictly OHLCV
                    if func_name == 'AD': # Accumulation/Distribution Line takes HLCV
                        inputs_for_func = [talib_inputs['high'], talib_inputs['low'], talib_inputs['close'], talib_inputs['volume']]
                    elif func_name == 'OBV': # On Balance Volume takes Close and Volume
                        inputs_for_func = [talib_inputs['close'], talib_inputs['volume']]
                else: # Default: assume 'close' is the primary input if not specified
                    inputs_for_func = [talib_inputs['close']]
                
                if not inputs_for_func or any(input_array.size == 0 for input_array in inputs_for_func):
                    logger.warning(f"Insufficient or empty input data for {func_name} with params {params}. Skipping.")
                    continue # Skip this parameter set

                # Construct keyword arguments for the TA-Lib function
                kwargs = {}
                col_suffix = ""

                if isinstance(params, (int, float)): # Single timeperiod (e.g., RSI, SMA)
                    kwargs['timeperiod'] = int(params)
                    col_suffix = str(int(params))
                elif isinstance(params, tuple): # Multiple parameters (e.g., MACD, STOCH, BBANDS)
                    if func_name == 'MACD':
                        kwargs['fastperiod'], kwargs['slowperiod'], kwargs['signalperiod'] = params
                        col_suffix = f"{params[0]}_{params[1]}_{params[2]}"
                    elif func_name == 'STOCH':
                        kwargs['fastk_period'], kwargs['slowk_period'], kwargs['slowd_period'] = params
                        col_suffix = f"{params[0]}_{params[1]}_{params[2]}"
                    elif func_name == 'BBANDS':
                        kwargs['timeperiod'], kwargs['nbdevup'] = params
                        kwargs['nbdevdn'] = params[1] # nbdevup and nbdevdn are often the same
                        col_suffix = f"{params[0]}_{params[1]}"
                    else:
                        logger.warning(f"Unhandled tuple parameters for {func_name}. Skipping {params}.")
                        continue
                else:
                    logger.warning(f"Unsupported parameter type for {func_name}: {type(params)}. Skipping.")
                    continue
                
                # Call the TA-Lib function with unwrapped inputs and kwargs
                results = func(*inputs_for_func, **kwargs)

                # Add results to DataFrame
                if isinstance(results, tuple): # Functions that return multiple outputs (e.g., MACD, BBANDS, STOCH)
                    # TA-Lib's get_func_info() would give us output names, but since it's not working,
                    # we hardcode common patterns or infer
                    output_names = []
                    if func_name == 'MACD': output_names = ['MACD_line', 'MACD_signal', 'MACD_hist']
                    elif func_name == 'BBANDS': output_names = ['BB_upper', 'BB_middle', 'BB_lower']
                    elif func_name == 'STOCH': output_names = ['STOCH_slowk', 'STOCH_slowd']
                    # Add more patterns here as needed for other multi-output functions

                    if len(results) != len(output_names):
                        logger.warning(f"Mismatched outputs for {func_name}. Expected {len(output_names)}, got {len(results)}. Using generic names.")
                        output_names = [f"{func_name}_output_{i+1}" for i in range(len(results))]

                    for i, output_name_base in enumerate(output_names):
                        df_with_indicators[f"{output_name_base}_{col_suffix}"] = results[i]
                else: # Functions that return a single output
                    df_with_indicators[f"{func_name}_{col_suffix}"] = results

        except Exception as e:
            # Catch any errors during indicator calculation
            logger.warning(f"Could not calculate TA-Lib indicator {func_name} with params {current_params_str}: {e}. Skipping.")
    
    return df_with_indicators