import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

def resample_logs(
    logs: List[Dict[str, Any]],
    sensor_keys: List[str],
    frequency_hz: float = 1.0,
    decay_rate: float = 0.05,
    null_value: float = 9999.0
) -> pd.DataFrame:
    """Resamples raw logs onto a uniform time grid with exponential decay for missing sensors.
    
    Args:
        logs: List of dicts, e.g., [{"time": 1716300000.0, "data": {"sensor.a": "3.5", ...}}, ...]
        sensor_keys: List of all tracked entity IDs.
        frequency_hz: Target resampling grid frequency in Hz.
        decay_rate: Lambda factor for exponential signal decay back to null_value.
        null_value: Sentinel value to represent offline/out-of-range sensor readings.
        
    Returns:
        pd.DataFrame: DataFrame with uniform time index and resampled, decayed sensor values.
    """
    if not logs:
        return pd.DataFrame(columns=sensor_keys)

    # Sort logs by timestamp ascending
    sorted_logs = sorted(logs, key=lambda x: x["time"])
    
    # Establish time grid
    start_time = sorted_logs[0]["time"]
    end_time = sorted_logs[-1]["time"]
    
    step = 1.0 / frequency_hz
    grid_times = np.arange(start_time, end_time + step / 2.0, step)
    if len(grid_times) == 0:
        return pd.DataFrame(columns=sensor_keys)
        
    # Initialize resampled grid
    resampled_data = {key: [] for key in sensor_keys}
    
    # Maintain state of last seen value and time for each sensor
    last_seen_val = {key: null_value for key in sensor_keys}
    last_seen_time = {key: start_time for key in sensor_keys}
    
    # Initialize binary sensor states (null is 0.0)
    for key in sensor_keys:
        if "binary_sensor." in key or "input_boolean." in key:
            last_seen_val[key] = 0.0

    # Pointer in raw logs
    raw_idx = 0
    num_logs = len(sorted_logs)
    
    for t in grid_times:
        # Collect all raw logs that happened at or before t since last step
        while raw_idx < num_logs and sorted_logs[raw_idx]["time"] <= t:
            log_data = sorted_logs[raw_idx]["data"]
            log_time = sorted_logs[raw_idx]["time"]
            
            for key in sensor_keys:
                if key in log_data and log_data[key] is not None:
                    try:
                        val_str = str(log_data[key])
                        # Handle categorical states if any, but focus on numeric
                        if val_str.lower() in ["on", "true", "home"]:
                            val = 1.0
                        elif val_str.lower() in ["off", "false", "not_home", "away"]:
                            val = 0.0
                        elif val_str.lower() in ["unknown", "unavailable", "null"]:
                            # Let it decay/nullify
                            continue
                        else:
                            val = float(val_str)
                        last_seen_val[key] = val
                        last_seen_time[key] = log_time
                    except ValueError:
                        pass
            raw_idx += 1
            
        # Apply exponential decay to t
        for key in sensor_keys:
            dt = t - last_seen_time[key]
            if "binary_sensor." in key or "input_boolean." in key:
                # Binary sensors don't decay exponentially. 
                # They just revert to 0.0 after a timeout (e.g., 30s of no updates)
                if dt > 30.0:
                    resampled_data[key].append(0.0)
                else:
                    resampled_data[key].append(last_seen_val[key])
            else:
                # Continuous distance/RSSI decays towards null_value
                v_decayed = last_seen_val[key] * np.exp(-decay_rate * dt) + null_value * (1.0 - np.exp(-decay_rate * dt))
                resampled_data[key].append(v_decayed)
                
    df = pd.DataFrame(resampled_data, index=grid_times)
    df.index.name = "time"
    return df

def strip_transitions(intervals: List[Dict[str, Any]], margin_seconds: float = 10.0) -> List[Dict[str, Any]]:
    """Trims boundary transition periods off the start and end of labeled training sessions."""
    trimmed = []
    for interval in intervals:
        start = interval["start_time"]
        end = interval["end_time"]
        duration = end - start
        if duration > 2.0 * margin_seconds + 5.0:  # Must have at least 5s left after margin removal
            trimmed.append({
                "id": interval.get("id"),
                "start_time": start + margin_seconds,
                "end_time": end - margin_seconds,
                "label": interval["label"]
            })
    return trimmed

def slice_windows(
    df: pd.DataFrame,
    label: str,
    window_steps: int = 15,
    step_size: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Slices a DataFrame into overlapping sequence windows of shape (window_steps, num_features).
    
    Returns:
        Tuple of (X, y) where:
            X is of shape (num_windows, window_steps, num_features)
            y is of shape (num_windows,) containing the string label
    """
    X_list = []
    num_rows = len(df)
    
    if num_rows < window_steps:
        return np.empty((0, window_steps, len(df.columns))), np.empty((0,))
        
    for i in range(0, num_rows - window_steps + 1, step_size):
        window = df.iloc[i : i + window_steps].to_numpy()
        X_list.append(window)
        
    X = np.array(X_list)
    y = np.full((len(X_list),), label, dtype=object)
    return X, y
