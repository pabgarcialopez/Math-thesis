# src/experiments/utils/logger.py

import os
import json
import datetime
from src.config import LOGS_PATH

def create_timestamped_dir(base_dir):
    """
    Creates a new subdirectory named with the current timestamp under base_dir.
    Returns the path to that new subdirectory.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(base_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def save_execution_log(log_data, filename, directory=LOGS_PATH):
    """
    Saves a dictionary of execution data to a JSON file.
    If directory is None, uses LOGS_DIR from config.
    """
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    log_data["timestamp"] = datetime.datetime.now().isoformat()
    filepath = os.path.join(directory, filename)
    with open(filepath, "w") as f:
        json.dump(log_data, f, indent=4)
