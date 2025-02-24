# src/utils/logger.py
import os
import json
import datetime
from src.config import LOGS_DIR  # Import the parameter from config.py

def get_timestamped_log_dir(base_directory=LOGS_DIR):
    """
    Create and return a new directory name based on the current timestamp
    inside the base directory.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(base_directory, timestamp)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def save_execution_log(log_data, filename="execution_log.json", directory=LOGS_DIR):
    """
    Save a dictionary of execution data to a JSON file in the specified directory.
    A timestamp is added automatically.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    log_data["timestamp"] = datetime.datetime.now().isoformat()
    filepath = os.path.join(directory, filename)
    with open(filepath, "w") as f:
        json.dump(log_data, f, indent=4)

def load_execution_log(filename="execution_log.json", directory=LOGS_DIR):
    """
    Load execution data from a JSON file in the specified directory.
    """
    filepath = os.path.join(directory, filename)
    with open(filepath, "r") as f:
        return json.load(f)