import os
import json
import datetime

DEFAULT_LOG_DIR = "logs"

def get_timestamped_log_dir(base_directory=DEFAULT_LOG_DIR):
    """
    Create and return a new directory name based on the current timestamp
    inside the base directory.
    """
    # Format the current time as YYYYMMDD_HHMMSS.
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(base_directory, timestamp)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def save_execution_log(log_data, filename="execution_log.json", directory=DEFAULT_LOG_DIR):
    """
    Save a dictionary of execution data to a JSON file in the specified directory.
    A timestamp is added automatically.
    """
    # Ensure the directory exists.
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Add a timestamp entry.
    log_data["timestamp"] = datetime.datetime.now().isoformat()
    filepath = os.path.join(directory, filename)
    with open(filepath, "w") as f:
        json.dump(log_data, f, indent=4)
    # print(f"Log saved to {filepath}")

def load_execution_log(filename="execution_log.json", directory=DEFAULT_LOG_DIR):
    """
    Load execution data from a JSON file in the specified directory and return it as a dictionary.
    """
    filepath = os.path.join(directory, filename)
    with open(filepath, "r") as f:
        return json.load(f)
