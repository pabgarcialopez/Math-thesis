import os
import json
import datetime

from src.experiments.config import SHOULD_LOG

# -----------------------------------------------------------------------------
# Directory logic
# -----------------------------------------------------------------------------

def create_subdirectory(name=None, *, parent, timestamped=False):
                
        subdir_name = ""
        if name: subdir_name += name
        if timestamped: subdir_name += datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        path = os.path.join(parent, subdir_name)
        
        # If directory already exists, don't create it again.
        if os.path.exists(path): return path
        
        # Create subdirectory
        os.makedirs(path, exist_ok=True)
        
        return path
    
# -----------------------------------------------------------------------------
# Data logging
# -----------------------------------------------------------------------------
    
def log_message(message, prefix="[INFO]"):
    print(f"{prefix} {message}")

def log_data(*, data, filename, directory):
    if SHOULD_LOG:
        assert os.path.exists(directory)
        filepath = os.path.join(directory, filename)
        with open(filepath, "w") as file:
            json.dump(data, file, indent=4, ensure_ascii=False)