# src/experiments/base_experiment.py
import os
from src.config import LOGS_DIR  # Import the logs directory from config
from src.utils.logger import get_timestamped_log_dir

class BaseExperiment:
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.run_dir = self.get_experiment_run_dir()

    def get_experiment_run_dir(self):
        """
        Creates and returns a timestamped run directory for the experiment.
        Directory structure: <LOGS_DIR>/<experiment_name>/<timestamp>/
        """
        base_dir = os.path.join(LOGS_DIR, self.experiment_name)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        run_dir = get_timestamped_log_dir(base_directory=base_dir)
        return run_dir

    def log_message(self, message, prefix="[INFO]"):
        print(f"{prefix} {message}")

    def run_experiment(self):
        """
        Abstract method. Subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement run_experiment()")

def project_history_to_boolean_function(config_history):
    """
    Given a list of 10-bit configuration strings, projects to the 5-bit tape portion and 
    creates a truth table (32 bits) where each bit is 1 if the corresponding 5-bit pattern 
    was observed, else 0. Returns the integer value of the truth table.
    """
    observed = set()
    for config in config_history:
        tape_bits = config[:5]
        observed.add(tape_bits)
    
    truth_table = ""
    for i in range(32):
        pattern = format(i, '05b')
        truth_table += "1" if pattern in observed else "0"
    
    return int(truth_table, 2)