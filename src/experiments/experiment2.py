# src/experiments/experiment2.py
import os
import glob
import json
from collections import defaultdict
from src.plotters import plot_frequency_histogram
from src.experiments.base_experiment import BaseExperiment, project_history_to_boolean_function
from src.config import LOGS_DIR  # Import the logs directory from config

def load_dataset(filepath):
    """
    Loads a dataset of functions from a text file.
    Each line should have the format: <function_code> <circuit_size>
    """
    dataset = {}
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                func_code = int(parts[0])
                circuit_size = int(parts[1])
                dataset[func_code] = circuit_size
    except Exception as e:
        print(f"Error loading dataset: {e}")
    return dataset

def load_json_logs(log_directory):
    """
    Recursively loads all JSON log files from the given directory.
    Returns a list of dictionaries.
    """
    logs = []
    if not os.path.exists(log_directory):
        return logs
    pattern = os.path.join(log_directory, '**', '*.json')
    files = glob.glob(pattern, recursive=True)
    for filepath in files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                logs.append(data)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
    return logs

def collect_projected_functions(log_directory, config_choice="final"):
    """
    For each log file, extracts the projected function if present;
    otherwise computes it from the configuration history.
    Returns a dictionary mapping projected function (int) to its occurrence count.
    """
    logs = load_json_logs(log_directory)
    freq_dict = defaultdict(int)
    for log in logs:
        if "projected_function" in log:
            func_val = log["projected_function"]
        elif "config_history" in log:
            history = log["config_history"]
            func_val = project_history_to_boolean_function(history)
        else:
            continue
        freq_dict[func_val] += 1
    return dict(freq_dict)

def analyze_representation(freq_dict, dataset):
    """
    Groups the frequency dictionary by circuit size using the dataset.
    Returns a dictionary where keys are circuit sizes and values are total frequency.
    """
    grouped_freq = defaultdict(int)
    missing = 0
    for func, count in freq_dict.items():
        if func in dataset:
            size = dataset[func]
            grouped_freq[size] += count
        else:
            grouped_freq["unknown"] += count
            missing += count
    if missing:
        print(f"[WARNING] {missing} functions from experiments were not found in the dataset.")
    return dict(grouped_freq)

def compute_ratios(grouped_freq, dataset):
    """
    Calculates and prints the ratio (experiment appearances / number of functions in dataset)
    for each circuit size.
    """
    dataset_counts = defaultdict(int)
    for func, size in dataset.items():
        dataset_counts[size] += 1

    print("\nRatio (experiment appearances / number of functions in dataset):")
    for size, freq in sorted(grouped_freq.items(), key=lambda x: (x[0] if isinstance(x[0], int) else 9999)):
        if isinstance(size, int) and size in dataset_counts and dataset_counts[size] > 0:
            ratio = freq / dataset_counts[size]
            print(f"Size {size}: {freq} / {dataset_counts[size]} = {ratio*100:.8f}%")
        else:
            print(f"Size {size}: {freq} (cannot compute ratio)")

# Define the dataset path (moved to the top-level data folder)
DATASET_PATH = os.path.join("data", "dataset_n5_10_puertas.txt")

class Experiment2(BaseExperiment):
    def __init__(self):
        # We call the BaseExperiment constructor with "experiment2" 
        # but we won't use its run_dir because we want to analyze experiment1 logs.
        super().__init__("experiment2")

    def run_experiment(self):
        # Instead of using self.run_dir, we point to the experiment1 logs.
        logs_path = os.path.join(LOGS_DIR, "experiment1")
        self.log_message(f"Analyzing logs from: {logs_path}")
        freq_dict = collect_projected_functions(logs_path, config_choice="final")
        total_experiments = sum(freq_dict.values())
        distinct_functions = len(freq_dict)
        self.log_message(f"Obtained {distinct_functions} distinct projected functions from {total_experiments} experiments.")

        dataset = load_dataset(DATASET_PATH)
        self.log_message(f"Loaded dataset with {len(dataset)} functions.")

        grouped_freq = analyze_representation(freq_dict, dataset)
        for size, freq in sorted(grouped_freq.items(), key=lambda x: (x[0] if isinstance(x[0], int) else 9999)):
            self.log_message(f"Circuit size {size}: {freq} occurrences.")

        compute_ratios(grouped_freq, dataset)
        plot_frequency_histogram(grouped_freq)
        self.log_message("Experiment 2 analysis completed.")

def run_experiment():
    exp = Experiment2()
    exp.run_experiment()

if __name__ == "__main__":
    run_experiment()