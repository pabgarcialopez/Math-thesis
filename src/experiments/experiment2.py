# src/experiments/experiment2.py

# This experiment tries to see how well the projected 5 bit to 1 boolean functions 
# are represented in the dataset of boolean functions from 5 bits to 1 computed with circuits
# from 1 to 10 gates. The apparent conclusion is that most projected functions are computed with
# a circuit with a higher number of gates, maybe showing that the projection does not work well,
# since with experiment 1 we have seen that the 10 bit to 1 functions seem "simple" because they
# have low values of equanimity and entanglement.

import os
import glob
import json
from collections import defaultdict
from src.plotters import plot_frequency_histogram
from src.experiments.base_experiment import BaseExperiment, project_history_to_boolean_function
from src.config import LOGS_DIR
from src.utils.logger import save_execution_log

def load_dataset(filepath):
    """
    Carga un dataset de funciones desde un archivo de texto.
    Cada línea: <function_code> <circuit_size>
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
    Carga recursivamente todos los .json en el directorio dado.
    Devuelve una lista de diccionarios.
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

def collect_projected_functions(log_directory):
    """
    Para cada log, extrae la función proyectada (si existe) o la computa desde el historial.
    Retorna dict {func_val -> count}.
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
    Agrupa la freq_dict por tamaño de circuito usando el dataset.
    Retorna dict {circuit_size -> count}.
    """
    unique_by_size = defaultdict(int)
    for func in freq_dict.keys():
        if func in dataset:
            size = dataset[func]
            unique_by_size[size] += 1
        else:
            unique_by_size["unknown"] += 1

    if "unknown" in unique_by_size:
        print(f"[WARNING] {unique_by_size['unknown']} functions from experiments were not found in the dataset.")
    return dict(unique_by_size)

DATASET_PATH = os.path.join("data", "dataset_n5_10_puertas.txt")

class Experiment2(BaseExperiment):
    def __init__(self):
        super().__init__("experiment2")

    def run_experiment(self):
        # Analiza logs de experiment1
        logs_path = os.path.join(LOGS_DIR, "experiment1")
        self.log_message(f"Analyzing logs from: {logs_path}")
        
        freq_dict = collect_projected_functions(logs_path)
        total_experiments = sum(freq_dict.values())
        distinct_functions = len(freq_dict)
        self.log_message(f"Obtained {distinct_functions} distinct projected functions from {total_experiments} experiments.")
        
        dataset = load_dataset(DATASET_PATH)
        self.log_message(f"Loaded dataset with {len(dataset)} functions.")
        
        grouped_freq = analyze_representation(freq_dict, dataset)
        for size, count in sorted(grouped_freq.items(), key=lambda x: (x[0] if isinstance(x[0], int) else 9999)):
            self.log_message(f"Circuit size {size}: {count} distinct occurrences.")
        
        # Cálculo de ratios
        unique_by_size_set = defaultdict(set)
        for func in freq_dict.keys():
            if func in dataset:
                size = dataset[func]
                unique_by_size_set[size].add(func)
            else:
                unique_by_size_set["unknown"].add(func)
        
        dataset_counts = defaultdict(int)
        for func, size in dataset.items():
            dataset_counts[size] += 1
        
        unique_ratios = {}
        ratio_details = {}
        for size, func_set in unique_by_size_set.items():
            count_unique = len(func_set)
            dataset_count = dataset_counts.get(size, 0)
            if isinstance(size, int) and dataset_count > 0:
                ratio = count_unique / dataset_count
                unique_ratios[size] = ratio
                ratio_details[size] = {
                    "unique_count": count_unique,
                    "dataset_count": dataset_count,
                    "ratio": ratio
                }
            else:
                unique_ratios[size] = None
                ratio_details[size] = {
                    "unique_count": count_unique,
                    "dataset_count": dataset_count,
                    "ratio": None
                }
        
        self.log_message("\nUnique Ratios (unique functions in logs / number of functions in dataset):")
        for size, details in sorted(ratio_details.items(), key=lambda x: (x[0] if isinstance(x[0], int) else 9999)):
            if details["ratio"] is not None:
                self.log_message(f"Size {size}: {details['unique_count']} / {details['dataset_count']} = {details['ratio']*100:.8f}%")
            else:
                self.log_message(f"Size {size}: {details['unique_count']} (cannot compute ratio)")
        
        summary_data = {
            "total_experiments": total_experiments,
            "distinct_projected_functions": distinct_functions,
            "dataset_size": len(dataset),
            "grouped_frequency": dict(grouped_freq),
            "unique_by_size": {size: len(func_set) for size, func_set in unique_by_size_set.items()},
            "dataset_counts": dict(dataset_counts),
            "unique_ratios": {size: (details["ratio"] if details["ratio"] is not None else "N/A") for size, details in ratio_details.items()}
        }
        
        from src.utils.logger import save_execution_log
        save_execution_log(summary_data, filename="experiment2_summary.json", directory=self.run_dir)
        self.log_message("Summary data saved as experiment2_summary.json")
        
        histogram_path = os.path.join(self.run_dir, "final_histogram.png")
        plot_frequency_histogram(grouped_freq, save_path=histogram_path)
        self.log_message(f"Final histogram saved to: {histogram_path}")
        self.log_message("Experiment 2 analysis completed.")

def run_experiment():
    exp = Experiment2()
    exp.run_experiment()

if __name__ == "__main__":
    run_experiment()
