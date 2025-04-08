# This experiment tries to see how well the projected 5 bit to 1 boolean functions 
# are represented in the dataset of boolean functions from 5 bits to 1 computed with circuits
# from 1 to 10 gates. The apparent conclusion is that most projected functions are computed with
# a circuit with a higher number of gates, maybe showing that the projection does not work well,
# since with experiment 1 we have seen that the 10 bit to 1 functions seem "simple" because they
# have low values of equanimity and entanglement.

# src/experiments/experiment2.py

import os
import glob
import json
from collections import defaultdict

from src.config import LOGS_PATH, DATA_PATH
from src.experiments.base_experiment import BaseExperiment
from src.experiments.utils.logger import save_execution_log
from src.experiments.utils.plotters import plot_frequency_histogram
from src.tm.utils import get_projected_history_function

class Experiment2(BaseExperiment):
    """
    This experiment analyzes logs produced by Experiment1, collecting the "projected functions"
    and comparing them to a dataset of 5-bit boolean functions. It then plots and logs the
    representation analysis (histogram, etc.).
    """

    def __init__(self):
        super().__init__("experiment2")
        self.experiment1_logs_path = os.path.join(LOGS_PATH, "experiment1")
        self.dataset_path = os.path.join(DATA_PATH, "dataset_n5_10_puertas.txt")

    def run_experiment(self):
        self.log_message(f"Experiment2 analyzing logs from: {self.experiment1_logs_path}")

        # 1) Collect projected functions from experiment1 logs
        freq_dict = self.collect_projected_functions(self.experiment1_logs_path)
        total_experiments = sum(freq_dict.values())
        distinct_functions = len(freq_dict)
        self.log_message(
            f"Obtained {distinct_functions} distinct projected functions "
            f"from {total_experiments} experiments."
        )

        # 2) Load dataset of known functions
        dataset = self.load_dataset(self.dataset_path)
        self.log_message(f"Loaded dataset with {len(dataset)} functions.")

        # 3) Analyze how frequently each function size appears
        grouped_freq = self.analyze_representation(freq_dict, dataset)

        # 4) Calculate ratios
        ratio_details, summary_data = self.calculate_ratios(freq_dict, dataset, grouped_freq)
        self.log_message("\nUnique Ratios (unique functions in logs / number of functions in dataset):")
        for size, details in sorted(ratio_details.items(), key=lambda x: (x[0] if isinstance(x[0], int) else 9999)):
            if details["ratio"] is not None:
                self.log_message(
                    f"Size {size}: {details['unique_count']} / {details['dataset_count']} "
                    f"= {details['ratio']*100:.8f}%"
                )
            else:
                self.log_message(f"Size {size}: {details['unique_count']} (cannot compute ratio)")

        # 5) Save final summary log if desired
        if self.should_log():
            self.log_data(summary_data, filename="experiment2_summary.json")
            self.log_message("Summary data saved as experiment2_summary.json")

        # 6) Plot a histogram of circuit sizes
        histogram_path = os.path.join(self.run_dir, "final_histogram.png")
        plot_frequency_histogram(grouped_freq, save_path=histogram_path)
        self.log_message(f"Final histogram saved to: {histogram_path}")

        self.log_message("Experiment 2 analysis completed.")

    # --------------------------------------------------------------------------
    # Helper Methods
    # --------------------------------------------------------------------------

    def load_dataset(self, filepath):
        """
        Loads a dataset of functions from a text file.
        Each line: <function_code> <circuit_size>
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
            self.log_message(f"Error loading dataset: {e}", prefix="[WARNING]")
        return dataset

    def load_json_logs(self, log_directory):
        """
        Recursively loads all .json files from the given directory.
        Returns a list of dictionaries (the parsed JSON).
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
                self.log_message(f"Error reading {filepath}: {e}", prefix="[WARNING]")
        return logs

    def collect_projected_functions(self, log_directory):
        """
        For each JSON log, extracts the 'projected_function' if present,
        or computes it from 'config_history' if not. Returns a dict {func_val -> count}.
        """
        from collections import defaultdict
        freq_dict = defaultdict(int)
        logs = self.load_json_logs(log_directory)
        for log in logs:
            if "turing_machine" not in log: continue
            config_history = log['turing_machine']['config_history']
            tape_length = log['turing_machine']['tape_bits']
            assert tape_length == 5, "Projection must be onto 5 bits for experiment 2"
            # Get the tape projection of the turing machine
            projected_history_func = get_projected_history_function(config_history, projection=range(tape_length))
            key = int("".join(map(str, projected_history_func)), 2)
            freq_dict[key] += 1
        return dict(freq_dict)

    def analyze_representation(self, freq_dict, dataset):
        """
        Groups freq_dict by circuit size using the dataset. 
        Returns dict {circuit_size -> count}.
        """
        from collections import defaultdict
        unique_by_size = defaultdict(int)
        for func in freq_dict.keys():
            if func in dataset:
                size = dataset[func]
                unique_by_size[size] += 1
            else:
                unique_by_size["unknown"] += 1

        if "unknown" in unique_by_size:
            self.log_message(
                f"[WARNING] {unique_by_size['unknown']} functions from logs not found in the dataset."
            )
        return dict(unique_by_size)

    def calculate_ratios(self, freq_dict, dataset, grouped_freq):
        """
        Calculates ratio of how many unique functions from the logs
        appear in the dataset, grouped by circuit size.
        Returns (ratio_details, summary_data).
        """
        from collections import defaultdict

        # Build a map of circuit sizes -> set of distinct functions in logs
        unique_by_size_set = defaultdict(set)
        for func_val in freq_dict.keys():
            if func_val in dataset:
                size = dataset[func_val]
                unique_by_size_set[size].add(func_val)
            else:
                unique_by_size_set["unknown"].add(func_val)

        # Count how many exist in the dataset for each circuit size
        dataset_counts = defaultdict(int)
        for func_code, size in dataset.items():
            dataset_counts[size] += 1

        ratio_details = {}
        for size, func_set in unique_by_size_set.items():
            count_unique = len(func_set)
            dataset_count = dataset_counts.get(size, 0)
            if isinstance(size, int) and dataset_count > 0:
                ratio = count_unique / dataset_count
                ratio_details[size] = {
                    "unique_count": count_unique,
                    "dataset_count": dataset_count,
                    "ratio": ratio
                }
            else:
                ratio_details[size] = {
                    "unique_count": count_unique,
                    "dataset_count": dataset_count,
                    "ratio": None
                }

        total_experiments = sum(freq_dict.values())
        distinct_functions = len(freq_dict)

        summary_data = {
            "total_experiments": total_experiments,
            "distinct_projected_functions": distinct_functions,
            "dataset_size": len(dataset),
            "grouped_frequency": grouped_freq,
            "unique_by_size": {size: len(func_set) for size, func_set in unique_by_size_set.items()},
            "dataset_counts": dict(dataset_counts),
            "unique_ratios": {
                size: (info["ratio"] if info["ratio"] is not None else "N/A")
                for size, info in ratio_details.items()
            }
        }
        return ratio_details, summary_data


def run_experiment():
    exp = Experiment2()
    exp.run_experiment()

if __name__ == "__main__":
    run_experiment()
