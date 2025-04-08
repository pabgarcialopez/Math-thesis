# src/experiments/base_experiment.py

import os
from src.config import LOGS_PATH, SHOULD_LOG, SHOULD_PLOT
from src.experiments.utils.logger import create_timestamped_dir, save_execution_log
from src.tm.utils import generate_turing_machines
from src.tm.utils import serialize_turing_machine

class BaseExperiment:
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        base_dir = os.path.join(LOGS_PATH, self.experiment_name)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        self.run_dir = create_timestamped_dir(base_dir)

    def log_message(self, message, prefix="[INFO]"):
        print(f"{prefix} {message}")

    def log_data(self, data, filename="experiment_log.json", directory=None):
        """
        Logs 'data' to a JSON file. By default, logs to self.run_dir,
        but if 'directory' is provided, logs to that subdirectory.
        """
        if directory is None:
            directory = self.run_dir
        save_execution_log(data, filename=filename, directory=directory)

    def create_config_subdir(self, config_label):
        """
        Creates and returns a subdirectory inside self.run_dir,
        labeled with 'config_label'.
        """
        config_dir = os.path.join(self.run_dir, config_label)
        os.makedirs(config_dir, exist_ok=True)
        return config_dir

    def save_plot(self, figure, filename, directory=None):
        """
        Saves a Matplotlib figure to 'filename'. By default in self.run_dir,
        or in 'directory' if provided.
        """
        if directory is None:
            directory = self.run_dir
        figure.savefig(os.path.join(directory, filename), bbox_inches='tight')

    def should_log(self):
        return SHOULD_LOG

    def run_experiment(self):
        raise NotImplementedError("Subclasses must implement run_experiment()")
    
    def run_and_collect(
        self,
        config,
        probabilities,
        num_machines,
        metric_callback,
        aggregate_callback=None,
        log_each_machine=True,
        directory=None
    ):
        """
        Iterates over 'probabilities'. For each probability 'p':
          1) Creates 'n_machines' Turing Machines via 'create_machine_fn(base_config, p)'
          2) Runs each machine, collects metrics via 'metric_callback'
          3) Optionally logs each machine's result if 'log_each_machine' is True
          4) Aggregates metrics if 'aggregate_callback' is provided

        Returns a list of dicts, each with:
          {
            "probability": p,
            "metrics_list": [ ... raw metrics for each machine ... ],
            "aggregated": ... result of aggregate_callback(...) if provided ...
          }
        """
        if directory is None:
            directory = self.run_dir

        results = []
        for idx, probability in enumerate(probabilities):
            metrics_list = []
            turing_machines = generate_turing_machines(num_machines, config, probability)
            for i, tm in enumerate(turing_machines):
                tm.run()
                metrics = metric_callback(tm)
                metrics_list.append(metrics)

                if self.should_log() and log_each_machine:
                    filename = f"prob_{idx+1}_machine_{i+1}.json"
                    self.log_data({
                        "turing_machine": serialize_turing_machine(tm),
                        "metrics": metrics
                    }, filename=filename, directory=directory)

            aggregated = None
            if aggregate_callback is not None:
                aggregated = aggregate_callback(metrics_list)

            results.append({
                "probability": float(probability),
                "metrics_list": metrics_list,
                "aggregated": aggregated
            })

        return results
