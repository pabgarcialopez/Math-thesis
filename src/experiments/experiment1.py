# src/experiments/experiment1.py

import os
import numpy as np
from tqdm import tqdm

from src.config import NUM_EXPERIMENTS, MIN_PROB, MAX_PROB, NUM_PROBS
from src.experiments.base_experiment import BaseExperiment
from src.experiments.utils.plotters import (
    plot_probabilities_vs_metrics,
    plot_equanimity_vs_entanglement_heatmap
)
from src.experiments.metrics.equanimities import (
    equanimity_importance,
    equanimity_subsets,
    equanimity_subsets_normalized
)
from src.experiments.metrics.entanglement import entanglement
from src.tm.utils import get_history_function


def metric_callback(tm):
    N = tm.config_bits  # e.g. tape_bits + head_bits + state_bits

    history_func = get_history_function(tm)
    eq_imp = equanimity_importance(history_func, N)
    eq_sub = equanimity_subsets(history_func, N)
    eq_sub_norm = equanimity_subsets_normalized(history_func, N)
    ent = entanglement(history_func, N)
    return {
        "eq_imp": eq_imp,
        "eq_sub": eq_sub,
        "eq_sub_norm": eq_sub_norm,
        "ent": ent,
        "outcome": tm.outcome,
    }

def aggregate_callback(metrics_list):
    if not metrics_list:
        return {}
    n = len(metrics_list)
    eq_imp_vals = [m["eq_imp"] for m in metrics_list]
    eq_sub_vals = [m["eq_sub"] for m in metrics_list]
    eq_sub_norm_vals = [m["eq_sub_norm"] for m in metrics_list]
    ent_vals = [m["ent"] for m in metrics_list]
    return {
        "mean_eq_imp": sum(eq_imp_vals)/n,
        "mean_eq_sub": sum(eq_sub_vals)/n,
        "mean_eq_sub_norm": sum(eq_sub_norm_vals)/n,
        "mean_ent": sum(ent_vals)/n
    }

class Experiment1(BaseExperiment):
    def __init__(self):
        super().__init__("experiment1")
        self.configs = [
            {"tape_bits": 5, "head_bits": 3, "state_bits": 2},
            # {"tape_bits": 7, "head_bits": 3, "state_bits": 2}
        ]

    def run_experiment(self):
        self.log_message(f"Running Experiment1. Logs in: {self.run_dir}")
        probabilities = np.linspace(MIN_PROB, MAX_PROB, NUM_PROBS)

        for config in self.configs:
            config_label = f"tape{config['tape_bits']}_head{config['head_bits']}_state{config['state_bits']}"
            config_dir = self.create_config_subdir(config_label)

            results = self.run_and_collect(
                config=config,
                probabilities=probabilities,
                num_machines=NUM_EXPERIMENTS,
                metric_callback=metric_callback,
                aggregate_callback=aggregate_callback,
                directory=config_dir,
            )

            self.process_experiment_results(config, probabilities, results, config_dir)

    def process_experiment_results(self, config, probabilities, results, config_dir):
        """
        Extracts aggregated metrics, plots them, and logs them if desired.
        Keeps run_experiment simpler and more readable.
        """
        avg_eq_imp = []
        avg_eq_sub = []
        avg_eq_sub_norm = []
        avg_ent = []

        heatmap_eq_imp = []
        heatmap_ent = []

        # results is a list of dicts: 
        # [
        #   { "prob": p, "metrics_list": [...], "aggregated": {...} },
        #   ...
        # ]
        for r in results:
            agg = r["aggregated"]
            if agg:
                avg_eq_imp.append(agg["mean_eq_imp"])
                avg_eq_sub.append(agg["mean_eq_sub"])
                avg_eq_sub_norm.append(agg["mean_eq_sub_norm"])
                avg_ent.append(agg["mean_ent"])

            # For heatmap, gather raw eq_imp/ent
            for m in r["metrics_list"]:
                heatmap_eq_imp.append(m["eq_imp"])
                heatmap_ent.append(m["ent"])

        if avg_eq_imp:
            metrics_plot_path = os.path.join(config_dir, "metrics_vs_transition_probability.png")
            plot_probabilities_vs_metrics(
                probabilities,
                avg_eq_imp,
                avg_eq_sub,
                avg_eq_sub_norm,
                avg_ent,
                save_path=metrics_plot_path
            )
        if heatmap_eq_imp:
            heatmap_plot_path = os.path.join(config_dir, "heatmap_eq_imp_vs_ent.png")
            plot_equanimity_vs_entanglement_heatmap(
                heatmap_eq_imp,
                heatmap_ent,
                bins=25,
                save_path=heatmap_plot_path
            )

        # Log aggregated data if desired
        if self.should_log():
            aggregated_log = {
                "config": config,
                "probabilities": list(probabilities),
                "avg_eq_imp": avg_eq_imp,
                "avg_eq_sub": avg_eq_sub,
                "avg_eq_sub_norm": avg_eq_sub_norm,
                "avg_ent": avg_ent
            }
            self.log_data(aggregated_log, filename="aggregated_metrics.json", directory=config_dir)


def run_experiment():
    exp = Experiment1()
    exp.run_experiment()

if __name__ == "__main__":
    run_experiment()
