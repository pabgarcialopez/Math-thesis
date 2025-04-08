# src/experiments/experiment3.py

# Experiment3 is about comparing the complexities of the Turing Machine’s full state vs. 
# its tape-only state, under varying transition probabilities, to see if ignoring head/state 
# bits drastically changes the measured complexity.

import os
import numpy as np
from tqdm import tqdm

import src.config as config
from src.experiments.base_experiment import BaseExperiment
from src.experiments.utils.plotters import (
    plot_probabilities_vs_metrics,
    plot_equanimity_vs_entanglement_heatmap,
    plot_probabilities_vs_metrics_difference,
    plot_metrics_comparison_vs_probability
)
from src.experiments.metrics.entanglement import entanglement
from src.experiments.metrics.equanimities import (
    equanimity_importance,
    equanimity_subsets,
    equanimity_subsets_normalized
)
from src.tm.utils import get_history_function, get_projected_history_function


def metric_callback(tm):
    """
    Computes metrics for both:
      - projected_history_func (the tape-only part)
      - full history_func (tape+head+state)

    Returns a dictionary containing both sets of metrics.
    """
    # Full config
    N = tm.config_bits
    full_hist = get_history_function(tm)
    eq_imp_full = equanimity_importance(full_hist, N)
    eq_sub_full = equanimity_subsets(full_hist, N)
    eq_sub_norm_full = equanimity_subsets_normalized(full_hist, N)
    ent_full = entanglement(full_hist, N)

    # Projected config (first tape_bits)
    tape_bits = tm.tape_bits
    proj_hist = get_projected_history_function(tm.config_history, projection=range(tape_bits))
    eq_imp_proj = equanimity_importance(proj_hist, tape_bits)
    eq_sub_proj = equanimity_subsets(proj_hist, tape_bits)
    eq_sub_norm_proj = equanimity_subsets_normalized(proj_hist, tape_bits)
    ent_proj = entanglement(proj_hist, tape_bits)

    return {
        "eq_imp_proj": eq_imp_proj,
        "eq_sub_proj": eq_sub_proj,
        "eq_sub_norm_proj": eq_sub_norm_proj,
        "ent_proj": ent_proj,

        "eq_imp_full": eq_imp_full,
        "eq_sub_full": eq_sub_full,
        "eq_sub_norm_full": eq_sub_norm_full,
        "ent_full": ent_full,

        "outcome": tm.outcome
    }

def aggregate_callback(metrics_list):
    """
    Averages the 'proj' and 'full' metrics over the entire list of Turing Machines.
    """
    if not metrics_list:
        return {}
    n = len(metrics_list)

    # We'll sum them up, then divide
    sums = {
        "eq_imp_proj": 0, "eq_sub_proj": 0, "eq_sub_norm_proj": 0, "ent_proj": 0,
        "eq_imp_full": 0, "eq_sub_full": 0, "eq_sub_norm_full": 0, "ent_full": 0
    }
    for m in metrics_list:
        for k in sums.keys():
            sums[k] += m[k]

    # Compute means
    for k in sums.keys():
        sums[k] /= n

    return {
        "mean_eq_imp_proj": sums["eq_imp_proj"],
        "mean_eq_sub_proj": sums["eq_sub_proj"],
        "mean_eq_sub_norm_proj": sums["eq_sub_norm_proj"],
        "mean_ent_proj": sums["ent_proj"],

        "mean_eq_imp_full": sums["eq_imp_full"],
        "mean_eq_sub_full": sums["eq_sub_full"],
        "mean_eq_sub_norm_full": sums["eq_sub_norm_full"],
        "mean_ent_full": sums["ent_full"]
    }


class Experiment3(BaseExperiment):
    """
    Compares complexities (equanimity, entanglement) of the Turing Machine’s
    full state vs. tape-only state, under varying transition probabilities.
    """

    def __init__(self):
        super().__init__("experiment3")
        self.configs = [
            {"tape_bits": 5, "head_bits": 3, "state_bits": 2},
            # {"tape_bits": 7, "head_bits": 3, "state_bits": 2},
        ]

    def run_experiment(self):
        self.log_message(f"Running Experiment3. Logs in: {self.run_dir}")
        probabilities = np.linspace(config.MIN_PROB, config.MAX_PROB, config.NUM_PROBS)

        for cfg in self.configs:
            config_label = f"tape{cfg['tape_bits']}_head{cfg['head_bits']}_state{cfg['state_bits']}"
            config_dir = self.create_config_subdir(config_label)

            # Collect metrics
            results = self.run_and_collect(
                config=cfg,
                probabilities=probabilities,
                num_machines=config.NUM_EXPERIMENTS,
                metric_callback=metric_callback,
                aggregate_callback=aggregate_callback,
                log_each_machine=True,
                directory=config_dir
            )

            # Process final results (plotting, difference arrays, etc.)
            self.process_experiment_results(cfg, probabilities, results, config_dir)

        self.log_message("Experiment 3 completed.")

    def process_experiment_results(self, cfg, probabilities, results, config_dir):
        """
        Extracts the aggregated 'proj' vs 'full' metrics, plots differences, logs data, etc.
        """
        # We'll store each probability's average proj metrics in arrays,
        # and each probability's average full metrics in arrays
        avg_eq_imp_proj = []
        avg_eq_sub_proj = []
        avg_eq_sub_norm_proj = []
        avg_ent_proj = []

        avg_eq_imp_full = []
        avg_eq_sub_full = []
        avg_eq_sub_norm_full = []
        avg_ent_full = []

        for r in results:
            agg = r["aggregated"]
            if not agg: # No data
                continue

            avg_eq_imp_proj.append(agg["mean_eq_imp_proj"])
            avg_eq_sub_proj.append(agg["mean_eq_sub_proj"])
            avg_eq_sub_norm_proj.append(agg["mean_eq_sub_norm_proj"])
            avg_ent_proj.append(agg["mean_ent_proj"])

            avg_eq_imp_full.append(agg["mean_eq_imp_full"])
            avg_eq_sub_full.append(agg["mean_eq_sub_full"])
            avg_eq_sub_norm_full.append(agg["mean_eq_sub_norm_full"])
            avg_ent_full.append(agg["mean_ent_full"])

        if avg_eq_imp_full:
            # 1) Plot full metrics vs probability
            full_metrics_path = os.path.join(config_dir, "metrics_full_vs_probability.png")
            plot_probabilities_vs_metrics(
                probabilities,
                avg_eq_imp_full,
                avg_eq_sub_full,
                avg_eq_sub_norm_full,
                avg_ent_full,
                save_path=full_metrics_path
            )

            # 2) Differences (full - proj)
            diff_eq_imp = [f - p for f, p in zip(avg_eq_imp_full, avg_eq_imp_proj)]
            diff_eq_sub = [f - p for f, p in zip(avg_eq_sub_full, avg_eq_sub_proj)]
            diff_eq_sub_norm = [f - p for f, p in zip(avg_eq_sub_norm_full, avg_eq_sub_norm_proj)]
            diff_ent = [f - p for f, p in zip(avg_ent_full, avg_ent_proj)]

            diff_path = os.path.join(config_dir, "metrics_difference_vs_probability.png")
            plot_probabilities_vs_metrics_difference(
                probabilities,
                diff_eq_imp,
                diff_eq_sub,
                diff_eq_sub_norm,
                diff_ent,
                save_path=diff_path
            )

            # 3) Grouped bar chart comparing proj vs full
            comparison_path = os.path.join(config_dir, "metrics_comparison_vs_probability.png")
            plot_metrics_comparison_vs_probability(
                probabilities,
                avg_eq_imp_proj, avg_eq_imp_full,
                avg_eq_sub_proj, avg_eq_sub_full,
                avg_eq_sub_norm_proj, avg_eq_sub_norm_full,
                avg_ent_proj, avg_ent_full,
                save_path=comparison_path
            )

        # Optionally log aggregated data
        if self.should_log():
            aggregated_data = {
                "config": cfg,
                "probabilities": probabilities.tolist(),
                "proj": {
                    "eq_imp": avg_eq_imp_proj,
                    "eq_sub": avg_eq_sub_proj,
                    "eq_sub_norm": avg_eq_sub_norm_proj,
                    "ent": avg_ent_proj
                },
                "full": {
                    "eq_imp": avg_eq_imp_full,
                    "eq_sub": avg_eq_sub_full,
                    "eq_sub_norm": avg_eq_sub_norm_full,
                    "ent": avg_ent_full
                }
            }
            self.log_data(aggregated_data, filename="aggregated_metrics.json", directory=config_dir)


def run_experiment():
    exp = Experiment3()
    exp.run_experiment()

if __name__ == "__main__":
    run_experiment()
