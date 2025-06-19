from pathlib import Path
import numpy as np

from src.experiments.base_experiment import Experiment
from src.experiments.config import LOGS_PATH
from src.experiments.utils.loader import load_json
from src.experiments.utils.logger import log_message, log_data
from src.experiments.utils.plotter import plot_heatmap
from src.experiments.utils.computing import equal_width_bins

class Experiment5(Experiment):
    """
    Classify TM runs into 3×3 bins of (steps vs. literals) and
    plot a categorical heatmap for ONE timestamp’s single config.
    """
    def __init__(self, timestamp: str):
        super().__init__("Experiment5")
        self.timestamp = timestamp
        # locate the single config directory under experiment1/<timestamp>/
        exp1_root = Path(LOGS_PATH) / "experiment1" / timestamp
        if not exp1_root.exists():
            raise FileNotFoundError(f"Experiment1 run not found: {exp1_root}")
        cfg_dirs = [d for d in exp1_root.iterdir() if d.is_dir()]
        if not cfg_dirs:
            raise RuntimeError(f"No config subfolder under {exp1_root}")
        if len(cfg_dirs) > 1:
            log_message(f"Found multiple configs under {exp1_root}, using first", prefix="[WARNING]")
        self.cfg_dir = cfg_dirs[0]

    def run_experiment(self):
        log_message(f"Experiment5 classifying runs for timestamp={self.timestamp}")
        # gather all steps & literals from each prob*/ folder
        steps = []
        lits  = []
        for prob_dir in sorted(self.cfg_dir.glob("prob*")):
            tm_list   = load_json(prob_dir / "turing_machines.json")
            comp_list = load_json(prob_dir / "tm_complexities.json")
            # assume they align 1:1
            for tm, comp in zip(tm_list, comp_list):
                steps.append(tm["num_steps"])
                lits.append(comp["literals"])

        total = len(steps)
        if total == 0:
            raise RuntimeError("No TMs found to classify!")

        # Build 3 equal‐width bins in steps & literals
        bins_steps = equal_width_bins(steps, 3)
        bins_lits  = equal_width_bins(lits,  3)

        # Class labels
        step_labels = ["Corta", "Media", "Larga"]
        lit_labels  = ["Fácil", "Moderada", "Difícil"]

        # Joint histogram 3×3
        H, xedges, yedges = np.histogram2d(steps, lits, bins=[bins_steps, bins_lits])

        # Save summary in JSON 
        summary = {
            "timestamp": self.timestamp,
            "total_TMs": total,
            "step_distribution": {
                step_labels[i]: int(H[i, :].sum()) for i in range(3)
            },
            "complexity_distribution": {
                lit_labels[j]: int(H[:, j].sum()) for j in range(3)
            },
            "joint_counts": {
                step_labels[i]: {
                    lit_labels[j]: int(H[i, j]) for j in range(3)
                }
                for i in range(3)
            }
        }
        
        log_data(
            data=summary,
            filename="experiment5_results.json",
            directory=self.run_dir
        )

        # Plot categorical heatmap
        title = f"Longitud de ejecución vs complejidad (FND) — n = {int(sum(self._extract_config_bits()))} bits"
        plot_heatmap(
            x=steps,
            y=lits,
            hexbin=False,
            bins_x=bins_steps,
            bins_y=bins_lits,
            class_labels_x=step_labels,
            class_labels_y=lit_labels,
            title=title,
            xlabel="Número de pasos de ejecución",
            ylabel="Complejidad (literales de la FND)",
            filename="heatmap_length_complexity.png",
            directory=self.plot_directory
        )

    def _extract_config_bits(self):
        """
        Helper to pull (tape, head, state) from the very first TM record.
        """
        # look in the first prob folder
        first_prob = next(self.cfg_dir.glob("prob*"))
        tm = load_json(first_prob / "turing_machines.json")[0]
        cfg = tm["config"]
        return cfg["tape_bits"], cfg["head_bits"], cfg["state_bits"]


def run_experiment():
    
    timestamp = "20250618_122014" # T1H0S2
    # timestamp = "20250618_122025" # T2H1S2
    # timestamp = "20250618_122045" # T4H2S2
    # timestamp = "20250618_122125" # T8H3S2
    
    # timestamp = "20250618_181419" # T1H0S3
    # timestamp = "20250618_181443" # T2H1S3
    # timestamp = "20250618_181456" # T4H2S3
    # timestamp = "20250618_181806" # T8H3S3
    
    exp = Experiment5(timestamp=timestamp)
    exp.run_experiment()


if __name__ == "__main__":
    run_experiment()
