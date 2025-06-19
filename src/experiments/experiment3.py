# src/experiments/experiment3.py

# Experiment3 is about comparing the complexities of the Turing Machine’s full state vs. 
# its tape-only state, under varying transition probabilities, to see if ignoring head/state 
# bits drastically changes the measured complexity.

from pathlib import Path

from src.experiments.base_experiment import Experiment
from src.experiments.utils.loader import load_json
from src.experiments.utils.logger import log_message, log_data
from src.experiments.utils.plotter import plot_series
from src.experiments.metrics.entanglement import entanglement
from src.experiments.metrics.equanimities import (
    equanimity_importance,
    equanimity_subsets,
    equanimity_subsets_normalized,
)
from src.experiments.config import LOGS_PATH

class Experiment3(Experiment):
    """
    Compara métricas de complejidad: estado completo vs solo cinta
    bajo distintas probabilidades de transición.
    """

    def __init__(self, *, timestamp: str):
        super().__init__("Experiment3")
        self.timestamp = timestamp

        # Path to aggregated_results.json from Experiment1
        agg_path = Path(LOGS_PATH) / "experiment1" / timestamp / "aggregated_results.json"
        log_message(f"Loading original metrics from {agg_path}")
        data = load_json(agg_path)
        self.original = data["per_transition_probability"]

        # Sorted transition probabilities
        self.transition_probabilities = sorted(float(tp) for tp in self.original.keys())

        # Get configuration directory
        run_dir = Path(LOGS_PATH) / "experiment1" / timestamp
        config_dirs = [d for d in run_dir.iterdir() if d.is_dir()]
        if not config_dirs:
            raise FileNotFoundError(f"No config directories in {run_dir}")
        config_dir = config_dirs[0]

        # Desde el primer prob_dir, extraer config para proyección
        first_tp = self.transition_probabilities[0]
        prob_dir = config_dir / f"prob{first_tp:.2f}"
        tm_json = load_json(prob_dir / "turing_machines.json")
        if not tm_json:
            raise ValueError(f"No machines in {prob_dir}")
        config = tm_json[0]["config"]
        self.config = config

        # Projection on tape bits
        self.projection = list(range(config["tape_bits"]))

    def _compute_projected_metrics_for_tp(self, tp: float) -> dict:
        """
        Recompute metrics sobre the tape-projected history function for a given transition probability `tp`.
        """
        # Ubicación de turing_machines.json de Experiment1
        base = Path(LOGS_PATH) / "experiment1" / self.timestamp
        # Unique config_dir
        cfg_dir = next(d for d in base.iterdir() if d.is_dir())
        machines = load_json(cfg_dir / f"prob{tp:.2f}" / "turing_machines.json")

        entanglement_values = []
        eq_importance_values = []
        eq_subsets_values = []
        eq_subsets_normalized_values = []

        for tm in machines:
            full_hf = tm["history_function"]
            proj_hf = [full_hf[i] for i in self.projection] # Tape projection
            # Compute Metrics
            entanglement_values.append(entanglement(proj_hf))
            eq_importance_values.append(equanimity_importance(proj_hf))
            eq_subsets_values.append(equanimity_subsets(proj_hf))
            eq_subsets_normalized_values.append(equanimity_subsets_normalized(proj_hf))

        n = len(entanglement_values)
        
        return {
            "avg_entanglement": sum(entanglement_values) / n,
            "avg_equanimities": {
                "importance":         sum(eq_importance_values) / n,
                "subsets":            sum(eq_subsets_values) / n,
                "subsets_normalized": sum(eq_subsets_normalized_values) / n,
            }
        }

    def run_experiment(self):
        log_message(f"Running Experiment3")

        comparison = {}
        full_ent, proj_ent = [], []
        full_imp, proj_imp = [], []
        full_sub, proj_sub = [], []
        full_norm, proj_norm = [], []

        for tp in self.transition_probabilities:
            orig = self.original[str(tp)]
            # Originals
            fe = orig["avg_entanglement"]
            fi = orig["avg_equanimities"]["importance"]
            fs = orig["avg_equanimities"]["subsets"]
            fn = orig["avg_equanimities"]["subsets_normalized"]

            # Projected
            proj = self._compute_projected_metrics_for_tp(tp)
            pe = proj["avg_entanglement"]
            pi = proj["avg_equanimities"]["importance"]
            ps = proj["avg_equanimities"]["subsets"]
            pn = proj["avg_equanimities"]["subsets_normalized"]

            comparison[tp] = {
                "original":  orig,
                "tape_projected": proj
            }

            full_ent.append(fe);   proj_ent.append(pe)
            full_imp.append(fi);   proj_imp.append(pi)
            full_sub.append(fs);   proj_sub.append(ps)
            full_norm.append(fn);  proj_norm.append(pn)

        # Save comparison
        log_data(
            data=comparison,
            filename="comparacion_estado_completo_vs_solo_cinta.json",
            directory=self.run_dir
        )

        # Define labels
        total_bits = self.config['tape_bits'] + self.config['head_bits'] + self.config['state_bits']
        tape_bits = self.config['tape_bits']
        labels = [f"{total_bits} bits", f"{tape_bits} bits de cinta"]
        
        x = self.transition_probabilities

        # 1) Entanglement
        plot_series(
            x=x,
            ys=[full_ent, proj_ent],
            labels=labels,
            title="Probabilidades de transición vs Comparativa de enredos",
            xlabel="Probabilidad de transición",
            ylabel="Enredo",
            filename="enredo_comparacion.png",
            directory=self.plot_directory,
        )
        # 2) Equanimity (importance)
        plot_series(
            x=x,
            ys=[full_imp, proj_imp],
            labels=labels,
            title="Probabilidades de transición vs Comparativa de ecuanimidad por importancia",
            xlabel="Probabilidad de transición",
            ylabel="Ecuanimidad",
            filename="ecuanimidad_importancia_comparacion.png",
            directory=self.plot_directory,
        )
        # 3) Equanimity (subsets)
        plot_series(
            x=x,
            ys=[full_sub, proj_sub],
            labels=labels,
            title="Probabilidades de transición vs Comparativa de ecuanimidad por subconjuntos",
            xlabel="Probabilidad de transición",
            ylabel="Ecuanimidad",
            filename="ecuanimidad_subconjuntos_comparacion.png",
            directory=self.plot_directory,
        )
        # 3) Equanimity (subsets normalized)
        plot_series(
            x=x,
            ys=[full_norm, proj_norm],
            labels=labels,
            title="Probabilidades de transición vs Comparativa de ecuanimidad por subconjuntos normalizada",
            xlabel="Probabilidad de transición",
            ylabel="Ecuanimidad",
            filename="ecuanimidad_normalizada_comparacion.png",
            directory=self.plot_directory,
        )


def run_experiment():
    timestamp = "20250618_122014" # T1H0S2
    # timestamp = "20250618_122025" # T2H1S2
    # timestamp = "20250618_122045" # T4H2S2
    # timestamp = "20250618_122125" # T8H3S2
    
    # timestamp = "20250618_181419" # T1H0S3
    # timestamp = "20250618_181443" # T2H1S3
    # timestamp = "20250618_181456" # T4H2S3
    # timestamp = "20250618_181806" # T8H3S3
    exp = Experiment3(timestamp=timestamp)
    exp.run_experiment()

if __name__ == "__main__":
    run_experiment()
