import time
import numpy as np
from src.experiments.base_experiment import Experiment
from src.experiments.utils.logger import log_message, log_data
from src.experiments.utils.plotter import plot_heatmap
from src.experiments.utils.computing import measure_minimal_dnf, generate_long_tm_transition_function
from src.tm.utils import (
    generate_turing_machine,
    get_history_function,
    get_num_steps,
    serialize_turing_machine,
    generate_random_binary_input,
)

class Experiment6(Experiment):
    """
    Generate “long‐running” Turing Machines via a custom transition generator,
    measure their execution lengths and minimal‐DNF complexities, then
    bucket and plot a categorical heatmap (steps vs. literals and minterms).
    """

    def __init__(
        self,
        config,
        halting_fraction_range,
        *,
        complexity_binning: str = "absolute"
    ):
        """
        :param config: dict with tape_bits, head_bits, state_bits
        :param halting_fraction_range: iterable of halting_fraction values
        :param complexity_binning: one of
            - "absolute": use theoretical bounds (0…n·2^{n-1} for literals,
                          0…2^{n-1} for minterms)
            - "observed": use observed min/max from this run
            - "percentile": use 0%,33.3%,66.7%,100% percentiles of observed data
        """
        super().__init__("Experiment6")
        self.config = config
        self.hf_range = halting_fraction_range
        if complexity_binning not in ("absolute", "observed", "percentile"):
            raise ValueError("complexity_binning must be 'absolute', 'observed' or 'percentile'")
        self.binning = complexity_binning

    def run_experiment(self):
        log_message(f"[Experiment6] Running config {self.config} with '{self.binning}' bins")
        num_per_hf = int(self.num_experiments_per_config / len(self.hf_range))

        tape  = self.config["tape_bits"]
        head  = self.config["head_bits"]
        state = self.config["state_bits"]
        n     = tape + head + state

        all_tms      = []
        complexities = []  # Dicts {hf, steps, minterms, literals}

        for hf in self.hf_range:
            for _ in range(num_per_hf):
                tf = generate_long_tm_transition_function(
                    num_states=2 ** state, halting_fraction=hf
                )
                inp = generate_random_binary_input(tape_length=tape)
                tm  = generate_turing_machine(
                    config=self.config,
                    binary_input=inp,
                    transition_function=tf
                )
                tm.run()
                steps     = get_num_steps(tm)
                history   = get_history_function(tm)
                minterms, literals = measure_minimal_dnf(history)

                all_tms.append(tm)
                complexities.append({
                    "halting_fraction": hf,
                    "steps":    steps,
                    "minterms": minterms,
                    "literals": literals
                })

        # Save raw and per‐TM data
        log_data(
            data=[serialize_turing_machine(tm) for tm in all_tms],
            filename="turing_machines.json",
            directory=self.run_dir
        )
        log_data(
            data=complexities,
            filename="tm_complexities.json",
            directory=self.run_dir
        )

        # Aggregate per‐hf
        agg = {}
        for hf in self.hf_range:
            group = [c for c in complexities if c["halting_fraction"] == hf]
            steps = [c["steps"]    for c in group]
            lits  = [c["literals"] for c in group]
            mins  = [c["minterms"] for c in group]
            k = len(group)
            agg[hf] = {
                "avg_steps":    sum(steps)/k,
                "max_steps":    max(steps),
                "avg_literals": sum(lits)/k,
                "max_literals": max(lits),
                "avg_minterms": sum(mins)/k,
                "max_minterms": max(mins),
            }
        log_data(
            data=agg,
            filename="experiment_results.json",
            directory=self.run_dir
        )

        # Prepare arrays for binning
        all_steps    = np.array([c["steps"]    for c in complexities], dtype=float)
        all_lits     = np.array([c["literals"] for c in complexities], dtype=float)
        all_minterms = np.array([c["minterms"] for c in complexities], dtype=float)

        # Decide bins for steps (always min y max observed)
        bins_steps = np.linspace(
            all_steps.min() - 0.5,
            all_steps.max() + 0.5,
            4
        )

        # Decide bins for literals & minterms based on strategy
        def make_bins(data, absolute_max):
            if self.binning == "absolute":
                return np.linspace(0, absolute_max, 4)
            if self.binning == "observed":
                mn, mx = data.min(), data.max()
                return np.linspace(mn - 0.5, mx + 0.5, 4)
            # Percentile
            return np.percentile(data, [0, 33.33, 66.67, 100.0])

        bins_literals  = make_bins(all_lits,  n * 2 ** (n - 1))
        bins_minterms  = make_bins(all_minterms, 2 ** (n - 1))

        labels_x = ["Corta", "Media", "Larga"]
        labels_y = ["Fácil", "Moderada", "Difícil"]

        # Heatmap: execution length vs literals
        plot_heatmap(
            x=all_steps,
            y=all_lits,
            hexbin=False,
            bins_x=bins_steps,
            bins_y=bins_literals,
            class_labels_x=labels_x,
            class_labels_y=labels_y,
            title=f"Longitud de ejecución vs Complejidad (n={n})",
            xlabel="Longitud de ejecución",
            ylabel="Literales de la FND mínima",
            filename="heatmap_steps_literals.png",
            directory=self.plot_directory
        )

        # Heatmap: execution length vs minterms
        plot_heatmap(
            x=all_steps,
            y=all_minterms,
            hexbin=False,
            bins_x=bins_steps,
            bins_y=bins_minterms,
            class_labels_x=labels_x,
            class_labels_y=labels_y,
            title=f"Longitud de ejecución vs Complejidad (n={n})",
            xlabel="Longitud de ejecución",
            ylabel="Minterms de la FND mínima",
            filename="heatmap_steps_minterms.png",
            directory=self.plot_directory
        )

        log_message("[Experiment6] done.")


def run_experiment():
    configs = [
        {"tape_bits": 1, "head_bits": 0, "state_bits": 2},
        {"tape_bits": 2, "head_bits": 1, "state_bits": 2},
        {"tape_bits": 4, "head_bits": 2, "state_bits": 2},
        {"tape_bits": 8, "head_bits": 3, "state_bits": 2},
        {"tape_bits": 1, "head_bits": 0, "state_bits": 3},
        {"tape_bits": 2, "head_bits": 1, "state_bits": 3},
        {"tape_bits": 4, "head_bits": 2, "state_bits": 3},
        {"tape_bits": 8, "head_bits": 3, "state_bits": 3},
    ]
    
    # halting_fracs = np.linspace(0.1, 0.9, 9)
    halting_fracs = np.linspace(0.1, 0.1, 1)

    for cfg in configs:
        exp = Experiment6(
            config=cfg,
            halting_fraction_range=halting_fracs,
            complexity_binning="absolute"
        )
        exp.run_experiment()
        time.sleep(3)


if __name__ == "__main__":
    run_experiment()
