# src/experiments/experiment6.py

import time
import numpy as np
from src.experiments.base_experiment import Experiment
from src.experiments.utils.logger import log_message, log_data
from src.experiments.utils.plotter import plot_heatmap
from src.experiments.utils.computing import measure_minimal_dnf, generate_long_tm_transition_function, counter_max_steps
from src.tm.utils import (
    generate_turing_machine,
    get_history_function,
    get_num_steps,
    serialize_turing_machine,
    generate_random_binary_input,
)
from src.tm.binary_counter import BinaryCounter

class Experiment6(Experiment):
    """
    Generate TMs via a custom tm_factory, measure lengths & minimal-DNF
    complexities, then bucket & plot a categorical heatmap.
    """
    def __init__(
        self,
        config: dict,
        halting_fraction_range: list,
        *,
        tm_factory=None,
        complexity_binning: str = "absolute"
    ):
        super().__init__("Experiment6")
        self.config = config
        self.hf_range = halting_fraction_range
        self.binning = complexity_binning
        if tm_factory is None:
            tm_factory = (
                lambda config, hf, inp: generate_turing_machine(
                    config=config,
                    binary_input=inp,
                    transition_function=generate_long_tm_transition_function(
                        num_states=2 ** config["state_bits"],
                        halting_fraction=hf,
                    ),
                )
            )
        self.tm_factory = tm_factory

    def run_experiment(self):
        log_message(
            f"[Experiment6] config={self.config}, bins={self.binning}, "
            f"factory={self.tm_factory.__name__}"
        )

        # Si el factory es de un solo TM, guardamos clasificación con datos numéricos
        if self.tm_factory.__name__ in ("binary_counter_factory", "alternating_factory"):
            tm = self.tm_factory(self.config, None, None)
            tm.run()
            steps = get_num_steps(tm)
            history = get_history_function(tm)
            terms, literals = measure_minimal_dnf(history)

            tape_bits = self.config["tape_bits"]
            n = tape_bits + self.config["head_bits"] + self.config["state_bits"]
            # máximos teóricos
            # max_steps = 2 ** tape_bits
            max_steps = counter_max_steps(tape_bits, exact=True)
            max_literals = n * 2 ** (n - 1)

            # categorías absolutas
            delta_s = max_steps / 3
            if steps <= delta_s:
                length_cat = "Corta"
            elif steps <= 2 * delta_s:
                length_cat = "Media"
            else:
                length_cat = "Larga"

            delta_l = max_literals / 3
            if literals <= delta_l:
                comp_cat = "Fácil"
            elif literals <= 2 * delta_l:
                comp_cat = "Moderada"
            else:
                comp_cat = "Difícil"

            classification = {
                "steps": steps,
                "terms": terms,
                "literals": literals,
                "max_steps_possible": max_steps,
                "max_literals_possible": max_literals,
                "length_category": length_cat,
                "complexity_category": comp_cat,
                "tm": serialize_turing_machine(tm)
            }

            log_data(
                data=classification,
                filename="tm_classification.json",
                directory=self.run_dir
            )
            log_message("[Experiment6] single‐TM classification done.")
            return

        # Si no, hacemos el flujo normal de batch + heatmaps
        num_per_hf = int(self.num_experiments_per_config / max(1, len(self.hf_range)))
        tape  = self.config["tape_bits"]
        head  = self.config["head_bits"]
        state = self.config["state_bits"]
        n     = tape + head + state

        all_tms      = []
        complexities = []

        for hf in self.hf_range:
            for _ in range(num_per_hf):
                inp = generate_random_binary_input(tape_length=tape)
                tm  = self.tm_factory(self.config, hf, inp)
                tm.run()
                steps = get_num_steps(tm)
                history = get_history_function(tm)
                terms, literals = measure_minimal_dnf(history)

                all_tms.append(tm)
                complexities.append({
                    "halting_fraction": hf,
                    "steps":    steps,
                    "terms": terms,
                    "literals": literals
                })

        # 1) Raw + per‐TM
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

        # 2) Agregados por hf
        agg = {}
        for hf in self.hf_range:
            group = [c for c in complexities if c["halting_fraction"] == hf]
            if not group:
                continue
            steps = [c["steps"]    for c in group]
            lits  = [c["literals"] for c in group]
            terms  = [c["terms"] for c in group]
            k = len(group)
            agg[hf] = {
                "avg_steps":    sum(steps)/k,
                "max_steps":    max(steps),
                "avg_literals": sum(lits)/k,
                "max_literals": max(lits),
                "avg_terms": sum(terms)/k,
                "max_terms": max(terms),
            }
        log_data(
            data=agg,
            filename="experiment_results.json",
            directory=self.run_dir
        )

        # 3) Bins and heatmaps
        all_steps = np.array([c["steps"]    for c in complexities], dtype=float)
        all_lits  = np.array([c["literals"] for c in complexities], dtype=float)
        all_terms = np.array([c["terms"] for c in complexities], dtype=float)

        bins_steps = np.linspace(0, counter_max_steps(tape), 4) if self.binning == "absolute" else \
            np.linspace(
                all_steps.min() - 0.5,
                all_steps.max() + 0.5,
                4
            )


        def make_bins(data, absolute_max):
            if self.binning == "absolute":
                return np.linspace(0, absolute_max, 4)
            if self.binning == "observed":
                mn, mx = data.min(), data.max()
                return np.linspace(mn - 0.5, mx + 0.5, 4)
            return np.percentile(data, [0, 33.33, 66.67, 100.0])

        bins_literals = make_bins(all_lits,  n * 2 ** (n - 1))
        bins_terms = make_bins(all_terms, 2 ** (n - 1))

        labels_x = ["Corta", "Media", "Larga"]
        labels_y = ["Fácil", "Moderada", "Difícil"]

        plot_heatmap(
            x=all_steps,
            y=all_lits,
            hexbin=False,
            bins_x=bins_steps,
            bins_y=bins_literals,
            class_labels_x=labels_x,
            class_labels_y=labels_y,
            title=f"Longitud vs Literales (n={n})",
            xlabel="Pasos",
            ylabel="Literales",
            filename="heatmap_steps_literals.png",
            directory=self.plot_directory
        )
        plot_heatmap(
            x=all_steps,
            y=all_terms,
            hexbin=False,
            bins_x=bins_steps,
            bins_y=bins_terms,
            class_labels_x=labels_x,
            class_labels_y=labels_y,
            title=f"Longitud vs Términos (n={n})",
            xlabel="Pasos",
            ylabel="Términos",
            filename="heatmap_steps_terms.png",
            directory=self.plot_directory
        )

        log_message("[Experiment6] batch heatmaps done.")


# ——— Ejemplos de tm_factory ———————————————————————————————

def binary_counter_factory(config, hf, inp):
    return BinaryCounter(config=config)

def alternating_factory(config, hf, inp):
    bc = BinaryCounter(config=config)

    def run_alt():
        while True:
            # Ejecuta un paso
            result = bc.step()

            # Guarda la configuración recién alcanzada
            bc.config_history.append(bc._get_configuration())

            # Comprueba si ya tenemos el patrón 0101…
            tape_str = ''.join(bc.tape[1:-1])
            if all(tape_str[i] == ("1" if i % 2 != 0 else "0")
                   for i in range(len(tape_str))):
                bc.outcome = "pattern found"
                break

            # Si la MT ha halta­do de forma natural, salimos
            if result is not None:        # result == "halt"
                break

    bc.run = run_alt
    return bc


if __name__ == "__main__":
    hf_range = list(np.linspace(0.1, 0.9, 9))
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

    for cfg in configs:
        # 1) Clasificación contador binario
        # exp_bin = Experiment6(cfg, [0], tm_factory=binary_counter_factory)
        # exp_bin.run_experiment()
        
        # 2) Clasificación patrón alternante
        # exp_alt = Experiment6(cfg, [0], tm_factory=alternating_factory)
        # exp_alt.run_experiment()
        
        # # 3) Heatmaps originales
        exp_long = Experiment6(cfg, hf_range)
        exp_long.run_experiment()
        
        time.sleep(3)
