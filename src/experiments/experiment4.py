"""
Experiment4 measures the minimal DNF complexity of boolean functions derived either from:
  - Turing Machines (“tm” mode), varying transition probabilities,
  - or purely random truth tables (“random” mode),
  - or both modes at once (“both” mode).

In “tm” mode, it observes how the transition probability influences the minimal DNF complexity
of the visited‐configurations function for each Turing Machine, given a partition of bits:
   tape_bits, head_bits, and state_bits.

In “random” mode, it simply measures the minimal DNF complexity of random boolean functions of
the same bit‐size (tape_bits + head_bits + state_bits), to compare how TMs might produce simpler
or more complex DNFs than random.

When mode == "both", we run both random and TM for each config, store aggregated data,
and produce a final plot comparing them.

We allow multiple configurations (each specifying tape_bits, head_bits, and state_bits). 
For each config, we repeat the process systematically.
"""

import os
import random
import numpy as np
from tqdm import tqdm

import src.experiments.config as config
from src.experiments.base_experiment import Experiment
from src.experiments.utils.plotter import (
    plot_length_vs_complexity_heatmap,
    plot_random_vs_tm_comparison,
    plot_terms_literals_freqs_histogram,
    plot_complexity_vs_probability,
    plot_bucket_histogram,
    plot_curve_with_max_line,
)
from src.experiments.utils.computing import measure_minimal_dnf
from src.tm.utils import get_history_function, generate_turing_machines, get_num_steps


class Experiment4(Experiment):
    """
    Experiment4 measures the minimal DNF complexity of boolean functions derived either from:
      - Turing Machines (“tm” mode), varying transition probabilities,
      - or purely random truth tables (“random” mode),
      - or both (“both” mode).

    In “tm” mode, the experiment observes how the transition probability influences the minimal DNF
    complexity of the visited-configurations function for each Turing Machine, given a partition of bits:
       tape_bits, head_bits, and state_bits.
    
    In “random” mode, it measures the minimal DNF complexity of random boolean functions of the same
    bit-size (tape_bits + head_bits + state_bits).

    When mode == "both", both modes are run and aggregated results are compared. The final comparison
    plot shows (in a 2×2 grid) the average and worst-case numbers of terms and literals versus total bits.
    """

    def __init__(self, mode="tm", configs=None):
        """
        :param mode: "tm", "random", or "both"
        :param configs: a list of dicts, each with keys:
            {"tape_bits": int, "head_bits": int, "state_bits": int}
        """
        super().__init__("experiment4")
        if configs is None:
            raise ValueError("Must provide `configs` (list of dicts) to Experiment4.")
        self.mode = mode
        self.configs = configs
        self.results_by_bits = {}  # For storing aggregated results by total_bits

    def run_experiment(self):
        self.log_message(f"[Experiment4] Running in mode '{self.mode}'. Logs in {self.run_dir}")
        trans_probs = np.linspace(config.MIN_PROB, config.MAX_PROB, config.NUM_PROBS)
        if self.mode in ("tm", "both"):
            for cfg in self.configs:
                tape_bits = cfg["tape_bits"]
                head_bits = cfg["head_bits"]
                state_bits = cfg["state_bits"]
                total_bits = tape_bits + head_bits + state_bits

                self.log_message(
                    f"[Experiment4 - tm] tape_bits={tape_bits}, head_bits={head_bits}, "
                    f"state_bits={state_bits} => total_bits={total_bits}"
                )
                avg_terms, avg_literals, tm_global_terms, tm_global_literals, num_steps = self.run_tm_mode_for_config(tape_bits, head_bits, state_bits, trans_probs)
                if self.mode == "both":
                    if total_bits not in self.results_by_bits:
                        self.results_by_bits[total_bits] = {}
                    worst_case_terms = max(tm_global_terms) if tm_global_terms else 0
                    worst_case_literals = max(tm_global_literals) if tm_global_literals else 0
                    self.results_by_bits[total_bits]["tm"] = (avg_terms, avg_literals, worst_case_terms, worst_case_literals, num_steps)

        if self.mode in ("random", "both"):
            for cfg in self.configs:
                tape_bits = cfg["tape_bits"]
                head_bits = cfg["head_bits"]
                state_bits = cfg["state_bits"]
                total_bits = tape_bits + head_bits + state_bits

                self.log_message(
                    f"[Experiment4 - random] tape_bits={tape_bits}, head_bits={head_bits}, "
                    f"state_bits={state_bits} => total_bits={total_bits}"
                )
                num_random_functions = config.NUM_EXPERIMENTS * len(trans_probs)
                avg_terms, avg_literals, rand_global_terms, rand_global_literals = self.run_random_mode_for_nbits(total_bits, num_random_functions)
                if self.mode == "both":
                    if total_bits not in self.results_by_bits:
                        self.results_by_bits[total_bits] = {}
                    worst_case_terms = max(rand_global_terms) if rand_global_terms else 0
                    worst_case_literals = max(rand_global_literals) if rand_global_literals else 0
                    self.results_by_bits[total_bits]["random"] = (avg_terms, avg_literals, worst_case_terms, worst_case_literals)

        if self.mode == "both":
            # Produce final 2x2 grid comparison plot
            comp_path = os.path.join(self.run_dir, "comparison_random_vs_tm.png")
            plot_random_vs_tm_comparison(self.results_by_bits, save_path=comp_path)
            self.log_message(f"Final comparison plot saved to: {comp_path}")

        self.log_message("Experiment4 completed.")

    # -------------------------------------------------------------------------
    # RANDOM MODE
    # -------------------------------------------------------------------------
    def run_random_mode_for_nbits(self, total_bits, num_functions):
        global_terms = []
        global_literals = []

        for i in tqdm(range(num_functions), desc=f"Random {total_bits}-bits", colour="blue"):
            table_size = 2 ** total_bits
            values = [random.choice([False, True]) for _ in range(table_size)]
            num_terms, num_literals = measure_minimal_dnf(values)
            global_terms.append(num_terms)
            global_literals.append(num_literals)
            if self.should_log():
                log_data = {
                    "mode": "random",
                    "total_bits": total_bits,
                    "truth_table": values,
                    "complexity": {"num_terms": num_terms, "total_literals": num_literals}
                }
                self.log_data(log_data, filename=f"random_{total_bits}bits_{i+1}.json")
        
        plot_terms_literals_freqs_histogram(
            global_terms,
            global_literals,
            total_bits,
            title=f"Random boolean functions ({total_bits} bits)",
            save_path=os.path.join(self.run_dir, f"random_complexity_histogram_{total_bits}bits.png")
        )

        self.save_frequency_percentages(global_terms, global_literals, mode_label=f"random_{total_bits}bits")
        avg_terms = sum(global_terms) / len(global_terms) if global_terms else 0
        avg_literals = sum(global_literals) / len(global_literals) if global_literals else 0
        return avg_terms, avg_literals, global_terms, global_literals

    # -------------------------------------------------------------------------
    # TM MODE
    # -------------------------------------------------------------------------
    def run_tm_mode_for_config(self, tape_bits, head_bits, state_bits, probabilities):
        total_bits = tape_bits + head_bits + state_bits
        avg_terms_per_prob = []
        avg_literals_per_prob = []
        global_terms = []
        global_literals = []
        steps = []

        for p_idx, prob in enumerate(tqdm(probabilities, desc=f"TM {total_bits}-bits", colour="green")):
            terms_list = []
            literals_list = []
            tms = generate_turing_machines(
                num_machines=config.NUM_EXPERIMENTS,
                config={'tape_bits': tape_bits, 'head_bits': head_bits, 'state_bits': state_bits},
                transition_probability=prob
            )
            for i, tm in enumerate(tms):
                tm.run()
                bool_vec = get_history_function(tm)
                num_terms, num_literals = measure_minimal_dnf(bool_vec)
                terms_list.append(num_terms)
                literals_list.append(num_literals)
                global_terms.append(num_terms)
                global_literals.append(num_literals)
                steps.append(get_num_steps(tm))
                if self.should_log():
                    log_data = {
                        "mode": "tm",
                        "prob": float(prob),
                        "tape_bits": tape_bits,
                        "head_bits": head_bits,
                        "state_bits": state_bits,
                        "complexity": {"num_terms": num_terms, "total_literals": num_literals}
                    }
                    self.log_data(log_data, filename=f"tm_{total_bits}bits_prob{p_idx+1}_{i+1}.json")
            avg_terms_per_prob.append(sum(terms_list) / len(terms_list))
            avg_literals_per_prob.append(sum(literals_list) / len(literals_list))
        
        plot_path = os.path.join(self.run_dir, f"tm_complexity_vs_prob_{total_bits}bits.png")
        plot_complexity_vs_probability(probabilities, avg_terms_per_prob, avg_literals_per_prob, save_path=plot_path)

        hist_title = f"Config: tape={tape_bits}, head={head_bits}, state={state_bits}"
        hist_path = os.path.join(self.run_dir, f"tm_histogram_{total_bits}bits.png")
        plot_terms_literals_freqs_histogram(
            global_terms,
            global_literals,
            total_bits,
            title=hist_title,
            save_path=hist_path
        )

        n = config.NUM_EXPERIMENTS

        # Plot steps vs probabilities
        save_path = os.path.join(self.run_dir, f"tm_steps_{total_bits}bits.png")
        title = 'TM mean total steps per transition probability'
        xlabel = 'Transition probabilities'
        ylabel = 'Mean number of steps'
        aggregated_steps = [sum(steps[i * n: (i + 1) * n]) / n for i in range(len(probabilities))]
        max_level = 2 ** total_bits
        plot_curve_with_max_line(probabilities, aggregated_steps, max_level, title, xlabel, ylabel, save_path)
    
        # Plot complexities vs probabilities
        save_path = os.path.join(self.run_dir, f"tm_complexity_{total_bits}bits.png")
        title = 'TM mean total DNF literals per transition probability'
        xlabel = 'Transition probabilities'
        ylabel = 'Mean number of literals in DNF'
        aggregated_complexities = [sum(global_literals[i * n: (i + 1) * n]) / n for i in range(len(probabilities))]
        max_level = total_bits * (2 ** (total_bits - 1))
        plot_curve_with_max_line(probabilities, aggregated_complexities, max_level, title, xlabel, ylabel, save_path)

        # Classes info for steps and difficulty of TM
        classes_info = {
            "step_class_labels": ["Short", "Medium", "Long"],
            "complexity_class_labels": ["Easy", "Moderate", "Hard"]
        }

        # Classify turing machines into classes based on execution length
        save_path = os.path.join(self.run_dir, f"tm_steps_buckets_{total_bits}bits.png")
        title = 'TM execution length classification'
        xlabel = 'Step classes'
        ylabel = 'Frequency'
        plot_bucket_histogram(steps, classes_info['step_class_labels'], title, xlabel, ylabel, save_path=save_path)
        
        # Classify turing machines into difficulty based on some metric
        save_path = os.path.join(self.run_dir, f"tm_complexity_buckets_{total_bits}bits.png")
        title = 'TM complexity classification'
        xlabel = 'Complexity classes'
        ylabel = 'Frequency'
        plot_bucket_histogram(global_literals, classes_info['complexity_class_labels'], title, xlabel, ylabel, save_path=save_path)

        # Plot heatmap steps-complexities   
        save_path = os.path.join(self.run_dir, f"tm_length_vs_complexities_{total_bits}bits.png")
        plot_length_vs_complexity_heatmap(steps, global_literals, classes_info, save_path=save_path)

        # Save data
        self.save_frequency_percentages(
            global_terms,
            global_literals,
            mode_label=f"tm_{total_bits}bits",
            tape_length=tape_bits,
            head_bits=head_bits,
            state_bits=state_bits
        )
        final_avg_terms = sum(avg_terms_per_prob) / len(avg_terms_per_prob) if avg_terms_per_prob else 0
        final_avg_literals = sum(avg_literals_per_prob) / len(avg_literals_per_prob) if avg_literals_per_prob else 0
        return final_avg_terms, final_avg_literals, global_terms, global_literals, steps

    # -------------------------------------------------------------------------
    # REUSED UTILS
    # -------------------------------------------------------------------------
    def save_frequency_percentages(self, terms_list, literals_list, mode_label,
                                   tape_length=None, head_bits=None, state_bits=None):
        total = len(terms_list)
        unique_terms, counts_terms = np.unique(terms_list, return_counts=True)
        unique_literals, counts_literals = np.unique(literals_list, return_counts=True)
        output_file = os.path.join(self.run_dir, f"{mode_label}_frequency_percentages.txt")
        with open(output_file, "w") as f:
            if tape_length is not None and head_bits is not None and state_bits is not None:
                n_bits = tape_length + head_bits + state_bits
                f.write("Configuration:\n")
                f.write(f"  tape_bits  = {tape_length}\n")
                f.write(f"  head_bits  = {head_bits}\n")
                f.write(f"  state_bits = {state_bits}\n")
                f.write(f"  total_bits = {n_bits}\n\n")
            f.write("Frecuencia y porcentajes de términos:\n")
            sum_terms = 0
            max_pair_terms = (0, 0)
            for ut, ct in zip(unique_terms, counts_terms):
                perc = (ct / total) * 100
                f.write(f"{ut}: {ct} ({perc:.2f}%)\n")
                sum_terms += ut * ct
                if ct > max_pair_terms[1]:
                    max_pair_terms = (ut, ct)
            mean_terms = sum_terms / total if total > 0 else 0
            f.write(f"Número medio de términos: {mean_terms:.2f}\n")
            f.write(f"Máximo # de términos observado: {max_pair_terms[0]} (con frecuencia {max_pair_terms[1]})\n\n")
            f.write("Frecuencia y porcentajes de literales:\n")
            sum_literals = 0
            max_pair_literals = (0, 0)
            for ul, cl in zip(unique_literals, counts_literals):
                perc = (cl / total) * 100
                f.write(f"{ul}: {cl} ({perc:.2f}%)\n")
                sum_literals += ul * cl
                if cl > max_pair_literals[1]:
                    max_pair_literals = (ul, cl)
            mean_literals = sum_literals / total if total > 0 else 0
            f.write(f"Número medio de literales: {mean_literals:.2f}\n")
            f.write(f"Máximo # de literales observado: {max_pair_literals[0]} (con frecuencia {max_pair_literals[1]})\n")
        self.log_message(f"Porcentajes de frecuencias ({mode_label}) guardados en: {output_file}")
    

def run_experiment():
    """
    Example usage: run in either 'tm', 'random' or 'both' mode, for multiple bit-partitions.
    Provide a list of configs, each specifying tape_bits, head_bits, and state_bits.
    """
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

    exp = Experiment4(
        mode="both",
        configs=configs
    )
    exp.run_experiment()


if __name__ == "__main__":
    run_experiment()
