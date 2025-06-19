# src/experiments/experiment6.py

from numpy import linspace
from src.experiments.base_experiment import Experiment
from src.experiments.utils.logger import log_message, log_data, create_subdirectory
from src.experiments.utils.plotter import plot_heatmap
from src.experiments.utils.computing import measure_minimal_dnf
from src.tm.utils import (
    generate_turing_machine,
    get_history_function,
    get_num_steps,
    serialize_turing_machine,
    generate_random_binary_input
)

class Experiment6(Experiment):
    """
    Generate “long‐running” Turing Machines via a custom transition generator,
    measure their execution lengths and minimal‐DNF complexities, then
    bucket and plot a categorical heatmap (steps vs. literals).
    """

    def __init__(self, configs, halting_fraction_range):
        """
        :param configs: list of dicts with tape_bits, head_bits, state_bits
        :param halting_fraction_range: iterable of halting_fraction values
        """
        super().__init__("Experiment6")
        assert configs and halting_fraction_range, \
            "Must pass configs, positive machines_per_hf, and nonempty halting_fraction_range"
        self.configs = configs
        self.hf_range = halting_fraction_range

    def run_experiment(self):
        log_message(f"[Experiment6] Generating long TMs for {len(self.configs)} configs")
        
        num_machines_per_hf = int(self.num_experiments_per_config / len(self.hf_range))

        for i, cfg in enumerate(self.configs):
            tape = cfg["tape_bits"]
            head = cfg["head_bits"]
            state = cfg["state_bits"]
            total_bits = tape + head + state
            label = f"C{i + 1}_T{tape}H{head}S{state}"

            # Create a subdirectory for this configuration
            cfg_dir = create_subdirectory(name=label, parent=self.run_dir)

            # Containers for all machines & data
            all_tms = []
            complexities = []  # list of dicts {hf, steps, minterms, literals}

            # Generate and run machines for each halting fraction
            for hf in self.hf_range:
                # Build transition functions designed to run long
                for _ in range(num_machines_per_hf):
                    tf = generate_long_tm_transition_function(2 ** state, halting_fraction=hf)
                    binary_input = generate_random_binary_input(tape_length=tape)
                    tm = generate_turing_machine(config=cfg, binary_input=binary_input, transition_function=tf)
                    tm.run()
                    steps = get_num_steps(tm)
                    history = get_history_function(tm)
                    minterms, literals = measure_minimal_dnf(history)

                    all_tms.append(tm)
                    complexities.append({
                        "halting_fraction": hf,
                        "steps": steps,
                        "minterms": minterms,
                        "literals": literals
                    })

            # 1) Save raw turing machines
            serialized = [serialize_turing_machine(tm) for tm in all_tms]
            log_data(data=serialized, filename="turing_machines.json", directory=cfg_dir)

            # 2) Save per‐TM complexities
            log_data(data=complexities, filename="tm_complexities.json", directory=cfg_dir)

            # 3) Aggregate per‐halting‐fraction
            agg = {}
            for hf in self.hf_range:
                group = [c for c in complexities if c["halting_fraction"] == hf]
                steps = [c["steps"] for c in group]
                lits  = [c["literals"] for c in group]
                mins  = [c["minterms"] for c in group]
                n = len(group)
                agg[hf] = {
                    "avg_steps":   sum(steps)/n,
                    "max_steps":   max(steps),
                    "avg_literals":sum(lits)/n,
                    "max_literals":max(lits),
                    "avg_minterms":sum(mins)/n,
                    "max_minterms":max(mins)
                }
            log_data(data=agg, filename="experiment_results.json", directory=cfg_dir)

            # 4) Prepare buckets for categorical heatmap
            all_steps = [c["steps"] for c in complexities]
            all_lits  = [c["literals"] for c in complexities]
            smin, smax = min(all_steps), max(all_steps)
            lmin, lmax = min(all_lits),  max(all_lits)
            delta_s = (smax - smin) / 3
            delta_l = (lmax - lmin) / 3
            bins_x = [smin, smin + delta_s, smin + 2*delta_s, smax]
            bins_y = [lmin, lmin + delta_l, lmin + 2*delta_l, lmax]
            labels_x = ["Corta", "Media", "Larga"]
            labels_y = ["Fácil", "Moderada", "Difícil"]

            # 5) plot categorical heatmap
            plot_heatmap(
                x=all_steps,
                y=all_lits,
                hexbin=False,
                bins_x=bins_x,
                bins_y=bins_y,
                class_labels_x=labels_x,
                class_labels_y=labels_y,
                title=f"Longitud de ejecución vs complejidad (FND) — n = {total_bits} bits",
                xlabel="Longitud de ejecución",
                ylabel="Complejidad (literales de la FND)",
                filename="heatmap_length_complexity.png",
                directory=self.plot_directory
            )

        log_message("[Experiment6] Done.")
        

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
    
    # Machines per halting fraction
    halting_fraction_range = linspace(0.1, 0.9, 9)
    exp = Experiment6(
        configs=configs, 
        halting_fraction_range=halting_fraction_range
    )
    exp.run_experiment()

if __name__ == "__main__":
    run_experiment()
