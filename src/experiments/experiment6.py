from base_experiment import Experiment
from src.experiments.utils.plotter import plot_experiment6_results
from src.tm.utils import generate_turing_machines_with_transitions
from src.experiments.utils.computing import measure_minimal_dnf
from src.tm.utils import get_history_function, get_num_steps
import numpy as np
from collections import defaultdict


class Experiment6(Experiment):
    def __init__(self, configs=None, machines_per_hf=None, hf_range=None):
        super().__init__("experiment6")
        assert len(configs) > 0 and machines_per_hf > 0 and len(hf_range) > 0
        self.configs = configs
        self.machines_per_hf = machines_per_hf
        self.hf_range = hf_range

    def _config_key(self, config):
        # sort items to make order invariant
        return tuple(sorted(config.items()))

    def run_experiment(self):
        # metrics[config_key][hf] => {lengths, dnf_terms, dnf_lits lists}
        metrics = defaultdict(lambda: defaultdict(lambda: {
            "lengths":   [],
            "dnf_terms": [],
            "dnf_lits":  []
        }))

        for config in self.configs:
            cfg_key = self._config_key(config)
            for hf in self.hf_range:
                # generate machines for this config and halting fraction
                tms = generate_turing_machines_with_transitions(
                    num_machines=self.machines_per_hf,
                    config=config,
                    halting_fraction=hf
                )
                for tm in tms:
                    tm.run()
                    # record execution length
                    metrics[cfg_key][hf]["lengths"].append(get_num_steps(tm))
                    # record DNF complexity
                    H = get_history_function(tm)
                    T, L = measure_minimal_dnf(H)
                    metrics[cfg_key][hf]["dnf_terms"].append(T)
                    metrics[cfg_key][hf]["dnf_lits"].append(L)

        # compute averages for each config and hf
        for cfg_key, hf_dict in metrics.items():
            for hf, m in hf_dict.items():
                count = len(m["lengths"])
                m["avg_lengths"]   = sum(m["lengths"])   / count
                m["avg_dnf_terms"] = sum(m["dnf_terms"]) / count
                m["avg_dnf_lits"]  = sum(m["dnf_lits"])  / count

        return metrics


def run_experiment():
    machines_per_hf = 50
    hf_range = np.linspace(0.1, 0.9, 9)
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
    exp = Experiment6(
        configs=configs,
        machines_per_hf=machines_per_hf,
        hf_range=hf_range
    )
    metrics = exp.run_experiment()
    
    plot_experiment6_results(
        metrics,
        configs,
        experiment=exp
    )
    

if __name__ == "__main__":
    run_experiment()
