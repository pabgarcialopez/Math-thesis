# src/experiments/experiment4.py

from pathlib import Path
import random

from src.experiments.base_experiment import Experiment
from src.experiments.utils.loader import load_json
from src.experiments.utils.logger import log_message, log_data
from src.experiments.utils.plotter import plot_series
from src.experiments.utils.computing import measure_minimal_dnf
from src.experiments.config import LOGS_PATH

class Experiment4(Experiment):
    """
    Compare minimal DNF complexity (minterms and literals) of:
      - TM history functions (loaded from Experiment1 logs),
      - Random boolean functions of equal bit-size.

    For each configuration specified by timestamp under LOGS_PATH/experiment1,
    computes mean and worst-case complexity for both modes, then plots:
      - literals (mean)
      - literals (worst)
      - minterms (mean)
      - minterms (worst)
    """

    def __init__(self, timestamps: list[str]):
        super().__init__("Experiment4")
        self.timestamps = timestamps
        self.exp1_dir = Path(LOGS_PATH) / "experiment1"
        if not self.exp1_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.exp1_dir}")

        # Prepare list of (timestamp, config_dir, total_bits)
        self.config_entries = []
        for ts in self.timestamps:
            run_dir = self.exp1_dir / ts
            if not run_dir.exists() or not run_dir.is_dir():
                raise FileNotFoundError(f"Run directory not found: {run_dir}")
            # Let's assume one config dir under run_dir
            cfg_dirs = [d for d in run_dir.iterdir() if d.is_dir()]
            if not cfg_dirs:
                raise FileNotFoundError(f"No configuration directory in {run_dir}")
            cfg_dir = cfg_dirs[0]
            # Load a sample to extract bit counts
            sample_prob = next(cfg_dir.glob('prob*'), None)
            if sample_prob is None:
                raise FileNotFoundError(f"No 'prob*' directories in {cfg_dir}")
            tm_list = load_json(sample_prob / 'turing_machines.json')
            if not tm_list:
                raise ValueError(f"No TMs in {sample_prob}/turing_machines.json")
            config = tm_list[0]['config']
            total_bits = config['tape_bits'] + config['head_bits'] + config['state_bits']
            self.config_entries.append((ts, cfg_dir, total_bits))

        self.results_by_bits = {}

    def run_experiment(self):
        log_message(f"Experiment4: computing minimal DNF for TM vs Random on {len(self.config_entries)} configs")

        for ts, cfg_dir, total_bits in self.config_entries:
            log_message(f" Processing config '{ts}' with {total_bits} bits")

            # TM mode: load and annotate each prob*/turing_machines.json with its complexities
            tm_overall_minterms = []
            tm_overall_literals = []
            for prob_dir in sorted(cfg_dir.glob('prob*')):
                tm_list = load_json(prob_dir / 'turing_machines.json')
                complexities = []
                for tm in tm_list:
                    hf = tm['history_function']
                    minterms, literals = measure_minimal_dnf(hf)
                    complexities.append({
                        "minterms": minterms,
                        "literals": literals
                    })
                    tm_overall_minterms.append(minterms)
                    tm_overall_literals.append(literals)

                # Write tm_complexities.json next to turing_machines.json
                log_data(
                    data=complexities,
                    filename="tm_complexities.json",
                    directory=prob_dir
                )

            tm_minterms = tm_overall_minterms
            tm_literals = tm_overall_literals

            tm_mean_minterms = sum(tm_minterms) / len(tm_minterms)
            tm_max_minterms  = max(tm_minterms)
            tm_mean_lit      = sum(tm_literals) / len(tm_literals)
            tm_max_lit       = max(tm_literals)

            # --- Random mode: generate random functions with same sample size ---
            sample_size = len(tm_minterms)
            rand_minterms = []
            rand_literals = []
            table_size = 2 ** total_bits
            for _ in range(sample_size):
                rand_history_function = [random.choice([False, True]) for _ in range(table_size)]
                minterms, literals = measure_minimal_dnf(rand_history_function)
                rand_minterms.append(minterms)
                rand_literals.append(literals)

            r_mean_minterms = sum(rand_minterms) / sample_size
            r_max_minterms  = max(rand_minterms)
            r_mean_lit      = sum(rand_literals) / sample_size
            r_max_lit       = max(rand_literals)

            # store aggregated results
            self.results_by_bits[total_bits] = {
                'tm': {
                    'mean_minterms': tm_mean_minterms,
                    'max_minterms':  tm_max_minterms,
                    'mean_literals': tm_mean_lit,
                    'max_literals':  tm_max_lit
                },
                'random': {
                    'mean_minterms': r_mean_minterms,
                    'max_minterms':  r_max_minterms,
                    'mean_literals': r_mean_lit,
                    'max_literals':  r_max_lit
                }
            }

        # save results to JSON
        log_data(
            data=self.results_by_bits,
            filename='experiment4_results.json',
            directory=self.run_dir
        )

        # prepare data for plotting
        bits = sorted(self.results_by_bits.keys())
        tm_mean_lit   = [self.results_by_bits[b]['tm']['mean_literals']     for b in bits]
        rand_mean_lit = [self.results_by_bits[b]['random']['mean_literals'] for b in bits]
        tm_max_lit    = [self.results_by_bits[b]['tm']['max_literals']      for b in bits]
        rand_max_lit  = [self.results_by_bits[b]['random']['max_literals']   for b in bits]

        tm_mean_minterms   = [self.results_by_bits[b]['tm']['mean_minterms']     for b in bits]
        rand_mean_minterms = [self.results_by_bits[b]['random']['mean_minterms'] for b in bits]
        tm_max_minterms    = [self.results_by_bits[b]['tm']['max_minterms']      for b in bits]
        rand_max_minterms  = [self.results_by_bits[b]['random']['max_minterms']   for b in bits]

        labels = ['Funciones aleatorias', 'Funciones de historial']

        # helper to plot comparison using plot_series
        def plot_comparison(x, ys, ylabel, title, filename):
            plot_series(
                x=x,
                ys=ys,
                labels=labels,
                title=title,
                xlabel='Número de bits',
                ylabel=ylabel,
                filename=filename,
                directory=self.plot_directory
            )

        # 1) Literals mean
        plot_comparison(
            bits,
            [rand_mean_lit, tm_mean_lit],
            ylabel='Número medio de literales',
            title='Comparativa media de literales: aleatorio vs historial',
            filename='lit_media_comparacion.png'
        )
        # 2) Literals worst-case
        plot_comparison(
            bits,
            [rand_max_lit, tm_max_lit],
            ylabel='Número de literales (caso peor)',
            title='Comparativa peor caso de literales: aleatorio vs historial',
            filename='lit_peor_comparacion.png'
        )
        # 3) Minterms mean
        plot_comparison(
            bits,
            [rand_mean_minterms, tm_mean_minterms],
            ylabel='Número medio de minterms',
            title='Comparativa media de minterms: aleatorio vs historial',
            filename='min_media_comparacion.png'
        )
        # 4) Minterms worst-case
        plot_comparison(
            bits,
            [rand_max_minterms, tm_max_minterms],
            ylabel='Número de minterms (caso peor)',
            title='Comparativa peor caso de minterms: aleatorio vs historial',
            filename='min_peor_comparacion.png'
        )


def run_experiment():
    timestamps = [
        "20250618_122014" # T1H0S2
        "20250618_122025" # T2H1S2
        "20250618_122045" # T4H2S2
        "20250618_122125" # T8H3S2

        "20250618_181419" # T1H0S3
        "20250618_181443" # T2H1S3
        "20250618_181456" # T4H2S3
        "20250618_181806" # T8H3S3
    ]
    exp = Experiment4(timestamps=timestamps)
    exp.run_experiment()
