from collections import defaultdict
from pathlib import Path
import time
from src.experiments.config import DATASET_PATH, LOGS_PATH
from src.experiments.base_experiment import Experiment
from src.experiments.utils.loader import list_dirs, load_json
from src.experiments.utils.logger import log_message, log_data
from src.experiments.utils.plotter import plot_histogram
from src.tm.utils import get_projected_history_function

DATASET = {}
def load_dataset():
    log_message('Loading dataset...')
    try:
        with open(DATASET_PATH, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split()
                if len(parts) < 2: continue
                function_code = int(parts[0])
                circuit_size = int(parts[1])
                DATASET[function_code] = circuit_size
    except Exception as e:
        log_message(f"Error loading dataset: {e}", prefix="[ERROR]")

class Experiment2(Experiment):
    """
    This experiment analyzes logs produced by Experiment1, collecting the configuration histories,
    translating them to the 5 bit projected history functions and comparing them to a dataset of 5-bit boolean functions.
    """

    def __init__(self, timestamp):
        super().__init__("experiment2")
        self.timestamp = timestamp
        self.projection = None
        
    def _get_config_histories_from_file(self, file_path):
        config_histories = []
        
        tm_json = load_json(file_path)
        for tm in tm_json:
            config_histories.append(tm['config_history'])
            
        # Save what the projection on to the tape needs to be
        self.projection = range(tm_json[0]['config']['tape_bits'])
        
        return config_histories            
        
    def _get_config_histories_from_run(self, run_dir, filename):
        config_histories = []
        for cfg in list_dirs(run_dir):
            for prob in list_dirs(cfg):
                file_path = prob / filename
                new_config_histories = self._get_config_histories_from_file(file_path)
                config_histories.extend(new_config_histories)
        return config_histories
    
    def get_config_histories(self, *, timestamp, filename):
        """
        Return a list of all (possibly repeated) history functions collected from LOGS_PATH/experiment1/<timestamp>,
        in the filename specified.
        """
        run_dir = Path(LOGS_PATH) / 'experiment1' / timestamp
        return self._get_config_histories_from_run(
            run_dir=run_dir, 
            filename=filename, 
        )
                
    def code_history_functions(self, history_functions):
        return [int(hf, 2) for hf in history_functions]            

    def run_experiment(self):
        log_message(f"Experiment2 analyzing logs from experiment1")

        # Collect raw history functions (duplicates allowed)
        config_histories = self.get_config_histories(
            timestamp=self.timestamp,
            filename='turing_machines.json',
        )
        
        # Obtain the projected history functions from the configurations
        projected_hist_funcs = [
            get_projected_history_function(
                config_history=config_history, 
                projection=self.projection)
            for config_history in config_histories
        ]
        
        # Obtain the projected history functions as strings
        projected_hist_funcs_as_strings = [
            ''.join(str(bit) for bit in projected_hist_func)
            for projected_hist_func in projected_hist_funcs
        ]

        unique_projected_hfs= set(projected_hist_funcs_as_strings)
        
        metadata = {
            'num_history_functions_total':             len(config_histories),
            'num_unique_projected_history_functions':  len(unique_projected_hfs),
        }

        # Project and code the unique ones
        coded = self.code_history_functions(unique_projected_hfs)

        # Count per circuit size, and unknowns
        counts = defaultdict(int)
        unknown = 0
        for code in coded:
            if code in DATASET: counts[DATASET[code]] += 1
            else: unknown += 1

        total_unique = len(coded)

        # Build count & percentage tables
        unknown_pct = (unknown / total_unique) * 100
        percentages = {
            size: (cnt / total_unique) * 100
            for size, cnt in counts.items()
        }

        metadata.update({
            'counts_by_circuit_size':       dict(counts),
            'percentages_by_circuit_size':  {str(size): pct for size, pct in percentages.items()},
            'unknown_count':                unknown,
            'unknown_percentage':           unknown_pct,
        })

        # Log metadata
        log_data(
            data=metadata,
            filename='experiment_results.json',
            directory=self.run_dir
        )

        # Finally, plot the known‐only histogram
        x = sorted(counts.keys())
        ys = [[counts[size] for size in x]]
        plot_histogram(
            x=x,
            ys=ys,
            colors=['green'],
            title=f'Tamaño de circuito vs. Nº de funciones de historial proyectadas sobre 5 bits de cinta',
            xlabel='Tamaño de circuito',
            ylabel='Número de funciones',
            filename='circuitos_VS_funciones_historial.png',
            directory=self.plot_directory,
        )



def run_experiment():
    
    timestamps = [
        "20250619_173223", # T5H3S2
    ]
    
    # Load the dataset just once
    load_dataset()
    
    for timestamp in timestamps:
        exp = Experiment2(timestamp=timestamp) 
        exp.run_experiment()
        time.sleep(3)

if __name__ == "__main__":
    run_experiment()
