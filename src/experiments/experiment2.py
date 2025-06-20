# This experiment tries to see how well the projected 5 bit to 1 boolean functions 
# are represented in the dataset of boolean functions from 5 bits to 1 computed with circuits
# from 1 to 10 gates. The apparent conclusion is that most projected functions are computed with
# a circuit with a higher number of gates, maybe showing that the projection does not work well,
# since with experiment 1 we have seen that the 10 bit to 1 functions seem "simple" because they
# have low values of equanimity and entanglement.

# src/experiments/experiment2.py

from collections import defaultdict
from pathlib import Path
from src.experiments.config import DATASET_PATH, LOGS_PATH
from src.experiments.base_experiment import Experiment
from src.experiments.utils.loader import list_dirs, load_json
from src.experiments.utils.logger import log_message, log_data
from src.experiments.utils.plotter import plot_histogram

class Experiment2(Experiment):
    """
    This experiment analyzes logs produced by Experiment1, collecting the "projected functions"
    and comparing them to a dataset of 5-bit boolean functions.
    """

    def __init__(self, timestamp):
        super().__init__("Experiment2")
        self.timestamp = timestamp
        self.dataset = self.load_dataset(DATASET_PATH)
        self.projection = None
        
    def load_dataset(self, dataset_path):
        log_message('Loading dataset...')
        dataset = {}
        try:
            with open(dataset_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    parts = line.split()
                    if len(parts) < 2: continue
                    function_code = int(parts[0])
                    circuit_size = int(parts[1])
                    dataset[function_code] = circuit_size
        except Exception as e:
            log_message(f"Error loading dataset: {e}", prefix="[ERROR]")
        return dataset
        
    def _get_history_functions_from_file(self, file_path):
        history_functions = []
        
        tm_json = load_json(file_path)
        for tm in tm_json:
            hf = tm['history_function']
            hf = ''.join(str(int(b)) for b in hf)
            history_functions.append(hf)
            
        # Save what the projection on to the tape needs to be
        self.projection = range(tm_json[0]['config']['tape_bits'])
        
        return history_functions            
        
    def _get_history_functions_from_run(self, run_dir, filename):
        history_functions = []
        for cfg in list_dirs(run_dir):
            for prob in list_dirs(cfg):
                file_path = prob / filename
                new_history_functions = self._get_history_functions_from_file(file_path)
                history_functions.extend(new_history_functions)
        return history_functions
    
    def get_history_functions(self, *, timestamp, filename):
        """
        Return a list of all (possibly repeated) history functions collected from LOGS_PATH/experiment1/<timestamp>,
        in the filename specified.
        """
        run_dir = Path(LOGS_PATH) / 'experiment1' / timestamp
        return self._get_history_functions_from_run(
            run_dir=run_dir, 
            filename=filename, 
        )
        
    def project_history_functions(self, history_functions):
        
        def project(string):
            projected_string = ""
            for index in self.projection:
                projected_string += string[index]
            return projected_string
    
        return [project(hf) for hf in history_functions]
                
    def code_history_functions(self, history_functions):
        return [int(hf, 2) for hf in history_functions]            

    def run_experiment(self):
        log_message(f"Experiment2 analyzing logs from experiment1")

        # Collect raw history functions (duplicates allowed)
        history_functions = self.get_history_functions(
            timestamp=self.timestamp,
            filename='turing_machines.json',
        )

        # Deduplicate
        unique_hfs = set(history_functions)
        metadata = {
            'num_history_functions_total':      len(history_functions),
            'num_history_functions_unique':     len(unique_hfs),
        }

        # Project and code the unique ones
        projected = self.project_history_functions(unique_hfs)
        coded     = self.code_history_functions(projected)

        # Count per circuit size, and unknowns
        counts = defaultdict(int)
        unknown = 0
        for code in coded:
            if code in self.dataset: counts[self.dataset[code]] += 1
            else: unknown += 1

        total_unique = len(coded)

        # 5) Build count & percentage tables
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
            title='Conteo de funciones de historial por tamaño mínimo de circuito',
            xlabel='Tamaño de circuito',
            ylabel='Número de funciones únicas',
            filename='circuitos_VS_funciones_historial.png',
            directory=self.plot_directory,
        )



def run_experiment():
    
    # timestamp = "20250618_122014" # T1H0S2
    # timestamp = "20250618_122025" # T2H1S2
    # timestamp = "20250618_122045" # T4H2S2
    # timestamp = "20250618_122125" # T8H3S2
    
    # timestamp = "20250618_181419" # T1H0S3
    # timestamp = "20250618_181443" # T2H1S3
    # timestamp = "20250618_181456" # T4H2S3
    # timestamp = "20250618_181806" # T8H3S3
    
    timestamp = "20250619_173223" # T5H3S2
    
    exp = Experiment2(timestamp=timestamp) 
    exp.run_experiment()

if __name__ == "__main__":
    run_experiment()
