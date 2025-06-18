# src/experiments/experiment1.py

from tqdm import tqdm
import time
from collections import defaultdict

from src.experiments.base_experiment import Experiment
from src.experiments.utils.plotter import plot_series, plot_heatmap
from src.experiments.utils.logger import log_message, log_data, create_subdirectory

from src.tm.utils import generate_turing_machines, get_history_function, serialize_turing_machine
from src.experiments.metrics.entanglement import entanglement
from src.experiments.metrics.equanimities import equanimity_importance, equanimity_subsets, equanimity_subsets_normalized

class Experiment1(Experiment):
    def __init__(self, configs=None):
        super().__init__("Experiment1")
        assert configs and len(configs) > 0
        self.configs = configs
        
        # Top‐level list of result‐dicts
        self.results = []
        # Auxiliary map: config_label to index in self.results
        self._results_index: dict[str,int] = {}
        
        # Top‐level list of aggregated results
        self.aggregated_results = {}

    def collect_results_for_TM(self, tm, config_label, transition_probability):
       
        serialized_tm = serialize_turing_machine(tm)
        # Save history function as a vector for the metrics computation
        history_function = serialized_tm['history_function']
        
        ent = entanglement(history_function)
        eq_imp = equanimity_importance(history_function)
        eq_sub = equanimity_subsets(history_function)
        eq_sub_norm = equanimity_subsets_normalized(history_function)

        # Find-or-create the config‐entry via the index map
        if config_label in self._results_index:
            entry = self.results[self._results_index[config_label]]
        else:
            entry = {
                'configuration': config_label,
                'transition_probabilities': {}
            }
            self.results.append(entry)
            self._results_index[config_label] = len(self.results) - 1  # Record its position

        # Find-or-create the transition probability slot
        tp_map = entry['transition_probabilities']
        if transition_probability not in tp_map:
            tp_map[transition_probability] = {
                'turing_machines':    [],
                'entanglements':      [],
                'equanimities': {
                    'importance':         [],
                    'subsets':            [],
                    'subsets_normalized': []
                }
            }
        slot = tp_map[transition_probability]

        # Append data
        slot['turing_machines'].append(serialized_tm)
        slot['entanglements'].append(ent)
        slot['equanimities']['importance'].append(eq_imp)
        slot['equanimities']['subsets'].append(eq_sub)
        slot['equanimities']['subsets_normalized'].append(eq_sub_norm)
            
    def collect_results(self, turing_machines, config_label, transition_probability):
        for tm in turing_machines:
            self.collect_results_for_TM(tm, config_label, transition_probability)

    def aggregate_results_per_config(self):
        per_config = []

        for result in self.results:
            config_label = result['configuration']
            tp_map = result['transition_probabilities']

            # Collect the per-TP means
            ent_means    = []
            imp_means    = []
            subs_means   = []
            norm_means   = []

            for tp, slot in tp_map.items():
                entanglements = slot['entanglements']
                equanimities  = slot['equanimities']
                # Mean for this transition probability
                ent_means.append(sum(entanglements) / len(entanglements))
                imp_means.append(sum(equanimities['importance']) / len(equanimities['importance']))
                subs_means.append(sum(equanimities['subsets']) / len(equanimities['subsets']))
                norm_means.append(sum(equanimities['subsets_normalized']) / len(equanimities['subsets_normalized']))

            # Now we average those TP-means
            n_tp          = len(ent_means)
            avg_ent       = sum(ent_means) / n_tp
            avg_imp       = sum(imp_means) / n_tp
            avg_subs      = sum(subs_means) / n_tp
            avg_subs_norm = sum(norm_means) / n_tp

            per_config.append({
                'configuration': config_label,
                'avg_entanglement': avg_ent,
                'avg_equanimities': {
                    'importance':         avg_imp,
                    'subsets':            avg_subs,
                    'subsets_normalized': avg_subs_norm
                }
            })

        self.aggregated_results['per_config'] = per_config

    def aggregate_results_per_transition_probability(self):
        # Factory that gives each TP its zeroed‐out slot
        def _new_slot():
            return {
                'avg_entanglement':    0.0,
                'avg_equanimities': {
                    'importance':         0.0,
                    'subsets':            0.0,
                    'subsets_normalized': 0.0
                }
            }

        per_tp = defaultdict(_new_slot)
        n_configs = len(self.results)

        # Accumulate per‐config per‐TP means
        for result in self.results:
            for tp, slot in result['transition_probabilities'].items():
                # Mean entanglement for this single config at this TP
                ent_mean = sum(slot['entanglements']) / len(slot['entanglements'])
                eqs = slot['equanimities']
                imp_mean  = sum(eqs['importance'])          / len(eqs['importance'])
                subs_mean = sum(eqs['subsets'])             / len(eqs['subsets'])
                norm_mean = sum(eqs['subsets_normalized'])  / len(eqs['subsets_normalized'])

                # Add into running totals
                per_tp[tp]['avg_entanglement'] += ent_mean
                per_tp[tp]['avg_equanimities']['importance']         += imp_mean
                per_tp[tp]['avg_equanimities']['subsets']            += subs_mean
                per_tp[tp]['avg_equanimities']['subsets_normalized'] += norm_mean

        # Divide by how many configurations we had
        for tp, slot in per_tp.items():
            slot['avg_entanglement'] /= n_configs
            ae = slot['avg_equanimities']
            ae['importance']         /= n_configs
            ae['subsets']            /= n_configs
            ae['subsets_normalized'] /= n_configs

        self.aggregated_results['per_transition_probability'] = per_tp
        
    def aggregate_results(self):
        self.aggregate_results_per_config()
        self.aggregate_results_per_transition_probability()
        
    def log_turing_machines(self, config_label, transition_probability, directory):
        idx = self._results_index[config_label]
        slot = self.results[idx]['transition_probabilities'][transition_probability]
        serialized_turing_machines = slot['turing_machines']
        
        turing_machines_data = [serialized for serialized in serialized_turing_machines]
        log_data(data=turing_machines_data, filename="turing_machines.json", directory=directory)

    def log_results(self, results, filename, directory):
        log_data(data=results, filename=filename, directory=directory)
        
    def plot_results(self):
        
        # 1. Transition probabilities vs equanimities
        # 2. Transition probabilities vs entanglement
        
        per_tp = self.aggregated_results['per_transition_probability']

        # Retrieve equanimities and entanglement per transition probability
        eq_importance = []
        eq_subsets = []
        eq_subsets_normalized = []
        entanglements = []
        for tp in self.transition_probabilities:
            slot = per_tp[tp]
            eq_importance.append(slot['avg_equanimities']['importance'])
            eq_subsets.append(slot['avg_equanimities']['subsets'])
            eq_subsets_normalized.append(slot['avg_equanimities']['subsets_normalized'])
            entanglements.append(slot['avg_entanglement'])
        
        # Plot 1.1
        x = self.transition_probabilities
        ys = [eq_importance, eq_subsets_normalized]
        labels = [
            "Ecuanimidad por importancia",
            "Ecuanimidad por subconjuntos normalizada",
        ]
        title = "Probabilidades de transición vs Ecuanimidades"
        xlabel = "Probabilidad de transición"
        ylabel = "Ecuanimidad"
        filename = "probabilidad_de_transicion_VS_ecuanimidades1.png"
        directory = self.plot_directory
        plot_series(
            x=x,
            ys=ys,
            labels=labels,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            filename=filename,
            directory=directory,
        )
        
        # Plot 1.2
        ys = [eq_subsets]
        labels = ["Ecuanimidad por subconjuntos",]
        title = "Probabilidades de transición vs Ecuanimidad por subconjuntos"
        filename = "probabilidad_de_transicion_VS_ecuanimidades2.png"
        plot_series(
            x=x,
            ys=ys,
            labels=labels,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            filename=filename,
            directory=directory,
        )
        
        # Plot 2
        
        x = self.transition_probabilities    
        ys = [entanglements]
        labels = ["Enredo"]
        title = "Probabilidades de transición vs Enredo"
        xlabel = "Probabilidad de transición"
        ylabel = "Enredo"
        filename = "probabilidad_de_transicion_VS_enredo.png"
        plot_series(
            x=x,
            ys=ys,
            labels=labels,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            filename=filename,
            directory=directory,
        )
        
        # 3. Configuration vs equanimities
        # 4. Configuration vs entanglement
        
        per_config = self.aggregated_results['per_config']
        
        # Retrieve equanimities and entanglement per config
        eq_importance = [slot['avg_equanimities']['importance'] for slot in per_config]
        eq_subsets = [slot['avg_equanimities']['subsets'] for slot in per_config]
        eq_subsets_normalized = [slot['avg_equanimities']['subsets_normalized'] for slot in per_config]
        entanglements = [slot['avg_entanglement'] for slot in per_config]
        
        # Plot 3.1
        x = [result['configuration'] for result in per_config]
        ys = [eq_importance, eq_subsets_normalized]
        labels = [
            "Ecuanimidad por importancia",
            "Ecuanimidad por subconjuntos normalizada",
        ]
        title = "Configuraciones vs Ecuanimidades"
        xlabel = "Configuración"
        ylabel = "Ecuanimidad"
        filename = "configuraciones_VS_ecuanimidades1.png"
        plot_series(
            x=x,
            ys=ys,
            labels=labels,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            filename=filename,
            directory=directory,
        )
        
        # Plot 3.2
        ys = [eq_subsets]
        labels = ["Ecuanimidad por subconjuntos"]
        title = "Configuraciones vs Ecuanimidad por subconjuntos"
        filename = "configuraciones_VS_ecuanimidades2.png"
        plot_series(
            x=x,
            ys=ys,
            labels=labels,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            filename=filename,
            directory=directory,
        )
        
        # Plot 4
        x = [result['configuration'] for result in per_config]
        ys = [entanglements]
        labels = ["Enredo"]
        title = "Configuraciones vs Enredo"
        xlabel = "Configuración"
        ylabel = "Enredo"
        filename = "configuraciones_VS_enredo.png"
        plot_series(
            x=x,
            ys=ys,
            labels=labels,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            filename=filename,
            directory=directory,
        )
        
        # 5. Heatmap of each equanimity vs entanglement (count)
        
        # Retrieve data
        eq_importance = []
        eq_subsets = []
        eq_subsets_normalized = []
        entanglements = []
        for result in self.results:
            for tp in result['transition_probabilities']:
                slot = result['transition_probabilities'][tp]
                eq_importance += slot['equanimities']['importance']
                eq_subsets += slot['equanimities']['subsets']
                eq_subsets_normalized += slot['equanimities']['subsets_normalized']
                entanglements += slot['entanglements']
                
        x = eq_importance
        y = entanglements
        title = "Ecuanimidad por importancia vs Enredo"
        xlabel = "Ecuanimidad por importancia"
        ylabel = "Enredo"
        filename = "ecuanimidad_importancia_VS_enredo.png"
        plot_heatmap(
            x=x,
            y=y,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            hexbin=True,
            cmap='Blues',
            filename=filename,
            directory=directory,
        )
        
        x = eq_subsets
        title = "Ecuanimidad por subconjuntos vs Enredo"
        xlabel = "Ecuanimidad por subconjuntos"
        filename = "ecuanimidad_subconjuntos_VS_enredo.png"
        plot_heatmap(
            x=x,
            y=y,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            hexbin=True,
            cmap='Reds',
            filename=filename,
            directory=directory,
        )
        
        x = eq_subsets_normalized
        title = "Ecuanimidad por subconjuntos normalizada vs Enredo"
        xlabel = "Ecuanimidad por subconjuntos normalizada"
        filename = "ecuanimidad_subconjuntos_normalizada_VS_enredo.png"
        plot_heatmap(
            x=x,
            y=y,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            hexbin=True,
            cmap='Greens',
            filename=filename,
            directory=directory,
        )
          
    def run_experiment(self):
        config_times = []
        global_time = time.time()
        
        log_message(f"Running Experiment1. Logs in: {self.run_dir}")
        num_machines = int(self.num_experiments_per_config / len(self.transition_probabilities))
        
        for i, config in enumerate(tqdm(self.configs, desc='Running configs')):
            start_time = time.time()
            config_label = f"C{i + 1}"
            config_dir_name = f"{config_label}_T{config['tape_bits']}H{config['head_bits']}S{config['state_bits']}"
            config_dir = create_subdirectory(name=config_dir_name, parent=self.run_dir)
            
            for transition_probability in tqdm(self.transition_probabilities, desc=f'   {config_label}'):
                transition_probability_label = f"prob{transition_probability:.2f}"
                transition_probability_dir = create_subdirectory(name=transition_probability_label, parent=config_dir)
            
                # Create all the TMs we need for this configuration and transition probability
                turing_machines = generate_turing_machines(
                    config=config,                    
                    transition_probability=transition_probability,
                    num_machines=num_machines
                )
                
                # Run all machines
                for tm in turing_machines:
                    tm.run()
                    
                # Collect results from the run TMs
                self.collect_results(turing_machines, config_label, transition_probability)
                
                # Log results
                self.log_turing_machines(config_label, transition_probability, transition_probability_dir)
            config_times.append(time.time() - start_time)
                
        # Aggregate results
        self.aggregate_results()
        
        global_time = time.time() - global_time
        
        # Log results and aggregated
        self.log_results(self.results, filename="results.json", directory=self.run_dir)
        self.log_results(self.aggregated_results, filename="aggregated_results.json", directory=self.run_dir)
        self.log_results({
            "configs": self.configs,
            "config_times": config_times,
            "global_time": global_time,
            "num_experiments_per_config": self.num_experiments_per_config,
            "num_machines_per_config_per_tp": num_machines,
        }, filename="experiment_metadata.json", directory=self.run_dir)
        
        # Plot results
        self.plot_results()

def run_experiment():
    exp = Experiment1(configs=[
            # {"tape_bits": 1, "head_bits": 0, "state_bits": 2},
            # {"tape_bits": 2, "head_bits": 1, "state_bits": 2},
            # {"tape_bits": 4, "head_bits": 2, "state_bits": 2},
            {"tape_bits": 8, "head_bits": 3, "state_bits": 2},

            # {"tape_bits": 1, "head_bits": 0, "state_bits": 3},
            # {"tape_bits": 2, "head_bits": 1, "state_bits": 3},
            # {"tape_bits": 4, "head_bits": 2, "state_bits": 3},
            # {"tape_bits": 8, "head_bits": 3, "state_bits": 3},
            
            
            # {"tape_bits": 5, "head_bits": 3, "state_bits": 2},
        ])
    
    exp.run_experiment()

if __name__ == "__main__":
    run_experiment()
