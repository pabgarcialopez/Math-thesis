# src/experiments/experiment1.py

import os
import numpy as np
import src.config as config
from tqdm import tqdm
from src.plotters import plot_metrics_vs_frequency, plot_equanimity_vs_entanglement_heatmap
from src.utils.logger import save_execution_log
from src.tm.generators import generate_tm_input_pairs
from src.tm.machine import TuringMachine
from src.tm.validators import equanimity_importance, equanimity_subsets, equanimity_subsets_normalized, entanglement
from src.experiments.base_experiment import BaseExperiment, project_history_to_boolean_function

class Experiment1(BaseExperiment):
    def __init__(self, tape_length, num_states, total_bits):
        super().__init__("experiment1")
        self.tape_length = tape_length
        self.num_states = num_states
        self.total_bits = total_bits

    def run_single_experiment(self, tm, trans_prob=None):
        """
        Ejecuta la MT y calcula métricas sobre su función de historial.
        """
        result = tm.run()
        history_func = tm.get_history_function()

        # Métricas sobre la función completa (tape+head+state)
        eq_imp = equanimity_importance(history_func, tm.total_config_bits)
        eq_sub = equanimity_subsets(history_func, tm.total_config_bits)
        eq_sub_norm = equanimity_subsets_normalized(history_func, tm.total_config_bits)
        ent = entanglement(history_func, tm.total_config_bits)

        # Formatear la transición para el log
        transition_function = {
            f"{state},{symbol}": [next_state, write_symbol, direction]
            for (state, symbol), (next_state, write_symbol, direction) in tm.transition_function.items()
        }
        num_steps = len(tm.config_history) - 1

        # Opcional: proyectar a 5 bits si quieres mantener la lógica anterior
        # con project_history_to_boolean_function(tm.config_history)
        projected_function = project_history_to_boolean_function(tm.config_history)

        log_data = {
            "transition_probability": trans_prob,
            "tm_parameters": {
                "tape_length": tm.tape_length,
                "num_states": tm.num_states,
                "input_symbols": list(tm.input_symbols),
                "blank_symbol": tm.blank_symbol,
                "initial_head_position": tm.initial_head_position,
                "accepting_states": tm.accepting_states,
                "transition_function": transition_function,
            },
            "input": tm.binary_input,
            "execution": {
                "num_steps": num_steps,
                "result": result,
            },
            "metrics": {
                "equanimity_importance": eq_imp,
                "equanimity_subsets": eq_sub,
                "equanimity_subsets_normalized": eq_sub_norm,
                "entanglement": ent,
            },
            "config_history": tm.config_history,
            "projected_function": projected_function  # si quieres usar la proyección a 5 bits
        }
        return log_data, (eq_imp, eq_sub, eq_sub_norm, ent)

    def run_experiment(self):
        self.log_message(f"Saving logs to: {self.run_dir}")

        trans_probs = np.linspace(config.MIN_PROB, config.MAX_PROB, config.NUM_PROBS)
        avg_eq_imp, avg_eq_sub, avg_eq_sub_norm, avg_ent = [], [], [], []
        heatmap_eq_imp, heatmap_ent = [], []

        # Iterar sobre cada probabilidad
        for p_idx, prob in enumerate(tqdm(trans_probs, desc="Experiment 1", colour="green")):
            eq_imp_list, eq_sub_list, eq_sub_norm_list, ent_list = [], [], [], []
            machines = generate_tm_input_pairs(
                config.NUM_EXPERIMENTS,
                trans_prob=prob,
                tape_length=self.tape_length,
                num_states=self.num_states,
                total_bits=self.total_bits
            )

            for i, tm in enumerate(machines):
                log_data, metrics = self.run_single_experiment(tm, trans_prob=prob)
                eq_imp_val, eq_sub_val, eq_sub_norm_val, ent_val = metrics
                eq_imp_list.append(eq_imp_val)
                eq_sub_list.append(eq_sub_val)
                eq_sub_norm_list.append(eq_sub_norm_val)
                ent_list.append(ent_val)
                heatmap_eq_imp.append(eq_imp_val)
                heatmap_ent.append(ent_val)

                filename = f"prob_{p_idx+1}_machine_{i+1}.json"
                save_execution_log(log_data, filename=filename, directory=self.run_dir)

            avg_eq_imp.append(sum(eq_imp_list) / len(eq_imp_list))
            avg_eq_sub.append(sum(eq_sub_list) / len(eq_sub_list))
            avg_eq_sub_norm.append(sum(eq_sub_norm_list) / len(eq_sub_norm_list))
            avg_ent.append(sum(ent_list) / len(ent_list))

        # Plots
        if config.GENERATE_PLOTS:
            metrics_plot_path = os.path.join(self.run_dir, "metrics_vs_transition_probability.png")
            plot_metrics_vs_frequency(trans_probs, avg_eq_imp, avg_eq_sub, avg_eq_sub_norm, avg_ent,
                                      save_path=metrics_plot_path)

            heatmap_plot_path = os.path.join(self.run_dir, "heatmap_eq_imp_vs_ent.png")
            plot_equanimity_vs_entanglement_heatmap(heatmap_eq_imp, heatmap_ent, bins=25,
                                                    save_path=heatmap_plot_path)

        self.log_message("Experiment 1 completed.")

def run_experiment():
    exp = Experiment1(tape_length=5, num_states=4, total_bits=10)
    exp.run_experiment()

if __name__ == "__main__":
    run_experiment()
