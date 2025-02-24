# src/experiments/experiment3.py

import os
import numpy as np
import src.config as config
from tqdm import tqdm
from src.plotters import (plot_metrics_vs_frequency, 
                          plot_equanimity_vs_entanglement_heatmap,
                          plot_metrics_difference_vs_probability,
                          plot_metrics_comparison_vs_probability)
from src.utils.logger import save_execution_log
from src.tm.generators import generate_tm_input_pairs
from src.tm.machine import TuringMachine
from src.tm.validators import (equanimity_importance, equanimity_subsets,
                               equanimity_subsets_normalized, entanglement)
from src.experiments.base_experiment import BaseExperiment

class Experiment3(BaseExperiment):
    def __init__(self, tape_length=5, num_states=4):
        super().__init__("experiment3")
        self.tape_length = tape_length
        self.num_states = num_states

    def run_single_experiment(self, tm, trans_prob=None):
        result = tm.run()
        history_func = tm.get_history_function()
        projected_history_func = tm.get_projected_history_function()

        # Métricas para la proyección (tape_length bits)
        eq_imp_5 = equanimity_importance(projected_history_func, tm.tape_length)
        eq_sub_5 = equanimity_subsets(projected_history_func, tm.tape_length)
        eq_sub_5_norm = equanimity_subsets_normalized(projected_history_func, tm.tape_length)
        ent_5 = entanglement(projected_history_func, tm.tape_length)

        # Métricas para la función completa (tape+head+state)
        eq_imp_10 = equanimity_importance(history_func, tm.total_config_bits)
        eq_sub_10 = equanimity_subsets(history_func, tm.total_config_bits)
        eq_sub_10_norm = equanimity_subsets_normalized(history_func, tm.total_config_bits)
        ent_10 = entanglement(history_func, tm.total_config_bits)

        transition_function = {
            f"{state},{symbol}": [next_state, write_symbol, direction]
            for (state, symbol), (next_state, write_symbol, direction) in tm.transition_function.items()
        }
        num_steps = len(tm.config_history) - 1

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
            "metrics_proj": {
                "eq_imp_proj": eq_imp_5,
                "eq_sub_proj": eq_sub_5,
                "eq_sub_norm_proj": eq_sub_5_norm,
                "ent_proj": ent_5,
            },
            "metrics_full": {
                "eq_imp_full": eq_imp_10,
                "eq_sub_full": eq_sub_10,
                "eq_sub_norm_full": eq_sub_10_norm,
                "ent_full": ent_10,
            },
            "config_history": tm.config_history
        }

        return log_data, (eq_imp_5, eq_sub_5, eq_sub_5_norm, ent_5), (eq_imp_10, eq_sub_10, eq_sub_10_norm, ent_10)

    def run_experiment(self):
        self.log_message(f"Saving logs to: {self.run_dir}")

        trans_probs = np.linspace(config.MIN_PROB, config.MAX_PROB, config.NUM_PROBS)
        
        avg_eq_imp_5, avg_eq_sub_5, avg_eq_sub_5_norm, avg_ent_5 = [], [], [], []
        avg_eq_imp_10, avg_eq_sub_10, avg_eq_sub_10_norm, avg_ent_10 = [], [], [], []
        
        for p_idx, prob in enumerate(tqdm(trans_probs, desc="Experiment 3", colour="green")):
            eq_imp_5_list, eq_sub_5_list, eq_sub_5_norm_list, ent_5_list = [], [], [], []
            eq_imp_10_list, eq_sub_10_list, eq_sub_10_norm_list, ent_10_list = [], [], [], []
            
            machines = generate_tm_input_pairs(
                config.NUM_EXPERIMENTS,
                trans_prob=prob,
                tape_length=self.tape_length,
                num_states=self.num_states
            )

            for i, tm in enumerate(machines):
                log_data, metrics5, metrics10 = self.run_single_experiment(tm, trans_prob=prob)
                eq_imp_5_val, eq_sub_5_val, eq_sub_5_norm_val, ent_5_val = metrics5
                eq_imp_10_val, eq_sub_10_val, eq_sub_10_norm_val, ent_10_val = metrics10

                eq_imp_5_list.append(eq_imp_5_val)
                eq_sub_5_list.append(eq_sub_5_val)
                eq_sub_5_norm_list.append(eq_sub_5_norm_val)
                ent_5_list.append(ent_5_val)

                eq_imp_10_list.append(eq_imp_10_val)
                eq_sub_10_list.append(eq_sub_10_val)
                eq_sub_10_norm_list.append(eq_sub_10_norm_val)
                ent_10_list.append(ent_10_val)

                filename = f"prob_{p_idx+1}_machine_{i+1}.json"
                save_execution_log(log_data, filename=filename, directory=self.run_dir)

            # Promedios
            avg_eq_imp_5.append(sum(eq_imp_5_list) / len(eq_imp_5_list))
            avg_eq_sub_5.append(sum(eq_sub_5_list) / len(eq_sub_5_list))
            avg_eq_sub_5_norm.append(sum(eq_sub_5_norm_list) / len(eq_sub_5_norm_list))
            avg_ent_5.append(sum(ent_5_list) / len(ent_5_list))

            avg_eq_imp_10.append(sum(eq_imp_10_list) / len(eq_imp_10_list))
            avg_eq_sub_10.append(sum(eq_sub_10_list) / len(eq_sub_10_list))
            avg_eq_sub_10_norm.append(sum(eq_sub_10_norm_list) / len(eq_sub_10_norm_list))
            avg_ent_10.append(sum(ent_10_list) / len(ent_10_list))

        if config.GENERATE_PLOTS:
            metrics_plot_path = os.path.join(self.run_dir, "metrics_full_vs_transition_probability.png")
            plot_metrics_vs_frequency(trans_probs, avg_eq_imp_10, avg_eq_sub_10, avg_eq_sub_10_norm, avg_ent_10,
                                      save_path=metrics_plot_path)
            
            # Diferencias (full - proj)
            diff_eq_imp = [m10 - m5 for m10, m5 in zip(avg_eq_imp_10, avg_eq_imp_5)]
            diff_eq_sub = [m10 - m5 for m10, m5 in zip(avg_eq_sub_10, avg_eq_sub_5)]
            diff_eq_sub_norm = [m10 - m5 for m10, m5 in zip(avg_eq_sub_10_norm, avg_eq_sub_5_norm)]
            diff_ent = [m10 - m5 for m10, m5 in zip(avg_ent_10, avg_ent_5)]
            diff_plot_path = os.path.join(self.run_dir, "metrics_difference_vs_transition_probability.png")
            plot_metrics_difference_vs_probability(trans_probs, diff_eq_imp, diff_eq_sub, diff_eq_sub_norm, diff_ent,
                                                   save_path=diff_plot_path)
            
            # Gráfica comparativa
            comparison_plot_path = os.path.join(self.run_dir, "metrics_comparison_vs_transition_probability.png")
            plot_metrics_comparison_vs_probability(
                trans_probs,
                avg_eq_imp_5, avg_eq_imp_10,
                avg_eq_sub_5, avg_eq_sub_10,
                avg_eq_sub_5_norm, avg_eq_sub_10_norm,
                avg_ent_5, avg_ent_10,
                save_path=comparison_plot_path
            )

        self.log_message("Experiment 3 completed.")

def run_experiment():
    exp = Experiment3(tape_length=5, num_states=4)
    exp.run_experiment()

if __name__ == "__main__":
    run_experiment()
