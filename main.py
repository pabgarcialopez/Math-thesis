#!/usr/bin/env python3
import numpy as np

from tm.generators import generate_tm_input_pairs
from tm.experiments import run_single_experiment, log_experiment
from tm.logger import get_timestamped_log_dir
from plotters import plot_metrics_vs_frequency, plot_equanimity_vs_entanglement_heatmap

# Import configuration parameters
import config

def run_experiments(trans_probs, num_experiments, run_directory):
    """
    Run experiments over the provided transition probabilities.
    For each probability value, generate a batch of Turing machines,
    run them, and accumulate metric values.
    
    Returns:
        A tuple containing:
          - trans_probs: The numpy array of probabilities.
          - avg_eq_imp, avg_eq_sub, avg_eq_sub_norm, avg_ent: Lists of average metrics per probability.
          - heatmap_eq_imp, heatmap_ent: Lists of values (for plotting a heatmap) from all experiments.
    """
    avg_eq_imp, avg_eq_sub, avg_eq_sub_norm, avg_ent = [], [], [], []
    heatmap_eq_imp, heatmap_ent = [], []

    for p_idx, prob in enumerate(trans_probs):
        eq_imp_list, eq_sub_list, eq_sub_norm_list, ent_list = [], [], [], []
        machines = generate_tm_input_pairs(num_experiments, trans_prob=prob)
        
        for i, machine in enumerate(machines):
            log_data, metrics = run_single_experiment(machine, trans_prob=prob)
            eq_imp_val, eq_sub_val, eq_sub_norm_val, ent_val = metrics
            
            eq_imp_list.append(eq_imp_val)
            eq_sub_list.append(eq_sub_val)
            eq_sub_norm_list.append(eq_sub_norm_val)
            ent_list.append(ent_val)
            
            heatmap_eq_imp.append(eq_imp_val)
            heatmap_ent.append(ent_val)
            
            filename = f"prob_{p_idx+1}_machine_{i+1}.json"
            log_experiment(log_data, filename, run_directory)
        
        avg_eq_imp.append(sum(eq_imp_list) / len(eq_imp_list))
        avg_eq_sub.append(sum(eq_sub_list) / len(eq_sub_list))
        avg_eq_sub_norm.append(sum(eq_sub_norm_list) / len(eq_sub_norm_list))
        avg_ent.append(sum(ent_list) / len(ent_list))

    return trans_probs, avg_eq_imp, avg_eq_sub, avg_eq_sub_norm, avg_ent, heatmap_eq_imp, heatmap_ent

def main():
    # Create an array of transition probabilities based on the config.
    trans_probs = np.linspace(config.MIN_PROB, config.MAX_PROB, config.NUM_PROBS)
    run_directory = get_timestamped_log_dir()
    print(f"Saving logs to: {run_directory}")
    
    # Run the experiments.
    (trans_probs,
     avg_eq_imp,
     avg_eq_sub,
     avg_eq_sub_norm,
     avg_ent,
     heatmap_eq_imp,
     heatmap_ent) = run_experiments(trans_probs, config.NUM_EXPERIMENTS, run_directory)
    
    # Generate plots if configured to do so.
    if config.GENERATE_PLOTS:
        metrics_plot_path = f"{run_directory}/metrics_vs_transition_probability.png"
        plot_metrics_vs_frequency(
            trans_probs, 
            avg_eq_imp, 
            avg_eq_sub, 
            avg_eq_sub_norm, 
            avg_ent,
            save_path=metrics_plot_path
        )
        heatmap_plot_path = f"{run_directory}/heatmap_eq_imp_vs_ent.png"
        plot_equanimity_vs_entanglement_heatmap(
            heatmap_eq_imp, 
            heatmap_ent, 
            bins=25,
            save_path=heatmap_plot_path
        )

if __name__ == "__main__":
    main()
