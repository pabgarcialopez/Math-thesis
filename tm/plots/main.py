# tm/plots/main.py

from tqdm import tqdm
import numpy as np

from tm.generators import generate_tm_input_pairs
from tm.experiments import run_single_experiment, log_experiment
from tm.plots.plots import plot_metrics_vs_frequency, plot_equanimity_vs_entanglement_heatmap
from tm.logger import get_timestamped_log_dir

def main():
    trans_probs = np.linspace(0.1, 1.0, 10)
    num_experiments_per_prob = 50
    
    run_directory = get_timestamped_log_dir()
    print(f"Saving logs to: {run_directory}")

    avg_eq_imp, avg_eq_sub, avg_eq_sub_norm, avg_ent = [], [], [], []
    heatmap_eq_imp, heatmap_ent = [], []

    for p_idx, prob in enumerate(tqdm(trans_probs, desc="Processing Transition Probabilities", colour='green')):
        eq_imp_list, eq_sub_list, eq_sub_norm_list, ent_list = [], [], [], []
        
        machines = generate_tm_input_pairs(num_experiments_per_prob, trans_prob=prob)
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

        avg_eq_imp.append(sum(eq_imp_list)/len(eq_imp_list))
        avg_eq_sub.append(sum(eq_sub_list)/len(eq_sub_list))
        avg_eq_sub_norm.append(sum(eq_sub_norm_list)/len(eq_sub_norm_list))
        avg_ent.append(sum(ent_list)/len(ent_list))

    plot_metrics_vs_frequency(
        trans_probs, 
        avg_eq_imp, 
        avg_eq_sub, 
        avg_eq_sub_norm, 
        avg_ent,
        save_path=f"{run_directory}/metrics_vs_transition_probability.png"
    )

    plot_equanimity_vs_entanglement_heatmap(
        heatmap_eq_imp, 
        heatmap_ent, 
        bins=25,
        save_path=f"{run_directory}/heatmap_eq_imp_vs_ent.png"
    )

if __name__ == "__main__":
    main()
