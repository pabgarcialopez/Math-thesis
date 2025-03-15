# src/experiments/utils/plotters.py
import matplotlib.pyplot as plt
import numpy as np

# ------------------------
# Experiment 1
# ------------------------

def plot_probabilities_vs_metrics(probabilities, eq_imp_values, eq_sub_values, eq_sub_norm_values, entanglement_values, save_path=None):
    """
    Plot the relation between transition frequency and the final metric values.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(probabilities, eq_imp_values, marker='o', label='Equanimity Importance')
    plt.plot(probabilities, eq_sub_values, marker='s', label='Equanimity Subsets')
    plt.plot(probabilities, eq_sub_norm_values, marker='^', label='Equanimity Subsets Normalized')
    plt.plot(probabilities, entanglement_values, marker='x', label='Entanglement')
    
    plt.xlabel('Transition Frequency')
    plt.ylabel('Metric Value')
    plt.title('Metrics vs Transition Frequency')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_equanimity_vs_entanglement_heatmap(equanimity_values, entanglement_values, bins=20, save_path=None):
    """
    Plot a hexbin heatmap showing the relation between equanimity (importance version) and entanglement.
    """
    plt.figure(figsize=(8, 6))
    hb = plt.hexbin(equanimity_values, entanglement_values, gridsize=bins, cmap='Reds', mincnt=1)
    plt.colorbar(hb, label='Count')
    plt.xlabel('Equanimity Importance')
    plt.ylabel('Entanglement')
    plt.title('Heatmap: Equanimity vs Entanglement')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

# ------------------------
# Experiment 2
# ------------------------

def plot_frequency_histogram(grouped_freq, title="Frecuencia de funciones por tamaño mínimo", save_path=None):
    """
    Generates and shows a histogram that represents the frequency of observed functions.
    """
    keys_numeric = [k for k in grouped_freq.keys() if isinstance(k, int)]
    keys_numeric.sort()
    labels = [str(k) for k in keys_numeric]
    values = [grouped_freq[k] for k in keys_numeric]
 
    if "desconocido" in grouped_freq:
        labels.append("desconocido")
        values.append(grouped_freq["desconocido"])
 
    plt.figure(figsize=(8, 6))
    plt.bar(labels, values, color='skyblue')
    plt.xlabel("Tamaño mínimo del circuito")
    plt.ylabel("Frecuencia observada")
    plt.title(title)
    plt.grid(axis='y', alpha=0.75)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_metrics_difference_vs_probability(probabilities, eq_imp_diff, eq_sub_diff, eq_sub_norm_diff, ent_diff, save_path=None):
    """
    Plot the difference between the metrics of 10-bit functions and 5-bit functions
    as a function of the transition probability.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(probabilities, eq_imp_diff, marker='o', label='Δ Equanimity Importance')
    plt.plot(probabilities, eq_sub_diff, marker='s', label='Δ Equanimity Subsets')
    plt.plot(probabilities, eq_sub_norm_diff, marker='^', label='Δ Equanimity Subsets Norm.')
    plt.plot(probabilities, ent_diff, marker='x', label='Δ Entanglement')
    
    plt.xlabel('Transition Probability')
    plt.ylabel('Metric Difference (10-bit - 5-bit)')
    plt.title('Difference of Metrics vs Transition Probability')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_metrics_comparison_vs_probability(probabilities, 
                                             eq_imp_5, eq_imp_10, 
                                             eq_sub_5, eq_sub_10,
                                             eq_sub_norm_5, eq_sub_norm_10,
                                             ent_5, ent_10,
                                             save_path=None):
    """
    Create grouped bar charts comparing the 5-bit and 10-bit average metrics for each transition probability.
    
    This function creates a figure with four subplots, one for each metric:
      - Equanimity Importance
      - Equanimity Subsets
      - Equanimity Subsets Normalized
      - Entanglement
      
    For each transition probability (displayed on the x-axis), two bars (side-by-side)
    represent the average value for 5-bit and 10-bit boolean functions.
    """
    x = np.arange(len(probabilities))
    width = 0.35  # width of each bar

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Comparison of Metrics: 10-bit vs 5-bit Functions", fontsize=16)

    # Equanimity Importance subplot
    ax = axes[0, 0]
    ax.bar(x - width/2, eq_imp_5, width, label='5-bit')
    ax.bar(x + width/2, eq_imp_10, width, label='10-bit')
    ax.set_title("Equanimity Importance")
    ax.set_xlabel("Transition Probability")
    ax.set_ylabel("Metric Value")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p:.2f}" for p in probabilities])
    ax.legend()
    ax.grid(True)

    # Equanimity Subsets subplot
    ax = axes[0, 1]
    ax.bar(x - width/2, eq_sub_5, width, label='5-bit')
    ax.bar(x + width/2, eq_sub_10, width, label='10-bit')
    ax.set_title("Equanimity Subsets")
    ax.set_xlabel("Transition Probability")
    ax.set_ylabel("Metric Value")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p:.2f}" for p in probabilities])
    ax.legend()
    ax.grid(True)

    # Equanimity Subsets Normalized subplot
    ax = axes[1, 0]
    ax.bar(x - width/2, eq_sub_norm_5, width, label='5-bit')
    ax.bar(x + width/2, eq_sub_norm_10, width, label='10-bit')
    ax.set_title("Equanimity Subsets Normalized")
    ax.set_xlabel("Transition Probability")
    ax.set_ylabel("Metric Value")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p:.2f}" for p in probabilities])
    ax.legend()
    ax.grid(True)

    # Entanglement subplot
    ax = axes[1, 1]
    ax.bar(x - width/2, ent_5, width, label='5-bit')
    ax.bar(x + width/2, ent_10, width, label='10-bit')
    ax.set_title("Entanglement")
    ax.set_xlabel("Transition Probability")
    ax.set_ylabel("Metric Value")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p:.2f}" for p in probabilities])
    ax.legend()
    ax.grid(True)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()