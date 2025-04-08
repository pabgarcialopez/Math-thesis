# src/experiments/utils/plotters.py
import matplotlib.pyplot as plt
import numpy as np
from src.config import SHOULD_PLOT

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
    if SHOULD_PLOT:
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
    if SHOULD_PLOT:
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
    if SHOULD_PLOT:
        plt.show()

def plot_probabilities_vs_metrics_difference(probabilities, eq_imp_diff, eq_sub_diff, eq_sub_norm_diff, ent_diff, save_path=None):
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
    if SHOULD_PLOT:
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
    if SHOULD_PLOT:
        plt.show()


# EXPERIMENT 4

def plot_complexity_vs_probability(trans_probs, avg_terms, avg_literals, save_path=None):
    """
    Plots line chart of (avg_terms, avg_literals) vs trans_probs.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(trans_probs, avg_terms, marker='o', label='Número de términos')
    plt.plot(trans_probs, avg_literals, marker='s', label='Total de literales')
    plt.xlabel('Probabilidad de transición')
    plt.ylabel('Complejidad (DNF)')
    plt.title('Complejidad mínima vs Probabilidad de transición')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if SHOULD_PLOT:
        plt.show()

def plot_terms_literals_freqs_histogram(terms_list, literals_list, n_bits, title, save_path=None):
    """
    Double histogram for #terms and #literals.
    """
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Terms
    if terms_list:
        t_min = min(terms_list)
        t_max = max(terms_list)
    else:
        t_min, t_max = 0, 0
    bins_terms = np.arange(t_min, t_max + 2) - 0.5
    axs[0].hist(terms_list, bins=bins_terms, color='skyblue', edgecolor='black', align='mid')
    axs[0].set_title(f'Histograma de términos ({title})')
    axs[0].set_xlabel('Número de términos')
    axs[0].set_ylabel('Frecuencia')
    axs[0].grid(True)

    # Literals
    if literals_list:
        l_min = min(literals_list)
        l_max = max(literals_list)
    else:
        l_min, l_max = 0, 0
    bins_literals = np.arange(l_min, l_max + 2) - 0.5
    axs[1].hist(literals_list, bins=bins_literals, color='salmon', edgecolor='black', align='mid')
    axs[1].set_title(f'Histograma de literales ({title})')
    axs[1].set_xlabel('Total de literales')
    axs[1].set_ylabel('Frecuencia')
    axs[1].grid(True)

    fig.suptitle(f'Distribución de la complejidad en funciones mínimas ({n_bits} bits)')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if SHOULD_PLOT:
        plt.show()

def plot_random_vs_tm_comparison(results_by_bits, save_path=None):
    """
    Given a dictionary results_by_bits mapping each total_bits value to a dict:
      {
         "random": (avg_terms, avg_literals, worst_case_terms, worst_case_literals),
         "tm": (avg_terms, avg_literals, worst_case_terms, worst_case_literals, num_steps)
      }
    this function produces a 2×2 grid plot:
      - Top left: Average # of terms vs. total bits
      - Top right: Average # of literals vs. total bits
      - Bottom left: Worst-case # of terms vs. total bits
      - Bottom right: Worst-case # of literals vs. total bits
    """
    sorted_bits = sorted(results_by_bits.keys())
    x_vals = sorted_bits
    rand_avg_terms = []
    tm_avg_terms = []
    rand_avg_literals = []
    tm_avg_literals = []
    rand_worst_terms = []
    tm_worst_terms = []
    rand_worst_literals = []
    tm_worst_literals = []

    for tb in sorted_bits:
        rand_data = results_by_bits[tb].get("random", (0, 0, 0, 0))
        tm_data = results_by_bits[tb].get("tm", (0, 0, 0, 0))
        rand_avg_terms.append(rand_data[0])
        rand_avg_literals.append(rand_data[1])
        rand_worst_terms.append(rand_data[2])
        rand_worst_literals.append(rand_data[3])
        tm_avg_terms.append(tm_data[0])
        tm_avg_literals.append(tm_data[1])
        tm_worst_terms.append(tm_data[2])
        tm_worst_literals.append(tm_data[3])

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    # Top left: Average terms
    axs[0, 0].plot(x_vals, rand_avg_terms, marker='o', label='Random')
    axs[0, 0].plot(x_vals, tm_avg_terms, marker='s', label='TM')
    axs[0, 0].set_title('Average Terms')
    axs[0, 0].set_xlabel('Total Bits')
    axs[0, 0].set_ylabel('Average # of Terms')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Top right: Average literals
    axs[0, 1].plot(x_vals, rand_avg_literals, marker='o', label='Random')
    axs[0, 1].plot(x_vals, tm_avg_literals, marker='s', label='TM')
    axs[0, 1].set_title('Average Literals')
    axs[0, 1].set_xlabel('Total Bits')
    axs[0, 1].set_ylabel('Average # of Literals')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Bottom left: Worst-case terms
    axs[1, 0].plot(x_vals, rand_worst_terms, marker='o', label='Random')
    axs[1, 0].plot(x_vals, tm_worst_terms, marker='s', label='TM')
    axs[1, 0].set_title('Worst-case Terms')
    axs[1, 0].set_xlabel('Total Bits')
    axs[1, 0].set_ylabel('Worst-case # of Terms')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Bottom right: Worst-case literals
    axs[1, 1].plot(x_vals, rand_worst_literals, marker='o', label='Random')
    axs[1, 1].plot(x_vals, tm_worst_literals, marker='s', label='TM')
    axs[1, 1].set_title('Worst-case Literals')
    axs[1, 1].set_xlabel('Total Bits')
    axs[1, 1].set_ylabel('Worst-case # of Literals')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    fig.tight_layout()
    if save_path: plt.savefig(save_path, bbox_inches='tight')
    if SHOULD_PLOT: plt.show()

def plot_curve_with_max_line(x_values, y_values, max_level, title, xlabel, ylabel, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, marker='o', label='Values')

    # Add y-values as text
    for x, y in zip(x_values, y_values):
        plt.text(x, y, f"{y:.2f}", ha='center', va='bottom', fontsize=9)

    # Add max-level horizontal line
    plt.axhline(max_level, color='r', linestyle='--', label=f'Max: {max_level:.2f}')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()

    if save_path: plt.savefig(save_path, bbox_inches='tight')
    if SHOULD_PLOT: plt.show()


def plot_bucket_histogram(sequence, classes, title, xlabel, ylabel, save_path=None):
    num_classes = len(classes)
    min_val, max_val = min(sequence), max(sequence)
    bucket_width = (max_val - min_val) / num_classes

    # Initialize bucket counts
    counts = [0] * num_classes

    # Count how many values fall into each bucket
    for val in sequence:
        idx = int((val - min_val) / bucket_width)
        if idx == num_classes:
            idx -= 1
        counts[idx] += 1

    # Plot bars using class names as labels
    plt.figure(figsize=(10, 6))
    bar_positions = range(num_classes)
    plt.bar(bar_positions, counts, width=0.6, edgecolor='black')

    # Custom tick labels and axis labels
    plt.xticks(bar_positions, classes)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis='y')

    # Show count on top of each bar
    for i, count in enumerate(counts):
        plt.text(i, count, str(count), ha='center', va='bottom', fontsize=9)

    if save_path: plt.savefig(save_path, bbox_inches='tight')
    if SHOULD_PLOT: plt.show()


def plot_length_vs_complexity_heatmap(steps, complexities, classes_info, save_path=None):
    assert len(steps) == len(complexities)

    step_class_labels = classes_info['step_class_labels']
    complexity_class_labels = classes_info['complexity_class_labels']
    num_step_classes = len(step_class_labels)
    num_complexity_classes = len(complexity_class_labels)

    def create_heatmap():
        step_m, step_M = min(steps), max(steps)
        complexity_m, complexity_M = min(complexities), max(complexities)
        step_delta = (step_M - step_m) / num_step_classes
        complexity_delta = (complexity_M - complexity_m) / num_complexity_classes

        heatmap = [[0 for _ in range(num_complexity_classes)] for _ in range(num_step_classes)]

        for step, complexity in zip(steps, complexities):
            step_class = int((step - step_m) / step_delta)
            complexity_class = int((complexity - complexity_m) / complexity_delta)

            if step_class == num_step_classes: step_class -= 1
            if complexity_class == num_complexity_classes: complexity_class -= 1

            heatmap[step_class][complexity_class] += 1

        return heatmap

    heatmap = np.array(create_heatmap())
    fig, ax = plt.subplots()
    im = ax.imshow(heatmap, cmap='Blues')

    # Label axes
    ax.set_xlabel("Complexity class")
    ax.set_ylabel("Step length class")
    ax.set_xticks(np.arange(num_complexity_classes))
    ax.set_yticks(np.arange(num_step_classes))
    ax.set_xticklabels(complexity_class_labels)
    ax.set_yticklabels(step_class_labels)

    # Annotate heatmap with counts
    for i in range(num_step_classes):
        for j in range(num_complexity_classes):
            ax.text(j, i, heatmap[i, j], ha="center", va="center", color="black")

    plt.title("Heatmap: Step Length vs Complexity")

    if save_path: plt.savefig(save_path, bbox_inches='tight')
    if SHOULD_PLOT: plt.show()
