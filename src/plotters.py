import matplotlib.pyplot as plt

def plot_metrics_vs_frequency(frequencies, eq_imp_values, eq_sub_values, eq_sub_norm_values, entanglement_values, save_path=None):
    """
    Plot the relation between transition frequency and the final metric values.
    
    Parameters:
        frequencies (list or np.array): The probability values for adding a transition.
        eq_imp_values (list or np.array): Equanimity Importance values.
        eq_sub_values (list or np.array): Equanimity Subsets values.
        eq_sub_norm_values (list or np.array): Equanimity Subsets Normalized values.
        entanglement_values (list or np.array): Entanglement values.
        save_path (str, optional): File path to save the plot image.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, eq_imp_values, marker='o', label='Equanimity Importance')
    plt.plot(frequencies, eq_sub_values, marker='s', label='Equanimity Subsets')
    plt.plot(frequencies, eq_sub_norm_values, marker='^', label='Equanimity Subsets Normalized')
    plt.plot(frequencies, entanglement_values, marker='x', label='Entanglement')
    
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
    
    Parameters:
        equanimity_values (list or np.array): Equanimity Importance values.
        entanglement_values (list or np.array): Entanglement values.
        bins (int): Number of bins (gridsize) for the hexbin plot.
        save_path (str, optional): File path to save the plot image.
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


# plotters.py
import matplotlib.pyplot as plt

def plot_frequency_histogram(grouped_freq, title="Frecuencia de funciones por tamaño mínimo"):
    """
    Genera y muestra un histograma (barras) que representa la frecuencia total
    de funciones observadas agrupadas por el tamaño mínimo del circuito.
 
    Args:
        grouped_freq (dict): Diccionario { circuit_size: frequency }.
        title (str): Título del gráfico.
    """
    # Ordenar las claves (pueden ser números o incluir 'desconocido')
    keys_numeric = [k for k in grouped_freq.keys() if isinstance(k, int)]
    keys_numeric.sort()
    labels = [str(k) for k in keys_numeric]
    values = [grouped_freq[k] for k in keys_numeric]
 
    # Si hay grupo "desconocido", se añade al final
    if "desconocido" in grouped_freq:
        labels.append("desconocido")
        values.append(grouped_freq["desconocido"])
 
    plt.figure(figsize=(8, 6))
    plt.bar(labels, values, color='skyblue')
    plt.xlabel("Tamaño mínimo del circuito")
    plt.ylabel("Frecuencia observada")
    plt.title(title)
    plt.grid(axis='y', alpha=0.75)
    plt.show()
