import os
import json
import glob
from collections import defaultdict
import matplotlib.pyplot as plt
from src.plotters import plot_frequency_histogram

# ============================================================
# Determinar la ruta base (raíz del proyecto)
# ============================================================
# __file__ es algo como "codeTFGmath/src/analysis/analysis.py"
# Queremos llegar a "codeTFGmath" (la raíz del proyecto).
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print("BASE_DIR:", BASE_DIR)

# Ahora definimos las rutas absolutas:
# Como los logs están en "logs/" a nivel de raíz, usamos:
LOG_DIR = os.path.join(BASE_DIR, "logs")
# print("LOG_DIR:", LOG_DIR)

# El dataset se encuentra en "src/analysis", ya que main.py está en src, pero el dataset
# se mantiene en la carpeta analysis dentro de src.
DATASET_PATH = os.path.join(BASE_DIR, "src", "analysis", "dataset_n5_10_puertas.txt")
# print("DATASET_PATH:", DATASET_PATH)

# ============================================================
# Funciones para recolectar y procesar los logs de los experimentos
# ============================================================
def load_experiment_logs(log_directory):
    """
    Recorre recursivamente el directorio de logs y carga todos los archivos JSON de experimentos,
    incluyendo aquellos en subdirectorios (por ejemplo, carpetas con timestamps).
    
    Args:
        log_directory (str): Ruta al directorio que contiene los logs.
        
    Returns:
        list: Lista de diccionarios (uno por log) con los datos de cada experimento.
    """
    logs = []
    if not os.path.exists(log_directory):
        return logs

    pattern = os.path.join(log_directory, '**', '*.json')
    
    files = glob.glob(pattern, recursive=True)
    
    for filepath in files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                logs.append(data)
        except Exception as e:
            print(f"Error leyendo {filepath}: {e}")
    return logs

def project_history_to_boolean_function(config_history):
    """
    Dada la historia de configuraciones (cada una de 10 bits, como string),
    proyecta la parte de la cinta (los primeros 5 bits) y construye la función
    booleana de 5 bits (tabla de verdad de 32 bits) definida de la siguiente manera:
      - Para cada uno de los 2^5 = 32 posibles patrones (ordenados lexicográficamente),
        se asigna 1 si dicho patrón aparece al menos una vez en la proyección,
        o 0 en caso contrario.
    
    Args:
        config_history (list of str): Lista de configuraciones de 10 bits (cada string).
        
    Returns:
        int: Representación entera de la función booleana de 32 bits.
    """

    # Obtenemos las configuraciones proyectadas
    observed = set()
    for config in config_history:
        tape_bits = config[:5]
        observed.add(tape_bits)
    
    # Construimos la tabla de verdad
    truth_table = ""
    for i in range(32):
        pattern = format(i, '05b')
        truth_table += "1" if pattern in observed else "0"
    
    return int(truth_table, 2)

def collect_projected_functions(log_directory, config_choice="final"):
    """
    Recolecta las funciones proyectadas de los experimentos.
    
    Para cada log, se busca:
      - Si existe el campo "projected_function", se usa directamente.
      - Si no, se utiliza "config_history" para calcular la función proyectada.
      
    Args:
        log_directory (str): Ruta al directorio donde se encuentran los logs.
        config_choice (str): Opción para elegir la configuración ("initial", "middle" o "final").
                             Se usa si se requiere extraer la proyección a partir del historial.
                             
    Returns:
        dict: Diccionario con las funciones proyectadas y su frecuencia.
              { function_int: count }
    """
    logs = load_experiment_logs(log_directory)
    freq_dict = defaultdict(int)
    for log in logs:
        if "projected_function" in log:
            func_val = log["projected_function"]
        elif "config_history" in log:
            history = log["config_history"]
            func_val = project_history_to_boolean_function(history)
        else:
            continue

        freq_dict[func_val] += 1

    return dict(freq_dict)

def load_dataset(filepath):
    """
    Carga el dataset de funciones a partir de un archivo de texto.
    
    Se espera que cada línea del archivo tenga el formato:
      <function_code> <circuit_size>
      
    Args:
        filepath (str): Ruta al archivo del dataset.
        
    Returns:
        dict: Diccionario con la información del dataset.
              { function_code (int): circuit_size (int) }
    """
    dataset = {}
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                func_code = int(parts[0])
                circuit_size = int(parts[1])
                dataset[func_code] = circuit_size
    except Exception as e:
        print(f"Error al cargar el dataset: {e}")
    return dataset

def analyze_representation(freq_dict, dataset):
    """
    Analiza la representación de las funciones obtenidas en los experimentos
    en relación con el dataset.
    
    Args:
        freq_dict (dict): Diccionario { function_int: count } obtenido de los experimentos.
        dataset (dict): Diccionario { function_int: circuit_size } obtenido del dataset.
        
    Returns:
        dict: Diccionario agrupado por tamaño mínimo del circuito.
              { circuit_size: total_frequency_observed }
    """
    grouped_freq = defaultdict(int)
    missing = 0
    for func, count in freq_dict.items():
        if func in dataset:
            size = dataset[func]
            grouped_freq[size] += count
        else:
            grouped_freq["desconocido"] += count
            missing += count
    if missing:
        print(f"Advertencia: {missing} funciones de los experimentos no se encontraron en el dataset.")
    return dict(grouped_freq)

def compute_ratios(grouped_freq, dataset):
    """
    Calcula el ratio (apariciones en experimentos / número de funciones en el dataset)
    para cada tamaño de circuito presente en el dataset.
    
    Args:
        grouped_freq (dict): Diccionario obtenido de los experimentos, donde la clave es el tamaño
                             y el valor es el número total de apariciones de funciones de ese tamaño.
        dataset (dict): Diccionario obtenido del dataset, donde la clave es la función (entero) y el valor
                        es el tamaño mínimo del circuito.
                        
    Imprime el ratio para cada tamaño.
    """
    # Agrupar el dataset por tamaño: contar cuántas funciones hay para cada tamaño.
    dataset_counts = defaultdict(int)
    for func, size in dataset.items():
        dataset_counts[size] += 1

    print("\nRatio (apariciones de experimentos / número de funciones en el dataset):")
    # Para cada tamaño presente en grouped_freq, se calcula el ratio si existe en dataset_counts.
    for size, freq in sorted(grouped_freq.items(), key=lambda x: (x[0] if isinstance(x[0], int) else 9999)):
        if isinstance(size, int) and size in dataset_counts and dataset_counts[size] > 0:
            ratio = freq / dataset_counts[size]
            print(f"Tamaño {size}: {freq} / {dataset_counts[size]} = {ratio*100:.8f}%")
        else:
            print(f"Tamaño {size}: {freq} (no se puede calcular ratio)")


def main():
    print("Recolectando funciones proyectadas de los experimentos...")
    freq_dict = collect_projected_functions(LOG_DIR, config_choice="final")
    print(f"Se han obtenido {len(freq_dict)} funciones distintas de {sum(freq_dict.values())} experimentos.")
    
    print("Cargando dataset...")
    dataset = load_dataset(DATASET_PATH)
    print(f"Dataset cargado con {len(dataset)} funciones.")
    
    print("Analizando representación (agrupado por tamaño mínimo del circuito)...")
    grouped_freq = analyze_representation(freq_dict, dataset)
    for size, freq in sorted(grouped_freq.items(), key=lambda x: (x[0] if isinstance(x[0], int) else 9999)):
        print(f"Tamaño {size}: {freq} apariciones")

    # Calcular y mostrar el ratio para cada tamaño
    compute_ratios(grouped_freq, dataset)
    
    print("Generando histograma...")
    plot_frequency_histogram(grouped_freq)

if __name__ == "__main__":
    main()
