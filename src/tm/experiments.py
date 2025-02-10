# tm/experiments.py

from src.analysis.analysis import project_history_to_boolean_function
from src.tm.metrics.equanimities import equanimity_importance, equanimity_subsets, equanimity_subsets_normalized
from src.tm.metrics.entanglement import entanglement
from src.tm.logger import save_execution_log

def run_single_experiment(tm, trans_prob=None):
    """
    Ejecuta la máquina de Turing 'tm', calcula las métricas y construye un diccionario log_data.
    Devuelve:
      log_data (dict): Información detallada del experimento, incluyendo:
                        - "config_history": El historial completo de configuraciones (cada configuración es un string de 10 bits).
                        - "projected_function": La proyección (tabla de verdad de 32 bits, representada como entero)
                                                obtenida a partir de todo el historial.
      metrics (tuple): (eq_imp, eq_sub, eq_sub_norm, ent)
    """
    # Ejecutar la máquina de Turing.
    result = tm.run()
    
    # Extraer el vector de historial.
    history_func = tm.get_history_function()

    # Calcular métricas.
    eq_imp = equanimity_importance(history_func, 10)
    eq_sub = equanimity_subsets(history_func, 10)
    eq_sub_norm = equanimity_subsets_normalized(history_func, 10)
    ent = entanglement(history_func, 10)

    # Preparar información de la función de transición.
    transition_function = {
        f"{state},{symbol}": [next_state, write_symbol, direction]
        for (state, symbol), (next_state, write_symbol, direction)
        in tm.transition_function.items()
    }
    
    num_steps = len(tm.config_history) - 1

    # --- Unificar el campo "projected_function" ---
    # Se utilizará el historial completo (tm.config_history) para calcular la proyección.
    # Si no hay historial (por alguna razón), se usa la configuración actual.
    if tm.config_history:
        projected_function = project_history_to_boolean_function(tm.config_history)
    else:
        projected_function = project_history_to_boolean_function([tm.get_configuration()])
    
    # Construir log_data con la información completa.
    log_data = {
        "transition_probability": trans_prob,  # Indica qué probabilidad se usó.
        "tm_parameters": {
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
        "config_history": tm.config_history,      # Guarda el historial completo (lista de strings de 10 bits)
        "projected_function": projected_function    # Proyección unificada (entero que representa la tabla de verdad de 32 bits)
    }

    return log_data, (eq_imp, eq_sub, eq_sub_norm, ent)


def display_metrics(machine_index, metrics):
    eq_imp, eq_sub, eq_sub_norm, ent = metrics
    header = f"\n{'-' * 20} Machine {machine_index + 1} {'-' * 20}"
    print(header)
    print(f"Equanimity = {eq_imp}")
    print(f"Equanimity (subsets) = {eq_sub}")
    print(f"Equanimity (subsets normalized) = {eq_sub_norm}")
    print(f"Entanglement = {ent}")

def log_experiment(log_data, filename, run_directory):
    save_execution_log(log_data, filename=filename, directory=run_directory)
