# tm/experiments.py

from tm.metrics.equanimities import equanimity_importance, equanimity_subsets, equanimity_subsets_normalized
from tm.metrics.entanglement import entanglement
from tm.logger import save_execution_log

def run_single_experiment(tm, trans_prob=None):
    """
    Runs the Turing Machine `tm`, computes metrics, and builds a log_data dict.
    Returns:
      log_data (dict): Detailed info for logging
      metrics (tuple): (eq_imp, eq_sub, eq_sub_norm, ent)
    """
    # Run the Turing Machine.
    result = tm.run()
    
    # Extract the history vector.
    vector = tm.get_history_vector()

    # Compute metrics.
    eq_imp = equanimity_importance(vector, 10)
    eq_sub = equanimity_subsets(vector, 10)
    eq_sub_norm = equanimity_subsets_normalized(vector, 10)
    ent = entanglement(vector, 10)

    # Build log_data, including the transition probability if provided.
    transition_function = {
        f"{state},{symbol}": [next_state, write_symbol, direction]
        for (state, symbol), (next_state, write_symbol, direction)
        in tm.transition_function.items()
    }
    
    num_steps = len(tm.config_history) - 1

    log_data = {
        "transition_probability": trans_prob,  # So we know which prob was used
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
        }
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
