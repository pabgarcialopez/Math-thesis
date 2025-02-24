# src/experiments/base_experiment.py
import os
from src.config import LOGS_DIR
from src.utils.logger import get_timestamped_log_dir

class BaseExperiment:
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.run_dir = self.get_experiment_run_dir()

    def get_experiment_run_dir(self):
        """
        Crea y devuelve un directorio con timestamp para la ejecución del experimento.
        Estructura: <LOGS_DIR>/<experiment_name>/<timestamp>/
        """
        base_dir = os.path.join(LOGS_DIR, self.experiment_name)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        run_dir = get_timestamped_log_dir(base_directory=base_dir)
        return run_dir

    def log_message(self, message, prefix="[INFO]"):
        print(f"{prefix} {message}")

    def run_experiment(self):
        """
        Método abstracto. Las subclases deben implementar run_experiment().
        """
        raise NotImplementedError("Subclasses must implement run_experiment()")

def project_history_to_boolean_function(config_history):
    """
    Dada una lista de configuraciones en binario (por ejemplo, total_config_bits),
    proyecta a los primeros 5 (por defecto) bits.
    """
    observed = set()
    for config in config_history:
        tape_bits = config[:5]  # 5 bits fijos. Cambiarlo segun convenga
        observed.add(tape_bits)
    
    truth_table = ""
    for i in range(32):
        pattern = format(i, '05b')
        truth_table += "1" if pattern in observed else "0"
    
    return int(truth_table, 2)
