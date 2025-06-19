from src.experiments.config import LOGS_PATH, TRANSITION_PROBABILITIES, NUM_EXPERIMENTS_PER_CONFIG
from src.experiments.utils.logger import create_subdirectory

class Experiment:
    def __init__(self, experiment_name):
        assert experiment_name
        
        # Directory containing all the executions for this experiment "experimentX"
        self.base_dir = create_subdirectory(name=experiment_name, parent=LOGS_PATH)
        # Directory containing a specific run of the experiment (timestamped)
        self.run_dir = create_subdirectory(parent=self.base_dir, timestamped=True)
        # Directory containing the plots for the experiment
        self.plot_directory = create_subdirectory(name="Plots", parent=self.run_dir)
        
        # Transition probabilities
        self.transition_probabilities = TRANSITION_PROBABILITIES
        
        # Number of experiments per configuration
        self.num_experiments_per_config = NUM_EXPERIMENTS_PER_CONFIG

    def run_experiment(self):
        raise NotImplementedError("Subclasses must implement run_experiment()")