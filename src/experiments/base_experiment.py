from src.experiments.config import LOGS_PATH, TRANSITION_PROBABILITIES, NUM_EXPERIMENTS_PER_CONFIG
from src.experiments.utils.logger import create_subdirectory
from src.tm.utils import generate_turing_machines, serialize_turing_machine

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
    
    def run_and_collect(
        self,
        config,
        probabilities,
        num_machines,
        metric_callback,
        aggregate_callback=None,
        log_each_machine=True,
        directory=None
    ):
        """
        Iterates over 'probabilities'. For each probability 'p':
          1) Creates 'n_machines' Turing Machines via 'create_machine_fn(base_config, p)'
          2) Runs each machine, collects metrics via 'metric_callback'
          3) Optionally logs each machine's result if 'log_each_machine' is True
          4) Aggregates metrics if 'aggregate_callback' is provided

        Returns a list of dicts, each with:
          {
            "probability": p,
            "metrics_list": [ ... raw metrics for each machine ... ],
            "aggregated": ... result of aggregate_callback(...) if provided ...
          }
        """
        if directory is None:
            directory = self.run_dir

        results = []
        for idx, probability in enumerate(probabilities):
            metrics_list = []
            turing_machines = generate_turing_machines(num_machines, config, probability)
            for i, tm in enumerate(turing_machines):
                tm.run()
                metrics = metric_callback(tm)
                metrics_list.append(metrics)

                if self.should_log() and log_each_machine:
                    filename = f"prob_{idx+1}_machine_{i+1}.json"
                    self.log_data({
                        "turing_machine": serialize_turing_machine(tm),
                        "metrics": metrics
                    }, filename=filename, directory=directory)

            aggregated = None
            if aggregate_callback is not None:
                aggregated = aggregate_callback(metrics_list)

            results.append({
                "probability": float(probability),
                "metrics_list": metrics_list,
                "aggregated": aggregated
            })

        return results
