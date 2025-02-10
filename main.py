from tm.generators import generate_tm_input_pairs
from tm.experiments import run_single_experiment, display_metrics, log_experiment
from tm.logger import get_timestamped_log_dir

def run_experiments():
    run_directory = get_timestamped_log_dir()
    # print(f"Saving logs to: {run_directory}")
    
    trans_prob = 0.7
    pairs = generate_tm_input_pairs(10, trans_prob=trans_prob)
    
    for i, pair in enumerate(pairs):
        # We pass trans_prob so it’s recorded in the log data
        log_data, metrics = run_single_experiment(pair, trans_prob=trans_prob)
        display_metrics(i, metrics)
        
        # Save log
        filename = f"execution_log_pair_{i+1}.json"
        log_experiment(log_data, filename, run_directory)

def main():
    run_experiments()

if __name__ == "__main__":
    main()
