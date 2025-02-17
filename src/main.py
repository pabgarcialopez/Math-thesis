#!/usr/bin/env python3
import sys
import os
import argparse
import importlib

def main():
    parser = argparse.ArgumentParser(description="Select which experiment to run.")
    parser.add_argument("--experiment", type=str, default="experiment1",
                        help="Name of the experiment module to run (e.g., experiment1 or experiment2)")
    args = parser.parse_args()

    try:
        experiment_module = importlib.import_module(f"src.experiments.{args.experiment}")
    except ImportError as e:
        print(f"Error: Could not import experiment '{args.experiment}'. Details: {e}")
        sys.exit(1)

    if hasattr(experiment_module, "run_experiment"):
        experiment_module.run_experiment()
    else:
        print(f"Error: The module src.experiments.{args.experiment} does not define run_experiment().")
        sys.exit(1)

if __name__ == "__main__":
    main()