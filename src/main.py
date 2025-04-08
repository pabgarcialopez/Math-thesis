#!/usr/bin/env python3
import sys
import argparse
from importlib import import_module

import sys

def main():

    parser = argparse.ArgumentParser(description="Select which experiment to run.")
    parser.add_argument("--experiment", required=True, type=str, help="Name of the experiment module to run")
    args = parser.parse_args()

    try:
        experiment_module = import_module(f"src.experiments.{args.experiment}")
    except ImportError as e:
        print(f"Error: could not import experiment '{args.experiment}'. Details: {e}")
        sys.exit(1)

    if hasattr(experiment_module, "run_experiment"):
        experiment_module.run_experiment()
    else:
        print(f"Error: The module src.experiments.{args.experiment} must define run_experiment()")
        sys.exit(1)

if __name__ == "__main__":
    main()