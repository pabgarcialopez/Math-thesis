# --------------------------------------------
# EXPERIMENTS CONFIG
# --------------------------------------------

# Logging options
import os
import numpy as np

LOGS_PATH = "logs"
DATA_PATH = "data"
DATASET_PATH = os.path.join(DATA_PATH, "dataset_n5_10_puertas.txt")

# Number of experiments per configuration
NUM_EXPERIMENTS_PER_CONFIG = 1000        

# Transition probabilities parameters.
MIN_PROB = 0.1
MAX_PROB = 1.0
PROB_STEP = 0.1
DEFAULT_TRANSITION_PROBABILITY = 0.5
TRANSITION_PROBABILITIES = np.linspace(MIN_PROB, MAX_PROB, num=int((MAX_PROB - MIN_PROB) / PROB_STEP)+1)

# Plotting and logging options
SHOULD_LOG = True
SHOULD_SAVE_PLOT = True
SHOULD_SHOW_PLOT = False

