import os
import time
import yaml
import argparse
from multiprocessing import Pool

import numpy as np

from tqdm import tqdm

from Simulation.simulator import Simulator

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

BASE_SEED = int(time.time())

def run_mc(run_index):
    # Set a unique random seed for each Monte Carlo run
    np.random.seed(BASE_SEED + run_index)
    simulation = Simulator(config, MC_run=run_index)
    simulation.run()

# --- Parse command-line argument ---
parser = argparse.ArgumentParser(description="Run spacecraft simulation trial from YAML.")
parser.add_argument("config_file", type=str, help="Path to the YAML configuration file.")
args = parser.parse_args()

# --- Load YAML configuration ---
with open(f"sim_configs/{args.config_file}.yaml", "r") as f:
    config = yaml.safe_load(f)

if __name__ == "__main__":
    
    if config.get("MC_runs", 0) > 0:
        print(f"Running Monte Carlo simulations with {config['MC_runs']} runs...")
        with Pool() as pool:
            list(tqdm(pool.imap_unordered(run_mc, range(config["MC_runs"])), total=config["MC_runs"]))
    else:
        simulation = Simulator(config)
        simulation.run()