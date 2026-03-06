import argparse
import yaml

from Simulation.simulator import Simulator

# --- Parse command-line argument ---
parser = argparse.ArgumentParser(description="Run spacecraft simulation trial from YAML.")
parser.add_argument("config_file", type=str, help="Path to the YAML configuration file.")
args = parser.parse_args()

# --- Load YAML configuration ---
print(f"Loading configuration from sim_configs/{args.config_file}.yaml...")
with open(f"sim_configs/{args.config_file}.yaml", "r") as f:
    config = yaml.safe_load(f)

if config.get("MC_runs", 0) > 0:
    print(f"Running Monte Carlo simulations with {config['MC_runs']} runs...")
    for i in range(config["MC_runs"]):
        print(f"--- Monte Carlo Run {i+1}/{config['MC_runs']} ---")
        simulation = Simulator(config, MC_run=i)
        simulation.run()
else:
    simulation = Simulator(config)
    simulation.run()