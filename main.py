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

simulation = Simulator(config)
simulation.run()