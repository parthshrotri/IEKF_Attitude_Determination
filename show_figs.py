import yaml
import pickle
import argparse
import matplotlib.pyplot as plt

from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("config_file")
args = parser.parse_args()

with open(f"sim_configs/{args.config_file}.yaml", "r") as f:
    config = yaml.safe_load(f)

output_dir = Path(config["output_files"]["dir"])
fig_dir = output_dir / "analysis_figs"

figs = list(fig_dir.glob("*.pkl"))
for fig_file in figs:
    fig = pickle.load(open(fig_file, 'rb'))
    fig.show()
plt.show()