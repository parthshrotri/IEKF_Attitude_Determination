import yaml
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

import utils.plot_utils as plot_utils

from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("config_file")
args = parser.parse_args()

with open(f"sim_configs/{args.config_file}.yaml", "r") as f:
    config = yaml.safe_load(f)

output_dir = Path(config["output_files"]["dir"])
fig_dir = output_dir / "analysis_figs"
fig_dir.mkdir(parents=True, exist_ok=True)
print("Making GIF from camera images...")
plot_utils.make_cam_gif(output_dir, fig_dir / "camera.gif", fps=5)

print("Loading simulation log data...")
data_file = output_dir / "simulation_log.npz"
sim_data = np.load(data_file, allow_pickle=True)

# Get truth
time_true_att, history_true_att = plot_utils.get_log_arrays(sim_data, "truth/icrf_to_body")
time_err_att, history_err_att = plot_utils.get_log_arrays(sim_data, "truth/icrf_to_body_err")

# Get estimate errors and covariances
time_est_error_MEKF, att_est_error_history_MEKF = plot_utils.get_log_arrays(sim_data, "est/MEKF/icrf_to_cam_error")
time_est_bias_MEKF, bias_est_history_MEKF = plot_utils.get_log_arrays(sim_data, "est/MEKF/gyro_bias_error")
time_MEKF_cov_diag, MEKF_cov_history_diag = plot_utils.get_log_arrays(sim_data, "est/MEKF/error_cov_diag")

time_est_error_LIEKF, att_est_error_history_LIEKF = plot_utils.get_log_arrays(sim_data, "est/LIEKF/icrf_to_cam_error")
time_est_bias_LIEKF, bias_est_history_LIEKF = plot_utils.get_log_arrays(sim_data, "est/LIEKF/gyro_bias_error")
time_LIEKF_cov_diag, LIEKF_cov_history_diag = plot_utils.get_log_arrays(sim_data, "est/LIEKF/error_cov_diag")

time_diff_MEKF_LIEKF, att_diff_history_MEKF_LIEKF = plot_utils.get_log_arrays(sim_data, "est/MEKF_LIEKF_icrf_to_cam_diff")

# Get control torque
time_torque, ctrl_torque_history = plot_utils.get_log_arrays(sim_data, "ctrl/torque")

print("Plotting results...")
# Plot Attitude Quaternion Components
plot_utils.plot_quaternion_components(
    time_true_att, history_true_att,
    title="True Attitude Quaternion Components (ICRF to Body)",
    label_prefix="Att",
    fig_path=fig_dir / "attitude_quaternion_components"
)

# Plot Attitude Point Error Quaternion Components
plot_utils.plot_quaternion_components(
    time_err_att, history_err_att,
    title="Attitude Pointing Error Quaternion Components (ICRF to Body)",
    label_prefix="Att Error",
    fig_path=fig_dir / "attitude_error_quaternion_components"
)

# Plot estimation errors and covariances
# MEKF Error Plots
plot_utils.plot_error_quaternion_components(
    time_est_error_MEKF, att_est_error_history_MEKF,
    title="Attitude Estimation Error Quaternion Components (ICRF to Camera) - MEKF",
    label_prefix="Att Est Error MEKF",
    fig_path=fig_dir / "attitude_estimation_error_quaternion_components_MEKF",
    error_cov_history_diag=(time_MEKF_cov_diag, MEKF_cov_history_diag[:,0:3])
)

plot_utils.plot_3_axes(
    time_est_bias_MEKF, bias_est_history_MEKF,
    title="Gyro Bias Estimation Error History - MEKF",
    ylabels=["Bias Error X (rad/s)", "Bias Error Y (rad/s)", "Bias Error Z (rad/s)"],
    fig_path=fig_dir / "gyro_bias_estimation_error_history_MEKF",
    error_cov_history_diag=(time_MEKF_cov_diag, MEKF_cov_history_diag[:,3:6])
)

# LIEKF Error Plots
plot_utils.plot_error_quaternion_components(
    time_est_error_LIEKF, att_est_error_history_LIEKF,
    title="Attitude Estimation Error Quaternion Components (ICRF to Camera) - LIEKF",
    label_prefix="Att Est Error LIEKF",
    fig_path=fig_dir / "attitude_estimation_error_quaternion_components_LIEKF",
    error_cov_history_diag=(time_LIEKF_cov_diag, LIEKF_cov_history_diag[:,0:3])
)

plot_utils.plot_3_axes(
    time_est_bias_LIEKF, bias_est_history_LIEKF,
    title="Gyro Bias Estimation Error History - LIEKF",
    ylabels=["Bias Error X (rad/s)", "Bias Error Y (rad/s)", "Bias Error Z (rad/s)"],
    fig_path=fig_dir / "gyro_bias_estimation_error_history_LIEKF",
    error_cov_history_diag=(time_LIEKF_cov_diag, LIEKF_cov_history_diag[:,3:6])
)

# Plot MEKF vs LIEKF attitude estimation error difference
plot_utils.plot_quaternion_components(
    time_diff_MEKF_LIEKF, att_diff_history_MEKF_LIEKF,
    title="Difference in Attitude Estimates (ICRF to Camera) - MEKF vs LIEKF",
    label_prefix="Att Est Diff MEKF-LIEKF",
    fig_path=fig_dir / "attitude_estimate_difference_quaternion_components_MEKF_LIEKF"
)

# Plot control torque
plot_utils.plot_3_axes(
    time_torque, ctrl_torque_history,
    title="Control Torque History",
    ylabels=["Torque X (Nm)", "Torque Y (Nm)", "Torque Z (Nm)"],
    fig_path=fig_dir / "control_torque_history"
)