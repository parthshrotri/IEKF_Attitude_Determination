import glob
import sys
import tqdm
import pickle
import numpy as np
import matplotlib.pyplot as plt

import glob
import imageio.v2 as imageio

def make_cam_gif(img_dir, output_path, fps=20):
    image_paths = sorted(glob.glob(f'{img_dir}/**/star_field_*.png', recursive=True))

    with imageio.get_writer(output_path, mode='I', fps=fps) as writer:
        for i in tqdm.tqdm(range(len(image_paths)), desc="Processing Frames"):
            image = imageio.imread(image_paths[i])
            writer.append_data(image)
    writer.close()

def get_log_arrays(log, key):
    """Return (time_array, data_array) for a given key"""
    timeseries = log[key].item()

    time_sec = np.array(timeseries["time"])
    data_array = np.array(timeseries["data"])
    return time_sec, data_array

def plot_3_axes(time, data_history, title, ylabels, fig_path, error_cov_history_diag=None):
    colors = ['m', 'g', 'b']
    if error_cov_history_diag is not None:
        time_cov, error_cov_diag = error_cov_history_diag
    fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for i in range(3):
        ax[i].plot(time, data_history[:,i], label=ylabels[i], color=colors[i])
        ax[i].set_ylabel(ylabels[i])
        ax[i].legend()

        if error_cov_history_diag is not None:
            sigma = np.sqrt(error_cov_diag[:,i])
            ax[i].fill_between(time_cov, -3*sigma, 3*sigma, label=fr'3$\sigma$', color=colors[i], alpha=0.2)
            
            # Boolean mask of divergence
            div_mask = np.abs(data_history[:, i]) > 3*sigma

            # Find start and end indices of contiguous True regions
            div_mask_int = div_mask.astype(int)
            edges = np.diff(div_mask_int)

            start_idxs = np.where(edges == 1)[0] + 1
            end_idxs   = np.where(edges == -1)[0] + 1

            # Handle edge cases
            if div_mask[0]:
                start_idxs = np.insert(start_idxs, 0, 0)
            if div_mask[-1]:
                end_idxs = np.append(end_idxs, len(div_mask))

            # Shade each divergence interval
            for start, end in zip(start_idxs, end_idxs):
                ax[i].axvspan(time[start], time[end-1], color='red', alpha=0.1)

            ax[i].legend()
    ax[2].set_xlabel("Time (s)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(fig_path.with_suffix('.png'))
    pickle.dump(fig, open(fig_path.with_suffix('.pkl'), 'wb'))
    plt.close()  # Close the figure to free up memory, since we'll load it again when we want to show it

def plot_error_quaternion_components(time, quat_error_history, title, label_prefix, fig_path, error_cov_history_diag=None):
    components = ['X', 'Y', 'Z']
    colors = ['m', 'g', 'b']
    if error_cov_history_diag is not None:
        time_cov, error_cov_diag = error_cov_history_diag
    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    delta_att = 2*quat_error_history[:,0:3]  # Convert quaternion error to small angle approximation for vector part
    for i in range(3):
        ax[i].plot(time, delta_att[:,i], label=f'{label_prefix} {components[i]} (rad)', linestyle='-', color=colors[i])
        ax[i].axhline(0 if i < 3 else 1, color='black', linestyle=':')
        ax[i].set_ylabel(f"{components[i]} Component Error (rad)")
    
        if error_cov_history_diag is not None:  # Only plot error covariance for vector part
            sigma = np.sqrt(error_cov_diag[:,i])

            # Plot error covariance diagonal elements
            ax[i].fill_between(time_cov, -3*sigma, 3*sigma, label=fr'3$\sigma$', color=colors[i], alpha=0.2)
            
            # Boolean mask of divergence
            div_mask = np.abs(delta_att[:, i]) > 3*sigma

            # Find start and end indices of contiguous True regions
            div_mask_int = div_mask.astype(int)
            edges = np.diff(div_mask_int)

            start_idxs = np.where(edges == 1)[0] + 1
            end_idxs   = np.where(edges == -1)[0] + 1

            # Handle edge cases
            if div_mask[0]:
                start_idxs = np.insert(start_idxs, 0, 0)
            if div_mask[-1]:
                end_idxs = np.append(end_idxs, len(div_mask))

            # Shade each divergence interval
            for start, end in zip(start_idxs, end_idxs):
                ax[i].axvspan(time[start], time[end-1], color='red', alpha=0.1)
            y_min, y_max = ax[i].get_ylim()
            ax[i].set_ylim(max(y_min, -np.pi), min(y_max, np.pi))
        ax[i].legend()
        
    fig.supxlabel("Time (s)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(fig_path.with_suffix('.png'))
    pickle.dump(fig, open(fig_path.with_suffix('.pkl'), 'wb'))
    plt.close()  # Close the figure to free up memory, since we'll load it again when we want to show it

def plot_quaternion_components(time, quat_history, title, label_prefix, fig_path):
    components = ['X', 'Y', 'Z', 'W']
    colors = ['r', 'g', 'b', 'm']

    fig, ax = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

    for i in range(4):
        ax[i].plot(time, quat_history[:,i], label=f'{label_prefix} {components[i]}', linestyle='-', color=colors[i])
        ax[i].axhline(0 if i < 3 else 1, color='black', linestyle=':')
        ax[i].set_ylabel(f"{components[i]} Component")
        ax[i].legend()
        
    fig.supxlabel("Time (s)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(fig_path.with_suffix('.png'))
    pickle.dump(fig, open(fig_path.with_suffix('.pkl'), 'wb'))
    plt.close()  # Close the figure to free up memory, since we'll load it again when we want to show it