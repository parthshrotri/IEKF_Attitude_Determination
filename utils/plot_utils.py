import glob
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

def plot_monte_carlo_att_results(run_data, title, ylabels, fig_path):
    colors = ['xkcd:lavender', 'xkcd:lime', 'xkcd:lightblue']

    num_runs = len(run_data)
    run_success = np.zeros(num_runs, dtype=bool)

    for i, run in enumerate(run_data):
        time_arrays = run["time_arrays"]
        data_arrays = run["data_arrays"]
        cov_diag_arrays = run["cov_diag_arrays"]
        
        data_arrays = 2*data_arrays[:,0:3]  # Convert quaternion error to small angle approximation for vector part

        error_outside_cov = np.any(np.abs(data_arrays) > 3*np.sqrt(cov_diag_arrays), axis=1)

        times_error_outside_cov = time_arrays[error_outside_cov]

        RMSE = np.sqrt(np.mean(data_arrays**2, axis=0))
        if np.any(RMSE > 1.0): # TBR: If RMSE of attitude error exceeds 1.0 rad, consider it a failure regardless of covariance consistency
            run_success[i] = False
            continue

        if len(times_error_outside_cov) > 0:
            time_diffs = np.diff(times_error_outside_cov)
            gap_threshold = 0.5  # If time gap between consecutive points is greater than this, consider it a separate segment
            segment_boundaries = np.where(time_diffs > gap_threshold)[0]
            segment_starts = np.insert(segment_boundaries + 1, 0, 0)
            segment_ends = np.append(segment_boundaries, len(times_error_outside_cov) - 1)

            longest_segment_length = 0
            for start, end in zip(segment_starts, segment_ends):
                segment_length = times_error_outside_cov[end] - times_error_outside_cov[start]
                if segment_length > longest_segment_length:
                    longest_segment_length = segment_length
                    if longest_segment_length > 20.0:  # If error exceeds 3-sigma for more than 20.0 seconds, consider it a divergence (TBR)
                        run_success[i] = False
                    else:
                        run_success[i] = True
        else:
            run_success[i] = True

    num_success = np.sum(run_success)
    num_fail = num_runs - num_success
    fail_rate = num_fail / num_runs

    fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    
    for i, run in enumerate(run_data):
        time_arrays = run["time_arrays"]
        data_arrays = run["data_arrays"]
        cov_diag_arrays = run["cov_diag_arrays"]

        data_arrays = 2*data_arrays[:,0:3]  # Convert quaternion error to small angle approximation for vector part

        if run_success[i]: 
            ax[0].plot(time_arrays, data_arrays[:,0], color='k', alpha=5/num_success, zorder=3)
            ax[1].plot(time_arrays, data_arrays[:,1], color='k', alpha=5/num_success, zorder=3)
            ax[2].plot(time_arrays, data_arrays[:,2], color='k', alpha=5/num_success, zorder=3)
            ax[0].fill_between(time_arrays, -3*np.sqrt(cov_diag_arrays[:,0]), 3*np.sqrt(cov_diag_arrays[:,0]), color=colors[0], alpha=1/num_success, zorder=2)
            ax[1].fill_between(time_arrays, -3*np.sqrt(cov_diag_arrays[:,1]), 3*np.sqrt(cov_diag_arrays[:,1]), color=colors[1], alpha=1/num_success, zorder=2)
            ax[2].fill_between(time_arrays, -3*np.sqrt(cov_diag_arrays[:,2]), 3*np.sqrt(cov_diag_arrays[:,2]), color=colors[2], alpha=1/num_success, zorder=2)
    ax[0].set_title(f"Failure Rate: {fail_rate:.1%}")

    ax[0].set_ylabel(ylabels[0])
    ax[1].set_ylabel(ylabels[1])
    ax[2].set_ylabel(ylabels[2])

    ax[2].set_xlabel("Time (s)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(fig_path.with_suffix('.png'))
    pickle.dump(fig, open(fig_path.with_suffix('.pkl'), 'wb'))
    plt.close()

def plot_monte_carlo_bias_results(run_data, title, ylabels, fig_path):
    colors = ['xkcd:lavender', 'xkcd:lime', 'xkcd:lightblue']

    num_runs = len(run_data)
    run_success = np.zeros(num_runs, dtype=bool)

    for i, run in enumerate(run_data):
        time_arrays = run["time_arrays"]
        data_arrays = run["data_arrays"]
        cov_diag_arrays = run["cov_diag_arrays"]
        
        error_outside_cov = np.any(np.abs(data_arrays) > 3*np.sqrt(cov_diag_arrays), axis=1)
        times_error_outside_cov = time_arrays[error_outside_cov]
        if len(times_error_outside_cov) > 0:
            time_diffs = np.diff(times_error_outside_cov)
            gap_threshold = 0.5  # If time gap between consecutive points is greater than this, consider it a separate segment
            segment_boundaries = np.where(time_diffs > gap_threshold)[0]
            segment_starts = np.insert(segment_boundaries + 1, 0, 0)
            segment_ends = np.append(segment_boundaries, len(times_error_outside_cov) - 1)

            longest_segment_length = 0
            for start, end in zip(segment_starts, segment_ends):
                segment_length = times_error_outside_cov[end] - times_error_outside_cov[start]
                if segment_length > longest_segment_length:
                    longest_segment_length = segment_length
                    if longest_segment_length > 10.0:  # If error exceeds 3-sigma for more than 10.0 seconds, consider it a divergence (TBR)
                        run_success[i] = False
                    else:
                        run_success[i] = True
        else:
                run_success[i] = True

    num_success = np.sum(run_success)
    num_fail = num_runs - num_success
    fail_rate = num_fail / num_runs

    fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

    for i, run in enumerate(run_data):
        time_arrays = run["time_arrays"]
        data_arrays = run["data_arrays"]
        cov_diag_arrays = run["cov_diag_arrays"]

        if run_success[i]: 
            ax[0].plot(time_arrays, data_arrays[:,0], color='k', alpha=5/num_success)
            ax[1].plot(time_arrays, data_arrays[:,1], color='k', alpha=5/num_success)
            ax[2].plot(time_arrays, data_arrays[:,2], color='k', alpha=5/num_success)
            ax[0].fill_between(time_arrays, -3*np.sqrt(cov_diag_arrays[:,0]), 3*np.sqrt(cov_diag_arrays[:,0]), color=colors[0], alpha=1/num_success)
            ax[1].fill_between(time_arrays, -3*np.sqrt(cov_diag_arrays[:,1]), 3*np.sqrt(cov_diag_arrays[:,1]), color=colors[1], alpha=1/num_success)
            ax[2].fill_between(time_arrays, -3*np.sqrt(cov_diag_arrays[:,2]), 3*np.sqrt(cov_diag_arrays[:,2]), color=colors[2], alpha=1/num_success)

    ax[0].set_title(f"Failure Rate: {fail_rate:.1%}")

    ax[0].set_ylabel(ylabels[0])
    ax[1].set_ylabel(ylabels[1])
    ax[2].set_ylabel(ylabels[2])

    ax[2].set_xlabel("Time (s)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(fig_path.with_suffix('.png'))
    pickle.dump(fig, open(fig_path.with_suffix('.pkl'), 'wb'))
    plt.close()