import yaml

import numpy as np
import utils.utils as utils
import astropy.units as u
import utils.utils as utils

from utils.utils import Quaternion

from scipy.linalg import expm

# Multiplicative Extended Kalman Filter for attitude estimation using star tracker and rate gyro measurements
class MEKF:
    def __init__(self, camera, MEKF_config):
        self.initialized = False

        self.camera = camera

        self.gyro_noise_sigma = float(MEKF_config["gyro_noise_sigma"])
        self.gyro_bias_rate_sigma = float(MEKF_config["gyro_bias_rate_sigma"])
    
        self.Q = np.block([[np.eye(3) * self.gyro_noise_sigma**2, np.zeros((3, 3))],
                          [np.zeros((3, 3)), np.eye(3) * self.gyro_bias_rate_sigma**2]])
        
        self.star_los_sigma = float(MEKF_config["star_los_sigma"])

    def initialize_filter(self, time, init_icrf_to_cam_est, init_error_cov):
        # Init Time
        self.time = time

        # Init States
        self.inertial_to_cam_est = init_icrf_to_cam_est

        self.gyro_bias_est = np.zeros(3)

        # Init Error Covariance
        self.error_cov = init_error_cov

        # Init Error State
        self.error_state = np.zeros(6)

        self.initialized = True
    
    def propagate(self, time, control_input):
        dt = (time - self.time).to(u.s).value
        self.time = time

        # Propagate attitude using gyro measurements
        omega_est = control_input - self.gyro_bias_est
        
        omega_est_quat = Quaternion(omega_est[0], omega_est[1], omega_est[2], 0)
        att_dot = omega_est_quat.mult(self.inertial_to_cam_est).as_array() * 0.5
        self.inertial_to_cam_est = Quaternion(*(self.inertial_to_cam_est.as_array() + att_dot * dt)).normalize()

        # Propagate error covariance
        F = np.block([[-utils.skew(omega_est), -np.eye(3)],
                      [np.zeros((3, 3)), np.zeros((3, 3))]])
        G = np.block([[-np.eye(3), np.zeros((3, 3))],
                      [np.zeros((3, 3)), np.eye(3)]])
        
        Q_discrete_00 = self.gyro_noise_sigma**2 * dt * np.eye(3) + self.gyro_bias_rate_sigma**2 * dt**3 / 3 * np.eye(3)
        Q_discrete_01 = -self.gyro_bias_rate_sigma**2 * dt**2 / 2 * np.eye(3)
        Q_discrete_10 = -self.gyro_bias_rate_sigma**2 * dt**2 / 2 * np.eye(3)
        Q_discrete_11 = self.gyro_bias_rate_sigma**2 * dt * np.eye(3)
        
        Q_discrete = np.block([[Q_discrete_00, Q_discrete_01],
                               [Q_discrete_10, Q_discrete_11]])
        
        phi = expm(F * dt)  # State transition matrix
        self.error_cov = phi @ self.error_cov @ phi.T + G @ Q_discrete @ G.T
        
    def update(self, time, hip_ids, measurements):
        # Reset error state for measurement update
        self.error_state = np.zeros(6)

        if len(hip_ids) < 2:
            return # Not enough stars for update
        
        # Get corresponding camera frame vectors from catalog
        star_icrf_vecs          = self.camera.star_catalog.get_stars_los_at_idx(time, hip_ids)
        expected_measurements   = star_icrf_vecs @ self.inertial_to_cam_est.as_dcm().T
    
        # Compute measurement residuals
        residuals = measurements - expected_measurements

        # Compute measurement sensitivity matrix
        H = np.zeros((3*len(hip_ids), 6))
        for i in range(len(hip_ids)):
            H[3*i:3*i+3, :3] = utils.skew(expected_measurements[i])
        
        R = np.eye(3*len(hip_ids)) * self.star_los_sigma**2

        # Compute Pre-fit residual covariance
        S = H @ self.error_cov @ H.T + R

        # Compute Kalman gain
        K = self.error_cov @ H.T @ np.linalg.inv(S)

        # Update error state
        self.error_state += K @ residuals.flatten()

        # Update error covariance Joseph form
        I_KH = np.eye(6) - K @ H
        self.error_cov = I_KH @ self.error_cov @ I_KH.T + K @ R @ K.T
        self.error_cov = (self.error_cov + self.error_cov.T) / 2  # Ensure symmetry

        # Correct attitude and gyro bias estimates
        delta_att_vec = self.error_state[:3] / 2
        att_correction = Quaternion(delta_att_vec[0], delta_att_vec[1], delta_att_vec[2], 1)

        new_inertial_to_cam_est = (att_correction.mult(self.inertial_to_cam_est)).normalize()
        self.inertial_to_cam_est = new_inertial_to_cam_est.ensure_positive_scalar()

        new_gyro_bias_est = self.gyro_bias_est + self.error_state[3:]
        self.gyro_bias_est = new_gyro_bias_est
        
        self.time = time    

# Left Invariant Extended Kalman Filter for attitude estimation using star tracker and rate gyro measurements
class LIEKF:
    def __init__(self, camera, LIEKF_config):
        self.initialized = False

        self.camera = camera

        self.gyro_noise_sigma = float(LIEKF_config["gyro_noise_sigma"])
        self.gyro_bias_rate_sigma = float(LIEKF_config["gyro_bias_rate_sigma"])
    
        self.Q = np.block([[np.eye(3) * self.gyro_noise_sigma**2, np.zeros((3, 3))],
                          [np.zeros((3, 3)), np.eye(3) * self.gyro_bias_rate_sigma**2]])
        
        self.star_los_sigma = float(LIEKF_config["star_los_sigma"])

    def initialize_filter(self, time, init_icrf_to_cam_est, init_error_cov):
        # Init Time
        self.time = time

        # Init States
        self.inertial_to_cam_est = init_icrf_to_cam_est
        self.gyro_bias_est = np.zeros(3)

        # Init Error Covariance
        self.error_cov = init_error_cov

        # Init Error State
        self.error_state = np.zeros(6)
        self.initialized = True
    
    def propagate(self, time, control_input):
        dt = (time - self.time).to(u.s).value
        self.time = time

        # Propagate attitude using gyro measurements
        omega_meas = control_input - self.gyro_bias_est
        
        omega_meas_quat = Quaternion(omega_meas[0], omega_meas[1], omega_meas[2], 0)
        att_dot = omega_meas_quat.mult(self.inertial_to_cam_est).as_array() * 0.5
        self.inertial_to_cam_est = Quaternion(*(self.inertial_to_cam_est.as_array() + att_dot * dt)).normalize()

        # Propagate error covariance
        F = np.block([[-utils.skew(omega_meas), -np.eye(3)],
                      [np.zeros((3, 3)), np.zeros((3, 3))]])
        G = np.block([[-np.eye(3), np.zeros((3, 3))],
                      [np.zeros((3, 3)), np.eye(3)]])
        
        Q_discrete_00 = self.gyro_noise_sigma**2 * dt * np.eye(3) + self.gyro_bias_rate_sigma**2 * dt**3 / 3 * np.eye(3)
        Q_discrete_01 = -self.gyro_bias_rate_sigma**2 * dt**2 / 2 * np.eye(3)
        Q_discrete_10 = -self.gyro_bias_rate_sigma**2 * dt**2 / 2 * np.eye(3)
        Q_discrete_11 = self.gyro_bias_rate_sigma**2 * dt * np.eye(3)
        
        Q_discrete = np.block([[Q_discrete_00, Q_discrete_01],
                               [Q_discrete_10, Q_discrete_11]])
        
        phi = expm(F * dt)  # State transition matrix
        self.error_cov = phi @ self.error_cov @ phi.T + G @ Q_discrete @ G.T
        
    def update(self, time, hip_ids, measurements):
        # Reset error state for measurement update
        self.error_state = np.zeros(6)

        if len(hip_ids) < 2:
            return # Not enough stars for update
        
        # Get corresponding camera frame vectors from catalog
        star_icrf_vecs          = self.camera.star_catalog.get_stars_los_at_idx(time, hip_ids)
        expected_measurements   = star_icrf_vecs @ self.inertial_to_cam_est.as_dcm().T
    
        # Compute measurement residuals
        residuals = measurements - expected_measurements

        # Compute measurement sensitivity matrix
        H = np.zeros((3*len(hip_ids), 6))
        for i in range(len(hip_ids)):
            H[3*i:3*i+3, :3] = utils.skew(expected_measurements[i])
        
        R = np.eye(3*len(hip_ids)) * self.star_los_sigma**2
        
        # Compute Pre-fit residual covariance
        S = H @ self.error_cov @ H.T + R

        # Compute Kalman gain
        K = self.error_cov @ H.T @ np.linalg.inv(S)

        # Update error state
        self.error_state += K @ residuals.flatten()

        # Update error covariance Joseph form
        I_KH = np.eye(6) - K @ H
        self.error_cov = I_KH @ self.error_cov @ I_KH.T + K @ R @ K.T
        self.error_cov = (self.error_cov + self.error_cov.T) / 2  # Ensure symmetry

        # Correct attitude and gyro bias estimates
        # LIEKF update
        delta_att_vec_quat  = Quaternion(*self.error_state[:3] / 2, 0)
        att_correction      = delta_att_vec_quat.pure_quaternion_exp()

        new_inertial_to_cam_est = (att_correction.mult(self.inertial_to_cam_est)).normalize()
        self.inertial_to_cam_est = new_inertial_to_cam_est.ensure_positive_scalar()

        new_gyro_bias_est = self.gyro_bias_est + self.error_state[3:]
        self.gyro_bias_est = new_gyro_bias_est
        
        self.time = time    
        
def get_attitude_from_stars(time, star_ids, star_meas_los, camera):
    """
    Estimate the spacecraft attitude based on observed star positions.

    Parameters:
    time : astropy.time.Time
        The current time for the observation.
    star_ids : list
        A list of HIP IDs corresponding to the observed stars.
    star_meas_los : np.ndarray
        An Nx3 array of line-of-sight vectors of observed stars in the camera frame.
    camera : cam.Camera
        The camera object containing intrinsic parameters and catalog.

    Returns:
    quaternion.quaternion
        The estimated attitude quaternion from inertial frame to camera frame.
    """
    if len(star_ids) < 2:
        raise ValueError("At least two stars are required for attitude determination.")
    
    years_since_epoch   = (time - camera.star_catalog.t_epoch).to(u.yr).value

    idxs = [np.where(camera.star_catalog.ids == hip_id)[0][0] for hip_id in star_ids]
    catalog_RA = camera.star_catalog.RA[idxs]
    catalog_DE = camera.star_catalog.DE[idxs]
    catalog_pmRA = camera.star_catalog.pmRA[idxs]
    catalog_pmDE = camera.star_catalog.pmDE[idxs]
    camera_sigma = camera.sigma_los

    alphas = catalog_RA + (catalog_pmRA / np.cos(catalog_DE)) * years_since_epoch
    deltas = catalog_DE + catalog_pmDE * years_since_epoch

    star_los_icrf = np.zeros((len(star_ids), 3))
    star_los_icrf[:, 0] = np.cos(alphas) * np.cos(deltas)
    star_los_icrf[:, 1] = np.sin(alphas) * np.cos(deltas)
    star_los_icrf[:, 2] = np.sin(deltas)

    # (Davenport's q-method) to find attitude
    weights = np.ones(len(star_ids)) / (camera_sigma**2)
    sigmas = np.ones(len(star_ids)) * camera_sigma
    att_quat, P = davenport(star_los_icrf, star_meas_los, weights, sigmas)
    return att_quat.ensure_positive_scalar(), P

def davenport(r_is, b_is, w_is, sigmas):
    B = np.zeros((3,3))
    z = np.zeros((3,1))
    for i in range(len(r_is)):
        r_i     = r_is[i]
        b_i     = b_is[i]
        w_i     = w_is[i]
        B       += w_i * np.outer(b_i, r_i)
        z       += w_i * np.array([np.cross(b_i, r_i)]).T

    K = np.block([[B + B.T - np.trace(B)*np.eye(3), z],
                  [z.T, np.trace(B)]])

    eigvals, eigvecs = np.linalg.eig(K)
    max_idx = np.argmax(eigvals)
    quat_vec = eigvecs[:,max_idx] / np.linalg.norm(eigvecs[:,max_idx])
    est_att_quat = Quaternion(*quat_vec).ensure_positive_scalar()

    lambda_0 = np.sum(w_is)

    B_0 = np.zeros((3, 3))
    M = np.zeros((3, 3))
    for i in range(len(r_is)):
        w_i = w_is[i]
        r_i = r_is[i]
        sigma_i = sigmas[i]

        b_true_i = est_att_quat.rotate_vector(r_i)
        B_0 += w_i * np.outer(b_true_i, b_true_i)
        M += w_i**2 * sigma_i**2 * (np.eye(3) - np.outer(b_true_i, b_true_i))
    l_min_B0_inv = np.linalg.inv(lambda_0 * np.eye(3) - B_0)

    P = l_min_B0_inv @ M @ l_min_B0_inv

    return est_att_quat, P