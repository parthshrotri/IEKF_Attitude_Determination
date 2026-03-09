from pathlib import Path
import yaml
import numpy as np
import spiceypy as spice
import astropy.units as u

import Simulation.dynamics as dyn

from Simulation.Vehicle.sensors.camera import Camera
from Simulation.Vehicle.sensors.rategyro import RateGyro

class Vehicle:
    def __init__(self, 
                 spice_dir,
                 output_dir,
                 vehicle_cfg_path,
                 t_samples,
                 init_icrf_to_body, 
                 init_ang_vel):

        # Load vehicle config
        with open(vehicle_cfg_path, "r") as f:
            vehicle_cfg = yaml.safe_load(f)["spacecraft"]
        
        parent = Path(vehicle_cfg_path).parent
        # Initialize sensors
        cameras = []
        for cur_cam in vehicle_cfg["sensors"]["cameras"]:
                cam_cfg = vehicle_cfg["sensors"]["cameras"][cur_cam]
                cam_cfg_loc = parent / cam_cfg["config_loc"]
                cameras.append(Camera(t_samples[0], cur_cam, cam_cfg_loc, cam_cfg["q_body_to_cam"], save_file=output_dir / cur_cam))
        gyros = []
        for cur_gyro in vehicle_cfg["sensors"]["rate_gyros"]:
                gyro_cfg = vehicle_cfg["sensors"]["rate_gyros"][cur_gyro]
                gyro_cfg_loc = parent / gyro_cfg["config_loc"]
                gyros.append(RateGyro(t_samples[0], cur_gyro, gyro_cfg_loc))
        self.sensors = cameras + gyros

        self.ref_camera = cameras[0]

        # Spacecraft-specific kernels
        for kernel in vehicle_cfg["kernels"]:
            spice.furnsh(str(spice_dir / kernel))

        # --- Get spacecraft position and velocity using SPICE ---
        et_samples = spice.str2et(t_samples.isot)
        pos_vels = np.array([
            spice.spkezr(str(vehicle_cfg["spice_id"]), et, "J2000", "NONE", "SSB")[0]
            for et in et_samples
        ])

        self.pos_vels           = pos_vels
        self.init_time          = t_samples[0]
        self.time               = self.init_time
        self.update_true_pos_vel(0)
        self.icrf_to_body_true  = init_icrf_to_body
        self.icrf_to_cam_true   = self.icrf_to_body_true.mult(self.ref_camera.q_body_to_cam).normalize()
        self.ang_vel_true       = init_ang_vel
        self.inertia            = np.array(vehicle_cfg["inertial_properties"]["inertia_tensor"])
        self.disturbance_torque_std = np.array(vehicle_cfg["rand_disturbance"]["torque"])

        self.set_control_torque(np.zeros(3))

    def update_true_pos_vel(self, idx):
        self.pos_true_icrf = np.array(self.pos_vels[idx, 0:3])
        self.vel_true_icrf = np.array(self.pos_vels[idx, 3:6])

    def update_true_att_and_rate(self, dt):
        disturbance_torque = np.random.normal(0, self.disturbance_torque_std, 3)
        new_att, new_ang_vel = dyn.att_prop(self, dt, self.ctrl_torque + disturbance_torque)

        self.icrf_to_body_true  = new_att
        self.icrf_to_cam_true   = self.icrf_to_body_true.mult(self.ref_camera.q_body_to_cam).normalize()
        self.ang_vel_true       = new_ang_vel

    def set_control_torque(self, ctrl_torque):
        self.ctrl_torque = ctrl_torque
    
    def update_time(self, new_time):
        self.time = new_time

    def get_sensor_measurements(self):
        measurements = []
        for sensor in self.sensors:
            meas = sensor.get_measurement(self)
            if meas is not None:
                measurements.append(meas)
        return measurements
    
    def propagate(self, idx, new_time):
        dt = (new_time - self.time).to(u.s).value

        # Update true states
        self.update_true_pos_vel(idx)
        self.update_true_att_and_rate(dt)
        self.update_time(new_time)