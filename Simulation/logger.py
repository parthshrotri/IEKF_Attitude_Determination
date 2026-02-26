import numpy as np

from Simulation.Vehicle.sensors.rategyro import RateGyro

class SimulationLogger:
    def __init__(self, init_time, save_file):
        self.save_file = save_file
        self.history = {}
        self.history["init_time"] = init_time

    def log(self, key, time, data):
        if key not in self.history:
            self.history[key] = {
                "time": [],
                "data": []
            }

        time_value = (time - self.history["init_time"]).sec

        self.history[key]["time"].append(time_value)

        self.history[key]["data"].append(
            data.copy() if isinstance(data, np.ndarray) else data
        )

    def log_truth(self, time, vehicle, fsw):
        self.log("truth/pos", time, vehicle.pos_true_icrf.copy())
        self.log("truth/vel", time, vehicle.vel_true_icrf.copy())
        self.log("truth/icrf_to_body", time, vehicle.icrf_to_body_true.as_array().copy())
        self.log("truth/icrf_to_cam", time, vehicle.icrf_to_cam_true.as_array().copy())
        self.log("truth/ang_vel", time, vehicle.ang_vel_true.copy())
        self.log("truth/ctrl_torque", time, vehicle.ctrl_torque.copy())

        acs = fsw.gnc_manager.att_controller
        current_target_icrf_to_body, _ = acs.get_target_attitude()
        if current_target_icrf_to_body is None:
            self.log("truth/icrf_to_body_err", time, np.array([0, 0, 0, 1]))
        else:
            self.log("truth/icrf_to_body_err", time, 
                            vehicle.icrf_to_body_true.mult(current_target_icrf_to_body.conjugate()).as_array().copy())
        
        for sensor in vehicle.sensors:
            if isinstance(sensor, RateGyro):
                self.log(f"truth/{sensor.name}_bias", time, sensor.get_gyro_bias().copy())

    def log_measurements(self, measurements):
        for meas in measurements:
            self.log(f"meas/{meas['sensor_name']}", meas["time"], meas["data"].copy())

    def log_fsw_history(self, time, vehicle, fsw):

        # Log GNC states and control outputs
        gnc = fsw.gnc_manager
        self.log("fsw/time", time, gnc.time)

        target_icrf_to_body, target_body_rate = gnc.att_controller.get_target_attitude()
        if target_icrf_to_body is None:
            target_icrf_to_body = vehicle.icrf_to_body_true
        if target_body_rate is None:
            target_body_rate = vehicle.ang_vel_true
        self.log("ctrl/target_icrf_to_body", time, target_icrf_to_body.as_array().copy())
        self.log("ctrl/target_body_rate", time, target_body_rate.copy())

        self.log("ctrl/torque", time, gnc.att_controller.get_ctrl_torque().copy())
        
        if gnc.MEKF.initialized:
            self.log("est/MEKF/icrf_to_cam", time, gnc.MEKF.inertial_to_cam_est.as_array().copy())
            self.log("est/MEKF/gyro_bias", time, gnc.MEKF.gyro_bias_est.copy())

            self.log("est/MEKF/icrf_to_cam_error", time, 
                        vehicle.icrf_to_cam_true.mult(gnc.MEKF.inertial_to_cam_est.conjugate()).as_array().copy())
            
            for sensor in vehicle.sensors:
                if isinstance(sensor, RateGyro):
                    self.log("est/MEKF/gyro_bias_error", time, sensor.get_gyro_bias() - gnc.MEKF.gyro_bias_est)
            
            self.log("est/MEKF/error_cov_diag", time, np.diag(gnc.MEKF.error_cov).copy())

        if gnc.LIEKF.initialized:
            self.log("est/LIEKF/icrf_to_cam", time, gnc.LIEKF.inertial_to_cam_est.as_array().copy())
            self.log("est/LIEKF/gyro_bias", time, gnc.LIEKF.gyro_bias_est.copy())

            self.log("est/LIEKF/icrf_to_cam_error", time, 
                        vehicle.icrf_to_cam_true.mult(gnc.LIEKF.inertial_to_cam_est.conjugate()).as_array().copy())
            for sensor in vehicle.sensors:
                if isinstance(sensor, RateGyro):
                    self.log("est/LIEKF/gyro_bias_error", time, sensor.get_gyro_bias() - gnc.LIEKF.gyro_bias_est)
            
            self.log("est/LIEKF/error_cov_diag", time, np.diag(gnc.LIEKF.error_cov).copy())
    
        if gnc.MEKF.initialized and gnc.LIEKF.initialized:
            self.log("est/MEKF_LIEKF_icrf_to_cam_diff", time, 
                        gnc.MEKF.inertial_to_cam_est.mult(gnc.LIEKF.inertial_to_cam_est.conjugate()).as_array().copy())

    def save_history(self):
        np.savez(self.save_file, **self.history)