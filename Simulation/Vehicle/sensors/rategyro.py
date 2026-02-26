import yaml
import numpy as np
import astropy.units as u

class RateGyro:
    def __init__(self, init_time, name, gyro_cfg_loc):
        '''
        Parameters:
        init_time : astropy.time.Time
            The initial time for the gyro.
        update_rate : float
            The update rate in Hz.
        noise_std : float
            The standard deviation of the measurement noise (rad/s).
        init_bias_std : float
            Turn-on bias standard deviation (rad/s).
        bias_rate_std : float
            The standard deviation of the rate of change of the bias (rad/s^2).
        '''
        self.name = name
        with open(f"sim_configs/{gyro_cfg_loc}", "r") as f:
            gyro_cfg = yaml.safe_load(f)

        self.update_rate    = gyro_cfg["update_rate"]  # Hz
        self.noise_std      = gyro_cfg["noise_std"]
        self.init_bias_std  = gyro_cfg["init_bias_std"]
        self.bias_rate_std  = gyro_cfg["bias_rate_std"]
        self.last_update    = init_time

        self.bias = np.random.normal(0, self.init_bias_std, 3)
        self.bias_rate_std = self.bias_rate_std

    def get_gyro_bias(self):
        return self.bias

    def get_measurement(self, spacecraft):
        time = spacecraft.time
        true_ang_vel = spacecraft.ang_vel_true

        if self.last_update is not None and (time - self.last_update).to(u.s).value <= 1/self.update_rate:
            return None
        else:
            noise = np.random.normal(0, self.noise_std, 3)
            bias_rate = np.random.normal(0, self.bias_rate_std, 3)
            if self.last_update is not None:
                dt = (time - self.last_update).to(u.s).value
                self.bias += bias_rate * dt
            measured_ang_vel = true_ang_vel + self.bias + noise
            self.last_update = time

            return_struct = {"sensor_name": self.name,
                             "type": "rate_gyro",
                             "time": time,
                             "data": measured_ang_vel}
            return return_struct