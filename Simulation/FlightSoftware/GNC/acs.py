import numpy as np
import astropy.units as u

class AttitudeController:
    def __init__(self, control_cfg):
        self.kp             = np.diag(control_cfg["KP"])
        self.kd             = np.diag(control_cfg["KD"])
        self.control_torque = np.array([0.0, 0.0, 0.0])  # Nm
        self.mode           = 0  # 0: No control, 1: Rate Control, 2: Attitude control
        self.target_icrf_to_body = None
        self.target_body_omega = None
        self.auto_torque        = True

    def get_ctrl_torque(self):
        return self.control_torque
    
    def get_target_attitude(self):
        return self.target_icrf_to_body, self.target_body_omega
    
    def set_acs_setpoints(self, new_mode, target_icrf_to_body, target_body_angular_velocity):
        self.mode                   = new_mode
        self.target_icrf_to_body    = target_icrf_to_body
        self.target_body_omega      = target_body_angular_velocity

    def set_auto_torque(self, auto_torque, manual_torque=None):
        self.auto_torque = auto_torque
        if not auto_torque and manual_torque is not None:
            self.control_torque = manual_torque

    def compute_control(self, t_now, current_icrf_to_body, current_body_angular_velocity):
        if self.auto_torque:
            if self.mode == 0:
                self.control_torque = np.array([0.0, 0.0, 0.0])
            elif self.mode == 1:
                self.last_update = t_now
                angular_rate_error  = self.target_body_omega - current_body_angular_velocity
                ctrl_torque         = self.kd@angular_rate_error
                self.control_torque = ctrl_torque
            elif self.mode == 2:
                self.last_update = t_now
                # Compute the quaternion error
                dq      =   current_icrf_to_body.mult(self.target_icrf_to_body.conjugate())
                dq_4    =   dq.s
                dq_vec  =   dq.vec

                angular_rate_error = self.target_body_omega - current_body_angular_velocity

                # Compute the control input
                ctrl_torque   =   -np.sign(dq_4)*self.kp@dq_vec + self.kd@angular_rate_error
                self.control_torque = ctrl_torque