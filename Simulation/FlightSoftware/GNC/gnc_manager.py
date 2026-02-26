import yaml
import numpy as np
import utils.utils as utils

from utils.utils import Quaternion
from Simulation.FlightSoftware.GNC import nav
from Simulation.FlightSoftware.GNC import acs
class GNC:
    def __init__(self, init_time, gnc_cfg_loc, vehicle, fsw):
        self.time = init_time
        self.vehicle = vehicle
        self.fsw = fsw

        self.ref_camera = vehicle.ref_camera

        # Load GNC config
        with open(f"sim_configs/{gnc_cfg_loc}", "r") as f:
            self.config = yaml.safe_load(f)

        self.star_catalog = utils.StarCatalog(self.config["star_catalog"]["loc"], 
                                              self.config["star_catalog"]["epoch"], 
                                              self.config["star_catalog"]["vmag_max"])
        
        # Create Attitude Controller and MEKF
        self.att_controller = acs.AttitudeController(self.config["att_controller"])
        self.MEKF           = nav.MEKF(self.ref_camera, self.config["MEKF"])
        self.LIEKF          = nav.LIEKF(self.ref_camera, self.config["LIEKF"])

        if self.config["MEKF"]["init_est"] != "AUTO":
            self.MEKF.initialize_filter(init_time, 
                                        Quaternion(*self.config["MEKF"]["init_est"]["att_est"]), 
                                        np.diag(np.array(self.config["MEKF"]["init_est"]["error_cov"], dtype=float)))
        if self.config["LIEKF"]["init_est"] != "AUTO":
            self.LIEKF.initialize_filter(init_time, 
                                         Quaternion(*self.config["LIEKF"]["init_est"]["att_est"]), 
                                         np.diag(np.array(self.config["LIEKF"]["init_est"]["error_cov"], dtype=float)))
        
        # Initialize target to be "OFF" (no control) until a command is received
        self.target = "OFF"

    def step(self, time, measurements):
        # Update current time
        self.time = time

        # Get current ACS mode and setpoints based on target
        mode, target_icrf_to_body, target_rate_body = self.get_acs_params()
        self.att_controller.set_acs_setpoints(mode, target_icrf_to_body, target_rate_body)

        # Compute control torque and send command to vehicle
        self.att_controller.compute_control(time, self.vehicle.icrf_to_body_true, self.vehicle.ang_vel_true)
        ctrl_torque = self.att_controller.get_ctrl_torque()
        self.fsw.add_to_command_queue(time, "SC_SET_TORQUE", ctrl_torque)

        # Process incoming measurements through the MEKF
        for meas in measurements:
            accept = self.MEKF_process_measurement(meas)
            if not accept:
                print(f"Measurement from {meas['sensor_name']} at time {meas['time']} was rejected by the MEKF.")
            accept = self.LIEKF_process_measurement(meas)
            if not accept:
                print(f"Measurement from {meas['sensor_name']} at time {meas['time']} was rejected by the LIEKF.")

    def LIEKF_process_measurement(self, measurement):
        if self.LIEKF.initialized:
            if measurement["type"] == "rate_gyro":
                self.LIEKF.propagate(measurement["time"],
                                    measurement["data"])
                return True
            elif measurement["type"] == "camera":
                self.LIEKF.update(measurement["time"],
                                 measurement["data"]["stars"][0], 
                                 measurement["data"]["stars"][1])
                return True
            else:
                return False
        else:
            if self.config["LIEKF"]["init_est"] == "AUTO":
                if measurement["type"] == "camera" and len(measurement["data"]["stars"][0]) >= 2: # Need at least 2 stars for initial attitude determination
                    init_att_est = nav.get_attitude_from_stars(measurement["time"], 
                                                            measurement["data"]["stars"][0],
                                                            measurement["data"]["stars"][1], 
                                                            self.ref_camera)
                    self.LIEKF.initialize_filter(measurement["time"], 
                                                init_att_est, 
                                                np.eye(6) * 0.01) #UPDATE THIS ERROR COV BASED ON DAVENPORT
                    return True
                else:
                    return False

    def MEKF_process_measurement(self, measurement):
        if self.MEKF.initialized:
            if measurement["type"] == "rate_gyro":
                self.MEKF.propagate(measurement["time"],
                                    measurement["data"])
                return True
            elif measurement["type"] == "camera":
                self.MEKF.update(measurement["time"],
                                 measurement["data"]["stars"][0], 
                                 measurement["data"]["stars"][1])
                return True
            else:
                return False
        else:
            if self.config["MEKF"]["init_est"] == "AUTO":
                if measurement["type"] == "camera" and len(measurement["data"]["stars"][0]) >= 2:  # Need at least 2 stars for initial attitude determination
                    init_att_est = nav.get_attitude_from_stars(measurement["time"], 
                                                            measurement["data"]["stars"][0],
                                                            measurement["data"]["stars"][1], 
                                                            self.ref_camera)
                    self.MEKF.initialize_filter(measurement["time"], 
                                                init_att_est, 
                                                np.eye(6) * 0.01) #UPDATE THIS ERROR COV BASED ON DAVENPORT
                    return True
                else:
                    return False

    def update_gnc_mode(self, new_target):
        self.target = new_target

    def get_acs_params(self):
        # Determine ACS mode and setpoints based on current target
        if type(self.target) == str and self.target.startswith("SPICE"):
            # Point towards a SPICE target
            mode                = 2
            planet              = int(self.target[5:])
            vec_to_planet_icrf  = utils.get_planet_position(self.time, planet) - self.vehicle.pos_true_icrf
            dir_to_planet_icrf  = utils.normalize_vector(vec_to_planet_icrf)
            # Update target attitude
            target_icrf_to_cam  = utils.align_vectors(np.array([0, 0, 1]), dir_to_planet_icrf)
            target_rate_body    = np.zeros(3)
        elif type(self.target) == str and self.target.startswith("HIP"):
            # Point towards a star
            mode                = 2
            star_icrf_vec       = self.star_catalog.get_stars_los_at_idx(self.time, [int(self.target[3:])]).flatten()
            target_icrf_to_cam  = utils.align_vectors(np.array([0, 0, 1]), star_icrf_vec)
            target_rate_body    = np.zeros(3)
        elif type(self.target) == str and self.target.startswith("RATE("):
            # Set a specific body rate
            mode                = 1
            target_icrf_to_cam  = None
            rate_vals           = self.target[5:-1].split(",")
            target_rate_body    = np.array([float(rate_vals[0]), 
                                            float(rate_vals[1]), 
                                            float(rate_vals[2])])
        elif type(self.target) == str and self.target.startswith("RA"):
            # Point towards a specific RA/DEC coordinate
            mode                = 2
            ra_dec              = self.target[3:].split(",DEC")
            ra                  = np.deg2rad(float(ra_dec[0]))
            dec                 = np.deg2rad(float(ra_dec[1]))
            target_vec_icrf     = np.zeros(3)
            target_vec_icrf[0]  = np.cos(ra) * np.cos(dec)
            target_vec_icrf[1]  = np.sin(ra) * np.cos(dec)
            target_vec_icrf[2]  = np.sin(dec)
            target_icrf_to_cam  = utils.align_vectors(np.array([0, 0, 1]), target_vec_icrf)
            target_rate_body    = np.zeros(3)
        elif type(self.target) == str and self.target.startswith("Q("):
            # Point towards a specific quaternion attitude
            mode                = 2
            quat_vals           = self.target[2:-1].split(",")
            target_icrf_to_cam  = Quaternion(float(quat_vals[0]), 
                                                float(quat_vals[1]), 
                                                float(quat_vals[2]), 
                                                float(quat_vals[3]))
            target_rate_body    = np.zeros(3)
        elif type(self.target) == str and self.target.startswith("OFF"):
            # Turn off control
            mode                = 0
            target_icrf_to_cam  = None
            target_rate_body    = None
        elif type(self.target) == str and self.target.startswith("TORQUE("):
            # Set a specific control torque (this is a direct override, not a mode)
            mode                = self.att_controller.mode  # Keep current mode
            target_icrf_to_cam  = self.att_controller.target_icrf_to_body.mult(self.ref_camera.q_body_to_cam).normalize() if self.att_controller.target_icrf_to_body is not None else None
            target_rate_body    = self.att_controller.target_body_omega if self.att_controller.target_body_omega is not None else None
            torque_vals         = self.target[7:-1].split(",")
            cmd_torque          = np.array([float(torque_vals[0]), 
                                            float(torque_vals[1]), 
                                            float(torque_vals[2])])
            self.att_controller.set_auto_torque(False, cmd_torque)
        elif type(self.target) == str and self.target.startswith("AUTO_TORQUE"):
            mode                = self.att_controller.mode  # Keep current mode
            target_icrf_to_cam  = self.att_controller.target_icrf_to_body.mult(self.ref_camera.q_body_to_cam).normalize() if self.att_controller.target_icrf_to_body is not None else None
            target_rate_body    = self.att_controller.target_body_omega if self.att_controller.target_body_omega is not None else None
            self.att_controller.set_auto_torque(True)
        else:
            raise ValueError("Invalid target type. Must be " \
                                "PLANET followed by planet number, " \
                                "'HIP' followed by star HIP ID, " \
                                "'RA' followed by Right Ascension, 'DEC' followed by Declination in Degrees, or " \
                                "'Q' followed by quaternion components," \
                                "'RATE' followed by body rates in radians per second,"
                                "OFF for no control.")

        if target_icrf_to_cam is not None:
            target_icrf_to_body = target_icrf_to_cam.mult(self.vehicle.ref_camera.q_body_to_cam.conjugate()).normalize()
            target_icrf_to_body = target_icrf_to_body.ensure_positive_scalar()
        else:
            target_icrf_to_body = None
        return mode, target_icrf_to_body, target_rate_body