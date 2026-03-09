import yaml
import astropy.units as u

from pathlib import Path
from Simulation.FlightSoftware.GNC import gnc_manager

class FSW:
    def __init__(self, init_time, fsw_cfg_loc, vehicle):
        # Load GNC config
        with open(fsw_cfg_loc, "r") as f:
            fsw_cfg = yaml.safe_load(f)

        gnc_config_loc = Path(fsw_cfg_loc).parent / "GNC.yaml"

        self.vehicle        = vehicle
        self.last_update    = None
        self.time           = init_time
        self.update_rate    = fsw_cfg["update_rate"]  # Hz
        self.gnc_manager    = gnc_manager.GNC(init_time, gnc_config_loc, vehicle, self)

        self.cmd_queue = []
        self.measurement_queue = []

    def step(self, time):
        if self.last_update is not None and (time - self.last_update).to(u.s).value <= 1/self.update_rate:
            return 
        self.last_update = time
        self.time = time
        self.parse_commands(time)
        self.gnc_manager.step(time, self.measurement_queue)
        self.measurement_queue = []  # Clear measurements after processing
        self.parse_commands(time)  # Check for any new commands that may have been added during GNC step
        self.measurement_queue = []

    def add_to_measurement_queue(self, measurement):
        self.measurement_queue.append(measurement)
        self.measurement_queue.sort(key=lambda x: x["time"])  # Ensure measurements are processed in chronological order

    def add_to_command_queue(self, cmd_time, cmd_type, cmd_data):
        self.cmd_queue.append((cmd_time, cmd_type, cmd_data))
        self.cmd_queue.sort(key=lambda x: x[0])  # Ensure commands are sorted by execution time

    def parse_commands(self, time):
        # Check if there are any commands in the queue that need to be executed at the current time
        for cmd in self.cmd_queue:
            if cmd[0] <= time:
                next_cmd = self.cmd_queue.pop(0)
                if next_cmd[1] == "GNC_ATT_CMD":
                    self.gnc_manager.update_gnc_mode(next_cmd[2])
                elif next_cmd[1] == "SC_SET_TORQUE":
                    self.vehicle.set_control_torque(next_cmd[2])
                # Add more command parsing logic here as needed
            else:
                break  # Since the list is sorted, we can stop checking further commands