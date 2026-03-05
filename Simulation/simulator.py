import yaml
import tqdm

import numpy as np
import spiceypy as spice
import astropy.units as u

from pathlib import Path
from astropy.time import Time

from utils.utils import Quaternion

from Simulation.Vehicle.vehicle import Vehicle
from Simulation.FlightSoftware.FSW import FSW
from Simulation.logger import SimulationLogger

class Simulator:
    def __init__(self, config):
        # --- Paths ---
        spice_dir = Path(config["input_files"]["spice_dir"])
        output_dir = Path(config["output_files"]["dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        self.save_file = Path(output_dir)
        self.save_file.parent.mkdir(parents=True, exist_ok=True)

        # --- Load SPICE kernels ---
        for kernel in config["input_files"]["base_spice_kernels"]:
            spice.furnsh(str(spice_dir / kernel))

        sim_cfg     = config["simulation"]
        vehicle_cfg = config["input_files"]["vehicle_config"]
        fsw_cfg     = config["input_files"]["fsw_config"]

        # --- Time setup ---
        t_start = Time(sim_cfg["start_time"], scale="utc")
        duration = sim_cfg["duration"] * u.s
        samples = int(duration.to(u.ms).value / sim_cfg["time_step"])
        t_samples = t_start + np.linspace(0, duration.to(u.s).value, samples) * u.s
        self.t_samples = t_samples
        self.dt = sim_cfg["time_step"] * u.s

        # Prepare command queue with actual times
        fsw_cmds = []
        for cmd in sim_cfg.get("cmds", []):
            cmd_time = t_start + cmd["time"] * u.s
            fsw_cmds.append((cmd_time, cmd["type"], cmd["cmd"]))
        fsw_cmds.sort(key=lambda x: x[0])  # Ensure commands are sorted by time
        self.fsw_cmds = fsw_cmds

        # --- Initialize Vehicle ---
        init_quat = Quaternion(*sim_cfg["init_states"]["att_states"])
        init_rates = np.array(sim_cfg["init_states"]["att_rates"])

        self.vehicle = Vehicle(
            spice_dir,
            output_dir,
            vehicle_cfg,
            t_samples,
            init_quat,
            init_rates)

        # --- Initialize FSW ---
        self.fsw = FSW(t_start, fsw_cfg, self.vehicle)

        #--- Initialize Logger ---
        self.logger = SimulationLogger(t_start, output_dir / "simulation_log.npz")

    def run(self):
        # --- Run simulation ---
        print("Running simulation...")
        for i in tqdm.tqdm(range(len(self.t_samples)), desc="Propagating"):
            self.step(i, self.t_samples[i])

        print("Finished Simulation. Saving history...")
        self.logger.save_history()
        print(f"Output saved to {self.save_file.parent}")
    
    def step(self, idx, new_time):
        # Propagate dynamics
        self.vehicle.propagate(idx, new_time)

        # Check for commands to execute at this time
        for cmd in self.fsw_cmds:
            if cmd[0] <= new_time:
                next_cmd = self.fsw_cmds.pop(0)
                dt = (new_time - self.t_samples[0]).to_value('sec')

                # Truncate to milliseconds
                dt_ms = int(dt * 1000)

                days = dt_ms // (24 * 3600 * 1000)
                hours = dt_ms // (3600 * 1000)
                minutes = (dt_ms % (3600 * 1000)) // (60 * 1000)
                seconds = (dt_ms % (60 * 1000)) / 1000  # keeps milliseconds

                label = f"T+{days:02d}:{hours:02d}:{minutes:02d}:{seconds:06.3f}"
                print(f"{label} Executing {next_cmd[1]} command: {next_cmd[2]}")
                self.fsw.add_to_command_queue(*next_cmd)
            
        # Get sensor measurements and add to FSW queue
        measurements = self.vehicle.get_sensor_measurements()
        for meas in measurements:
            self.fsw.add_to_measurement_queue(meas)
        self.logger.log_measurements(measurements)

        self.fsw.step(new_time)
        self.logger.log_fsw_history(new_time, self.vehicle, self.fsw)
        
        # Log true states        
        self.logger.log_truth(new_time, self.vehicle, self.fsw)