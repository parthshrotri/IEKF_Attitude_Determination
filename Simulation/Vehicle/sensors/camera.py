import yaml
import numpy as np
import astropy.time as time
import astropy.units as u
import matplotlib.pyplot as plt
from adjustText import adjust_text

import utils.utils as utils

from utils.utils import Quaternion

class Camera:
    def __init__(self, init_time, name, cam_cfg_loc, q_body_to_cam, save_file):
        self.name = name
        
        with open(f"sim_configs/{cam_cfg_loc}", "r") as f:
            cam_cfg = yaml.safe_load(f)

        self.q_body_to_cam = Quaternion(*q_body_to_cam)

        self.planets     = np.array(cam_cfg["catalog"]["planets_to_include"])
        self.num_planets = len(self.planets)

        self.dx         = cam_cfg["cam_params"]["dx"]
        self.dy         = cam_cfg["cam_params"]["dy"]
        self.alpha      = cam_cfg["cam_params"]["alpha"]
        self.nrows      = cam_cfg["cam_params"]["nrows"]
        self.ncols      = cam_cfg["cam_params"]["ncols"]
        self.up         = cam_cfg["cam_params"]["up"]
        self.vp         = cam_cfg["cam_params"]["vp"]
        self.half_fov   = cam_cfg["cam_params"]["half_fov"]
        self.vmag_lim   = cam_cfg["catalog"]["vmag_max"]
        self.bore_cam   = cam_cfg["cam_params"]["bore_cam"]
        self.cam_K      = np.array([[self.dx, self.alpha, self.up],
                                    [0, self.dy, self.vp],
                                    [0, 0, 1]])
        self.sigma_los      = float(cam_cfg["cam_params"]["sigma_los"])
        self.update_rate    = cam_cfg["update_rate"]  # Hz
        self.last_update    = init_time  # time of last update

        self.star_catalog   = utils.StarCatalog(cam_cfg["catalog"]["loc"], cam_cfg["catalog"]["epoch"], self.vmag_lim)

        self.save_file      = save_file

    def get_stars(self, icrf_to_cam_true, t_now):
        years_since_epoch   = (t_now - self.star_catalog.t_epoch).to(u.yr).value

        RAs = self.star_catalog.RA + (self.star_catalog.pmRA / np.cos(self.star_catalog.DE)) * years_since_epoch
        DECs = self.star_catalog.DE + self.star_catalog.pmDE * years_since_epoch

        star_los_icrf       = np.zeros((self.star_catalog.length, 3))
        star_los_icrf[:, 0] = np.cos(RAs) * np.cos(DECs)
        star_los_icrf[:, 1] = np.sin(RAs) * np.cos(DECs)
        star_los_icrf[:, 2] = np.sin(DECs)

        stars_los_cam   = star_los_icrf @ icrf_to_cam_true.as_dcm().T
        mask            = stars_los_cam[:, 2] > np.cos(2*np.deg2rad(self.half_fov))

        star_ids        = self.star_catalog.ids[mask]
        stars_los_cam   = stars_los_cam[mask]

        stars_los_cam_noisy = np.nan*np.ones((len(stars_los_cam), 3))
        stars_pixs_noisy    = np.nan*np.ones((len(stars_los_cam), 2))

        for i in range(len(stars_los_cam)):
            star_los_cam = stars_los_cam[i]

            noise = np.random.multivariate_normal(np.zeros(3), 
                                                  self.sigma_los * (np.eye(3) - np.outer(star_los_cam, star_los_cam)))
            star_los_cam_noisy = utils.normalize_vector(star_los_cam + noise)

            star_pix_noisy  = self.cam_K @ star_los_cam_noisy
            star_pix_noisy  = star_pix_noisy / star_pix_noisy[2]

            if star_pix_noisy[0] >= 0 and star_pix_noisy[0] <= self.ncols and \
            star_pix_noisy[1] >= 0 and star_pix_noisy[1] <= self.nrows:
                stars_pixs_noisy[i]     = [star_pix_noisy[0], star_pix_noisy[1]]
                stars_los_cam_noisy[i]  = star_los_cam_noisy
            else:
                star_ids[i] = np.nan

        star_ids            = star_ids[~np.isnan(star_ids)]
        stars_pixs_noisy    = stars_pixs_noisy[~np.isnan(stars_pixs_noisy).any(axis=1)]
        stars_los_cam_noisy  = stars_los_cam_noisy[~np.isnan(stars_los_cam_noisy).any(axis=1)]

        return star_ids, stars_pixs_noisy, stars_los_cam_noisy
    
    def get_planets(self, sc_truth_pos, icrf_to_cam_true, t_now):
        planets_los_icrf    = np.zeros((self.num_planets, 3))  # km

        for i in range(self.num_planets):
            planets_pos_icrf    = utils.get_planet_position(t_now, self.planets[i])
            planets_los_icrf[i] = utils.normalize_vector(planets_pos_icrf - sc_truth_pos)

        planets_los_cam = planets_los_icrf @ icrf_to_cam_true.as_dcm().T
        mask            = planets_los_cam[:, 2] > np.cos(2*np.deg2rad(self.half_fov))

        planets_ids     = self.planets[mask].astype(float)
        planets_los_cam = planets_los_cam[mask]

        planets_los_cam_noisy   = np.nan*np.ones((len(planets_los_cam), 3))
        planets_pixs_noisy      = np.nan*np.ones((len(planets_los_cam), 2))

        for i in range(len(planets_ids)):
            planet_los_cam = planets_los_cam[i]

            noise = np.random.multivariate_normal(np.zeros(3), 
                                                  self.sigma_los * (np.eye(3) - np.outer(planet_los_cam, planet_los_cam)))
            planet_los_cam_noisy = utils.normalize_vector(planet_los_cam + noise)

            planet_pix_noisy = self.cam_K @ planet_los_cam_noisy
            planet_pix_noisy = planet_pix_noisy / planet_pix_noisy[2]

            if planet_pix_noisy[0] >= 0 and planet_pix_noisy[0] <= self.ncols and \
            planet_pix_noisy[1] >= 0 and planet_pix_noisy[1] <= self.nrows:
                planets_pixs_noisy[i]       = [planet_pix_noisy[0], planet_pix_noisy[1]]
                planets_los_cam_noisy[i]    = planet_los_cam_noisy
            else:
                planets_ids[i] = np.nan

        planets_ids          = planets_ids[~np.isnan(planets_ids)]
        planets_pixs_noisy   = planets_pixs_noisy[~np.isnan(planets_pixs_noisy).any(axis=1)]
        planets_los_cam_noisy= planets_los_cam_noisy[~np.isnan(planets_los_cam_noisy).any(axis=1)]

        return planets_ids, planets_pixs_noisy, planets_los_cam_noisy

    def get_measurement(self, spacecraft):
        '''
        Gets the camera measurement at the current time step. Returns None if the camera is not due for an update yet.
        
        :param self: Camera object
        :param spacecraft: Spacecraft object
        '''
        pos_true_icrf = spacecraft.pos_true_icrf
        icrf_to_body_true = spacecraft.icrf_to_body_true
        t_now = spacecraft.time

        icrf_to_cam_true = icrf_to_body_true.mult(self.q_body_to_cam)

        if (t_now - self.last_update).to(u.s).value <= 1/self.update_rate:
            return None
        star_ids, star_pixs, star_los       = self.get_stars(icrf_to_cam_true, t_now)
        planet_ids, planet_pixs, planet_los = self.get_planets(pos_true_icrf, icrf_to_cam_true, t_now)
        self.last_update = t_now

        self.plot_star_field(star_ids, star_pixs, planet_ids, planet_pixs, t_now)

        stars = [star_ids, star_los]
        planets = [planet_ids, planet_los]
        return_struct = {"sensor_name": self.name,
                         "type": "camera",
                         "time": t_now,
                         "data": {"stars": stars, 
                                 "planets": planets}}
        return return_struct

    def plot_star_field(self, star_ids, star_pixs, planet_ids, planet_pixs, t_now):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_facecolor('black')

        texts = [None] * (len(star_pixs) + len(planet_pixs))
        for i in range(len(star_pixs)):
            ax.scatter(star_pixs[i, 0], 
                       star_pixs[i, 1], 
                       s=5, color="white", marker='*')
            texts[i] = ax.text(star_pixs[i, 0], 
                               star_pixs[i, 1], 
                               str(int(star_ids[i])), color='white', fontsize=6)
        for i in range(len(planet_pixs)):
            ax.scatter(planet_pixs[i, 0], 
                       planet_pixs[i, 1], 
                       s=20, color="xkcd:electric lime", marker='*')
            texts[len(star_pixs) + i] = ax.text(planet_pixs[i, 0],
                                               planet_pixs[i, 1], 
                                               str(int(planet_ids[i])), color='xkcd:electric lime', fontsize=8)
            
        ax.scatter(self.up, self.vp, s=50, color='red', marker='+')
        ax.set_xlim(0, self.ncols)
        ax.set_ylim(0, self.nrows)
        ax.set_aspect('equal', adjustable='box')
        ax.invert_yaxis()
        ax.set_title('Simulated Star Field @t=' + t_now.isot)
        ax.set_xlabel('Pixel U -->')
        ax.set_ylabel('<-- Pixel V')
        ax.set_xticks(np.arange(0, self.ncols+1, 64))
        ax.set_yticks(np.arange(0, self.nrows+1, 64))
        plt.xticks(rotation=90)
        adjust_text(texts, 
                    x=np.concatenate((star_pixs[:, 0], planet_pixs[:, 0])),
                    y=np.concatenate((star_pixs[:, 1], planet_pixs[:, 1])),
                    autoalign='xy')
        ax.grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        save_file = self.save_file / f'star_field_{t_now.isot}.png'
        save_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_file, dpi=300)
        plt.close()

# if __name__ == "__main__":
#     # Load SPICE kernels
#     spice.furnsh("data/kernels/naif0012.tls")
#     spice.furnsh("data/kernels/de440.bsp")

#     cam = Camera(7310, 7310, 0, 2192, 2192, 1096.5, 1096.5, 10, "data/catalog/hip.csv", vmag_lim=6.5, planets=[1, 2, 3, 4, 5, 6])
#     t_start = time.Time("2025-07-15T00:00:00.000", scale="utc")
#     t_end   = time.Time("2025-08-15T00:00:00.000", scale="utc")
#     t_range = np.linspace(t_start.jd, t_end.jd, 32)
#     truth_pos = np.array([0, 0, 0])
#     truth_att = Quaternion(0, 0, 0, 1)
#     for i in range(len(t_range)):
#         t_now = time.Time(t_range[i], format='jd', scale='utc')
#         star_ids, star_pixs, planet_ids, planet_pixs = cam.get_stars_planets(truth_pos, truth_att, t_now)
#         starfield = cam.plot_star_field(star_ids, star_pixs, planet_ids, planet_pixs, t_now, save_path=f'Thesis/outputs/star_imgs/star_field_{t_now.isot}.png')
#         att_est = nav.get_attitude_from_stars(t_now, star_ids, star_pixs, cam)
#         print(truth_att, att_est)