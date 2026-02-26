import pandas as pd
import numpy as np
import spiceypy as spice

import astropy.units as u
from astropy.time import Time

def angle_between(v1, v2):
    """
    Calculate the angle in radians between two vectors.
    
    Parameters:
    v1, v2 : array-like
        Input vectors.
        
    Returns:
    float
        The angle in radians between the two vectors.
    """
    v1_u = normalize_vector(v1)
    v2_u = normalize_vector(v2)
    dot_product = np.dot(v1_u, v2_u)
    return np.arccos(dot_product)

def quaternion_diff(q1, q2):
    """
    Compute the difference between two quaternions.
    
    Parameters:
    q1, q2 : Quaternion
        Input quaternions.
        
    Returns:
    Quaternion
        The quaternion representing the rotation from q2 to q1.
    """
    return q1.mult(q2.conjugate())

def normalize_vector(v):
    """
    Normalize a vector to have unit length.
    
    Parameters:
    v : array-like
        Input vector.
        
    Returns:
    np.ndarray
        The normalized vector.
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def skew(v):
    """
    Generate a skew-symmetric matrix from a 3D vector.
    
    Parameters:
    v : array-like
        A 3D vector.
        
    Returns:
    np.ndarray
        A 3x3 skew-symmetric matrix.
    """
    v = v.flatten()
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def align_vectors(v1, v2):
    """
    Compute the rotation matrix that aligns vector v1 to vector v2.
    
    Parameters:
    v1, v2 : array-like
        Input vectors to be aligned.
        
    Returns:
    np.ndarray
        A 3x3 rotation matrix that aligns v1 to v2.
    """
    v1_u = normalize_vector(v1)
    v2_u = normalize_vector(v2)
    
    v = np.cross(v1_u, v2_u)
    s = np.linalg.norm(v)
    c = np.dot(v1_u, v2_u)
    
    if np.isclose(c, 1.0):
        return Quaternion(0, 0, 0, 1)  # No rotation needed

    skew_mat = skew(v)

    R = np.eye(3) + skew_mat + np.dot(skew_mat, skew_mat) * ((1 - c) / (s ** 2))
    return Quaternion.from_dcm(R)

def get_planet_position(time, planet_id):
    """
    Get the position of a planet at a given time using SPICE toolkit.
    
    Parameters:
    time : astropy.time.Time
        The time at which to get the planet position.
    planet_id : int
        The SPICE ID of the planet.
        
    Returns:
    np.ndarray
        The position vector of the planet in km.
    """

    planet_state, _ = spice.spkezr(str(planet_id), spice.str2et(time.isot), 'J2000', 'NONE', 'SSB')
    return np.array(planet_state[0:3])  # km

def cartesian_to_spherical(vec):
    """
    Convert a Cartesian vector to spherical coordinates (r, theta, phi).
    
    Parameters:
    vec : array-like
        A 3D Cartesian vector.
        
    Returns:
    np.ndarray
        A vector containing the spherical coordinates (r, theta, phi).
    """
    x, y, z = vec
    r = np.linalg.norm(vec)
    theta = np.arccos(z / r) if r != 0 else 0.0  # polar angle
    phi = np.arctan2(y, x)  # azimuthal angle
    return np.array([r, theta, phi])

class Quaternion:
    """A class representing a scalar last quaternion."""
    def __init__(self, x, y, z, s):
        self.x = x
        self.y = y
        self.z = z
        self.s = s

        self.vec = np.array([x, y, z])
    
    def as_array(self):
        return np.array([self.x, self.y, self.z, self.s])

    def __add__(self, other):
        return Quaternion(self.x + other.x, self.y + other.y, self.z + other.z, self.s + other.s)

    def __neg__(self):
        return Quaternion(-self.x, -self.y, -self.z, -self.s)
        
    def from_dcm(dcm):
        """Create a quaternion from a direction cosine matrix."""
        m = dcm
        tr = m[0, 0] + m[1, 1] + m[2, 2]

        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            s = 0.25 * S
            x = (m[2, 1] - m[1, 2]) / S
            y = (m[0, 2] - m[2, 0]) / S
            z = (m[1, 0] - m[0, 1]) / S
        elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
            S = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            x = 0.25 * S
            y = (m[0, 1] + m[1, 0]) / S
            z = (m[0, 2] + m[2, 0]) / S
            s = (m[2, 1] - m[1, 2]) / S
        elif m[1, 1] > m[2, 2]:
            S = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            s = (m[0, 2] - m[2, 0]) / S
            x = (m[0, 1] + m[1, 0]) / S
            y = 0.25 * S
            z = (m[1, 2] + m[2, 1]) / S
        else:
            S = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            s = (m[1, 0] - m[0, 1]) / S
            x = (m[0, 2] + m[2, 0]) / S
            y = (m[1, 2] + m[2, 1]) / S
            z = 0.25 * S
        return Quaternion(x, y, z, s)

    def __repr__(self):
        return f"({self.x}, {self.y}, {self.z}, {self.s})"

    def conjugate(self):    
        """Compute the conjugate of a quaternion."""
        x, y, z, s = self.x, self.y, self.z, self.s
        return Quaternion(-x, -y, -z, s)
    
    def quaternion_cross_matrix(self):
        vec, s = self.vec, self.s
        return np.block([[s*np.eye(3) + skew(vec), -vec.reshape(3, 1)],
                         [vec.reshape(1, 3), s]]).T
    
    def mult(self, right_quat):
        """THE NOT HAMILTON PRODUCT (X) in Fundamentals of Spacecraft Attitude Determination and Control, F. Landis Markley, John L. Crassidis, 2014."""
        x1, y1, z1, s1 = self.x, self.y, self.z, self.s
        right_quat = right_quat.as_array()

        w_1_cross = self.quaternion_cross_matrix()
        new_quat = w_1_cross @ right_quat
        return Quaternion(*new_quat)
    
    def rotate_vector(self, vec):
        """Rotate a vector by the quaternion."""
        return self.mult(Quaternion(*vec, 0)).mult(self.conjugate()).vec
    
    def as_dcm(self):
        """Convert the quaternion to a direction cosine matrix."""
        x, y, z, s = self.x, self.y, self.z, self.s
        return np.array([[s**2 + x**2 - y**2 - z**2 , 2*(x*y + s*z), 2*(x*z - s*y)],
                         [2*(x*y - s*z), s**2 - x**2 + y**2 - z**2, 2*(y*z + s*x)], 
                         [2*(x*z + s*y), 2*(y*z - s*x), s**2 - x**2 - y**2 + z**2]])
    
    def normalize(self):
        norm = np.linalg.norm(self.as_array())
        if norm == 0:
            print("Warning: Attempting to normalize a zero quaternion. Returning the original quaternion.")
            return self
        return Quaternion(*(self.as_array() / norm))
    
    def ensure_positive_scalar(self):
        if self.s < 0:
            return -self
        return self
    
    def pure_quaternion_exp(self):
        """Compute the exponential of a pure quaternion (0, v)."""
        # Ensure the quaternion is pure
        if self.s != 0:
            raise ValueError("Input quaternion must be pure (scalar part must be zero).")
        v_norm = np.linalg.norm(self.vec)
        if v_norm == 0:
            return Quaternion(0, 0, 0, 1)  # exp(0) = 1
        exp_s = np.cos(v_norm)
        exp_v = np.sin(v_norm) * (self.vec / v_norm)
        return Quaternion(exp_v[0], exp_v[1], exp_v[2], exp_s)
    
class StarCatalog:
    def __init__(self, file_path, epoch, vmag_lim=6.5):
        """
        Reads the Hipparcos catalog from a CSV file and returns a pandas DataFrame.
        
        Parameters:
        file_path (str): The path to the CSV file containing the Hipparcos catalog.
        
        Returns:
        tuple: Arrays containing catalog IDs, right ascension, declination, proper motion in RA, and proper motion in DE.
        """
        hip_df = pd.read_csv(file_path, skiprows=[1], header=0)

        self.t_epoch = Time(epoch, scale="tt")

        catalog     = hip_df[hip_df['Vmag'] <= vmag_lim].reset_index(drop=True)
        self.ids    = catalog['HIP'].to_numpy().astype(np.float64)
        self.RA     = ((catalog['_RAJ2000'].to_numpy())*u.deg).to(u.rad).value
        self.DE     = ((catalog['_DEJ2000'].to_numpy())*u.deg).to(u.rad).value
        self.pmRA   = ((catalog['pmRA'].to_numpy())*u.mas).to(u.rad).value
        self.pmDE   = ((catalog['pmDE'].to_numpy())*u.mas).to(u.rad).value
        self.length = len(self.ids)
        self.vmag_max = catalog['Vmag'].max()

    def get_stars_los_at_idx(self, t_now, star_ids):
        years_since_epoch   = (t_now - self.t_epoch).to(u.yr).value

        idxs = [np.where(self.ids == hip_id)[0][0] for hip_id in star_ids]
        catalog_RA = self.RA[idxs]
        catalog_DE = self.DE[idxs]
        catalog_pmRA = self.pmRA[idxs]
        catalog_pmDE = self.pmDE[idxs]

        RA = catalog_RA + (catalog_pmRA / np.cos(catalog_DE)) * years_since_epoch
        DE = catalog_DE + catalog_pmDE * years_since_epoch

        star_los_icrf = np.zeros((len(star_ids), 3))
        star_los_icrf[:, 0] = np.cos(RA) * np.cos(DE)
        star_los_icrf[:, 1] = np.sin(RA) * np.cos(DE)
        star_los_icrf[:, 2] = np.sin(DE)

        return star_los_icrf
