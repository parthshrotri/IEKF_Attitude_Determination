import numpy as np

from scipy.integrate import solve_ivp
from utils.utils import Quaternion

def att_dt(t, state, J, torque):
    q = Quaternion(*state[0:4])
    omega = Quaternion(*state[4:7], 0)

    L = torque

    omega_dot   =  np.linalg.inv(J) @ (L - np.cross(omega.vec,  J @ omega.vec))

    q_dot       =  1/2 * (omega.mult(q).as_array())

    return np.hstack((q_dot, omega_dot))

def att_prop(vehicle, dt, torque):
    state = np.hstack((vehicle.icrf_to_body_true.as_array(), vehicle.ang_vel_true))
    sol = solve_ivp(att_dt, (0, dt), state, args=(vehicle.inertia, torque), method='RK45')
    new_state = sol.y[:, -1]
    new_q = Quaternion(*new_state[0:4])
    new_q = new_q.normalize()
    new_omega = new_state[4:7]
    return new_q.ensure_positive_scalar(), new_omega

