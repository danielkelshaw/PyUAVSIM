import numpy as np
from .parameters import uav_parameters as uav
from .utility.rotations import Quaternion2Rotation


class UAVDynamics:

    def __init__(self, ts):

        self.ts_sim = ts
        self.state = np.array([
            [uav.px0],
            [uav.py0],
            [uav.pz0],
            [uav.u0],
            [uav.v0],
            [uav.w0],
            [uav.e0],
            [uav.e1],
            [uav.e2],
            [uav.e3],
            [uav.p0],
            [uav.q0],
            [uav.r0]
        ])

    def update(self, forces_moments):

        delta_t = self.ts_sim
        k1 = self._derivatives(self.state, forces_moments)
        k2 = self._derivatives(self.state + delta_t / 2.0 * k1, forces_moments)
        k3 = self._derivatives(self.state + delta_t / 2.0 * k2, forces_moments)
        k4 = self._derivatives(self.state + delta_t * k3, forces_moments)
        self.state += delta_t / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        # normalize the quaternion
        e0 = self.state.item(6)
        e1 = self.state.item(7)
        e2 = self.state.item(8)
        e3 = self.state.item(9)
        norm = np.sqrt(e0 ** 2 + e1 ** 2 + e2 ** 2 + e3 ** 2)

        self.state[6][0] = self.state.item(6) / norm
        self.state[7][0] = self.state.item(7) / norm
        self.state[8][0] = self.state.item(8) / norm
        self.state[9][0] = self.state.item(9) / norm

    @staticmethod
    def _derivatives(state, forces_moments):

        # extract states
        u = state.item(3)
        v = state.item(4)
        w = state.item(5)
        e0 = state.item(6)
        e1 = state.item(7)
        e2 = state.item(8)
        e3 = state.item(9)
        p = state.item(10)
        q = state.item(11)
        r = state.item(12)

        fx = forces_moments.item(0)
        fy = forces_moments.item(1)
        fz = forces_moments.item(2)
        l = forces_moments.item(3)
        m = forces_moments.item(4)
        n = forces_moments.item(5)

        # position kinematics
        p_dot = np.matmul(Quaternion2Rotation(state[6:10]), state[3:6])
        px_dot = p_dot.item(0)
        py_dot = p_dot.item(1)
        pz_dot = p_dot.item(2)

        # position dynamics
        u_dot = r * v - q * w + fx / uav.mass
        v_dot = p * w - r * u + fy / uav.mass
        w_dot = q * u - p * v + fz / uav.mass

        # rotational kinematics
        e0_dot = 0.5 * (-p * e1 - q * e2 - r * e3)
        e1_dot = 0.5 * (p * e0 + r * e2 - q * e3)
        e2_dot = 0.5 * (q * e0 - r * e1 + p * e3)
        e3_dot = 0.5 * (r * e0 + q * e1 - p * e2)

        # rotatonal dynamics
        p_dot = (uav.gamma1 * p * q
                 - uav.gamma2 * q * r
                 + uav.gamma3 * l
                 + uav.gamma4 * n)

        q_dot = (uav.gamma5 * p * r
                 - uav.gamma6 * (p ** 2 - r ** 2)
                 + m / uav.Jy)

        r_dot = (uav.gamma7 * p * q
                 - uav.gamma1 * q * r
                 + uav.gamma4 * l
                 + uav.gamma8 * n)

        # state derivative vector
        x_dot = np.array([
            [px_dot],
            [py_dot],
            [pz_dot],
            [u_dot],
            [v_dot],
            [w_dot],
            [e0_dot],
            [e1_dot],
            [e2_dot],
            [e3_dot],
            [p_dot],
            [q_dot],
            [r_dot]
        ])

        return x_dot
