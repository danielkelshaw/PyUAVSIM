import numpy as np
from .parameters import uav_parameters as uav
from .messages import MsgState
from .utility.rotations import Quaternion2Rotation, Quaternion2Euler


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

        self.wind = np.array([[0.0], [0.0], [0.0]])
        self._update_velocity()

        self.forces = np.array([[0.0], [0.0], [0.0]])

        self.v_air = uav.u0
        self.alpha = 0
        self.beta = 0

        self.true_state = MsgState()

    def update(self, delta, wind):

        forces_moments = self._forces_moments(delta)

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

        self._update_velocity(wind)
        self._update_true_state()

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

    def _update_velocity(self, wind=np.zeros((6, 1))):

        steady_state = wind[0:3]
        gust = wind[3:6]

        rot = Quaternion2Rotation(self.state[6:10])
        wind_bframe = np.matmul(rot.T, steady_state)
        wind_bframe += gust

        v_air = self.state[3:6] - wind_bframe
        ur = v_air.item(0)
        vr = v_air.item(1)
        wr = v_air.item(2)

        self.v_air = np.sqrt(ur ** 2 + vr ** 2 + wr ** 2)

        if ur == 0:
            self.alpha = np.sign(wr) * np.pi / 2
        else:
            self.alpha = np.arctan(wr / ur)

        tmp = np.sqrt(ur ** 2 + wr ** 2)
        if tmp == 0:
            self.beta = np.sign(vr) * np.pi / 2
        else:
            self.beta = np.arcsin(vr / tmp)

    def _forces_moments(self, delta):

        p = self.state.item(10)
        q = self.state.item(11)
        r = self.state.item(12)

        delta_e = delta.item(0)
        delta_a = delta.item(1)
        delta_r = delta.item(2)
        delta_t = delta.item(3)

        rot = Quaternion2Rotation(self.state[6:10])
        g_vector = np.array([[0.0], [0.0], [uav.mass * uav.g0]])
        f_gravity = np.matmul(rot.T, g_vector)

        fx = f_gravity.item(0)
        fy = f_gravity.item(1)
        fz = f_gravity.item(2)

        qbar = 0.5 * uav.rho * self.v_air ** 2
        c_alpha = np.cos(self.alpha)
        s_alpha = np.sin(self.alpha)

        p_ndim = p * uav.b / (2 * self.v_air)
        q_ndim = q * uav.c / (2 * self.v_air)
        r_ndim = r * uav.b / (2 * self.v_air)

        tmp1 = np.exp(-uav.M * (self.alpha - uav.alpha0))
        tmp2 = np.exp(uav.M * (self.alpha + uav.alpha0))
        sigma = (1 + tmp1 + tmp2) / ((1 + tmp1) * (1 + tmp2))

        cl = ((1 - sigma) * (uav.C_L_0 + uav.C_L_alpha * self.alpha)
              + sigma * 2 * np.sign(self.alpha) * s_alpha ** 2 * c_alpha)

        cd = uav.C_D_p + ((uav.C_L_0 + uav.C_L_alpha * self.alpha) ** 2
                          / (np.pi * uav.e * uav.AR))

        f_lift = qbar * uav.S_wing * (cl
                                      + uav.C_L_q * q_ndim
                                      + uav.C_L_delta_e * delta_e)

        f_drag = qbar * uav.S_wing * (cd
                                      + uav.C_D_q * q_ndim
                                      + uav.C_D_delta_e * delta_e)

        fx = fx - c_alpha * f_drag + s_alpha * f_lift
        fz = fz - s_alpha * f_drag - c_alpha * f_lift

        fy = fy + qbar * uav.S_wing * (uav.C_Y_0
                                       + uav.C_Y_beta * self.beta
                                       + uav.C_Y_p * p_ndim
                                       + uav.C_Y_r * r_ndim
                                       + uav.C_Y_delta_a * delta_a
                                       + uav.C_Y_delta_r * delta_r)

        My = qbar * uav.S_wing * uav.c * (uav.C_m_0
                                          + uav.C_m_alpha * self.alpha
                                          + uav.C_m_q * q_ndim
                                          + uav.C_m_delta_e * delta_e)

        Mx = qbar * uav.S_wing * uav.b * (uav.C_ell_0
                                          + uav.C_ell_beta * self.beta
                                          + uav.C_ell_p * p_ndim
                                          + uav.C_ell_r * r_ndim
                                          + uav.C_ell_delta_a * delta_a
                                          + uav.C_ell_delta_r * delta_r)

        Mz = qbar * uav.S_wing * uav.b * (uav.C_n_0
                                          + uav.C_n_beta * self.beta
                                          + uav.C_n_p * p_ndim
                                          + uav.C_n_r * r_ndim
                                          + uav.C_n_delta_a * delta_a
                                          + uav.C_n_delta_r * delta_r)

        p_thrust, p_torque = self._motor_thrust_torque(self.v_air, delta_t)
        fx += p_thrust
        Mx += -p_torque

        self.forces[0] = fx
        self.forces[1] = fy
        self.forces[2] = fz

        return np.array([[fx, fy, fz, Mx, My, Mz]]).T

    def _motor_thrust_torque(self, va, delta_t):

        v_in = uav.V_max * delta_t

        a = uav.C_Q0 * uav.rho * np.power(uav.D_prop, 5) / ((2. * np.pi) ** 2)
        b = ((uav.C_Q1 * uav.rho * np.power(uav.D_prop, 4)
              / (2. * np.pi)) * va + uav.KQ ** 2 / uav.R_motor)
        c = (uav.C_Q2 * uav.rho * np.power(uav.D_prop, 3) * va ** 2
             - (uav.KQ / uav.R_motor) * v_in + uav.KQ * uav.i0)

        omega = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        J = 2 * np.pi * self.v_air / (omega * uav.D_prop)

        C_T = uav.C_T2 * J ** 2 + uav.C_T1 * J + uav.C_T0
        C_Q = uav.C_Q2 * J ** 2 + uav.C_Q1 * J + uav.C_Q0

        n = omega / (2 * np.pi)
        thrust = uav.rho * n ** 2 * np.power(uav.D_prop, 4) * C_T
        torque = uav.rho * n ** 2 * np.power(uav.D_prop, 5) * C_Q
        return thrust, torque

    def _update_true_state(self):

        phi, theta, psi = Quaternion2Euler(self.state[6:10])
        pdot = np.matmul(Quaternion2Rotation(self.state[6:10]), self.state[3:6])

        self.true_state.px = self.state.item(0)
        self.true_state.py = self.state.item(1)
        self.true_state.h = -self.state.item(2)

        self.true_state.phi = phi
        self.true_state.theta = theta
        self.true_state.psi = psi

        self.true_state.v_air = self.v_air
        self.true_state.alpha = self.alpha
        self.true_state.beta = self.beta

        self.true_state.p = self.state.item(10)
        self.true_state.q = self.state.item(11)
        self.true_state.r = self.state.item(12)

        self.true_state.v_ground = np.linalg.norm(pdot)
        self.true_state.gamma = np.arcsin(pdot.item(2)
                                          / self.true_state.v_ground)
        self.true_state.chi = np.arctan2(pdot.item(1), pdot.item(0))

        self.true_state.wx = self.wind.item(0)
        self.true_state.wy = self.wind.item(1)
