import numpy as np
from .messages import MsgState, MsgSensors
from .parameters.uav_parameters import UAVParams
from .parameters.sensor_parameters import SensorParams
from .utility.rotations import Quaternion2Rotation, Quaternion2Euler
from .utility.rotations import Euler2Rotation


class UAVDynamics:

    def __init__(self, uav_params, ts):

        assert isinstance(uav_params, UAVParams)

        self.uav = uav_params
        self.ts_sim = ts

        self.state = np.array([
            [self.uav.px0],
            [self.uav.py0],
            [self.uav.pz0],
            [self.uav.u0],
            [self.uav.v0],
            [self.uav.w0],
            [self.uav.e0],
            [self.uav.e1],
            [self.uav.e2],
            [self.uav.e3],
            [self.uav.p0],
            [self.uav.q0],
            [self.uav.r0]
        ])

        self.wind = np.array([[0.0], [0.0], [0.0]])
        self._update_velocity()

        self.forces = np.array([[0.0], [0.0], [0.0]])

        self.v_air = self.uav.u0
        self.alpha = 0
        self.beta = 0

        self.true_state = MsgState()

        self._sp = SensorParams()
        self.sensors = MsgSensors()
        self.t_gps = 1e6
        self.gps_eta_x = 0.0
        self.gps_eta_y = 0.0
        self.gps_eta_z = 0.0

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

    def get_sensors(self):

        phi, theta, psi = Quaternion2Euler(self.state[6:10])
        pdot = np.matmul(Quaternion2Rotation(self.state[6:10]), self.state[3:6])

        # simulate gyroscopes
        self.sensors.gyro_x = (self.state.item(10)
                               + np.random.normal(self._sp.gyro_bias_x,
                                                  self._sp.gyro_sigma))

        self.sensors.gyro_y = (self.state.item(11)
                               + np.random.normal(self._sp.gyro_bias_y,
                                                  self._sp.gyro_sigma))

        self.sensors.gyro_z = (self.state.item(12)
                               + np.random.normal(self._sp.gyro_bias_z,
                                                  self._sp.gyro_sigma))

        # simulate accelerometers
        self.sensors.accel_x = (self.forces.item(0) / self.uav.mass
                                + self.uav.g0 * np.sin(theta)
                                + np.random.normal(0, self._sp.accel_sigma))

        self.sensors.accel_y = (self.forces.item(1) / self.uav.mass
                                + self.uav.g0 * np.cos(theta) * np.sin(phi)
                                + np.random.normal(0, self._sp.accel_sigma))

        self.sensors.accel_z = (self.forces.item(2) / self.uav.mass
                                + self.uav.g0 * np.cos(theta) * np.cos(phi)
                                + np.random.normal(0, self._sp.accel_sigma))

        # simulate magnetometers
        rot_mag = Euler2Rotation(0.0, np.radians(-66), np.radians(12.5))
        mag_inertial = np.matmul(rot_mag.T, np.array([[1.0], [0.0], [0.0]]))
        rot = Quaternion2Rotation(self.state[6:10])
        mag_body = np.matmul(rot.T, mag_inertial)

        self.sensors.mag_x = (mag_body.item(0)
                              + np.random.normal(0, self._sp.mag_sigma))

        self.sensors.mag_y = (mag_body.item(1)
                              + np.random.normal(0, self._sp.mag_sigma))

        self.sensors.mag_z = (mag_body.item(2)
                              + np.random.normal(0, self._sp.mag_sigma))

        # simulate pressure sensors
        self.sensors.p_static = (-self.uav.rho * self.uav.g0
                                 * self.state.item(2)
                                 + np.random.normal(0, self._sp.p_static_sigma))

        self.sensors.p_diff = (0.5 * self.uav.rho * self.v_air ** 2
                               + np.random.normal(0, self._sp.p_diff_sigma))

        # simulate gps sensor
        if self.t_gps > self._sp.ts_gps:
            self.gps_eta_x = (np.exp(-self._sp.gps_beta * self._sp.ts_gps)
                              * self.gps_eta_x
                              + np.random.normal(0, self._sp.gps_sigma_x))

            self.gps_eta_y = (np.exp(-self._sp.gps_beta * self._sp.ts_gps)
                              * self.gps_eta_y
                              + np.random.normal(0, self._sp.gps_sigma_y))

            self.gps_eta_z = (np.exp(-self._sp.gps_beta * self._sp.ts_gps)
                              * self.gps_eta_z
                              + np.random.normal(0, self._sp.gps_sigma_z))

            self.sensors.gps_x = self.state.item(0) + self.gps_eta_x
            self.sensors.gps_y = self.state.item(1) + self.gps_eta_y
            self.sensors.gps_z = self.state.item(2) + self.gps_eta_z
            self.sensors.gps_v = np.linalg.norm(
                self.state[3:6] + np.random.normal(0, self._sp.gps_sigma_v)
            )

            self.sensors.gps_course = np.arctan2(pdot.item(1), pdot.item(2))
            self.sensors.gps_course += np.random.normal(
                0, self._sp.gps_sigma_course
            )

            self.t_gps = 0.0
        else:
            self.t_gps += self.ts_sim
        return self.sensors

    def _derivatives(self, state, forces_moments):

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
        u_dot = r * v - q * w + fx / self.uav.mass
        v_dot = p * w - r * u + fy / self.uav.mass
        w_dot = q * u - p * v + fz / self.uav.mass

        # rotational kinematics
        e0_dot = 0.5 * (-p * e1 - q * e2 - r * e3)
        e1_dot = 0.5 * (p * e0 + r * e2 - q * e3)
        e2_dot = 0.5 * (q * e0 - r * e1 + p * e3)
        e3_dot = 0.5 * (r * e0 + q * e1 - p * e2)

        # rotatonal dynamics
        p_dot = (self.uav.gamma1 * p * q
                 - self.uav.gamma2 * q * r
                 + self.uav.gamma3 * l
                 + self.uav.gamma4 * n)

        q_dot = (self.uav.gamma5 * p * r
                 - self.uav.gamma6 * (p ** 2 - r ** 2)
                 + m / self.uav.Jy)

        r_dot = (self.uav.gamma7 * p * q
                 - self.uav.gamma1 * q * r
                 + self.uav.gamma4 * l
                 + self.uav.gamma8 * n)

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
        g_vector = np.array([[0.0], [0.0], [self.uav.mass * self.uav.g0]])
        f_gravity = np.matmul(rot.T, g_vector)

        fx = f_gravity.item(0)
        fy = f_gravity.item(1)
        fz = f_gravity.item(2)

        qbar = 0.5 * self.uav.rho * self.v_air ** 2
        c_alpha = np.cos(self.alpha)
        s_alpha = np.sin(self.alpha)

        p_ndim = p * self.uav.b / (2 * self.v_air)
        q_ndim = q * self.uav.c / (2 * self.v_air)
        r_ndim = r * self.uav.b / (2 * self.v_air)

        tmp1 = np.exp(-self.uav.M * (self.alpha - self.uav.alpha0))
        tmp2 = np.exp(self.uav.M * (self.alpha + self.uav.alpha0))
        sigma = (1 + tmp1 + tmp2) / ((1 + tmp1) * (1 + tmp2))

        cl = ((1 - sigma) * (self.uav.C_L_0 + self.uav.C_L_alpha * self.alpha)
              + sigma * 2 * np.sign(self.alpha) * s_alpha ** 2 * c_alpha)

        cd = (self.uav.C_D_p
              + ((self.uav.C_L_0 + self.uav.C_L_alpha * self.alpha) ** 2
                 / (np.pi * self.uav.e * self.uav.AR)))

        f_lift = qbar * self.uav.S_wing * (
                cl
                + self.uav.C_L_q * q_ndim
                + self.uav.C_L_delta_e * delta_e
        )

        f_drag = qbar * self.uav.S_wing * (
                cd
                + self.uav.C_D_q * q_ndim
                + self.uav.C_D_delta_e * delta_e
        )

        fx = fx - c_alpha * f_drag + s_alpha * f_lift
        fz = fz - s_alpha * f_drag - c_alpha * f_lift

        fy = fy + qbar * self.uav.S_wing * (
                self.uav.C_Y_0
                + self.uav.C_Y_beta * self.beta
                + self.uav.C_Y_p * p_ndim
                + self.uav.C_Y_r * r_ndim
                + self.uav.C_Y_delta_a * delta_a
                + self.uav.C_Y_delta_r * delta_r
        )

        My = qbar * self.uav.S_wing * self.uav.c * (
                self.uav.C_m_0
                + self.uav.C_m_alpha * self.alpha
                + self.uav.C_m_q * q_ndim
                + self.uav.C_m_delta_e * delta_e
        )

        Mx = qbar * self.uav.S_wing * self.uav.b * (
                self.uav.C_ell_0
                + self.uav.C_ell_beta * self.beta
                + self.uav.C_ell_p * p_ndim
                + self.uav.C_ell_r * r_ndim
                + self.uav.C_ell_delta_a * delta_a
                + self.uav.C_ell_delta_r * delta_r
        )

        Mz = qbar * self.uav.S_wing * self.uav.b * (
                self.uav.C_n_0
                + self.uav.C_n_beta * self.beta
                + self.uav.C_n_p * p_ndim
                + self.uav.C_n_r * r_ndim
                + self.uav.C_n_delta_a * delta_a
                + self.uav.C_n_delta_r * delta_r
        )

        p_thrust, p_torque = self._motor_thrust_torque(self.v_air, delta_t)
        fx += p_thrust
        Mx += -p_torque

        self.forces[0] = fx
        self.forces[1] = fy
        self.forces[2] = fz

        return np.array([[fx, fy, fz, Mx, My, Mz]]).T

    def _motor_thrust_torque(self, va, delta_t):

        v_in = self.uav.V_max * delta_t

        a = (self.uav.C_Q0 * self.uav.rho
             * np.power(self.uav.D_prop, 5) / ((2. * np.pi) ** 2))

        b = ((self.uav.C_Q1 * self.uav.rho
              * np.power(self.uav.D_prop, 4) / (2. * np.pi)) * va
             + self.uav.KQ ** 2 / self.uav.R_motor)

        c = (self.uav.C_Q2 * self.uav.rho
             * np.power(self.uav.D_prop, 3) * va ** 2
             - (self.uav.KQ / self.uav.R_motor) * v_in
             + self.uav.KQ * self.uav.i0)

        omega = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        J = 2 * np.pi * self.v_air / (omega * self.uav.D_prop)

        C_T = self.uav.C_T2 * J ** 2 + self.uav.C_T1 * J + self.uav.C_T0
        C_Q = self.uav.C_Q2 * J ** 2 + self.uav.C_Q1 * J + self.uav.C_Q0

        n = omega / (2 * np.pi)
        thrust = self.uav.rho * n ** 2 * np.power(self.uav.D_prop, 4) * C_T
        torque = self.uav.rho * n ** 2 * np.power(self.uav.D_prop, 5) * C_Q
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
