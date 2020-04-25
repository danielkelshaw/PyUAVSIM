import numpy as np
from scipy import stats
from .messages.msg_state import MsgState
from .utility.wrap import wrap


class Observer:

    def __init__(self, uav_params, sensor_params, ts_control):

        self.uav = uav_params

        self.estimated_state = MsgState()

        self.lpf_gyro_x = AlphaFilter(alpha=0.2)
        self.lpf_gyro_y = AlphaFilter(alpha=0.2)
        self.lpf_gyro_z = AlphaFilter(alpha=0.2)

        self.lpf_accel_x = AlphaFilter(alpha=0.3)
        self.lpf_accel_y = AlphaFilter(alpha=0.3)
        self.lpf_accel_z = AlphaFilter(alpha=0.3)

        self.lpf_p_static = AlphaFilter(alpha=0.9)
        self.lpf_p_diff = AlphaFilter(alpha=0.5)

        self.attitude_ekf = EkfAttitude(uav_params, sensor_params, ts_control)
        self.position_ekf = EkfPosition(uav_params, sensor_params, ts_control)

    def update(self, measurement):

        self.estimated_state.p = (self.lpf_gyro_x.update(measurement.gyro_x)
                                  - self.estimated_state.bx)

        self.estimated_state.q = (self.lpf_gyro_y.update(measurement.gyro_y)
                                  - self.estimated_state.by)

        self.estimated_state.r = (self.lpf_gyro_z.update(measurement.gyro_z)
                                  - self.estimated_state.bz)

        p_static = self.lpf_p_static.update(measurement.p_static)
        p_diff = self.lpf_p_diff.update(measurement.p_diff)

        self.estimated_state.h = p_static / self.uav.rho / self.uav.g0
        self.estimated_state.v_air = np.sqrt(2.0 * p_diff / self.uav.rho)

        self.attitude_ekf.update(measurement, self.estimated_state)
        self.position_ekf.update(measurement, self.estimated_state)

        self.estimated_state.alpha = self.estimated_state.theta
        self.estimated_state.beta = 0.0
        self.estimated_state.bx = 0.0
        self.estimated_state.by = 0.0
        self.estimated_state.bz = 0.0

        return self.estimated_state


class AlphaFilter:

    def __init__(self, alpha=0.5, y0=0.0):

        self.alpha = alpha
        self.y = y0

    def update(self, u):

        self.y = self.alpha * self.y + (1 - self.alpha) * u
        return self.y


class EkfAttitude:

    def __init__(self, uav_params, sensor_params, ts_control):

        self._sp = sensor_params
        self.uav = uav_params
        self.n_steps = 2

        self.Q = 1e-9 * np.diag([1.0, 1.0])
        self.Q_gyro = self._sp.gyro_sigma ** 2 * np.diag([1.0, 1.0, 1.0])
        self.R_accel = self._sp.accel_sigma ** 2 * np.diag([1.0, 1.0, 1.0])

        self.xhat = np.array([[0.0], [0.0]])
        self.P = np.diag([1.0, 1.0])

        self.ts = ts_control / self.n_steps
        self.acc_threshold = stats.chi2.isf(q=0.01, df=3)

    def update(self, measurement, state):

        self.propogate_model(measurement, state)
        self.measurement_update(measurement, state)

        state.phi = self.xhat.item(0)
        state.theta = self.xhat.item(1)

    def f(self, x, measurement, state):

        phi = x.item(0)
        theta = x.item(1)

        p = measurement.gyro_x - state.bx
        q = measurement.gyro_y - state.by
        r = measurement.gyro_z - state.bz

        G = np.array([
            [1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
            [0.0, np.cos(phi), -np.sin(phi)]
        ])

        _f = np.matmul(G, np.array([[p], [q], [r]]))

        return _f

    def h(self, x, measurement, state):

        phi = x.item(0)
        theta = x.item(1)

        p = measurement.gyro_x - state.bx
        q = measurement.gyro_y - state.by
        r = measurement.gyro_z - state.bz

        va = np.sqrt(2 * measurement.p_diff / self.uav.rho)

        _h = np.array([
            [q * va * np.sin(theta) + self.uav.g0 * np.sin(theta)],
            [r * va * np.cos(theta) - p * va * np.sin(theta)
             - self.uav.g0 * np.cos(theta) * np.sin(phi)],
            [-q * va * np.cos(theta)
             - self.uav.g0 * np.cos(theta) * np.cos(phi)]
        ])

        return _h

    def propogate_model(self, measurement, state):

        for i in range(0, self.n_steps):
            phi = self.xhat.item(0)
            theta = self.xhat.item(1)

            self.xhat += self.ts * self.f(self.xhat, measurement, state)

            A = jacobian(self.f, self.xhat, measurement, state)
            G = np.array([
                [1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
                [0.0, np.cos(phi), -np.sin(phi)]
            ])

            Adiscrete = (np.eye(2) + self.ts * A + (self.ts ** 2)
                         * np.matmul(A, A) / 2.0)
            Gdisrete = G * self.ts

            self.P = (np.matmul(np.matmul(Adiscrete, self.P), Adiscrete.T)
                      + self.ts ** 2 * self.Q
                      + np.matmul(np.matmul(Gdisrete, self.Q_gyro), Gdisrete.T))

    def measurement_update(self, measurement, state):

        h = self.h(self.xhat, measurement, state)
        C = jacobian(self.h, self.xhat, measurement, state)
        y = np.array([[measurement.accel_x,
                       measurement.accel_y,
                       measurement.accel_z]]).T

        S_inv = np.linalg.inv(self.R_accel
                              + np.matmul(np.matmul(C, self.P), C.T))

        if np.matmul(np.matmul((y - h).T, S_inv), (y - h)) < self.acc_threshold:
            L = np.matmul(np.matmul(self.P, C.T), S_inv)
            tmp = np.eye(2) - np.matmul(L, C)

            self.P = (np.matmul(np.matmul(tmp, self.P), tmp.T)
                      + np.matmul(np.matmul(L, self.R_accel), L.T))

            self.xhat += np.matmul(L, (y - h))


class EkfPosition:

    def __init__(self, uav_params, sensor_params, ts_control):

        self._sp = sensor_params
        self.uav = uav_params
        self.n_steps = 10
        self.ts = ts_control / self.n_steps

        self.Q = np.diag([
            0.1,
            0.1,
            0.1,
            0.0001,
            0.1,
            0.1,
            0.0001
        ])

        self.R_gps = np.diag([
            self._sp.gps_sigma_x ** 2,
            self._sp.gps_sigma_y ** 2,
            self._sp.gps_sigma_v ** 2,
            self._sp.gps_sigma_course ** 2
        ])

        self.R_pseudo = np.diag([1e-6, 1e-6])

        self.xhat = np.array([
            [self.uav.px0],
            [self.uav.py0],
            [self.uav.u0],
            [0.0],
            [0.0],
            [0.0],
            [self.uav.psi0]
        ])

        self.P = np.eye(7)

        self.gps_old_x = 9999
        self.gps_old_y = 9999
        self.gps_old_v = 9999
        self.gps_old_course = 9999

        self.ps_threshold = stats.chi2.isf(q=0.01, df=2)
        self.gps_threshold = 1e6

    def update(self, measurement, state):

        self.propogate_model(measurement, state)
        self.measurement_update(measurement, state)

        state.px = self.xhat.item(0)
        state.py = self.xhat.item(1)
        state.v_ground = self.xhat.item(2)

        state.chi = self.xhat.item(3)
        state.wx = self.xhat.item(4)
        state.wy = self.xhat.item(5)
        state.psi = self.xhat.item(6)

    def f(self, x, measurement, state):

        vg = x.item(2)
        chi = x.item(3)
        wx = x.item(4)
        wy = x.item(5)
        psi = x.item(6)

        psidot = ((state.q * np.sin(state.phi) + state.r * np.cos(state.phi))
                  / np.cos(state.theta))

        vgdot = ((state.v_air * np.cos(psi) + wx)
                 * (-psidot * state.v_air * np.sin(psi))
                 + (state.v_air * np.sin(psi) + wy)
                 * (psidot * state.v_air * np.cos(psi))) / vg

        _f = np.array([
            [vg * np.cos(chi)],
            [vg * np.sin(chi)],
            [vgdot],
            [(self.uav.g0 / vg) * np.tan(state.phi) * np.cos(chi - psi)],
            [0.0],
            [0.0],
            [psidot]
        ])

        return _f

    def h_gps(self, x, measurement, state):

        px = x.item(0)
        py = x.item(1)
        vg = x.item(2)
        chi = x.item(3)

        _h = np.array([
            [px],
            [py],
            [vg],
            [chi],
        ])

        return _h

    def h_pseudo(self, x, measurement, state):

        px = x.item(0)
        py = x.item(1)
        vg = x.item(2)
        chi = x.item(3)
        wx = x.item(4)
        wy = x.item(5)
        psi = x.item(6)

        _h = np.array([
            [state.v_air * np.cos(psi) + wx - vg * np.cos(chi)],
            [state.v_air * np.sin(psi) + wy - vg * np.sin(chi)],
        ])

        return _h

    def propogate_model(self, measurement, state):

        for i in range(0, self.n_steps):
            self.xhat += self.ts * self.f(self.xhat, measurement, state)

            A = jacobian(self.f, self.xhat, measurement, state)
            Adiscrete = (np.eye(7) + self.ts * A
                         + np.matmul((self.ts ** 2) * A, A) / 2.0)

            self.P = (np.matmul(np.matmul(Adiscrete, self.P), Adiscrete.T)
                      + self.ts ** 2 * self.Q)

    def measurement_update(self, measurement, state):

        h = self.h_pseudo(self.xhat, measurement, state)
        C = jacobian(self.h_pseudo, self.xhat, measurement, state)
        y = np.array([[0, 0]]).T
        S_inv = np.linalg.inv(self.R_pseudo
                              + np.matmul(np.matmul(C, self.P), C.T))

        if np.matmul(np.matmul((y - h).T, S_inv), (y - h)) < self.ps_threshold:

            L = np.matmul(np.matmul(self.P, C.T), S_inv)
            tmp = np.eye(7) - np.matmul(L, C)
            self.P = (np.matmul(np.matmul(tmp, self.P), tmp.T)
                      + np.matmul(np.matmul(L, self.R_pseudo), L.T))

            self.xhat += np.matmul(L, (y - h))

        if ((measurement.gps_x != self.gps_old_x)
                or (measurement.gps_y != self.gps_old_y)
                or (measurement.gps_v != self.gps_old_v)
                or (measurement.gps_course != self.gps_old_course)):

            h = self.h_gps(self.xhat, measurement, state)
            C = jacobian(self.h_gps, self.xhat, measurement, state)
            y_chi = wrap(measurement.gps_course, h[3, 0])

            y = np.array([[
                measurement.gps_x,
                measurement.gps_y,
                measurement.gps_v,
                y_chi
            ]]).T

            S_inv = np.linalg.inv(self.R_gps
                                  + np.matmul(np.matmul(C, self.P), C.T))

            if np.matmul(np.matmul((y - h).T, S_inv),
                         (y - h)) < self.gps_threshold:

                L = np.matmul(np.matmul(self.P, C.T), S_inv)
                tmp = np.eye(7) - np.matmul(L, C)
                self.P = (np.matmul(np.matmul(tmp, self.P), tmp.T)
                          + np.matmul(np.matmul(L, self.R_gps), L.T))

                self.xhat += np.matmul(L, (y - h))

            self.gps_old_x = measurement.gps_x
            self.gps_old_y = measurement.gps_y
            self.gps_old_v = measurement.gps_v
            self.gps_old_course = measurement.gps_course


def jacobian(func, x, measurement, state):

    f = func(x, measurement, state)
    m = f.shape[0]
    n = x.shape[0]

    eps = 0.01
    J = np.zeros((m, n))

    for i in range(0, n):
        x_eps = np.copy(x)
        x_eps[i][0] += eps

        f_eps = func(x_eps, measurement, state)
        df = (f_eps - f) / eps

        J[:, i] = df[:, 0]

    return J