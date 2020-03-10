import numpy as np
from .uav_dynamics import UAVDynamics
from .utility.rotations import Quaternion2Rotation


class Catapult(UAVDynamics):

    def __init__(self, uav_params, sensor_params, ts):

        super().__init__(uav_params, sensor_params, ts)

        self.mag_acc = 300

        rot = Quaternion2Rotation(self.state[6:10])
        g_vector = np.array([[0.0], [0.0], [self.uav.mass * self.uav.g0]])
        f_gravity = np.matmul(rot, g_vector)

        applied_vector = np.array([[self.uav.mass * self.mag_acc], [0.0], [0.0]])
        f_applied = np.matmul(rot, applied_vector)

        print(f_applied)
        print(f_gravity)

        self.fm_final = f_applied - f_gravity

        self._update_true_state()

    def launch(self):

        forces_moments = np.concatenate((self.fm_final,
                                         np.array([[0], [0], [0]])))

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

        self._update_velocity()
        self._update_true_state()

    def update(self, *args):

        rot = Quaternion2Rotation(self.state[6:10])
        g_vector = np.array([[0.0], [0.0], [self.uav.mass * self.uav.g0]])
        f_gravity = np.matmul(rot.T, g_vector)

        fx = f_gravity.item(0)
        fy = f_gravity.item(1)
        fz = f_gravity.item(2)

        forces_moments = np.array([[fx, fy, fz, 0, 0, 0]]).T

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

        self._update_velocity()
        self._update_true_state()
