import numpy as np


class SensorParams:

    def __init__(self):

        self.accel_sigma = 0.0025 * 9.81

        self.gyro_bias_x = 0.0
        self.gyro_bias_y = 0.0
        self.gyro_bias_z = 0.0
        self.gyro_sigma = np.radians(0.1)

        self.p_static_sigma = 10
        self.p_diff_sigma = 2

        self.mag_beta = np.radians(1.0)
        self.mag_sigma = np.radians(0.03)

        self.ts_gps = 1.0
        self.gps_beta = 1 / 1100
        self.gps_sigma_x = 0.21
        self.gps_sigma_y = 0.21
        self.gps_sigma_z = 0.40
        self.gps_sigma_v = 0.05
        self.gps_sigma_course = self.gps_sigma_v / 10
