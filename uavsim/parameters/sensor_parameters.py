import yaml
import numpy as np


class SensorParams:

    def __init__(self, filename):

        self.filename = filename
        self._load_file()

    def _load_file(self):

        with open(self.filename) as file:
            params = yaml.load(file, Loader=yaml.FullLoader)

        self.accel_sigma = params['accelerometer']['accel_sigma'] * 9.81

        self.gyro_bias_x = params['gyroscope']['gyro_bias_x']
        self.gyro_bias_y = params['gyroscope']['gyro_bias_y']
        self.gyro_bias_z = params['gyroscope']['gyro_bias_z']
        self.gyro_sigma = np.radians(params['gyroscope']['gyro_sigma'])

        self.mag_beta = np.radians(params['magnetometer']['mag_beta'])
        self.mag_sigma = np.radians(params['magnetometer']['mag_sigma'])

        self.ts_gps = params['gps']['ts_gps']
        self.gps_beta = params['gps']['gps_beta']
        self.gps_sigma_x = params['gps']['gps_sigma_x']
        self.gps_sigma_y = params['gps']['gps_sigma_y']
        self.gps_sigma_z = params['gps']['gps_sigma_z']
        self.gps_sigma_v = params['gps']['gps_sigma_v']
        self.gps_sigma_course = self.gps_sigma_v / 10

        self.p_static_sigma = params['pressure']['p_static_sigma']
        self.p_diff_sigma = params['pressure']['p_diff_sigma']
