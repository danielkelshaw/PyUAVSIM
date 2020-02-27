from ..utility.compute_models import *


class ControlParams:

    def __init__(self, uav, uav_dyn, trim_state, trim_input, ts):

        self.uav = uav
        self.uav_dyn = uav_dyn
        self.trim_state = trim_state
        self.trim_input = trim_input
        self.ts = ts

        self.tf_params = self._get_tf_params()
        self.load_params()

    def _get_tf_params(self):
        return compute_tf_model(self.uav, self.uav_dyn, self.trim_state,
                                self.trim_input, self.ts)

    def load_params(self):

        self.gravity = self.uav.g0
        self.rho = self.uav.rho
        self.sigma = 0.05
        self.Va0 = self.tf_params.va_trim

        # roll loop
        self.wn_roll = 7
        self.zeta_roll = 0.707
        self.roll_kp = self.wn_roll ** 2 / self.tf_params.a_phi2
        self.roll_kd = (2.0 * self.zeta_roll * self.wn_roll
                        - self.tf_params.a_phi1) / self.tf_params.a_phi2

        # course loop
        self.wn_course = self.wn_roll / 20.0
        self.zeta_course = 1.0
        self.course_kp = (2.0 * self.zeta_course * self.wn_course
                          * self.Va0 / self.gravity)
        self.course_ki = self.wn_course ** 2 * self.Va0 / self.gravity

        # sideslip loop
        self.wn_sideslip = 0.5
        self.zeta_sideslip = 5.0
        self.sideslip_ki = self.wn_sideslip ** 2 / self.tf_params.a_beta2
        self.sideslip_kp = (2.0 * self.zeta_sideslip * self.wn_sideslip
                            - self.tf_params.a_beta1) / self.tf_params.a_beta2

        # yaw damper
        self.yaw_damper_tau_r = 0.5
        self.yaw_damper_kp = 0.5

        # pitch loop
        self.wn_pitch = 24.0
        self.zeta_pitch = 0.707
        self.pitch_kp = ((self.wn_pitch ** 2 - self.tf_params.a_theta2)
                         / self.tf_params.a_theta3)
        self.pitch_kd = (2.0 * self.zeta_pitch * self.wn_pitch
                          - self.tf_params.a_theta1) / self.tf_params.a_theta3
        self.K_theta_DC = (self.pitch_kp * self.tf_params.a_theta3
                           / (self.tf_params.a_theta2
                              + self.pitch_kp * self.tf_params.a_theta3))
        # altitude loop
        self.wn_altitude = self.wn_pitch / 30.0
        self.zeta_altitude = 1.0
        self.altitude_kp = (2.0 * self.zeta_altitude * self.wn_altitude
                            / self.K_theta_DC / self.Va0)
        self.altitude_ki = self.wn_altitude ** 2 / self.K_theta_DC / self.Va0
        self.altitude_zone = 10.0

        # airspeed hold
        self.wn_airspeed_throttle = 3.0
        self.zeta_airspeed_throttle = 2.0
        self.airspeed_throttle_kp = ((2.0 * self.zeta_airspeed_throttle
                                      * self.wn_airspeed_throttle
                                      - self.tf_params.a_V1)
                                     / self.tf_params.a_V2 )
        self.airspeed_throttle_ki = (self.wn_airspeed_throttle ** 2
                                     / self.tf_params.a_V2)
