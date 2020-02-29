import numpy as np
from .utility.transfer_function import TransferFunction
from .utility.wrap import wrap
from .pid_control import PIControl, PDControlRate
from .messages.msg_state import MsgState


class Autopilot:

    def __init__(self, control_params, ts_control):

        self.cps = control_params
        self.ts_control = ts_control

        # lateral controllers
        self.roll_aileron = PDControlRate(kp=self.cps.roll_kp,
                                          kd=self.cps.roll_kd,
                                          limit=np.radians(45))

        self.course_roll = PIControl(kp=self.cps.course_kp,
                                     ki=self.cps.course_ki,
                                     ts=self.ts_control,
                                     limit=np.radians(30))

        self.yaw_damper = TransferFunction(
            num=np.array([[self.cps.yaw_damper_kp, 0]]),
            den=np.array([[1, 1 / self.cps.yaw_damper_tau_r]]),
            ts=self.ts_control
        )

        # longitudinal controllers
        self.pitch_elevator = PDControlRate(kp=self.cps.pitch_kp,
                                            kd=self.cps.pitch_kd,
                                            limit=np.radians(45))

        self.altitude_pitch = PIControl(kp=self.cps.altitude_kp,
                                        ki=self.cps.altitude_ki,
                                        ts=self.ts_control,
                                        limit=np.radians(30))

        self.airspeed_throttle = PIControl(kp=self.cps.airspeed_throttle_kp,
                                           ki=self.cps.airspeed_throttle_ki,
                                           ts=self.ts_control,
                                           limit=1.0)

        self.commanded_state = MsgState()

    def update(self, cmd, state):

        # lateral autopilot
        chi_c = wrap(cmd.course_command, state.chi)
        phi_c = self._saturate(cmd.phi_feedforward
                               + self.course_roll.update(chi_c, state.chi),
                               -np.radians(30), np.radians(30))

        delta_a = self.roll_aileron.update(phi_c, state.phi, state.p)
        delta_r = self.yaw_damper.update(state.r)

        # longitudinal autopilot
        h_c = self._saturate(cmd.altitude_command,
                             state.h - self.cps.altitude_zone,
                             state.h + self.cps.altitude_zone)

        theta_c = self.altitude_pitch.update(h_c, state.h)
        delta_e = self.pitch_elevator.update(theta_c, state.theta, state.q)
        delta_t = self.airspeed_throttle.update(cmd.airspeed_command,
                                                state.v_air)
        delta_t = self._saturate(delta_t, 0.0, 1.0)

        # construct output and commanded states
        delta = np.array([[delta_e], [delta_a], [delta_r], [delta_t]])
        self.commanded_state.h = cmd.altitude_command
        self.commanded_state.v_air = cmd.airspeed_command
        self.commanded_state.phi = phi_c
        self.commanded_state.theta = theta_c
        self.commanded_state.chi = cmd.course_command

        return delta, self.commanded_state

    @staticmethod
    def _saturate(input, low_limit, up_limit):
        # TODO:>> Potentially replace with np.clip()
        if input <= low_limit:
            output = low_limit
        elif input >= up_limit:
            output = up_limit
        else:
            output = input
        return output
