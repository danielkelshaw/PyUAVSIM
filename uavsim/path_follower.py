import numpy as np
from .messages.msg_autopilot import MsgAutopilot
from .utility.wrap import wrap


class PathFollower:

    def __init__(self):

        self.chi_inf = np.radians(50)
        self.k_path = 0.05
        self.k_orbit = 10.0
        self.g0 = 9.81

        self.autopilot_commands = MsgAutopilot()

    def update(self, path, state):

        if path.type == 'line':
            self._follow_straight_line(path, state)
        elif path.type == 'orbit':
            self._follow_orbit(path, state)
        return self.autopilot_commands

    def _follow_straight_line(self, path, state):

        self.autopilot_commands.airspeed_command = path.v_air

        chi_q = np.arctan2(path.line_direction.item(1),
                           path.line_direction.item(0))
        chi_q = wrap(chi_q, state.chi)

        path_error = (-np.sin(chi_q) * (state.px - path.line_origin.item(0))
                      + np.cos(chi_q) * (state.py - path.line_origin.item(1)))

        self.autopilot_commands.course_command = (
                chi_q - self.chi_inf * (2 / np.pi)
                * np.arctan(self.k_path * path_error)
        )

        self.autopilot_commands.altitude_command = (
                -path.line_origin.item(2)
                - np.sqrt((path.line_origin.item(0) - state.px) ** 2
                          + (path.line_origin.item(1) - state.py) ** 2)
                * path.line_direction.item(2)
                / np.sqrt(path.line_direction.item(0) ** 2
                          + path.line_direction.item(1) ** 2)
        )

        self.autopilot_commands.phi_feedforward = 0.0

    def _follow_orbit(self, path, state):

        if path.orbit_direction == 'CW':
            direction = 1.0
        else:
            direction = -1.0

        self.autopilot_commands.airspeed_command = path.v_air

        d = np.sqrt((state.px - path.orbit_centre.item(0)) ** 2
                    + (state.py - path.orbit_centre.item(1)) ** 2)

        varphi = np.arctan2(state.py - path.orbit_centre.item(1),
                            state.px - path.orbit_centre.item(0))
        varphi = wrap(varphi, state.chi)

        orbit_error = (d - path.orbit_radius) / path.orbit_radius

        self.autopilot_commands.course_command = (
            varphi + direction
            * (np.pi / 2.0 + np.arctan(self.k_orbit * orbit_error))
        )

        self.autopilot_commands.altitude_command = path.orbit_centre.item(2)

        if orbit_error < 0.25:
            self.autopilot_commands.phi_feedforward = (
                direction
                * np.arctan(path.v_air ** 2 / self.g0 / path.orbit_radius)
            )
        else:
            self.autopilot_commands.phi_feedforward = 0.0
