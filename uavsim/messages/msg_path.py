import numpy as np


class MsgPath:

    def __init__(self):

        self.type = 'line'
        self.v_air = 25

        self.line_origin = np.array([[0.0, 0.0, 0.0]]).T
        self.line_direction = np.array([[1.0, 1.0, 1.0]]).T

        self.orbit_centre = np.array([[0.0, 0.0, 0.0]]).T
        self.orbit_radius = 100

        self.orbit_direction = 'CW'

        self.flag_path_changed = True
