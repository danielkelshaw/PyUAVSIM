import numpy as np
from .parameters import uav_parameters as UAV


class UAVDynamics:

    def __init__(self, ts):

        self.ts_sim = ts

        self.state = np.array([
            [UAV.px0],
            [UAV.py0],
            [UAV.pz0],
            [UAV.u0],
            [UAV.v0],
            [UAV.w0],
            [UAV.e0],
            [UAV.e1],
            [UAV.e2],
            [UAV.e3],
            [UAV.p0],
            [UAV.q0],
            [UAV.r0]
        ])

    def update(self, forces_moments):
        pass

    def _flight_derivatives(self, state, forces_moments):
        pass
