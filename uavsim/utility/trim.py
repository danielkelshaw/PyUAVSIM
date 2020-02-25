import numpy as np
from copy import deepcopy
from scipy.optimize import minimize
from .rotations import Euler2Quaternion


def compute_trim(dyn, v_air, gamma):

    dynamics = deepcopy(dyn)

    e0 = Euler2Quaternion(0, gamma, 0)

    _state = np.array([
        [dynamics.state.item(0)],
        [dynamics.state.item(1)],
        [dynamics.state.item(2)],
        [dynamics.v_air],
        [0.0],
        [0.0],
        [e0.item(0)],
        [e0.item(1)],
        [e0.item(2)],
        [e0.item(3)],
        [0.0],
        [0.0],
        [0.0],
    ])

    _delta = np.array([
        [0.0],
        [0.0],
        [0.0],
        [0.5],
    ])

    _x = np.concatenate((_state, _delta), axis=0)

    cons = ({'type': 'eq',
             'fun': lambda x: np.array([
                 x[3] ** 2 + x[4] ** 2 + x[5] ** 2 - v_air ** 2,
                 x[4],
                 x[6] ** 2 + x[7] ** 2 + x[8] ** 2 + x[9] ** 2 - 1,
                 x[7],
                 x[9],
                 x[10],
                 x[11],
                 x[12]
             ])
             })

    res = minimize(trim_objective_fun, _x, method='SLSQP',
                   args=(dynamics, v_air, gamma),
                   constraints=cons, options={'ftol': 1e-6})

    trim_state = np.array([res.x[0:13]]).T
    trim_input = np.array([res.x[13:17]]).T

    return trim_state, trim_input


def trim_objective_fun(x, dynamics, v_air, gamma):

    state = x[0:13]
    delta = x[13:17]

    xdot = np.array([[0, 0, -v_air * np.sin(gamma), 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0]]).T

    dynamics.state = state
    dynamics._update_velocity()

    forces_moments = dynamics._forces_moments(delta)
    f = dynamics._derivatives(state, forces_moments)

    state_diff = xdot - f
    obj = np.linalg.norm(state_diff[3:13]) ** 2

    return obj
