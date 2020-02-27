import numpy as np
from uavsim.utility.rotations import Euler2Quaternion, Quaternion2Euler
from uavsim.utility.transfer_function import TransferFunction
from uavsim.parameters.tf_params import TFParams


def compute_tf_model(uav, uav_dyn, trim_state, trim_input, ts):

    # trim values
    uav_dyn.state = trim_state
    uav_dyn._update_velocity()
    v_air_trim = uav_dyn.v_air
    alpha_trim = uav_dyn.alpha
    phi, theta_trim, psi = Quaternion2Euler(trim_state[6:10])

    # define transfer function constants
    a_phi1 = (-0.5 * uav.rho * v_air_trim**2 * uav.S_wing
              * uav.b * uav.C_p_p * uav.b / 2.0 / v_air_trim)
    a_phi2 = (0.5 * uav.rho * v_air_trim**2
              * uav.S_wing * uav.b * uav.C_p_delta_a)

    a_theta1 = (-uav.rho * v_air_trim ** 2 * uav.c * uav.S_wing
                / 2.0 / uav.Jy * uav.C_m_q * uav.c / 2.0 / v_air_trim)
    a_theta2 = (-uav.rho * v_air_trim ** 2 * uav.c
                * uav.S_wing / 2.0 / uav.Jy * uav.C_m_alpha)
    a_theta3 = (uav.rho * v_air_trim ** 2 * uav.c
                * uav.S_wing / 2.0 / uav.Jy * uav.C_m_delta_e)

    a_beta1 = (-(uav.rho * v_air_trim * uav.S_wing)
               / 2.0 / uav.mass * uav.C_Y_beta)
    a_beta2 = ((uav.rho * v_air_trim ** 2 * uav.S_wing)
               / 2.0 / uav.mass * uav.C_Y_delta_r)

    # Compute transfer function coefficients using new propulsion model
    delta_e_trim = trim_input.item(0)
    delta_t_trim = trim_input.item(3)

    a_V1 = uav.rho * v_air_trim * uav.S_wing / uav.mass * (
            uav.C_D_0
            + uav.C_D_alpha * alpha_trim
            + uav.C_D_delta_e * delta_e_trim
            ) - dT_dVa(uav_dyn, v_air_trim, delta_t_trim) / uav.mass
    a_V2 = dT_ddelta_t(uav_dyn, v_air_trim, delta_t_trim) / uav.mass
    a_V3 = uav.g0 * np.cos(theta_trim - alpha_trim)

    tf_params = TFParams()

    tf_params.x_trim = trim_state
    tf_params.u_trim = trim_input
    tf_params.va_trim = v_air_trim
    tf_params.alpha_trim = alpha_trim
    tf_params.theta_trim = theta_trim

    tf_params.a_phi1 = a_phi1
    tf_params.a_phi2 = a_phi2

    tf_params.a_theta1 = a_theta1
    tf_params.a_theta2 = a_theta2
    tf_params.a_theta3 = a_theta3

    tf_params.a_beta1 = a_beta1
    tf_params.a_beta2 = a_beta2

    tf_params.a_V1 = a_V1
    tf_params.a_V2 = a_V2
    tf_params.a_V3 = a_V3

    tf_params.ts = ts


    # define transfer functions
    tf_params.T_phi_delta_a = TransferFunction(np.array([[a_phi2]]),
                                               np.array([[1, a_phi1, 0]]), ts)

    tf_params.T_chi_phi = TransferFunction(np.array([[uav.g0 / v_air_trim]]),
                                           np.array([[1, 0]]), ts)

    tf_params.T_beta_delta_r = TransferFunction(np.array([[a_beta2]]),
                                                np.array([[1, a_beta1]]), ts)

    tf_params.T_theta_delta_e = TransferFunction(np.array([[a_theta3]]),
                                                 np.array([[1, a_theta1,
                                                            a_theta2]]), ts)

    tf_params.T_h_theta = TransferFunction(np.array([[v_air_trim]]),
                                           np.array([[1, 0]]), ts)

    tf_params.T_h_Va = TransferFunction(np.array([[theta_trim]]),
                                        np.array([[1, 0]]), ts)

    tf_params.T_Va_delta_t = TransferFunction(np.array([[a_V2]]),
                                              np.array([[1, a_V1]]), ts)

    tf_params.T_Va_theta = TransferFunction(np.array([[-a_V3]]),
                                            np.array([[1, a_V1]]), ts)

    return tf_params


def compute_ss_model(uav_dyn, trim_state, trim_input):

    x_euler = euler_state(trim_state)
    A = df_dx(uav_dyn, x_euler, trim_input)
    B = df_du(uav_dyn, x_euler, trim_input)

    # extract longitudinal states (u, w, q, theta, pd)
    A_lon = A[np.ix_([3, 5, 10, 7, 2], [3, 5, 10, 7, 2])]
    B_lon = B[np.ix_([3, 5, 10, 7, 2], [0, 3])]

    # change pd to h
    for i in range(0, 5):
        A_lon[i, 4] = -A_lon[i, 4]
        A_lon[4, i] = -A_lon[4, i]
    for i in range(0, 2):
        B_lon[4, i] = -B_lon[4, i]

    # extract lateral states (v, p, r, phi, psi)
    A_lat = A[np.ix_([4, 9, 11, 6, 8], [4, 9, 11, 6, 8])]
    B_lat = B[np.ix_([4, 9, 11, 6, 8], [1, 2])]

    return A_lon, B_lon, A_lat, B_lat


def euler_state(x_quat):

    x_euler = np.zeros((12, 1))
    x_euler[0:6] = np.copy(x_quat[0:6])

    phi, theta, psi = Quaternion2Euler(x_quat[6:10])
    x_euler[6] = phi
    x_euler[7] = theta
    x_euler[8] = psi

    x_euler[9:12] = np.copy(x_quat[10:13])

    return x_euler


def quaternion_state(x_euler):

    x_quat = np.zeros((13, 1))
    x_quat[0:6] = np.copy(x_euler[0:6])

    phi = x_euler.item(6)
    theta = x_euler.item(7)
    psi = x_euler.item(8)

    quat = Euler2Quaternion(phi, theta, psi)
    x_quat[6:10] = quat
    x_quat[10:13] = np.copy(x_euler[9:12])

    return x_quat


def f_euler(uav_dyn, x_euler, input):

    # return 12x1 dynamics (as if state were Euler state)
    x_quat = quaternion_state(x_euler)

    uav_dyn.state = x_quat
    uav_dyn._update_velocity()
    f = uav_dyn._derivatives(x_quat, uav_dyn._forces_moments(input))
    f_euler_ = euler_state(f)

    eps = 0.001
    e = x_quat[6:10]

    phi = x_euler.item(6)
    theta = x_euler.item(7)
    psi = x_euler.item(8)

    dTheta_dquat = np.zeros((3, 4))

    for j in range(0, 4):
        tmp = np.zeros((4, 1))
        tmp[j][0] = eps

        e_eps = (e + tmp) / np.linalg.norm(e + tmp)
        phi_eps, theta_eps, psi_eps = Quaternion2Euler(e_eps)
        dTheta_dquat[0][j] = (phi_eps - phi) / eps
        dTheta_dquat[1][j] = (theta_eps - theta) / eps
        dTheta_dquat[2][j] = (psi_eps - psi) / eps

    f_euler_[6:9] = np.copy(np.matmul(dTheta_dquat, f[6:10]))

    return f_euler_


def df_dx(uav_dyn, x_euler, input):

    # take partial of f_euler with respect to x_euler
    eps = 0.01
    A = np.zeros((12, 12))  # Jacobian of f wrt x
    f = f_euler(uav_dyn, x_euler, input)

    for i in range(0, 12):
        x_eps = np.copy(x_euler)
        x_eps[i][0] += eps
        f_eps = f_euler(uav_dyn, x_eps, input)
        df = (f_eps - f) / eps
        A[:, i] = df[:, 0]

    return A


def df_du(uav_dyn, x_euler, delta):

    # take partial of f_euler with respect to delta
    eps = 0.01
    B = np.zeros((12, 4))  # Jacobian of f wrt u
    f = f_euler(uav_dyn, x_euler, delta)

    for i in range(0, 4):
        delta_eps = np.copy(delta)
        delta_eps[i, 0] += eps
        f_eps = f_euler(uav_dyn, x_euler, delta_eps)
        df = (f_eps - f) / eps
        B[:, i] = df[:, 0]

    return B


def dT_dVa(uav_dyn, v_air, delta_t):

    # returns the derivative of motor thrust with respect to Va
    eps = 0.01
    T_eps, Q_eps = uav_dyn._motor_thrust_torque(v_air + eps, delta_t)
    T, Q = uav_dyn._motor_thrust_torque(v_air, delta_t)

    return (T_eps - T) / eps


def dT_ddelta_t(uav_dyn, v_air, delta_t):

    # returns the derivative of motor thrust with respect to delta_t
    eps = 0.01
    T_eps, Q_eps = uav_dyn._motor_thrust_torque(v_air, delta_t + eps)
    T, Q = uav_dyn._motor_thrust_torque(v_air, delta_t)

    return (T_eps - T) / eps
