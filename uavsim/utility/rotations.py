import numpy as np
import scipy.linalg as linalg


def Quaternion2Euler(quaternion):

    e0 = quaternion.item(0)
    e1 = quaternion.item(1)
    e2 = quaternion.item(2)
    e3 = quaternion.item(3)

    phi = np.arctan2(2.0 * (e0 * e1 + e2 * e3),
                     e0 ** 2.0 + e3 ** 2.0 - e1 ** 2.0 - e2 ** 2.0)
    theta = np.arcsin(2.0 * (e0 * e2 - e1 * e3))
    psi = np.arctan2(2.0 * (e0 * e3 + e1 * e2),
                     e0 ** 2.0 + e1 ** 2.0 - e2 ** 2.0 - e3 ** 2.0)

    return phi, theta, psi


def Euler2Quaternion(phi, theta, psi):

    e0 = (np.cos(psi / 2.0) * np.cos(theta / 2.0) * np.cos(phi / 2.0)
          + np.sin(psi / 2.0) * np.sin(theta / 2.0) * np.sin(phi / 2.0))

    e1 = (np.cos(psi / 2.0) * np.cos(theta/ 2.0) * np.sin(phi / 2.0)
          - np.sin(psi / 2.0) * np.sin(theta / 2.0) * np.cos(phi / 2.0))

    e2 = (np.cos(psi / 2.0) * np.sin(theta / 2.0) * np.cos(phi / 2.0)
          + np.sin(psi / 2.0) * np.cos(theta / 2.0) * np.sin(phi / 2.0))

    e3 = (np.sin(psi / 2.0) * np.cos(theta / 2.0) * np.cos(phi / 2.0)
          - np.cos(psi / 2.0) * np.sin(theta / 2.0) * np.sin(phi / 2.0))

    return np.array([[e0],[e1],[e2],[e3]])


def Quaternion2Rotation(quaternion):

    e0 = quaternion.item(0)
    e1 = quaternion.item(1)
    e2 = quaternion.item(2)
    e3 = quaternion.item(3)

    R = np.array([[e1 ** 2.0 + e0 ** 2.0 - e2 ** 2.0 - e3 ** 2.0,
                   2.0 * (e1 * e2 - e3 * e0),
                   2.0 * (e1 * e3 + e2 * e0)],
                  [2.0 * (e1 * e2 + e3 * e0),
                   e2 ** 2.0 + e0 ** 2.0 - e1 ** 2.0 - e3 ** 2.0,
                   2.0 * (e2 * e3 - e1 * e0)],
                  [2.0 * (e1 * e3 - e2 * e0),
                   2.0 * (e2 * e3 + e1 * e0),
                   e3 ** 2.0 + e0 ** 2.0 - e1 ** 2.0 - e2 ** 2.0]])

    R = R / linalg.det(R)

    return R


def Euler2Rotation(phi, theta, psi):

    c_phi = np.cos(phi)
    s_phi = np.sin(phi)
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    c_psi = np.cos(psi)
    s_psi = np.sin(psi)

    R_roll = np.array([[1, 0, 0],
                       [0, c_phi, -s_phi],
                       [0, s_phi, c_phi]])

    R_pitch = np.array([[c_theta, 0, s_theta],
                        [0, 1, 0],
                        [-s_theta, 0, c_theta]])

    R_yaw = np.array([[c_psi, -s_psi, 0],
                      [s_psi, c_psi, 0],
                      [0, 0, 1]])

    R = np.matmul(R_yaw, np.matmul(R_pitch, R_roll))

    return R
