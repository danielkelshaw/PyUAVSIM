import numpy as np
from ..utility.rotations import Euler2Quaternion

# Initial conditions for UAV
px0 = 0.0
py0 = 0.0
pz0 = -100.0
u0 = 25.0
v0 = 0.0
w0 = 0.0
phi0 = 0.0
theta0 = 0.0
psi0 = 0.0
p0 = 0.0
q0 = 0.0
r0 = 0.0
Va0 = np.sqrt(u0 ** 2 + v0 ** 2 + w0 ** 2)
e = Euler2Quaternion(phi0, theta0, psi0)
e0 = e.item(0)
e1 = e.item(1)
e2 = e.item(2)
e3 = e.item(3)

# Physical Parameters
mass = 11
Jx = 0.8244
Jy = 1.135
Jz = 1.759
Jxz = 0.1204
S_wing = 0.55
b = 2.8956
c = 0.18994
S_prop = 0.2027
rho = 1.2682
e = 0.9
AR = (b ** 2) / S_wing
g0 = 9.8

# longitudinal parameters
C_L_0 = 0.23
C_D_0 = 0.043
C_m_0 = 0.0135
C_L_alpha = 5.61
C_D_alpha = 0.03
C_m_alpha = -2.74
C_L_q = 7.95
C_D_q = 0.0
C_m_q = -38.21
C_L_delta_e = 0.13
C_D_delta_e = 0.0135
C_m_delta_e = -0.99
M = 50.0
alpha0 = 0.47
epsilon = 0.16
C_D_p = 0.0

# lateral parameters
C_Y_0 = 0.0
C_ell_0 = 0.0
C_n_0 = 0.0
C_Y_beta = -0.98
C_ell_beta = -0.13
C_n_beta = 0.073
C_Y_p = 0.0
C_ell_p = -0.51
C_n_p = 0.069
C_Y_r = 0.0
C_ell_r = 0.25
C_n_r = -0.095
C_Y_delta_a = 0.075
C_ell_delta_a = 0.17
C_n_delta_a = -0.011
C_Y_delta_r = 0.19
C_ell_delta_r = 0.0024
C_n_delta_r = -0.069

# propeller thrust parameters
D_prop = 20*(0.0254)

K_V = 145.0
KQ = (1.0 / K_V) * 60.0 / (2.0 * np.pi)
R_motor = 0.042
i0 = 1.5

ncells = 12.0
V_max = 3.7 * ncells

C_Q2 = -0.01664
C_Q1 = 0.004970
C_Q0 = 0.005230
C_T2 = -0.1079
C_T1 = -0.06044
C_T0 = 0.09357

# Calculation Variables
gamma = Jx * Jz - (Jxz ** 2)
gamma1 = (Jxz * (Jx - Jy + Jz)) / gamma
gamma2 = (Jz * (Jz - Jy) + (Jxz ** 2)) / gamma
gamma3 = Jz / gamma
gamma4 = Jxz / gamma
gamma5 = (Jz - Jx) / Jy
gamma6 = Jxz / Jy
gamma7 = ((Jx - Jy) * Jx + (Jxz ** 2)) / gamma
gamma8 = Jx / gamma
