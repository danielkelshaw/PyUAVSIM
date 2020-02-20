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
gravity = 9.8

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
