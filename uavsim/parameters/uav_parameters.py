import yaml
import numpy as np
from ..utility.rotations import Euler2Quaternion


class UAVParams:

    def __init__(self, filename):

        self.filename = filename
        self._load_file()

    def _load_file(self):

        with open(self.filename) as file:
            params = yaml.load(file, Loader=yaml.FullLoader)

        # loading initial conditions
        self.px0 = params['initial_conditions']['px0']
        self.py0 = params['initial_conditions']['py0']
        self.pz0 = params['initial_conditions']['pz0']

        self.u0 = params['initial_conditions']['u0']
        self.v0 = params['initial_conditions']['v0']
        self.w0 = params['initial_conditions']['w0']

        self.phi0 = params['initial_conditions']['phi0']
        self.theta0 = params['initial_conditions']['theta0']
        self.psi0 = params['initial_conditions']['psi0']

        self.p0 = params['initial_conditions']['p0']
        self.q0 = params['initial_conditions']['q0']
        self.r0 = params['initial_conditions']['r0']

        self.Va0 = np.sqrt(self.u0 ** 2 + self.v0 ** 2 + self.w0 ** 2)
        self.e = Euler2Quaternion(self.phi0, self.theta0, self.psi0)
        self.e0 = self.e.item(0)
        self.e1 = self.e.item(1)
        self.e2 = self.e.item(2)
        self.e3 = self.e.item(3)

        # loading physical parameters
        self.mass = params['physical_params']['mass']

        self.Jx = params['physical_params']['Jx']
        self.Jy = params['physical_params']['Jy']
        self.Jz = params['physical_params']['Jz']
        self.Jxz = params['physical_params']['Jxz']

        self.S_wing = params['physical_params']['S_wing']
        self.b = params['physical_params']['b']
        self.c = params['physical_params']['c']
        self.e = params['physical_params']['e']
        self.AR = (self.b ** 2) / self.S_wing

        self.S_prop = params['physical_params']['S_prop']

        self.rho = params['physical_params']['rho']
        self.g0 = params['physical_params']['g0']

        # loading longitudinal parameters
        self.C_L_0 = params['longitudinal_params']['C_L_0']
        self.C_D_0 = params['longitudinal_params']['C_D_0']
        self.C_m_0 = params['longitudinal_params']['C_m_0']

        self.C_L_alpha = params['longitudinal_params']['C_L_alpha']
        self.C_D_alpha = params['longitudinal_params']['C_D_alpha']
        self.C_m_alpha = params['longitudinal_params']['C_m_alpha']

        self.C_L_q = params['longitudinal_params']['C_L_q']
        self.C_D_q = params['longitudinal_params']['C_D_q']
        self.C_m_q = params['longitudinal_params']['C_m_q']

        self.C_L_delta_e = params['longitudinal_params']['C_L_delta_e']
        self.C_D_delta_e = params['longitudinal_params']['C_D_delta_e']
        self.C_m_delta_e = params['longitudinal_params']['C_m_delta_e']

        self.M = params['longitudinal_params']['M']
        self.alpha0 = params['longitudinal_params']['alpha0']
        self.epsilon = params['longitudinal_params']['epsilon']
        self.C_D_p = params['longitudinal_params']['C_D_p']

        # loading lateral parameters
        self.C_Y_0 = params['lateral_params']['C_Y_0']
        self.C_ell_0 = params['lateral_params']['C_ell_0']
        self.C_n_0 = params['lateral_params']['C_n_0']

        self.C_Y_beta = params['lateral_params']['C_Y_beta']
        self.C_ell_beta = params['lateral_params']['C_ell_beta']
        self.C_n_beta = params['lateral_params']['C_n_beta']

        self.C_Y_p = params['lateral_params']['C_Y_p']
        self.C_ell_p = params['lateral_params']['C_ell_p']
        self.C_n_p = params['lateral_params']['C_n_p']

        self.C_Y_r = params['lateral_params']['C_Y_r']
        self.C_ell_r = params['lateral_params']['C_ell_r']
        self.C_n_r = params['lateral_params']['C_n_r']

        self.C_Y_delta_a = params['lateral_params']['C_Y_delta_a']
        self.C_ell_delta_a = params['lateral_params']['C_ell_delta_a']
        self.C_n_delta_a = params['lateral_params']['C_n_delta_a']

        self.C_Y_delta_r = params['lateral_params']['C_Y_delta_r']
        self.C_ell_delta_r = params['lateral_params']['C_ell_delta_r']
        self.C_n_delta_r = params['lateral_params']['C_n_delta_r']

        # loading prop params
        self.D_prop = params['prop_params']['D_prop']

        self.K_V = params['prop_params']['K_V']
        self.KQ = (1.0 / self.K_V) * 60.0 / (2.0 * np.pi)
        self.R_motor = params['prop_params']['R_motor']
        self.i0 = params['prop_params']['i0']

        self.ncells = params['prop_params']['ncells']
        self.V_max = 3.7 * self.ncells

        self.C_Q2 = params['prop_params']['C_Q2']
        self.C_Q1 = params['prop_params']['C_Q1']
        self.C_Q0 = params['prop_params']['C_Q0']
        self.C_T2 = params['prop_params']['C_T2']
        self.C_T1 = params['prop_params']['C_T1']
        self.C_T0 = params['prop_params']['C_T0']

        # calculation variables
        self.gamma = self.Jx * self.Jz - (self.Jxz ** 2)

        self.gamma1 = (self.Jxz * (self.Jx - self.Jy + self.Jz)) / self.gamma
        self.gamma2 = (self.Jz * (self.Jz - self.Jy)
                       + (self.Jxz ** 2)) / self.gamma
        self.gamma3 = self.Jz / self.gamma
        self.gamma4 = self.Jxz / self.gamma
        self.gamma5 = (self.Jz - self.Jx) / self.Jy
        self.gamma6 = self.Jxz / self.Jy
        self.gamma7 = ((self.Jx - self.Jy) * self.Jx
                       + (self.Jxz ** 2)) / self.gamma
        self.gamma8 = self.Jx / self.gamma

        self.C_p_0 = self.gamma3 * self.C_ell_0 + self.gamma4 * self.C_n_0
        self.C_p_beta = (self.gamma3 * self.C_ell_beta
                         + self.gamma4 * self.C_n_beta)
        self.C_p_p = self.gamma3 * self.C_ell_p + self.gamma4 * self.C_n_p
        self.C_p_r = self.gamma3 * self.C_ell_r + self.gamma4 * self.C_n_r
        self.C_p_delta_a = (self.gamma3 * self.C_ell_delta_a
                            + self.gamma4 * self.C_n_delta_a)
        self.C_p_delta_r = (self.gamma3 * self.C_ell_delta_r
                            + self.gamma4 * self.C_n_delta_r)
        self.C_r_0 = self.gamma4 * self.C_ell_0 + self.gamma8 * self.C_n_0
        self.C_r_beta = (self.gamma4 * self.C_ell_beta
                         + self.gamma8 * self.C_n_beta)
        self.C_r_p = self.gamma4 * self.C_ell_p + self.gamma8 * self.C_n_p
        self.C_r_r = self.gamma4 * self.C_ell_r + self.gamma8 * self.C_n_r
        self.C_r_delta_a = (self.gamma4 * self.C_ell_delta_a
                            + self.gamma8 * self.C_n_delta_a)
        self.C_r_delta_r = (self.gamma4 * self.C_ell_delta_r
                            + self.gamma8 * self.C_n_delta_r)
