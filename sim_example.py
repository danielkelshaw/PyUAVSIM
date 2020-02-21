import uavsim
import numpy as np


sim_timestep = 0.01
uav_dynamics = uavsim.UAVDynamics(sim_timestep)

sim_time = 0.0
n_steps = int(15 / sim_timestep)

print('Starting Simulation...')

n = 0
while n < (n_steps + 1):

    fx = 10
    fy = 0
    fz = 0
    Mx = 0
    My = 0
    Mz = 0

    forces_moments = np.array([[fx, fy, fz, Mx, My, Mz]]).T
    uav_dynamics.update(forces_moments)

    if n % 100 == 0:
        print('t = {:.3f}\t'.format(sim_time)
              + '\t'.join(['{:.3f}'.format(i)
                           for i in uav_dynamics.state.squeeze()]))

    sim_time += sim_timestep
    n += 1
