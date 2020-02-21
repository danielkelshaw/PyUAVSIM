import uavsim
import numpy as np


sim_timestep = 0.01
uav_dynamics = uavsim.UAVDynamics(sim_timestep)
wind_sim = uavsim.WindSimulator(sim_timestep)

sim_time = 0.0
n_steps = int(15 / sim_timestep)

print('Starting Simulation...')

n = 0
while n < (n_steps + 1):

    delta_e = -0.2
    delta_a = 0.0
    delta_r = 0.0

    delta = np.array([[delta_e, delta_a, delta_r]]).T

    wind = wind_sim.update()
    uav_dynamics.update(delta, wind)

    if n % 100 == 0:
        print('t = {:.3f}\t'.format(sim_time)
              + '\t'.join(['{:.3f}'.format(i)
                           for i in uav_dynamics.state.squeeze()]))

    sim_time += sim_timestep
    n += 1

print('Simulation Done...')
