import uavsim
import numpy as np
from uavsim.utility.trim import compute_trim


sim_timestep = 0.01

uav = uavsim.UAVParams('params.yaml')

uav_dynamics = uavsim.UAVDynamics(uav, sim_timestep)
wind_sim = uavsim.WindSimulator(sim_timestep)

trim_state, trim_delta = compute_trim(uav_dynamics, 25, 0)
uav_dynamics.state = trim_state
delta = trim_delta

uav_viewer = uavsim.UAVViewer()

sim_time = 0.0
n_steps = int(30 / sim_timestep)

print('Starting Simulation...')

n = 0
while n < (n_steps + 1):

    # delta_e = -0.2
    # delta_a = 0.0
    # delta_r = 0.005
    # delta_t = 0.5
    #
    # delta = np.array([[delta_e, delta_a, delta_r, delta_t]]).T

    wind = wind_sim.update()
    uav_dynamics.update(delta, wind)

    uav_viewer.update(uav_dynamics.true_state)

    if n % 100 == 0:
        print('t = {:.3f}\t'.format(sim_time)
              + '\t'.join(['{:.3f}'.format(i)
                           for i in uav_dynamics.state.squeeze()]))

    sim_time += sim_timestep
    n += 1

print('Simulation Done...')
