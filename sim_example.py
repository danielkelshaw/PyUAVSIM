import uavsim
import numpy as np


sim_timestep = 0.01
uav_dynamics = uavsim.UAVDynamics(sim_timestep)
wind_sim = uavsim.WindSimulator(sim_timestep)
plotter = uavsim.Plotter(title='Example', initial_state=uav_dynamics.state)

sim_time = 0.0
n_steps = int(15 / sim_timestep)

print('Starting Simulation...')
uav_dynamics.state[3] = 50

n = 0
while n < (n_steps + 1):

    delta_e = 0.0
    delta_a = 0.0
    delta_r = 0.0
    delta_t = 0.0

    delta = np.array([[delta_e, delta_a, delta_r, delta_t]]).T

    # wind = wind_sim.update()
    wind = np.zeros((6, 1))
    uav_dynamics.update(delta, wind)
    plotter.update(uav_dynamics.state, sim_time)

    if n % 100 == 0:
        print('t = {:.3f}\t'.format(sim_time)
              + '\t'.join(['{:.3f}'.format(i)
                           for i in uav_dynamics.state.squeeze()]))

    sim_time += sim_timestep
    n += 1

print('Simulation Done...')

plotter.plot_vel()
plotter.plot_quarternions()
plotter.plot_pqr()