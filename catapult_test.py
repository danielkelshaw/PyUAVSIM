import uavsim
import numpy as np

sim_timestep = 0.01

uav_params = uavsim.UAVParams('catapult_params.yaml')
sensor_params = uavsim.SensorParams('sensor_params.yaml')

cat = uavsim.Catapult(uav_params, sensor_params, sim_timestep)

uav_viewer = uavsim.UAVViewer()

sim_time = 0.0
n_steps = int(30 / sim_timestep)

print('Starting Simulation...')

print(cat.fm_final)

print(np.concatenate((cat.fm_final, np.array([[0], [0], [0]]))))

n = 0
while n < (n_steps + 1):

    if n < 20:
        cat.launch()
    else:
        cat.update()

    cat._update_true_state()
    uav_viewer.update(cat.true_state)

    n += 1
