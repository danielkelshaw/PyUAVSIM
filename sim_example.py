import uavsim
import numpy as np
from uavsim.utility.trim import compute_trim
from uavsim.path_follower import PathFollower
from uavsim.messages.msg_path import MsgPath


sim_timestep = 0.01

uav_params = uavsim.UAVParams('params.yaml')
sensor_params = uavsim.SensorParams('sensor_params.yaml')

uav_dynamics = uavsim.UAVDynamics(uav_params, sensor_params, sim_timestep)
wind_sim = uavsim.WindSimulator(sim_timestep)

trim_state, trim_delta = compute_trim(uav_dynamics, 25, 0)
uav_dynamics.state = trim_state
delta = trim_delta

control_params = uavsim.ControlParams(uav_params, uav_dynamics, trim_state,
                                      trim_delta, sim_timestep)

autopilot = uavsim.Autopilot(control_params, sim_timestep)
observer = uavsim.Observer(uav_params, sensor_params, sim_timestep)
path_follower = PathFollower()

uav_viewer = uavsim.UAVViewer()
path_viewer = uavsim.PathViewer()

path = MsgPath()
path.type = 'orbit'

path.type = 'orbit'
if path.type == 'line':
    path.line_origin = np.array([[0.0, 0.0, -100.0]]).T
    path.line_direction = np.array([[0.5, 1.0, 0.0]]).T
    path.line_direction = path.line_direction \
                          / np.linalg.norm(path.line_direction)
elif path.type == 'orbit':
    path.orbit_centre = np.array([[0.0, 0.0, 100.0]]).T
    path.orbit_radius = 300.0
    path.orbit_direction = 'CW'

sim_time = 0.0
n_steps = int(60 / sim_timestep)

print('Starting Simulation...')

n = 0
while n < (n_steps + 1):

    measurements = uav_dynamics.get_sensors()
    estimated_state = observer.update(measurements)

    commands = path_follower.update(path, estimated_state)
    delta, commanded_state = autopilot.update(commands, estimated_state)

    wind = wind_sim.update()
    uav_dynamics.update(delta, wind)

    path_viewer.update(uav_dynamics.true_state, path)

    if n % 100 == 0:
        print(f't = {sim_time:.3f}')

    sim_time += sim_timestep
    n += 1

print('Simulation Done...')
