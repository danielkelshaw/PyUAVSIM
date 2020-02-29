import uavsim
import numpy as np
from uavsim.utility.trim import compute_trim
from uavsim.utility.signals import Signal
from uavsim.messages.msg_autopilot import MsgAutopilot


sim_timestep = 0.01

uav = uavsim.UAVParams('params.yaml')
uav_dynamics = uavsim.UAVDynamics(uav, sim_timestep)
wind_sim = uavsim.WindSimulator(sim_timestep)

# TODO:>> Place trim calculations within UAV instantiation
trim_state, trim_delta = compute_trim(uav_dynamics, 25, 0)
uav_dynamics.state = trim_state
delta = trim_delta
saved_trim = delta.copy()

control_params = uavsim.ControlParams(uav, uav_dynamics, trim_state,
                                      trim_delta, sim_timestep)

autopilot = uavsim.Autopilot(control_params, sim_timestep)

uav_viewer = uavsim.UAVViewer()

commands = MsgAutopilot()
va_command = Signal(offset=25.0, amplitude=3.0, start_time=2.0, frequency=0.01)
h_command = Signal(offset=100.0, amplitude=10.0, start_time=0.0, frequency=0.05)
chi_command = Signal(offset=np.radians(180), amplitude=np.radians(45),
                     start_time=5.0, frequency=0.02)

sim_time = 0.0
n_steps = int(30 / sim_timestep)

print('Starting Simulation...')

n = 0
while n < (n_steps + 1):

    commands.airspeed_command = va_command.square(sim_time)
    commands.course_command = chi_command.square(sim_time)
    commands.altitude_command = h_command.square(sim_time)

    measurements = uav_dynamics.get_sensors()
    estimated_state = uav_dynamics.true_state
    delta, commanded_state = autopilot.update(commands, estimated_state)

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
