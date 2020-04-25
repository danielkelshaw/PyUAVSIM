# PyUAVSIM: Flight Dynamics

<img src="media/trimmed_state.gif" />

## **Features:**
PyUAVSIM provides an extensible 6DOF flight dynamics simulator for fixed-wing aircraft.

**Trim State Calculation:**
<br>Aircraft can be initialised in a trimmed state through the use of gradient-based optimisation.

**Control Surface Manipulation:**
<br>The state of the aircraft can be altered through control surface manipulation.

**Sensor Suite Simuation:**
<br>The simulation provides support for simulation of various onboard sensors, including IMU.

**Sensor Fusion:**
<br>Extended Kalman Filters can be used in conjunction with simulated sensor data to provide state estimation.

**PID Control / Autopilot:**
<br>Autopilot can be used to achieve a commanded state - this is facilitated through the use of PID control.

## **Requirements:**
- [ ] NumPy
- [ ] SciPy
- [ ] PyOpenGL
- [ ] pyqtgraph
- [ ] yaml
