import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Plotter:

    def __init__(self, title, initial_state):

        self.state = initial_state
        self.time = np.array([0])

    def update(self, state, sim_time):
        self.state = np.append(self.state, state, axis=1)
        self.time = np.append(self.time, sim_time)

    def plot_vel(self):

        fig, axs = plt.subplots(3, 1, figsize=(12, 10))

        axs[0].plot(self.time, self.state[3, :])
        axs[1].plot(self.time, self.state[4, :])
        axs[2].plot(self.time, self.state[5, :])

        plt.show()

    def plot_quarternions(self):

        fig, axs = plt.subplots(4, 1, figsize=(16, 10))

        axs[0].plot(self.time, self.state[6, :])
        axs[1].plot(self.time, self.state[7, :])
        axs[2].plot(self.time, self.state[8, :])
        axs[3].plot(self.time, self.state[9, :])

        plt.show()

    def plot_pqr(self):

        fig, axs = plt.subplots(3, 1, figsize=(12, 10))

        axs[0].plot(self.time, self.state[10, :])
        axs[1].plot(self.time, self.state[11, :])
        axs[2].plot(self.time, self.state[12, :])

        plt.show()
