import numpy as np


class Signal:

    def __init__(self,
                 amplitude=1.0,
                 frequeny=1.0,
                 start_time=1.0,
                 duration=0.01):

        self.amplitude = amplitude
        self.frequency = frequeny
        self.period = 1 / frequeny
        self.start_time = start_time
        self.duration = duration

        self.last_time = start_time

    def step(self, time):

        if time >= self.start_time:
            y = self.amplitude
        else:
            y = 0.0
        return y

    def sinusoid(self, time):

        if time >= self.start_time:
            y = self.amplitude * np.sin(self.frequency * time)
        else:
            y = 0.0
        return y

    def square(self, time):

        if time < self.start_time:
            y = 0.0
        elif time < self.last_time + self.period / 2.0:
            y = self.amplitude
        else:
            y = -self.amplitude
        if time >= self.last_time + self.period:
            self.last_time = time
        return y

    def impulse(self, time):

        if ((time >= self.start_time)
                and (time <= self.start_time + self.duration)):
            y = self.amplitude
        else:
            y = 0.0
        return y
