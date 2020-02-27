import numpy as np


class BasePID:

    def __init__(self, kp, ki, kd, ts, limit):

        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.ts = ts
        self.limit = limit

        self.e_delay = 0.0
        self.integrator = 0.0

    def update(self, *args):
        raise NotImplementedError('BasePID::update()')

    def _saturate(self, u):

        if u >= self.limit:
            u_sat = self.limit
        elif u <= -self.limit:
            u_sat = -self.limit
        else:
            u_sat = u
        return u_sat


class PIDControl(BasePID):

    def __init__(self, kp=0.0, ki=0.0, kd=0.0, ts=0.01, sigma=0.05, limit=1.0):

        super().__init__(kp=kp, ki=ki, kd=kd, ts=ts, limit=limit)

        self.edot_delay = 0.0

        self.a1 = (2.0 * sigma - ts) / (2.0 * sigma + ts)
        self.a2 = 2.0 / (2.0 * sigma + ts)

    def update(self, y_ref, y):

        error = y_ref - y

        self.integrator += (self.ts / 2) * (error + self.e_delay)
        edot = self.a1 + self.edot_delay + self.a2 * (error - self.e_delay)

        u = (self.kp * error
             + self.ki * self.integrator
             + self.kd * edot)

        u_sat = self._saturate(u)

        if np.abs(self.ki) > 0.0001:
            self.integrator += (self.ts / self.ki) * (u_sat - u)

        self.e_delay = error

        return u_sat


class PIControl(BasePID):

    def __init__(self, kp=0.0, ki=0.0, ts=0.01, limit=1.0):

        super().__init__(kp=kp, ki=ki, kd=0, ts=ts, limit=limit)

    def update(self, y_ref, y):

        error = y_ref - y

        self.integrator += (self.ts / 2) * (error - self.e_delay)

        u = self.kp * error + self.ki * self.integrator
        u_sat = self._saturate(u)

        if np.abs(self.ki) > 0.0001:
            self.integrator += (self.ts / self.ki) * (u_sat - u)

        self.e_delay = error

        return u_sat


class PDControlRate(BasePID):

    def __init__(self, kp=0.0, kd=0.0, limit=1.0):

        super().__init__(kp=kp, ki=0, kd=kd, ts=0, limit=limit)

    def update(self, y_ref, y, ydot):

        error = y_ref - y
        u = self.kp * error - self.kd * ydot
        u_sat = self._saturate(u)

        return u_sat
