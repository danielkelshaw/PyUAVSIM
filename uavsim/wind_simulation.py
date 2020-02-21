import numpy as np
from .utility.transfer_function import TransferFunction


class WindSimulator:

    def __init__(self, ts, gust_flag=True):

        self.ts = ts
        self.gust_flag = gust_flag
        self.steady_state = np.array([[0.0, 0.0, 0.0]]).T

        va = 25

        lu = 200.0
        lv = 200.0
        lw = 50.0

        if self.gust_flag:
            sigma_u = 1.06
            sigma_v = 1.06
            sigma_w = 0.7
        else:
            sigma_u = 0
            sigma_v = 0
            sigma_w = 0

        a1 = sigma_u * np.sqrt(2.0 * va / lu)
        a2 = sigma_v * np.sqrt(3.0 * va / lv)
        a3 = a2 * va / np.sqrt(3.0) / lv
        a4 = sigma_w * np.sqrt(3.0 * va / lw)
        a5 = a4 * va / np.sqrt(3.0) / lw
        b1 = va / lu
        b2 = va / lv
        b3 = va / lw

        self.u_w = TransferFunction(num=np.array([[a1]]),
                                    den=np.array([[1, b1]]),
                                    ts=ts)

        self.v_w = TransferFunction(num=np.array([[a2, a3]]),
                                    den=np.array([[1, 2 * b2, b2 ** 2.0]]),
                                    ts=ts)

        self.w_w = TransferFunction(num=np.array([[a4, a5]]),
                                    den=np.array([[1, 2 * b3, b3 ** 2.0]]),
                                    ts=ts)

    def update(self):

        gust = np.array([[self.u_w.update(np.random.randn())],
                         [self.v_w.update(np.random.randn())],
                         [self.w_w.update(np.random.randn())]])

        return np.concatenate((self.steady_state, gust))
