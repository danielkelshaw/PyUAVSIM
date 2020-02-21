import numpy as np


class TransferFunction:

    def __init__(self, num, den, ts):

        m = num.shape[1]
        n = den.shape[1]

        self.state = np.zeros((n - 1, 1))

        if den[0][0] != 1:
            den /= den[0][0]
            num /= den[0][0]

        self.num = num
        self.den = den

        self._A = np.eye(n - 1)
        self._B = np.zeros((n - 1, 1))
        self._C = np.zeros((1, n - 1))

        self._B[0][0] = ts

        if m == n:
            self._D = num[0][0]

            for i in range(0, m):
                self._C[0][n - i - 2] = (num[0][m - i - 1] -
                                         num[0][0] * den[0][n - i - 2])

            for i in range(0, n - 1):
                self._A[0][i] += - ts * den[0][i + 1]

            for i in range(1, n - 1):
                self._A[i][i - 1] += ts

        else:
            self._D = 0.0

            for i in range(0, m):
                self._C[0][n - i - 2] = num[0][m - i - 1]

            for i in range(0, n - 1):
                self._A[0][i] += - ts * den[0][i]

            for i in range(1, n - 1):
                self._A[i][i - 1] += ts

    def update(self, u):

        self.state = np.matmul(self._A, self.state) + (self._B * u)
        y = np.matmul(self._C, self.state) + self._D * u

        return y[0][0]
