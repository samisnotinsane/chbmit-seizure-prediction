import numpy as np
from tqdm import tqdm

class ARMA:
    def __init__(self, window_width, order, memory, seed):
        self.N = window_width
        self.n_i = order
        self.m = memory
        self.seed = seed

    def spin(self, sig=None, fs=None, a_scale=1.8):
        np.random.seed(self.seed)

        fp = fs / self.N                                           # Prediction frequency
        t_s = 1 / fs                                               # Input signal time period

        a_k_list = []                                              # Sequential buffer for  MA samples
        a_h_k_list = []                                            # Sequential buffer for AR signal samples
        k_list = []                                                # Sequential buffer for time index in prediction point _k

        n_c = sig.shape[0]                                         # Number of input channels
        n = sig.shape[1]                                           # Total number of input samples
        a = a_scale * np.random.randn(n_c, self.n_i)               # Initialise AR coefficients
        c = self.m * np.ones(self.m)                               # Initialise MA coefficients
        c = c / c.sum()

        Ik = self.N                                                # Window width
        for _k in tqdm(range(Ik + self.n_i, n)):                         # Sliding window
            if (_k % self.N == 0):                                 # Decimation policy: _k occurs once every N samples
                w_start = _k - Ik - self.n_i + 1                   # Starting index of sliding window (end index is maintained by _k)
                a_h = np.zeros((n_c, self.n_i))                    # AR parameter estimates from samples in window
                for _i in range(n_c):                              # Iterate channels
                    x_t = sig[_i, w_start:_k]                       # Multi-channel window over input signal
                    N_w = len(x_t)
                    ymat = np.zeros((N_w - self.n_i, self.n_i))
                    yb = np.zeros((N_w - self.n_i, self.n_i))
                    for _c in range(self.n_i, 0, -1):                   # Past dependency of AR up to model order
                        ymat[ : , self.n_i - _c] = x_t[self.n_i - _c : -_c]
                    yb = x_t[self.n_i:]
                    a_h[_i] = np.linalg.pinv(ymat) @ yb            # Least squares solution to optimal parameters via Moore-Penrose Pseudo Inverse
                a_k = np.zeros((n_c, self.n_i))
                a_h_k_idx = len(a_h_k_list) - 1                    # Index to most recent block of AR parameters of shape: (n_c, n_i)
                for _j in range(self.m):                                # MA smoothing of AR parameters going back m units of time, in timescale k
                    if len(a_h_k_list) > self.m:                        # Only begin smoothing once unit of time elapsed is greater than m
                        a_k = c[_j] * a_h_k_list[a_h_k_idx - _j]
                k_list.append(_k)
                a_h_k_list.append(a_h)
                a_k_list.append(a_k)
        k = np.array(k_list)
        a_h_k = np.array(a_h_k_list)
        a_k = np.array(a_k_list)
        return k, a_h_k, a_k # return prediction times, AR, MA
