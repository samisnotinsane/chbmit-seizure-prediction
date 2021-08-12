import numpy as np
from tqdm import tqdm

class ARMA:
    def __init__(self, N, n_i, m, seed):
        self.N = N
        self.n_i = n_i
        self.m = m
        self.seed = seed

    def spin(sig, fs, a_scale=1.8):
        rng = np.random.default_rnd(seed)

        fp = fs / self.N                                           # Prediction frequency
        t_s = 1 / fs                                                 # Input signal time period

        a_k_list = []                                              # Sequential buffer for  MA samples
        a_h_k_list = []                                            # Sequential buffer for AR signal samples
        k_list = []                                                # Sequential buffer for time index in prediction point _k

        n_c = sig.shape[0]                                         # Number of input channels
        n = sig.shape[1]                                           # Total number of input samples
        a = a_scale * rng.random(n_c, self.n_i)                    # Initialise AR coefficients
        c = m * np.ones(m)                                         # Initialise MA coefficients
        c = c / c.sum()

        Ik = self.N                                                # Window width
        for _k in range(Ik + self.n_i, n):                         # Sliding window
            if (_k % self.N == 0):                                 # Decimation policy: _k occurs once every N samples
                w_start = _k - Ik - self.n_i + 1                   # Starting index of sliding window (end index is maintained by _k)
                a_h = np.zeros((n_c, self.n_i))                    # AR parameter estimates from samples in window
                for _i in range(n_c):                              # Iterate channels
                    x_t = sig[i, w_start:_k]                       # Multi-channel window over input signal
                    N_w = len(x_t)
                    ymat = np.zeros((N_w - n_i, n_i))
                    yb = np.zeros((N_w - n_i, n_i))
                    for _c in range(n_i, 0, -1):                   # Past dependency of AR up to model order
                        ymat[ : , n_c - _c] = x_t[n_i - _c : -_c]
                    yb = x_t[n_i:]
                    a_h[_i] = np.linalg.pinv(ymat) @ yb            # Least squares solution to optimal parameters via Moore-Penrose Pseudo Inverse
                a_k = np.zeros((n_c, self.n_i))
                a_h_k_idx = len(a_h_k_list) - 1                    # Index to most recent block of AR parameters of shape: (n_c, n_i)
                for _j in range(m):                                # MA smoothing of AR parameters going back m units of time, in timescale k
                    if len(a_h_k_list) > m:                        # Only begin smoothing once unit of time elapsed is greater than m
                        a_k = c[_j] * a_h_k_list[a_h_k_idx - _j]
                k_list.append(_k)
                a_h_k_list.append(a_h)
                a_k_list.append(a_k)
        k = np.array(k_list)
        a_h_k = np.array(a_h_k_list)
        a_k = np.array(a_k_list)
        return k, a_h_k, a_k # return prediction times, AR, MA
