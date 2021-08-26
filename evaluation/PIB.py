import numpy as np
from tqdm import tqdm
import yasa

class PIB:
    def __init__(self, fs, sliding_window=35, bands=None, band_names=['Delta', 'Theta', 'Alpha', 'Beta', 'Low Gamma', 'High Gamma'], fft_window='hann', fft_window_duration=20):
        self.fs = fs
        self.sliding_window = sliding_window
        self.bands = bands
        self.window = fft_window
        self.fft_window = fft_window_duration
        self.band_names = band_names
        
    def spin(self, sig=None) -> np.ndarray:
        N = self.fs*self.sliding_window             # fs = N*fp (N must be a natural number)
        fp = self.fs/N                              # prediction frequency
        n = sig.shape[1]                            # Total number of input samples
#         print('N:', N)
#         print('Prediction time period (s):', 1/fp)
#         print('Prediction freq (Hz):', fp)
        bandpower_list = []
        Ik = N                                      # Window width
        for _k in tqdm(range(Ik, n)):               # Sliding window
            if (_k % N == 0):                       # Decimation policy: _k occurs once every N samples
                w_start = _k - Ik                   # Starting index of sliding window (end index is maintained by _k)
                x_t = sig[:,w_start:_k]
                df = yasa.bandpower(x_t, sf=self.fs, win_sec=self.fft_window, bands=self.bands, bandpass=True, relative=True, kwargs_welch={'window': self.window})
                df = df[self.band_names]
                bandpower_list.append(df.to_numpy())
        return np.vstack(bandpower_list)