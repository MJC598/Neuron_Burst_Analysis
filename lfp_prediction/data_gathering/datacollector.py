from abc import ABC, abstractmethod
from typing import Union, Tuple, List
import numpy as np
from scipy import signal
from scipy.signal import hilbert, butter, filtfilt
from scipy.stats import stats


def get_norm_factor(full_lfp: np.ndarray,
                    mu: np.ndarray,
                    std: np.ndarray,
                    hf_cutoff: float = 200.,
                    nfft: int = 256,
                    sample_rate: float = 1000.) -> np.ndarray:
    lfp = (full_lfp - mu) / std
    axis = 0 if lfp.shape[0] >= lfp.shape[-1] else -1
    f, Pxx = signal.welch(lfp, fs=sample_rate, window='hamming', nperseg=nfft, scaling='spectrum', axis=axis)
    f = f.ravel()
    Pxx = Pxx.ravel()

    f_cutoff = max(np.argmax(Pxx), np.ndarray([1]))
    Pmax = Pxx[f_cutoff]
    idx = np.arange(f_cutoff, f.size)
    result = stats.linregress(np.log(f[idx]), np.log(Pxx[idx]))
    b = result.intercept
    a = -result.slope

    f_cutoff = np.exp((b - np.log(Pmax)) / a)
    idx = f > f_cutoff
    Pfit = Pxx.copy()
    Pfit[idx] = np.exp(b) / f[idx] ** a
    Pfit[~idx] = Pmax
    idx = f > hf_cutoff
    Pfit[idx] = np.min(Pfit[~idx])
    norm_factor = np.square(Pmax / Pfit)
    return norm_factor


class DataCollector(ABC):
    def __init__(self, datapath: str = None):
        if not datapath:
            raise ValueError('No Datapath Specified')
        self.datapath = datapath
        self.data = None
        self.threshold = None
        self.global_mean = None
        self.global_std = None
        self.norm_factor = None

    @abstractmethod
    def filter_data(self,
                    filter_type: str = None,
                    freq_band: Union[np.ndarray, Tuple[int, int], List[int]] = None):
        pass

    @abstractmethod
    def get_data(self):
        pass

    def get_threshold(self, data: np.ndarray = None, scalar: int = 1) -> Tuple[float, float]:
        if data is None:
            data = self.data[0]
        z, a = butter(4, np.array([55, 85]), btype='bandpass', output='ba', fs=1000)
        axis = 0 if data.shape[0] >= data.shape[-1] else -1
        lfp = filtfilt(z, a, data, axis=axis)
        lfp_amplitude = abs(hilbert(lfp, axis=axis))
        self.threshold = np.mean(lfp_amplitude) + (scalar * np.std(lfp_amplitude))
        return -self.threshold, self.threshold

    def _oversample(self, data):
        z, a = signal.butter(4, np.array([55, 85]), btype='bandpass', output='ba', fs=1000)
        lfp = signal.filtfilt(z, a, data, axis=0)
        burst_samples = np.any(abs(lfp) > self.threshold, axis=0)
        return sum(burst_samples)
