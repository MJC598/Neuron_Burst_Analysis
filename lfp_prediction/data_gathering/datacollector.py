from abc import ABC, abstractmethod
from typing import Union, Tuple, List
import numpy as np
from scipy import signal
from scipy.signal import hilbert, butter, filtfilt


class DataCollector(ABC):
    def __init__(self, datapath: str = None):
        if not datapath:
            raise ValueError('No Datapath Specified')
        self.datapath = datapath
        self.data = None
        self.threshold = None
        self.global_mean = None
        self.global_std = None

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
        lfp = filtfilt(z, a, data, axis=0)
        lfp_amplitude = abs(hilbert(lfp, axis=0))
        self.threshold = np.mean(lfp_amplitude) + (scalar * np.std(lfp_amplitude))
        return -self.threshold, self.threshold

    def _oversample(self, data):
        z, a = signal.butter(4, np.array([55, 85]), btype='bandpass', output='ba', fs=1000)
        lfp = signal.filtfilt(z, a, data, axis=0)
        burst_samples = np.any(abs(lfp) > self.threshold, axis=0)
        return sum(burst_samples)
