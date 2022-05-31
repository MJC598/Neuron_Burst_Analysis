from typing import Union, Tuple, List
import numpy as np
from scipy import signal, io

from .datacollector import DataCollector, get_norm_factor
from lfp_prediction.config import params


class MatlabCollector(DataCollector):
    def __init__(self, path: str = None):
        super().__init__(path)

    def filter_data(self,
                    filter_type: str = None,
                    freq_band: Union[np.ndarray, Tuple[int, int], List[int]] = None,
                    filter_rate: int = 400, only_bursts: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        z, a = signal.butter(4, freq_band, btype='bandpass', output='ba', fs=1000)
        inputs = []
        outputs = []
        burst_samples = 0
        i = 0
        t = params.PREVIOUS_TIME + 2
        k = params.LOOK_AHEAD + 2

        if self.data.shape[0] < self.data.shape[-1]:
            sample = self.data.T  # Currently, we assume any numpy array is a 1D time series set
        else:
            sample = self.data

        if filter_type == 'non-causal':
            lfp = signal.filtfilt(z, a, sample, axis=0)
        elif filter_type == 'causal':
            lfp = signal.lfilter(z, a, sample, axis=0)
        else:
            lfp = sample

        while lfp.shape[0] > t + k:
            if only_bursts and self._oversample(sample[i:t + k, :]) == 0:  # done to include only bursts
                i += filter_rate  # params.PREVIOUS_TIME
                t += filter_rate  # params.PREVIOUS_TIME
                continue
            burst_samples += self._oversample(sample[i:t + k, :])
            inputs.append(np.diff((sample[i:t, :] - self.global_mean) / self.global_std, n=2, axis=0))

            if filter_type == 'decomposition':
                outputs.append(
                    signal.lfilter(z,
                                   a,
                                   np.diff((lfp[t:t + k, :] - self.global_mean) / self.global_std,
                                           n=2,
                                           axis=0),
                                   axis=0)
                )
            elif filter_type == 'raw':  # Raw Condition
                # outputs.append((lfp[i + k:t + k, :] - self.global_mean) / self.global_std)
                outputs.append(np.diff((lfp[t:t + k, :] - self.global_mean) / self.global_std, n=2, axis=0))
            else:  # Non Causal Full Filter and
                outputs.append(np.diff(lfp[t:t+k, :], n=2, axis=0))
            i += filter_rate  # params.PREVIOUS_TIME
            t += filter_rate  # params.PREVIOUS_TIME

        inputs = np.transpose(np.stack(inputs, axis=0), (0, 2, 1))
        outputs = np.transpose(np.stack(outputs, axis=0), (0, 2, 1))
        # print(burst_samples)
        # print(inputs.shape)
        # print(outputs.shape)
        return inputs, outputs

    def get_data(self, threshold: int = 2, column: int = None) -> np.ndarray:
        """
        Retrieve the data from the path specified during instantiation of the class.
        This expects a .mat file with the key 'LFP_seg'
        :param column: Needless, applies to textcollector
        :param threshold: Number of zscores away to be considered a burst
        :return: numpy array (samples, ) where each sample is a (N,1)
        """
        try:
            mat = io.loadmat(self.datapath)['LFP_seg']
        except FileNotFoundError:
            print('File {} not found'.format(self.datapath))
            raise
        self.data = np.concatenate(mat)
        axis = 0 if self.data[0].shape[0] >= self.data[0].shape[-1] else -1
        self.global_std = np.std(np.concatenate(self.data, axis=axis))
        self.global_mean = np.mean(np.concatenate(self.data, axis=axis))
        self.get_threshold(scalar=2)
        self.data = np.concatenate(self.data, axis=axis)
        self.norm_factor = get_norm_factor(self.data, self.global_mean, self.global_std)
        return self.data

