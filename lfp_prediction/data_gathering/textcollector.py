from typing import Union, Tuple, List
import numpy as np
from scipy import signal
import re

from lfp_prediction.config import params
from lfp_prediction.data_gathering.datacollector import DataCollector


class TextCollector(DataCollector):
    def __init__(self, path: str = None):
        super().__init__(path)

    def filter_data(self,
                    filter_type: str = None,
                    freq_band: Union[np.ndarray, Tuple[int, int], List[int]] = None,
                    filter_rate: int = 50, only_bursts: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        z, a = signal.butter(4, freq_band, btype='bandpass', output='ba', fs=1000)
        inputs = []
        outputs = []
        burst_samples = 0
        i = 0
        t = params.PREVIOUS_TIME
        k = params.LOOK_AHEAD

        sample = self.data  # Currently, we assume any numpy array is a 1D time series set

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
            inputs.append((sample[i:t, :] - self.global_mean) / self.global_std)
            # x.append(lfp[i:t,:])
            if filter_type == 'decomposition':
                outputs.append(
                    signal.lfilter(z, a, (lfp[t:t + k, :] - self.global_mean) / self.global_std, axis=0)
                )
            else:  # Non Causal Full Filter and Raw Condition
                # y1.append(norm_lfp[t:t+k,:])
                # outputs.append((lfp[i + k:t + k, :] - self.global_mean) / self.global_std)
                outputs.append((lfp[i + k:t + k, :] - self.global_mean) / self.global_std)
            i += filter_rate  # params.PREVIOUS_TIME
            t += filter_rate  # params.PREVIOUS_TIME

        inputs = np.transpose(np.stack(inputs, axis=0), (0, 2, 1))
        outputs = np.transpose(np.stack(outputs, axis=0), (0, 2, 1))
        # print(burst_samples)
        # print(inputs.shape)
        # print(outputs.shape)
        return inputs, outputs

    def get_data(self, threshold: int = 2, column: int = None) -> np.ndarray:
        with open(self.datapath, 'r') as f:
            data = f.readlines()
        if column:
            data = [re.findall(r'\d+', d)[column-1] for d in data]
            self.data = np.array(list(map(float, data))).reshape((-1, 1))
        else:
            self.data = np.array(list(map(float, data))).reshape((-1, 1))

        self.global_std = np.std(self.data)
        self.global_mean = np.mean(self.data)
        self.get_threshold(self.data, scalar=threshold)
        return self.data
