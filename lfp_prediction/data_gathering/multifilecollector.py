from typing import Union, Tuple, List, Dict
import numpy as np
from scipy import signal
import re

from lfp_prediction.config import params
from lfp_prediction.data_gathering.datacollector import DataCollector, get_norm_factor


class MultifileCollector(DataCollector):
    def __init__(self, paths: List[str] = None):
        self.datapaths = paths
        super().__init__(paths[0])  # Assumes the first file path is the LFPs

    def filter_data(self,
                    filter_type: str = None,
                    freq_band: Union[np.ndarray, Tuple[int, int], List[int]] = None,
                    filter_rate: int = 400, only_bursts: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        z, a = signal.butter(4, freq_band, btype='bandpass', output='ba', fs=1000)
        inputs = []
        outputs = []
        burst_samples = 0
        i = 0
        t = params.PREVIOUS_TIME
        k = params.LOOK_AHEAD

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
            inputs.append((sample[i:t, :] - self.global_mean) / self.global_std)

            if filter_type == 'decomposition':
                outputs.append(
                    signal.lfilter(z, a, (lfp[t:t + k, :] - self.global_mean) / self.global_std, axis=0)
                )
            elif filter_type == 'raw':  # Raw Condition
                outputs.append((lfp[i + k:t + k, :] - self.global_mean) / self.global_std)
            else:  # Non Causal Full Filter and
                outputs.append(lfp[i+k:t+k, :])
            i += filter_rate  # params.PREVIOUS_TIME
            t += filter_rate  # params.PREVIOUS_TIME

        inputs = np.transpose(np.stack(inputs, axis=0), (0, 2, 1))
        outputs = np.transpose(np.stack(outputs, axis=0), (0, 2, 1))
        print(inputs.shape)
        print(outputs.shape)
        return inputs, outputs

    def get_data(self, threshold: int = 2, column: Dict[str, int] = None) -> np.ndarray:
        for dp in self.datapaths:
            with open(dp, 'r') as f:
                data = f.readlines()
            if dp in column.keys():
                data = [re.findall(r'\d+', d)[column[dp]-1] for d in data]
            if self.data is None:
                self.data = np.array(list(map(float, data))).reshape((-1, 1))
            else:
                data = np.array(list(map(float, data))).reshape((-1, 1))
                self.data = np.concatenate((self.data, data), axis=1)

        self.global_std = np.std(self.data, axis=0)
        self.global_mean = np.mean(self.data, axis=0)
        self.get_threshold(self.data[:, 0], scalar=threshold)
        self.norm_factor = get_norm_factor(self.data[:, 0], self.global_mean[0], self.global_std[0])
        return self.data
