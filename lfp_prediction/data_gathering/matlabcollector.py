from typing import Union, Tuple, List
import numpy as np
from scipy import signal, io

from .datacollector import DataCollector
from lfp_prediction.config import params


class MatlabCollector(DataCollector):
    def __init__(self, path: str = None):
        super().__init__(path)

    def filter_data(self,
                    filter_type: str = None,
                    freq_band: Union[np.ndarray, Tuple[int, int], List[int]] = None,
                    filter_rate: int = 50, only_bursts: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter the data provided
        :param only_bursts: only allow for burst parameters
        :param filter_type: String describing the type of filtering required.
                            Expects ['non-causal', 'causal', 'decomposition', 'raw']
        :param freq_band: Frequency range to filter, only uses 2 values, a lower and higher limit
        :param filter_rate: The rate at which to jump to the next value in the segment.
                            Higher values means less samples lower values means more samples
        :return: Tuple of numpy arrays containing the inputs and labels
        """
        z, a = signal.butter(4, freq_band, btype='bandpass', output='ba', fs=1000)
        inputs = []
        outputs = []
        burst_samples = 0

        for sample in self.data:
            if sample.shape[0] < (params.PREVIOUS_TIME + params.LOOK_AHEAD):
                continue
            i = 0
            t = params.PREVIOUS_TIME
            k = params.LOOK_AHEAD

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
                    # y1.append(norm_lfp[i:t,:])
                    outputs.append((lfp[i + k:t + k, :] - self.global_mean) / self.global_std)
                i += filter_rate  # params.PREVIOUS_TIME
                t += filter_rate  # params.PREVIOUS_TIME

        inputs = np.transpose(np.stack(inputs, axis=0), (0, 2, 1))
        outputs = np.transpose(np.stack(outputs, axis=0), (0, 2, 1))
        print(burst_samples)
        print(inputs.shape)
        print(outputs.shape)
        return inputs, outputs

    def get_data(self, threshold: int = 2) -> np.ndarray:
        """
        Retrieve the data from the path specified during instantiation of the class.
        This expects a .mat file with the key 'LFP_seg'
        :param threshold: Number of zscores away to be considered a burst
        :return: numpy array (samples, ) where each sample is a (N,1)
        """
        try:
            mat = io.loadmat(self.datapath)['LFP_seg']
        except FileNotFoundError:
            print('File {} not found'.format(self.datapath))
            raise
        self.global_std = np.std(np.concatenate(np.concatenate(mat)))
        self.global_mean = np.mean(np.concatenate(np.concatenate(mat)))
        self.data = np.concatenate(mat)
        self.get_threshold(scalar=2)
        return self.data

