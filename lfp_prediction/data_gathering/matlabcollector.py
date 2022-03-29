from typing import Union, Tuple, List
import numpy as np
from scipy import signal, io
import pandas as pd

from datacollector import DataCollector
from lfp_prediction.config import params


class MatlabCollector(DataCollector):
    def __init__(self, path: str = None):
        super().__init__(path)
        self.global_mean = None
        self.global_std = None

    def filter_data(self,
                    data: pd.DataFrame,
                    filter_type: str = None,
                    freq_band: Union[np.ndarray, Tuple[int, int], List[int]] = None,
                    filter_rate: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter the data provided
        :param data: pandas DataFrame containing the raw LFP segments with each segment as
                     a row and each timestep as a column
        :param filter_type: String describing the type of filtering required.
                            Expects ['non-causal', 'causal', 'decomposition', 'raw']
        :param freq_band: Frequency range to filter, only uses 2 values, a lower and higher limit
        :param filter_rate: The rate at which to jump to the next value in the segment.
                            Higher values means less samples lower values means more samples
        :return: Tuple of numpy arrays containing the inputs and labels
        """

        data = data.dropna(axis=0, subset=[str(params.PREVIOUS_TIME + params.LOOK_AHEAD)])
        labels = data.copy()

        data = data.apply(lambda x: (x - self.global_mean) / self.global_std)  # Normalize Raw Data
        z, a = signal.butter(4, freq_band, btype='bandpass', output='ba', fs=1000)

        if filter == 'non-causal':
            labels = labels.apply(lambda x: signal.filtfilt(z, a, x))
        elif filter == 'causal':
            labels = labels.apply(lambda x: signal.lfilter(z, a, x))
        labels = labels.apply(lambda x: (x - self.global_mean) / self.global_std)  # Normalize Labels

        t = params.PREVIOUS_TIME
        k = params.LOOK_AHEAD

        inputs = []
        outputs = []
        for column in range(len(data.columns)):
            if int(column) + t + k > len(data.columns):
                break
            data = data.dropna(axis=0, subset=[np.arange(column, column+t+k+1)])
            labels = labels.dropna(axis=0, subset=[np.arange(column, column+t+k+1)])
            inputs.extend(data[data[column:column+t]].to_numpy())
            if filter == 'decomposition':
                outputs.extend(labels.apply(lambda x: signal.lfilter(z, a, x[t:t+k])).to_numpy())
            outputs.extend(labels[t:t+k])
            column += filter_rate
            t += filter_rate

        inputs = np.transpose(np.stack(inputs, axis=0), (0, 2, 1))
        outputs = np.transpose(np.stack(outputs, axis=0), (0, 2, 1))
        print(inputs.shape)
        print(outputs.shape)
        return inputs, outputs

    def get_data(self) -> pd.DataFrame:
        """
        Retrieve the data from the path specified during instantiation of the class.
        This expects a .mat file with the key 'LFP_seg'
        :return: pandas DataFrame of the raw data with each segment as a row and each timestep as a column
        """
        try:
            mat = io.loadmat(self.datapath)['LFP_seg']
        except FileNotFoundError:
            print('File {} not found'.format(self.datapath))
            raise
        self.global_std = np.std(np.concatenate(np.concatenate(mat)))
        self.global_mean = np.mean(np.concatenate(np.concatenate(mat)))
        mat_dict = {i: np.concatenate(arr) for i, arr in enumerate(np.concatenate(mat))}
        df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in mat_dict.items()])).transpose()
        return df
