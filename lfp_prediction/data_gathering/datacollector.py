from abc import ABC, abstractmethod
from typing import Union, Tuple, List
import numpy as np
from pandas import DataFrame


class DataCollector(ABC):
    def __init__(self, datapath: str = None):
        if not datapath:
            raise ValueError('No Datapath Specified')
        self.datapath = datapath

    @abstractmethod
    def filter_data(self,
                    data: DataFrame,
                    filter_type: str = None,
                    freq_band: Union[np.ndarray, Tuple[int, int], List[int]] = None):
        pass

    @abstractmethod
    def get_data(self):
        pass
