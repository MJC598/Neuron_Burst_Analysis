import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from typing import Union, Sequence, Optional, Tuple, List, Dict

from lfp_prediction.data_gathering import MatlabCollector, TextCollector, NumpyCollector, MultifileCollector


class LFPData(Dataset):
    def __init__(
            self,
            data_path: str,
            split: str,
            filter_type: str = "non-causal",
            frequency_band: Union[np.ndarray, Tuple[int, int], List[int]] = None,
            filter_rate: int = 5,
            column: int = None,
            only_bursts: bool = False
    ):
        if data_path.endswith(".npy") or data_path.endswith(".npz"):
            dc = NumpyCollector(data_path)
        elif data_path.endswith(".mat"):
            dc = MatlabCollector(data_path)
        elif data_path.endswith(".txt"):
            dc = TextCollector(data_path)
        else:
            raise ValueError("File does not have a .npy, .npz, .mat, or .txt extension")
        dc.get_data(column=column)

        self.norm_factor = None if not dc.norm_factor.any() else torch.Tensor(dc.norm_factor)

        data, labels = dc.filter_data(filter_type=filter_type,
                                      freq_band=frequency_band,
                                      filter_rate=filter_rate,
                                      only_bursts=only_bursts)
        split_idx = int(.75 * data.shape[0])
        if split == "train":
            self.data = torch.tensor(data[:split_idx], dtype=torch.float)
            self.labels = torch.tensor(labels[:split_idx], dtype=torch.float)
        elif split == "validation":
            self.data = torch.tensor(data[split_idx:], dtype=torch.float)
            self.labels = torch.tensor(labels[split_idx:], dtype=torch.float)
        else:
            self.data = torch.tensor(data, dtype=torch.float)
            self.labels = torch.tensor(labels, dtype=torch.float)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx, :, :], self.labels[idx, :, :]


class LFPDataset(LightningDataModule):
    def __init__(
            self,
            data_path: str,
            train_batch_size: int = 8,
            val_batch_size: int = 8,
            patch_size: Union[int, Sequence[int]] = 300,
            num_workers: int = 16,
            pin_memory: bool = False,
            filter_type: str = "non-causal",
            filter_rate: int = 5,
            column: int = None,
            only_bursts: bool = False,
            **kwargs
    ) -> None:
        super().__init__()
        self.train_dataset = None
        self.val_dataset = None
        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.filter_type = filter_type
        self.filter_rate = filter_rate
        self.column = column
        self.only_bursts = only_bursts
        self.norm_factor = None

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = LFPData(self.data_dir, split='train', filter_type=self.filter_type,
                                     frequency_band=(55, 85), filter_rate=self.filter_rate,
                                     column=self.column, only_bursts=self.only_bursts)

        self.val_dataset = LFPData(self.data_dir, split='validation', filter_type=self.filter_type,
                                   frequency_band=(55, 85), filter_rate=self.filter_rate,
                                   column=self.column, only_bursts=self.only_bursts)

        self.norm_factor = self.train_dataset.norm_factor

    def get_norm_factor(self):
        return self.norm_factor

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory
        )


class LFPITNData(Dataset):
    def __init__(
            self,
            data_paths: List[str],
            split: str,
            filter_type: str = "non-causal",
            frequency_band: Union[np.ndarray, Tuple[int, int], List[int]] = None,
            filter_rate: int = 5,
            column: Dict[str, int] = None,
            only_bursts: bool = False
    ):
        dc = MultifileCollector(data_paths)
        dc.get_data(column=column)

        self.norm_factor = dc.norm_factor

        data, labels = dc.filter_data(filter_type=filter_type,
                                      freq_band=frequency_band,
                                      filter_rate=filter_rate,
                                      only_bursts=only_bursts)
        split_idx = int(.75 * data.shape[0])
        if split == "train":
            self.data = torch.tensor(data[:split_idx], dtype=torch.float)
            self.labels = torch.tensor(labels[:split_idx], dtype=torch.float)
        elif split == "validation":
            self.data = torch.tensor(data[split_idx:], dtype=torch.float)
            self.labels = torch.tensor(labels[split_idx:], dtype=torch.float)
        else:
            self.data = torch.tensor(data, dtype=torch.float)
            self.labels = torch.tensor(labels, dtype=torch.float)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx, :, :], self.labels[idx, :, :]


class MultiFeatureDataset(LightningDataModule):
    def __init__(
            self,
            data_paths: List[str],
            train_batch_size: int = 8,
            val_batch_size: int = 8,
            patch_size: Union[int, Sequence[int]] = 300,
            num_workers: int = 16,
            pin_memory: bool = False,
            filter_type: str = "non-causal",
            filter_rate: int = 5,
            column: Dict[str, int] = None,
            only_bursts: bool = False,
            **kwargs
    ) -> None:
        super().__init__()
        self.train_dataset = None
        self.val_dataset = None
        self.data_paths = data_paths
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.filter_type = filter_type
        self.filter_rate = filter_rate
        self.column = column
        self.only_bursts = only_bursts
        self.norm_factor = None

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = LFPITNData(self.data_paths, split='train', filter_type=self.filter_type,
                                        frequency_band=(55, 85), filter_rate=self.filter_rate,
                                        column=self.column, only_bursts=self.only_bursts)

        self.val_dataset = LFPITNData(self.data_paths, split='validation', filter_type=self.filter_type,
                                      frequency_band=(55, 85), filter_rate=self.filter_rate,
                                      column=self.column, only_bursts=self.only_bursts)

        self.norm_factor = self.train_dataset.norm_factor

    def get_norm_factor(self):
        return self.norm_factor

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory
        )