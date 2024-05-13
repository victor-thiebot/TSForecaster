import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from tsforecaster.data.timeseries import TimeSeriesDataset


class DataModule:
    def __init__(
        self,
        train_dataset: TimeSeriesDataset,
        val_dataset: TimeSeriesDataset,
        test_dataset: TimeSeriesDataset,
        batch_size: int,
        num_workers: int = 0,
    ):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __repr__(self):
        return (
            f"DataModule(train_dataset={self.train_dataset}, "
            f"val_dataset={self.val_dataset}, test_dataset={self.test_dataset}, "
            f"batch_size={self.batch_size}, num_workers={self.num_workers})"
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class TimeSeriesDatasetTorch(Dataset):
    def __init__(self, dataset: TimeSeriesDataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x, y = self.dataset[idx]
        # x_values = x.values
        # y_values = y.values

        # if not isinstance(x_values, np.ndarray):
        #     raise ValueError(
        #         f"TimeSeries values must be a numpy array, but got {type(x_values)}"
        #     )

        # # Add channel dimension to the input data
        # x_values = x_values.reshape(1, -1)

        # return torch.from_numpy(x_values).float(), torch.from_numpy(y_values).float()
        x_values = x.values.reshape(self.dataset.n_channels_x, -1)
        y_values = y.values.reshape(self.dataset.n_channels_y, -1)

        return torch.from_numpy(x_values).float(), torch.from_numpy(y_values).float()
