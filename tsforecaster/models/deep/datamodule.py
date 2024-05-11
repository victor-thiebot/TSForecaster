import torch
from torch.utils.data import Dataset, DataLoader

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

        x = self.dataset[idx].values
        return torch.from_numpy(x).float()


class CroppedTimeSeriesDatasetTorch(Dataset):
    def __init__(
        self,
        dataset: TimeSeriesDataset,
        window_length: int,
        stride: int = 1,
    ):
        self.dataset = dataset
        self.window_length = window_length
        self.stride = stride

    def __len__(self):
        return sum(
            (len(ts) - self.window_length) // self.stride + 1 for ts in self.dataset
        )

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ts_idx = 0
        window_idx = idx
        while (
            window_idx
            >= (len(self.dataset[ts_idx]) - self.window_length) // self.stride + 1
        ):
            window_idx -= (
                len(self.dataset[ts_idx]) - self.window_length
            ) // self.stride + 1
            ts_idx += 1

        start = window_idx * self.stride
        end = start + self.window_length
        x = self.dataset[ts_idx].values[start:end]
        return torch.from_numpy(x).float()
