import numpy as np
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    """Time series dataset class. This class is used to create a PyTorch dataset object.

    Args:
        Dataset (torch.utils.data.Dataset): PyTorch dataset class
    """
    
    def __init__(self, x_data, y_data, seq_length):
        self.x_data = x_data
        self.y_data = y_data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.x_data) - self.seq_length

    def __getitem__(self, idx):
        x = self.x_data[idx : idx + self.seq_length]
        y = self.y_data[idx + self.seq_length]

        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        if x_tensor.dim() == 2:
            x_tensor = x_tensor.unsqueeze(0)

        return x_tensor, y_tensor
