import torch
from torch.utils.data import Dataset
import numpy as np


class TimeSeriesDatasetCreator:
    def __init__(self, X, y, input_seq_length, output_seq_length):
        self.X = X
        self.y = y
        self.input_seq_length = input_seq_length
        self.output_seq_length = output_seq_length

    def create_sequences(self):
        X_sequences = []
        y_sequences = []

        for i in range(
            len(self.X) - self.input_seq_length - self.output_seq_length + 1
        ):
            input_end_idx = i + self.input_seq_length
            output_end_idx = input_end_idx + self.output_seq_length

            input_seq = self.X[i:input_end_idx]
            output_seq = self.y[input_end_idx:output_end_idx]

            X_sequences.append(input_seq)
            y_sequences.append(output_seq)

        return np.array(X_sequences), np.array(y_sequences)


class TimeSeriesDatasetPyTorch(Dataset, TimeSeriesDatasetCreator):
    def __init__(self, X, y, input_seq_length, output_seq_length):
        super().__init__(X, y, input_seq_length, output_seq_length)
        self.X_sequences, self.y_sequences = self.create_sequences()
        self.X_sequences = torch.from_numpy(self.X_sequences).float()
        self.y_sequences = torch.from_numpy(self.y_sequences).float()

    def __len__(self):
        return len(self.X_sequences)

    def __getitem__(self, idx):
        return self.X_sequences[idx], self.y_sequences[idx]


class TimeSeriesDatasetSklearn(TimeSeriesDatasetCreator):
    def __init__(self, X, y, input_seq_length, output_seq_length):
        super().__init__(X, y, input_seq_length, output_seq_length)
        self.X_sequences, self.y_sequences = self.create_sequences()
