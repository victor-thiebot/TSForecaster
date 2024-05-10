import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from torch.utils.data import Dataset
from tsforecaster.data.dataset import TimeSeriesDataset


class DataPreprocessor:
    """Data preprocessor class to load, preprocess, and split time series data."""

    def __init__(self, data_path, header_row=0, date_column="Date", freq=None):
        self.data_path = data_path
        self.header_row = header_row
        self.date_column = date_column
        self.freq = freq
        self.data, self.variable_names = self._load_data()

    def _load_data(self):
        data = pd.read_csv(self.data_path, header=self.header_row)
        variable_names = list(data.columns)
        data = self._convert_date_column(data)
        data = data.set_index(self.date_column)

        if self.freq is None:
            self.freq = pd.infer_freq(data.index)

        if self.freq is not None:
            data = data.asfreq(self.freq)

        data = data.interpolate()

        return data, variable_names

    def _convert_date_column(self, data):
        if self.date_column in data.columns:
            data[self.date_column] = pd.to_datetime(data[self.date_column])
        return data

    def get_data(self):
        return self.data

    def get_variable_names(self):
        return self.variable_names

    def preprocess(
        self, detrend=False, deseasonalize=False, normalize=False, method="min_max"
    ):
        if detrend:
            self.detrend()
        if deseasonalize:
            self.deseasonalize()
        if normalize:
            self.normalize(method=method)

    def detrend(self):
        detrended_data = self.data.copy()
        for column in detrended_data.columns:
            series = detrended_data[column]
            stl = STL(series, robust=True)
            res = stl.fit()
            detrended_data[column] = series - res.trend
        self.data = detrended_data

    def deseasonalize(self):
        deseasonal_data = self.data.copy()
        for column in deseasonal_data.columns:
            series = deseasonal_data[column]
            stl = STL(series, robust=True)
            res = stl.fit()
            deseasonal_data[column] = series - res.seasonal
        self.data = deseasonal_data

    def normalize(self, method="min_max"):
        if method == "min_max":
            self._min_max_scaling()
        elif method == "sigmoid":
            self._sigmoid_scaling()
        elif method == "z_score":
            self._z_score_scaling()
        elif method == "robust":
            self._robust_scaling()
        elif method == "tanh":
            self._tanh_scaling()
        else:
            raise ValueError(f"Unsupported normalization method: {method}")

    def _min_max_scaling(self):
        normalized_data = self.data.copy()
        for column in normalized_data.columns:
            series = normalized_data[column]
            min_value = series.min()
            max_value = series.max()
            normalized_data[column] = (series - min_value) / (max_value - min_value)
        self.data = normalized_data

    def _sigmoid_scaling(self):
        normalized_data = self.data.copy()
        for column in normalized_data.columns:
            series = normalized_data[column]
            normalized_data[column] = expit(series)
        self.data = normalized_data

    def _z_score_scaling(self):
        normalized_data = self.data.copy()
        for column in normalized_data.columns:
            series = normalized_data[column]
            mean = series.mean()
            std = series.std()
            normalized_data[column] = (series - mean) / std
        self.data = normalized_data

    def _robust_scaling(self):
        normalized_data = self.data.copy()
        for column in normalized_data.columns:
            series = normalized_data[column]
            median = series.median()
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            normalized_data[column] = (series - median) / iqr
        self.data = normalized_data

    def _tanh_scaling(self):
        normalized_data = self.data.copy()
        for column in normalized_data.columns:
            series = normalized_data[column]
            normalized_data[column] = np.tanh(series)
        self.data = normalized_data

    def plot_data(self, figsize=(12, 6)):
        plt.figure(figsize=figsize)

        if len(self.data.shape) == 1:
            # Univariate time series
            plt.plot(self.data)
            plt.xlabel("Time")
            plt.ylabel(self.variable_names[0])
            plt.title(f"{self.variable_names[0]} Time Series")
        else:
            # Multivariate time series
            for i in range(self.data.shape[1]):
                plt.subplot(self.data.shape[1], 1, i + 1)
                plt.plot(self.data[:, i])
                plt.xlabel("Time")
                plt.ylabel(self.variable_names[i])
                plt.title(f"{self.variable_names[i]} Time Series")
            plt.tight_layout()

        plt.show()

    def split_data(self, train_ratio, seq_length):
        train_size = int(len(self.data) * train_ratio)
        train_data, test_data = self.data[:train_size], self.data[train_size:]

        train_x, train_y = train_data[:, :-1], train_data[:, -1]
        test_x, test_y = test_data[:, :-1], test_data[:, -1]

        train_dataset = TimeSeriesDataset(train_x, train_y, seq_length)
        test_dataset = TimeSeriesDataset(test_x, test_y, seq_length)

        return train_dataset, test_dataset
