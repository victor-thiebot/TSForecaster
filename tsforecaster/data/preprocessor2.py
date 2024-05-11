import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

from sklearn.model_selection import train_test_split

from tsforecaster.utils.plotting import plot_time_series


class DataPreprocessor:

    def __init__(self, data_path, header_row=0, date_column="Date", freq=None):
        self.data_path = data_path
        self.header_row = header_row
        self.date_column = date_column
        self.freq = freq
        self.data = self._load_data()
        self.variable_names = list(self.data.columns)
        self._preprocess_data()
        self.trend = None
        self.seasonality = None

    def _load_data(self):
        return pd.read_csv(self.data_path, header=self.header_row)

    def _preprocess_data(self):
        self.data = self._convert_date_column(self.data)
        self.data = self._sort_data(self.data)
        self.data = self._set_index(self.data)
        self._handle_frequency()
        self.data = self._interpolate_missing_values(self.data)

    def _convert_date_column(self, data):
        if self.date_column in data.columns:
            data[self.date_column] = pd.to_datetime(data[self.date_column])
        return data

    def _sort_data(self, data):
        data = data.sort_values(by=self.date_column)
        return data

    def _set_index(self, data):
        data = data.set_index(self.date_column)
        return data

    def _infer_frequency(self):
        if self.freq is None:
            self.freq = pd.infer_freq(self.data.index)

    def _set_frequency(self):
        if self.freq is None:
            warnings.warn(
                "Failed to infer the frequency of the time series. Please check the date column and consider providing the frequency explicitly."
            )
        if self.freq is not None:
            self.data = self.data.asfreq(self.freq)

    def _handle_frequency(self):
        self._infer_frequency()
        self._set_frequency()

    def _interpolate_missing_values(self, data):
        data = data.interpolate()
        return data

    def get_data(self):
        return self.data

    def get_variable_names(self):
        return self.variable_names

    def get_freq(self):
        return self.freq

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
        trend_data = pd.DataFrame(index=self.data.index, columns=self.data.columns)
        for column in detrended_data.columns:
            series = detrended_data[column]
            stl = STL(series, robust=True)
            res = stl.fit()
            trend_data[column] = res.trend
            detrended_data[column] = series - res.trend
        self.trend = trend_data
        self.data = detrended_data

    def deseasonalize(self):
        deseasonal_data = self.data.copy()
        seasonality_data = pd.DataFrame(
            index=self.data.index, columns=self.data.columns
        )
        for column in deseasonal_data.columns:
            series = deseasonal_data[column]
            stl = STL(series, robust=True)
            res = stl.fit()
            seasonality_data[column] = res.seasonal
            deseasonal_data[column] = series - res.seasonal
        self.seasonality = seasonality_data
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
        plot_time_series(
            self.data,
            title="Preprocessed Data",
            xlabel="Time",
            ylabel="Value",
            figsize=figsize,
        )


class DataSplitter:
    def __init__(
        self,
        X,
        y,
        input_seq_length,
        output_seq_length,
        test_size=0.2,
        random_state=None,
    ):
        self.X = X
        self.y = y
        self.input_seq_length = input_seq_length
        self.output_seq_length = output_seq_length
        self.test_size = test_size
        self.random_state = random_state

    def split_data(self):
        # Calculate the number of complete sequences in the dataset
        num_sequences = len(self.X) - self.input_seq_length - self.output_seq_length + 1

        # Split the data into train and test sets
        train_size = int((1 - self.test_size) * num_sequences)
        X_train = self.X[:train_size + self.input_seq_length - 1]
        X_test = self.X[train_size:]
        y_train = self.y[self.input_seq_length - 1 : train_size + self.input_seq_length + self.output_seq_length - 1]
        y_test = self.y[train_size + self.input_seq_length - 1:]

        return X_train, X_test, y_train, y_test

    def create_sklearn_datasets(self):
        X_train, X_test, y_train, y_test = self.split_data()
        X_train_seq, y_train_seq = self._create_sequences(X_train, y_train)
        X_test_seq, y_test_seq = self._create_sequences(X_test, y_test)
        return X_train_seq, X_test_seq, y_train_seq, y_test_seq

    def _create_sequences(self, X, y):
        input_seq = []
        target_seq = []
        for i in range(len(X) - self.input_seq_length - self.output_seq_length + 1):
            input_seq.append(X[i : i + self.input_seq_length])
            target_seq.append(y[i + self.input_seq_length : i + self.input_seq_length + self.output_seq_length])
        return np.array(input_seq), np.array(target_seq)

    def create_pytorch_datasets(self):
        X_train, X_test, y_train, y_test = self.split_data()
        train_dataset = TimeSeriesDatasetPyTorch(
            X_train, y_train, self.input_seq_length, self.output_seq_length
        )
        test_dataset = TimeSeriesDatasetPyTorch(
            X_test, y_test, self.input_seq_length, self.output_seq_length
        )
        return train_dataset, test_dataset


class DataSplitter:
    def __init__(self, X, y, test_size=0.2, random_state=None):
        self.X = X
        self.y = y
        self.test_size = test_size
        self.random_state = random_state

    def split_data(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )
        return X_train, X_test, y_train, y_test
