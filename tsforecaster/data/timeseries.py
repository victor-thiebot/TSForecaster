from typing import Any, Optional, Union, List, Tuple
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


class TimeSeries:
    """Base data structure for time series

    Attributes:
        values: Time series values. This can be a vector in the case of univariate time series, or
            a matrix for multivariate time series, with shape (Time, dimension).
        timestamps: Timestamps corresponding to each time step in the time series.
        item_id: Identifies the time series, usually with a string.
        metadata: Additional metadata associated with the time series.
    """

    values: np.ndarray
    timestamps: np.ndarray
    item_id: Any
    metadata: Optional[dict] = None

    def __init__(
        self,
        values: np.ndarray,
        timestamps: np.ndarray,
        item_id: Any = None,
        metadata: Optional[dict] = None,
    ):
        self.values = np.asarray(values, np.float32)
        self.timestamps = np.asarray(timestamps, np.float32)
        assert len(self.values) == len(
            self.timestamps
        ), "Values and timestamps must have the same length."
        self.item_id = item_id
        self.metadata = metadata or {}

    def __len__(self):
        return len(self.values)

    def copy(self) -> "TimeSeries":
        return TimeSeries(
            values=self.values.copy(),
            timestamps=self.timestamps.copy(),
            item_id=self.item_id,
            metadata=self.metadata.copy(),
        )

    def to_json(self, path: Union[Path, str]):
        path = Path(path).expanduser()
        data = {
            "values": self.values.tolist(),
            "timestamps": self.timestamps.tolist(),
            "item_id": self.item_id,
            "metadata": self.metadata,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def from_json(cls, path: Union[Path, str]):
        path = Path(path).expanduser()
        with open(path, "r") as f:
            data = json.load(f)
        return cls(
            values=np.array(data["values"]),
            timestamps=np.array(data["timestamps"]),
            item_id=data["item_id"],
            metadata=data.get("metadata", {}),
        )

    def __repr__(self):
        return f"TimeSeries(item_id={self.item_id!r}, shape={self.shape})"

    def __getitem__(self, key):
        if isinstance(key, int):
            key = slice(key, key + 1)
        return TimeSeries(
            values=self.values[key],
            timestamps=self.timestamps[key],
            item_id=f"{self.item_id}_slice" if self.item_id else None,
            metadata=self.metadata,
        )

    def append(self, ts: "TimeSeries"):
        return TimeSeries(
            values=np.concatenate([self.values, ts.values], axis=0),
            timestamps=np.concatenate([self.timestamps, ts.timestamps], axis=0),
            item_id=f"({self.item_id},{ts.item_id})",
            metadata={**self.metadata, **ts.metadata},
        )

    @property
    def shape(self):
        if self.values.ndim == 1:
            T, ts_channels = self.values.shape[0], 1
        elif self.values.ndim == 2:
            T, ts_channels = self.values.shape
        else:
            raise ValueError("values must be a vector or matrix.")
        return T, ts_channels

    def plot(self, title: str = "", figsize: tuple = (10, 6)):
        T, ts_channels = self.shape
        fig, axs = plt.subplots(ts_channels, 1, figsize=figsize, sharex=True)
        if ts_channels == 1:
            axs = np.array([axs])

        for i in range(ts_channels):
            values = self.values if ts_channels == 1 else self.values[:, i]
            axs[i].plot(self.timestamps, values)
            axs[i].set_ylabel(f"Channel {i+1}")

        axs[-1].set_xlabel("Timestamp")
        fig.suptitle(title)
        fig.tight_layout()
        return fig, axs


class TimeSeriesDataset:
    """Collection of Time Series."""

    def __init__(self, x_list: List[TimeSeries], y_list: List[TimeSeries]):
        assert len(x_list) == len(
            y_list
        ), "Input and output time series must have the same length."
        self.x_list = x_list
        self.y_list = y_list

    def __len__(self):
        return len(self.x_list)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.x_list[idx], self.y_list[idx]
        elif isinstance(idx, slice):
            return TimeSeriesDataset(self.x_list[idx], self.y_list[idx])
        else:
            raise TypeError("Invalid argument type.")

    def __iter__(self):
        return iter(zip(self.x_list, self.y_list))

    def __repr__(self):
        return f"TimeSeriesDataset(num_series={len(self)})"

    def copy(self):
        return TimeSeriesDataset(
            [ts.copy() for ts in self.x_list], [ts.copy() for ts in self.y_list]
        )

    def append(self, x: TimeSeries, y: TimeSeries):
        self.x_list.append(x)
        self.y_list.append(y)

    def extend(self, ts_dataset: "TimeSeriesDataset"):
        self.x_list.extend(ts_dataset.x_list)
        self.y_list.extend(ts_dataset.y_list)

    def save_to_directory(
        self,
        directory: Union[Path, str],
        x_filename_prefix: str = "x_",
        y_filename_prefix: str = "y_",
    ):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for i, (x, y) in enumerate(tqdm(self, desc="Saving time series")):
            x_filename = f"{x_filename_prefix}{i}.json"
            y_filename = f"{y_filename_prefix}{i}.json"
            x.to_json(directory / x_filename)
            y.to_json(directory / y_filename)

    @classmethod
    def load_from_directory(
        cls,
        directory: Union[Path, str],
        x_filename_prefix: str = "x_",
        y_filename_prefix: str = "y_",
    ):
        directory = Path(directory)
        x_list = []
        y_list = []
        for i in range(len(list(directory.glob(f"{x_filename_prefix}*.json")))):
            x_file = directory / f"{x_filename_prefix}{i}.json"
            y_file = directory / f"{y_filename_prefix}{i}.json"
            x = TimeSeries.from_json(x_file)
            y = TimeSeries.from_json(y_file)
            x_list.append(x)
            y_list.append(y)
        return cls(x_list, y_list)

    @property
    def shape(self):
        return [(x.shape, y.shape) for x, y in self]


def split_train_val_test(
    dataset: TimeSeriesDataset,
    val_portion: float = 0.2,
    test_portion: float = 0.2,
    split_method: str = "sequential",
    verbose: bool = True,
) -> Tuple[TimeSeriesDataset, TimeSeriesDataset, TimeSeriesDataset]:
    if split_method != "sequential":
        raise NotImplementedError(f"split_method={split_method} is not supported.")

    train_size = int(len(dataset) * (1 - val_portion - test_portion))
    val_size = int(len(dataset) * val_portion)

    train_dataset = TimeSeriesDataset(
        dataset.x_list[:train_size], dataset.y_list[:train_size]
    )
    val_dataset = TimeSeriesDataset(
        dataset.x_list[train_size : train_size + val_size],
        dataset.y_list[train_size : train_size + val_size],
    )
    test_dataset = TimeSeriesDataset(
        dataset.x_list[train_size + val_size :], dataset.y_list[train_size + val_size :]
    )

    if verbose:
        print(
            f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}"
        )

    return train_dataset, val_dataset, test_dataset


def ts_random_crop(
    ts_x: TimeSeries,
    ts_y: TimeSeries,
    length: int,
    num_crops: int = 1,
) -> Tuple[TimeSeriesDataset, TimeSeriesDataset]:
    assert len(ts_x) == len(
        ts_y
    ), "Input and output time series must have the same length."

    T = len(ts_x)
    if T < length:
        return [], []

    idx_end = np.random.randint(low=length, high=T, size=num_crops)

    out_x = [
        TimeSeries(
            values=ts_x.values[i - length : i],
            timestamps=ts_x.timestamps[i - length : i],
        )
        for i in idx_end
    ]
    out_y = [
        TimeSeries(
            values=ts_y.values[i - length : i],
            timestamps=ts_y.timestamps[i - length : i],
        )
        for i in idx_end
    ]

    return out_x, out_y


def ts_rolling_window(
    ts_x: TimeSeries,
    ts_y: TimeSeries,
    window_length: int,
    stride: int = 1,
) -> Tuple[TimeSeriesDataset, TimeSeriesDataset]:
    assert len(ts_x) == len(
        ts_y
    ), "Input and output time series must have the same length."

    values_windows_x = _rolling_window(
        ts_x.values, window_length=window_length, stride=stride
    )
    timestamps_windows_x = _rolling_window(
        ts_x.timestamps, window_length=window_length, stride=stride
    )
    values_windows_y = _rolling_window(
        ts_y.values, window_length=window_length, stride=stride
    )
    timestamps_windows_y = _rolling_window(
        ts_y.timestamps, window_length=window_length, stride=stride
    )
    out_x = [
        TimeSeries(values=values_windows_x[i], timestamps=timestamps_windows_x[i])
        for i in range(len(values_windows_x))
    ]
    out_y = [
        TimeSeries(values=values_windows_y[i], timestamps=timestamps_windows_y[i])
        for i in range(len(values_windows_y))
    ]

    return out_x, out_y


def ts_split(
    ts_x: TimeSeries,
    ts_y: TimeSeries,
    indices_or_sections,
) -> Tuple[TimeSeriesDataset, TimeSeriesDataset]:
    assert len(ts_x) == len(
        ts_y
    ), "Input and output time series must have the same length."

    values_split_x = np.split(ts_x.values, indices_or_sections)
    timestamps_split_x = np.split(ts_x.timestamps, indices_or_sections)

    values_split_y = np.split(ts_y.values, indices_or_sections)
    timestamps_split_y = np.split(ts_y.timestamps, indices_or_sections)

    out_x = [
        TimeSeries(values=values_split_x[i], timestamps=timestamps_split_x[i])
        for i in range(len(values_split_x))
    ]
    out_y = [
        TimeSeries(values=values_split_y[i], timestamps=timestamps_split_y[i])
        for i in range(len(values_split_y))
    ]

    return out_x, out_y


def ts_to_array(
    ts_x: TimeSeriesDataset,
    ts_y: TimeSeriesDataset,
) -> Tuple[np.array, np.array]:
    assert len(ts_x) == len(
        ts_y
    ), "Input and output time series must have the same length."

    out_x = np.array([ts.values for ts in ts_x])
    out_y = np.array([ts.values for ts in ts_y])

    return out_x, out_y


def _rolling_window(
    ts_array: np.ndarray,
    window_length: int,
    stride: int = 1,
) -> np.ndarray:
    assert len(ts_array) >= window_length

    shape = (((len(ts_array) - window_length) // stride) + 1, window_length)
    strides = (ts_array.strides[0] * stride, ts_array.strides[0])
    return np.lib.stride_tricks.as_strided(ts_array, shape=shape, strides=strides)
