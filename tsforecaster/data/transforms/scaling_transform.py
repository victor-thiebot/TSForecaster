from typing import Iterable
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer
from scipy.special import inv_boxcox
import numpy as np

from tsforecaster.data.timeseries import TimeSeries
from tsforecaster.data.transforms.base import TimeSeriesTransform


class ScalingTransform(TimeSeriesTransform):
    def __init__(self, scaling_method="min-max", **kwargs):
        self.scaling_method = scaling_method
        self.scaler = None
        self.kwargs = kwargs
        self.metadata = {}

    def fit(self, ts: TimeSeries):
        # Fit the scaler using the input data
        if self.scaling_method == "min-max":
            self.scaler = MinMaxScaler(**self.kwargs)
            self.scaler.fit(ts.values.reshape(-1, 1))
        elif self.scaling_method == "standardization":
            self.scaler = StandardScaler(**self.kwargs)
            self.scaler.fit(ts.values.reshape(-1, 1))
        elif self.scaling_method == "log":
            # No fitting required for log transformation
            pass
        elif self.scaling_method == "box-cox":
            self.kwargs.setdefault("standardize", False)
            self.scaler = PowerTransformer(method="box-cox", **self.kwargs)
            self.scaler.fit(ts.values.reshape(-1, 1))
        else:
            raise ValueError(f"Unsupported scaling method: {self.scaling_method}")

    def transform(self, ts: TimeSeries) -> TimeSeries:
        # Apply scaling to the TimeSeries object
        if self.scaling_method == "log":
            scaled_values = np.log(ts.values)
        elif self.scaler is None:
            raise ValueError(
                "Scaler has not been fitted. Call 'fit' before 'transform'."
            )
        else:
            scaled_values = self.scaler.transform(ts.values.reshape(-1, 1)).flatten()

        self.metadata[ts.item_id] = {
            "scaling_method": self.scaling_method,
            "scaler": self.scaler,
            "original_shape": ts.values.shape,
        }
        return TimeSeries(
            values=scaled_values,
            timestamps=ts.timestamps,
            item_id=ts.item_id,
            metadata=ts.metadata,
        )

    # def inverse_transform(self, ts: TimeSeries) -> TimeSeries:
    #     # Inverse the scaling transformation
    #     item_metadata = self.metadata.get(ts.item_id)
    #     if item_metadata is None:
    #         raise ValueError(f"No scaling metadata found for item_id: {ts.item_id}")

    #     if item_metadata["scaling_method"] == "log":
    #         inverse_values = np.exp(ts.values)
    #     else:
    #         scaler = item_metadata["scaler"]
    #         inverse_values = scaler.inverse_transform(
    #             ts.values.reshape(-1, 1)
    #         ).flatten()

    #     original_shape = item_metadata["original_shape"]
    #     inverse_values = inverse_values.reshape(original_shape)

    #     return TimeSeries(
    #         values=inverse_values,
    #         timestamps=ts.timestamps,
    #         item_id=ts.item_id,
    #         metadata=ts.metadata,
    #     )
    def inverse_transform(self, ts: TimeSeries) -> TimeSeries:
        # Inverse the scaling transformation
        item_metadata = self.metadata.get(ts.item_id)
        if item_metadata is None:
            raise ValueError(f"No scaling metadata found for item_id: {ts.item_id}")

        if item_metadata["scaling_method"] == "log":
            inverse_values = np.exp(ts.values)
        else:
            scaler = item_metadata["scaler"]
            inverse_values = scaler.inverse_transform(
                ts.values.reshape(-1, 1)
            ).flatten()

        return TimeSeries(
            values=inverse_values,
            timestamps=ts.timestamps,
            item_id=ts.item_id,
            metadata=ts.metadata,
        )
