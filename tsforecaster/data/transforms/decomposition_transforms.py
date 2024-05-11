from statsmodels.tsa.seasonal import STL, seasonal_decompose
from typing import Optional

from tsforecaster.data.timeseries import TimeSeries
from tsforecaster.data.transforms.base import TimeSeriesTransform


class DecompositionTransform(TimeSeriesTransform):
    def __init__(
        self, decomposition_method="STL", period: Optional[int] = None, **kwargs
    ):
        self.decomposition_method = decomposition_method
        self.period = period
        self.kwargs = kwargs
        self.metadata = {}

    def transform(self, ts: TimeSeries) -> TimeSeries:
        # Apply decomposition to the TimeSeries object
        if self.decomposition_method == "STL":
            self.kwargs.setdefault("robust", False)
            if self.period is None:
                raise ValueError("Period must be specified for STL decomposition.")
            decomposition = STL(ts.values, period=self.period, **self.kwargs).fit()
        elif self.decomposition_method == "classical":
            if self.period is None:
                raise ValueError(
                    "Period must be specified for classical decomposition."
                )
            decomposition = seasonal_decompose(
                ts.values, model="additive", period=self.period, **self.kwargs
            )
        elif self.decomposition_method == "multiplicative":
            if self.period is None:
                raise ValueError(
                    "Period must be specified for multiplicative decomposition."
                )
            decomposition = seasonal_decompose(
                ts.values, model="multiplicative", period=self.period, **self.kwargs
            )
        else:
            raise ValueError(
                f"Unsupported decomposition method: {self.decomposition_method}"
            )

        self.metadata[ts.item_id] = {
            "decomposition_method": self.decomposition_method,
            "period": self.period,
            "decomposition": decomposition,
        }

        return TimeSeries(
            values=decomposition.trend,
            timestamps=ts.timestamps,
            item_id=ts.item_id,
            metadata=ts.metadata,
        )

    # def inverse_transform(self, ts: TimeSeries) -> TimeSeries:
    #     # Inverse the decomposition transformation
    #     item_metadata = self.metadata.get(ts.item_id)
    #     if item_metadata is None:
    #         raise ValueError(
    #             f"No decomposition metadata found for item_id: {ts.item_id}"
    #         )

    #     if item_metadata["decomposition_method"] in [
    #         "STL",
    #         "classical",
    #         "multiplicative",
    #     ]:
    #         decomposition = item_metadata["decomposition"]

    #         reconstructed_values = (
    #             decomposition.trend + decomposition.seasonal + decomposition.resid
    #         )
    #     return TimeSeries(
    #         values=reconstructed_values,
    #         timestamps=ts.timestamps,
    #         item_id=ts.item_id,
    #         metadata=ts.metadata,
    #     )

    def inverse_transform(self, ts: TimeSeries) -> TimeSeries:
        # Inverse the decomposition transformation
        item_metadata = self.metadata.get(ts.item_id)
        if item_metadata is None:
            raise ValueError(
                f"No decomposition metadata found for item_id: {ts.item_id}"
            )

        decomposition = item_metadata["decomposition"]

        if item_metadata["decomposition_method"] in ["STL", "classical"]:
            reconstructed_values = (
                decomposition.trend + decomposition.seasonal + decomposition.resid
            )
        elif item_metadata["decomposition_method"] == "multiplicative":
            reconstructed_values = (
                decomposition.trend * decomposition.seasonal * decomposition.resid
            )
        else:
            raise ValueError(
                f"Unsupported decomposition method: {item_metadata['decomposition_method']}"
            )

        return TimeSeries(
            values=reconstructed_values,
            timestamps=ts.timestamps,
            item_id=ts.item_id,
            metadata=ts.metadata,
        )
