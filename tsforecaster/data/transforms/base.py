from typing import List, Optional, Union, Iterable, Iterator
from tsforecaster.data.timeseries import TimeSeries
import pandas as pd


class TimeSeriesTransform(object):
    def __call__(self, ts_iterable: Iterable[TimeSeries]) -> Iterator[TimeSeries]:
        for ts in ts_iterable:
            yield self.transform(ts.copy())

    def transform(self, ts: TimeSeries) -> TimeSeries:
        raise NotImplementedError()

    def __add__(self, other: "TimeSeriesTransform") -> "TimeSeriesTransform":
        return Chain([self, other])


class Chain(TimeSeriesTransform):
    def __init__(
        self,
        ts_transforms: List[TimeSeriesTransform],
    ) -> None:
        self.ts_transforms = []
        for trans in ts_transforms:
            # flatten chains
            if isinstance(trans, Chain):
                self.ts_transforms.extend(trans.ts_transforms)
            else:
                self.ts_transforms.append(trans)

    def __repr__(self) -> str:
        return f"Chain({self.ts_transforms})"

    def __call__(self, data_it: Iterable[TimeSeries]) -> Iterator[TimeSeries]:
        tmp = data_it
        for trans in self.ts_transforms:
            tmp = trans(tmp)
        return tmp

    def fit(self, ts: TimeSeries):
        for trans in self.ts_transforms:
            if hasattr(trans, "fit"):
                trans.fit(ts)

    def transform(self, ts: TimeSeries) -> TimeSeries:
        tmp = ts.copy()
        for trans in self.ts_transforms:
            tmp = trans.transform(tmp)
        return tmp

    def inverse_transform(self, ts: TimeSeries) -> TimeSeries:
        tmp = ts.copy()
        for trans in reversed(self.ts_transforms):
            if hasattr(trans, "inverse_transform"):
                tmp = trans.inverse_transform(tmp)
        return tmp
