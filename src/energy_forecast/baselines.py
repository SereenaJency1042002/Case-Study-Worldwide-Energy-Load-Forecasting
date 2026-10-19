"""Reference forecasts: seasonal naive and the official ENTSO-E day-ahead forecast."""

from __future__ import annotations

import pandas as pd


class SeasonalNaive:
    """Forecast each hour with the actual load ``season_hours`` earlier.

    season_hours=24 is "same hour yesterday", 168 is "same hour last week".
    """

    def __init__(self, season_hours: int) -> None:
        self.season_hours = season_hours
        self.name = f"seasonal_naive_{season_hours}h"

    def fit(self, y: pd.Series) -> None:
        """Nothing to learn; present for the common model interface."""

    def predict(self, history: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
        if self.season_hours < len(index):
            raise ValueError(
                f"season ({self.season_hours} h) must cover the horizon "
                f"({len(index)} h)"
            )
        source = index - pd.Timedelta(hours=self.season_hours)
        values = history.reindex(source)
        if values.isna().any():
            raise ValueError(f"{self.name}: history does not cover {source[0]}")
        return pd.Series(values.to_numpy(), index=index, name=self.name)


class EntsoeForecast:
    """The TSOs' own day-ahead load forecast published on ENTSO-E.

    This is the benchmark to beat. It is published on D-1, before the forecast
    cutoff, so looking it up for day D uses no future information.
    """

    name = "entsoe_day_ahead"

    def __init__(self, forecast: pd.Series) -> None:
        self.forecast = forecast

    def fit(self, y: pd.Series) -> None:
        """Nothing to learn; present for the common model interface."""

    def predict(self, history: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
        values = self.forecast.reindex(index)
        if values.isna().any():
            missing = index[values.isna()]
            raise ValueError(
                f"ENTSO-E forecast missing for {len(missing)} hours, first {missing[0]}"
            )
        return pd.Series(values.to_numpy(), index=index, name=self.name)
