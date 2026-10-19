"""Forecasting models behind one common interface.

Every model (baselines, LightGBM, and later spotforecast2 or foundation-model
wrappers) implements ``Forecaster``, so the backtest treats them all the same.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd
from lightgbm import LGBMRegressor

from energy_forecast.features import build_features


class Forecaster(Protocol):
    name: str

    def fit(self, y: pd.Series) -> None:
        """Train on the hourly UTC series ``y``."""

    def predict(self, history: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
        """Forecast the hours in ``index`` using only ``history`` (known values)."""


class LightGBMForecaster:
    """Day-ahead LightGBM on local calendar features and >= 48 h lags."""

    name = "lightgbm"

    def __init__(self, tz: str, seed: int = 42) -> None:
        self.tz = tz
        self.model = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            random_state=seed,
            n_jobs=1,
            deterministic=True,
            force_row_wise=True,
            verbose=-1,
        )

    def fit(self, y: pd.Series) -> None:
        X = build_features(y, y.index, self.tz).dropna()
        self.model.fit(X, y.loc[X.index])

    def predict(self, history: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
        X = build_features(history, index, self.tz)
        if X.isna().any().any():
            raise ValueError("history too short to build features for the horizon")
        return pd.Series(self.model.predict(X), index=index, name=self.name)
