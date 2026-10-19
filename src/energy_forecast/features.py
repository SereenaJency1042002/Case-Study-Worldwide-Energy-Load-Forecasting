"""Leak-free features for day-ahead load forecasting.

Calendar features use LOCAL time (derived from the UTC index), so the daily
pattern stays aligned across daylight-saving switches.

All lags are >= 48 h. The forecast for local day D is made with actual load known
only up to a cutoff on D-1 (default 12 h before D starts). The furthest target
hour is then at most 12 + 25 = 37 h after the cutoff, so every 48 h+ lag is known
at forecast time and the same features serve training and prediction. If a lag
ever reaches past the cutoff, its feature is NaN and the model refuses to predict.
"""

from __future__ import annotations

import pandas as pd

LAGS = (48, 72, 168)
MIN_LAG = 48


def calendar_features(index: pd.DatetimeIndex, tz: str) -> pd.DataFrame:
    """Hour, weekday, month and weekend flag in local time ``tz``."""
    local = index.tz_convert(tz)
    return pd.DataFrame(
        {
            "hour": local.hour,
            "weekday": local.dayofweek,
            "month": local.month,
            "is_weekend": (local.dayofweek >= 5).astype(int),
        },
        index=index,
    )


def lag_features(
    history: pd.Series, index: pd.DatetimeIndex, lags: tuple[int, ...] = LAGS
) -> pd.DataFrame:
    """Lagged load and a trailing 24 h mean, evaluated at ``index``.

    ``history`` holds only values known at forecast time; ``index`` may lie
    after its end. Values not covered by ``history`` come out as NaN.
    """
    if min(lags) < MIN_LAG:
        raise ValueError(f"lags must be >= {MIN_LAG} h to avoid leakage")
    grid = pd.date_range(
        min(history.index[0], index[0]), max(history.index[-1], index[-1]),
        freq="h",
    )
    full = history.reindex(grid)
    out = {f"lag_{lag}": full.shift(lag) for lag in lags}
    out[f"mean_24_lag_{MIN_LAG}"] = full.shift(MIN_LAG).rolling(24).mean()
    return pd.DataFrame(out).reindex(index)


def build_features(
    history: pd.Series, index: pd.DatetimeIndex, tz: str
) -> pd.DataFrame:
    """Calendar + lag features for every timestamp in ``index``."""
    return calendar_features(index, tz).join(lag_features(history, index))
