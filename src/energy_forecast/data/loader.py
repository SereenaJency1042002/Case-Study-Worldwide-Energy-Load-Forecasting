"""Load hourly electricity load from ENTSO-E, with a local CSV cache.

All series returned here are hourly, UTC-indexed, gap-free and strictly positive.
The API key is read from the ``ENTSOE_API_KEY`` environment variable, and only
when a download is actually needed.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("data") / "raw"
MAX_FILL_HOURS = 2


class DataError(Exception):
    """Raised when load data is missing, malformed or cannot be fetched."""


def validate_series(y: pd.Series) -> None:
    """Raise DataError unless ``y`` is hourly, UTC, gap-free and positive."""
    if not isinstance(y.index, pd.DatetimeIndex):
        raise DataError("index must be a DatetimeIndex")
    if str(y.index.tz) != "UTC":
        raise DataError(f"index must be in UTC, got tz={y.index.tz}")
    if y.index.has_duplicates or not y.index.is_monotonic_increasing:
        raise DataError("index must be strictly increasing")
    steps = y.index[1:] - y.index[:-1]
    if (steps != pd.Timedelta(hours=1)).any():
        raise DataError("index must be hourly without gaps")
    if y.isna().any():
        raise DataError(f"series contains {int(y.isna().sum())} missing values")
    if (y <= 0).any():
        raise DataError("load values must be strictly positive")


def to_hourly(raw: pd.Series) -> pd.Series:
    """Average sub-hourly data to hourly and fill gaps of at most 2 hours.

    Incomplete hours at the end (data not yet published) are dropped. Longer
    gaps inside the series raise DataError instead of being filled silently.
    """
    hourly = raw.tz_convert("UTC").resample("1h").mean()
    hourly = hourly.loc[: hourly.last_valid_index()]
    filled = hourly.interpolate(limit=MAX_FILL_HOURS, limit_area="inside")
    if filled.isna().any():
        missing = filled.index[filled.isna()]
        raise DataError(
            f"{len(missing)} hours missing in gaps longer than {MAX_FILL_HOURS} h, "
            f"first at {missing[0]}"
        )
    n_filled = int(hourly.isna().sum())
    if n_filled:
        logger.info("Interpolated %d short-gap hours", n_filled)
    filled.name = "load_mw"
    return filled


def fetch_load(
    zone: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> pd.Series:
    """Return hourly actual load for ``zone`` in [start, end), using a CSV cache.

    ``zone`` is an ENTSO-E bidding-zone code understood by entsoe-py, e.g. "DE_LU".
    """
    return _fetch_cached(
        "load", "query_load", "Actual Load", zone, start, end, cache_dir
    )


def fetch_day_ahead_forecast(
    zone: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> pd.Series:
    """Return the TSOs' hourly day-ahead load forecast for ``zone`` in [start, end)."""
    return _fetch_cached(
        "forecast", "query_load_forecast", "Forecasted Load", zone, start, end, cache_dir
    )


def _fetch_cached(
    kind: str,
    query: str,
    column: str,
    zone: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: Path,
) -> pd.Series:
    """Download ``column`` via entsoe-py method ``query`` unless already cached."""
    path = cache_dir / f"{kind}_{zone}_{start:%Y%m%d}_{end:%Y%m%d}.csv"
    if path.exists():
        logger.info("Reading cached %s from %s", kind, path)
        return read_cached(path)

    api_key = os.environ.get("ENTSOE_API_KEY")
    if not api_key:
        raise DataError("ENTSOE_API_KEY environment variable is not set")

    from entsoe import EntsoePandasClient

    logger.info("Downloading %s %s %s -> %s from ENTSO-E", zone, kind, start, end)
    client = EntsoePandasClient(api_key=api_key)
    try:
        raw = getattr(client, query)(zone, start=start, end=end)
    except Exception as exc:
        raise DataError(f"ENTSO-E {kind} download failed for {zone}: {exc}") from exc
    if isinstance(raw, pd.DataFrame):
        raw = raw[column]

    y = to_hourly(raw)
    validate_series(y)
    cache_dir.mkdir(parents=True, exist_ok=True)
    y.to_csv(path, index_label="timestamp_utc")
    return y


def read_cached(path: Path) -> pd.Series:
    """Read a cached load CSV written by ``fetch_load`` and validate it."""
    df = pd.read_csv(path, index_col=0)
    y = df["load_mw"].astype(float)
    y.index = pd.to_datetime(df.index, utc=True)
    validate_series(y)
    return y
