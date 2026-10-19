import numpy as np
import pandas as pd
import pytest

from energy_forecast.data.loader import (
    DataError,
    fetch_day_ahead_forecast,
    fetch_load,
    to_hourly,
    validate_series,
)


def _quarter_hourly(hours: int) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=hours * 4, freq="15min", tz="UTC")
    return pd.Series(np.full(len(index), 100.0), index=index)


def test_validate_rejects_naive_index(weekly_load):
    naive = weekly_load.tz_localize(None)
    with pytest.raises(DataError):
        validate_series(naive)


def test_to_hourly_averages_and_fills_short_gap():
    raw = _quarter_hourly(10)
    raw.iloc[8:12] = np.nan  # one missing hour
    y = to_hourly(raw)
    assert len(y) == 10 and not y.isna().any()
    validate_series(y)


def test_to_hourly_raises_on_long_gap():
    raw = _quarter_hourly(20)
    raw.iloc[8:28] = np.nan  # five missing hours
    with pytest.raises(DataError):
        to_hourly(raw)


def test_fetch_reads_cache_without_api_key(tmp_path, monkeypatch, weekly_load):
    monkeypatch.delenv("ENTSOE_API_KEY", raising=False)
    start, end = pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-03-01", tz="UTC")
    weekly_load.to_csv(tmp_path / "load_DE_LU_20240101_20240301.csv", index_label="timestamp_utc")

    y = fetch_load("DE_LU", start, end, cache_dir=tmp_path)
    pd.testing.assert_series_equal(y, weekly_load, check_freq=False, check_names=False)


def test_fetch_forecast_reads_its_own_cache(tmp_path, monkeypatch, weekly_load):
    monkeypatch.delenv("ENTSOE_API_KEY", raising=False)
    start, end = pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-03-01", tz="UTC")
    weekly_load.to_csv(
        tmp_path / "forecast_DE_LU_20240101_20240301.csv", index_label="timestamp_utc"
    )

    y = fetch_day_ahead_forecast("DE_LU", start, end, cache_dir=tmp_path)
    pd.testing.assert_series_equal(y, weekly_load, check_freq=False, check_names=False)


def test_fetch_without_key_or_cache_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("ENTSOE_API_KEY", raising=False)
    start, end = pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-02-01", tz="UTC")
    with pytest.raises(DataError, match="ENTSOE_API_KEY"):
        fetch_load("DE_LU", start, end, cache_dir=tmp_path)
