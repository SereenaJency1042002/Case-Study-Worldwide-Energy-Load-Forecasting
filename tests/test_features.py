import pandas as pd
import pytest

from energy_forecast.features import build_features, calendar_features, lag_features


def test_spring_dst_day_has_23_local_hours():
    # 2024-03-31 in Berlin: 00:00 local = 23:00 UTC the day before, hour 02 skipped
    index = pd.date_range("2024-03-30 23:00", periods=23, freq="h", tz="UTC")
    hours = calendar_features(index, "Europe/Berlin")["hour"].tolist()
    assert hours == [0, 1] + list(range(3, 24))


def test_autumn_dst_day_has_25_local_hours():
    # 2024-10-27 in Berlin: hour 02 occurs twice
    index = pd.date_range("2024-10-26 22:00", periods=25, freq="h", tz="UTC")
    hours = calendar_features(index, "Europe/Berlin")["hour"].tolist()
    assert hours == [0, 1, 2, 2] + list(range(3, 24))


def test_forecast_features_do_not_use_load_after_cutoff(weekly_load):
    origin = weekly_load.index[30 * 24]
    cutoff = origin - pd.Timedelta(hours=12)
    index = pd.date_range(origin, periods=24, freq="h")
    history = weekly_load[weekly_load.index < cutoff]
    with_future = weekly_load.copy()
    with_future[with_future.index >= cutoff] = -1.0  # poison values after cutoff

    expected = build_features(history, index, "Europe/Berlin")
    actual = build_features(with_future, index, "Europe/Berlin")
    pd.testing.assert_frame_equal(expected, actual)
    assert not expected.isna().any().any()


def test_lags_shorter_than_horizon_are_rejected(weekly_load):
    with pytest.raises(ValueError):
        lag_features(weekly_load, weekly_load.index, lags=(1, 24))
