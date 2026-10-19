import pandas as pd
import pytest

from energy_forecast.baselines import EntsoeForecast, SeasonalNaive
from energy_forecast.models import LightGBMForecaster


def _split(
    y: pd.Series, day: int, gap_hours: int = 12
) -> tuple[pd.Series, pd.DatetimeIndex]:
    """History known before the cutoff, and the 24 target hours of ``day``."""
    origin = y.index[day * 24]
    cutoff = origin - pd.Timedelta(hours=gap_hours)
    return y[y.index < cutoff], pd.date_range(origin, periods=24, freq="h")


def test_seasonal_naive_repeats_last_week(weekly_load):
    history, index = _split(weekly_load, 40)
    forecast = SeasonalNaive(168).predict(history, index)
    expected = weekly_load.reindex(index - pd.Timedelta(hours=168)).to_numpy()
    assert (forecast.to_numpy() == expected).all()
    assert forecast.index.equals(index)


def test_seasonal_naive_season_must_cover_horizon(weekly_load):
    history, index = _split(weekly_load, 40)
    with pytest.raises(ValueError):
        SeasonalNaive(12).predict(history, index)


def test_lightgbm_is_deterministic_and_complete(weekly_load):
    history, index = _split(weekly_load, 50)
    forecasts = []
    for _ in range(2):
        model = LightGBMForecaster(tz="Europe/Berlin")
        model.fit(history)
        forecasts.append(model.predict(history, index))
    pd.testing.assert_series_equal(forecasts[0], forecasts[1])
    assert len(forecasts[0]) == 24 and not forecasts[0].isna().any()


def test_lightgbm_refuses_when_history_ends_too_early(weekly_load):
    # with a 30 h gap the 48 h lag of the last target hour is not yet known
    history, index = _split(weekly_load, 50, gap_hours=30)
    model = LightGBMForecaster(tz="Europe/Berlin")
    model.fit(history)
    with pytest.raises(ValueError):
        model.predict(history, index)


def test_entsoe_forecast_returns_published_values(weekly_load):
    published = weekly_load * 1.01
    history, index = _split(weekly_load, 40)
    forecast = EntsoeForecast(published).predict(history, index)
    assert (forecast.to_numpy() == published.reindex(index).to_numpy()).all()


def test_entsoe_forecast_missing_hour_raises(weekly_load):
    history, index = _split(weekly_load, 40)
    published = weekly_load.drop(index[5])
    with pytest.raises(ValueError, match="missing"):
        EntsoeForecast(published).predict(history, index)
