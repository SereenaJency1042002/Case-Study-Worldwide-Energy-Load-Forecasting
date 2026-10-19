import pandas as pd

from energy_forecast.backtest import daily_mae, rolling_origin_backtest, summarize
from energy_forecast.baselines import SeasonalNaive

TZ = "Europe/Berlin"


class RecordingNaive(SeasonalNaive):
    """Seasonal naive that records what data it was given."""

    def __init__(self) -> None:
        super().__init__(168)
        self.fit_ends: list[pd.Timestamp] = []
        self.calls: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def fit(self, y: pd.Series) -> None:
        self.fit_ends.append(y.index[-1])

    def predict(self, history: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
        self.calls.append((history.index[-1], index[0]))
        return super().predict(history, index)


def test_backtest_follows_challenge_timing(weekly_load):
    model = RecordingNaive()
    preds = rolling_origin_backtest(
        weekly_load, [model], "2024-01-29", "2024-02-12", TZ,
        gap_hours=12, refit_every_days=7,
    )

    assert len(preds) == 14 * 24
    # local midnight 2024-01-29 (CET) is 23:00 UTC the day before
    assert preds.index[0] == pd.Timestamp("2024-01-28 23:00", tz="UTC")
    assert len(model.fit_ends) == 2  # refit on day 0 and day 7
    for last_known, first_target in model.calls:
        assert last_known < first_target - pd.Timedelta(hours=12)
    assert summarize(preds).loc["seasonal_naive_168h", "mae"] == 0.0
    assert daily_mae(preds).shape == (14, 1)


def test_dst_day_has_23_hours(weekly_load_spring):
    preds = rolling_origin_backtest(
        weekly_load_spring, [SeasonalNaive(168)], "2024-03-30", "2024-04-01", TZ
    )
    counts = preds.groupby("target_day").size().to_dict()
    assert counts == {"2024-03-30": 24, "2024-03-31": 23}
