"""Rolling-origin backtest that follows the daily challenge timing.

For each target day D (00:00-23:00 local time, so 23, 24 or 25 hours) every
model forecasts D using only actual load known before the cutoff, ``gap_hours``
before D starts (default: D-1 12:00 local). All models see the same days,
cutoffs and refit schedule.

Run from the project root:
    python -m energy_forecast.backtest
"""

from __future__ import annotations

import argparse
import logging
from itertools import pairwise
from pathlib import Path

import pandas as pd

from energy_forecast.baselines import EntsoeForecast, SeasonalNaive
from energy_forecast.data.loader import (
    DataError,
    fetch_day_ahead_forecast,
    fetch_load,
    validate_series,
)
from energy_forecast.metrics import compute_all
from energy_forecast.models import Forecaster, LightGBMForecaster

logger = logging.getLogger(__name__)

# Pinned defaults, so every run (and every team member) gets identical results.
ZONE = "DE_LU"
TZ = "Europe/Berlin"
DATA_START = "2023-01-01"  # UTC
DATA_END = "2026-10-19"  # UTC, exclusive
TEST_START = "2026-09-21"  # first local target day
TEST_END = "2026-10-19"  # local, exclusive
GAP_HOURS = 12
REFIT_EVERY_DAYS = 7
TRAIN_WINDOW_DAYS = 730


def rolling_origin_backtest(
    y: pd.Series,
    models: list[Forecaster],
    test_start: str,
    test_end: str,
    tz: str,
    gap_hours: int = GAP_HOURS,
    refit_every_days: int = REFIT_EVERY_DAYS,
    train_window_days: int = TRAIN_WINDOW_DAYS,
) -> pd.DataFrame:
    """Return hourly predictions for every local day in [test_start, test_end).

    Columns: ``target_day``, ``actual`` and one column per model name. Models
    are refit on the trailing ``train_window_days`` every ``refit_every_days``.
    """
    validate_series(y)
    midnights = pd.date_range(test_start, test_end, freq="D", tz=tz)
    frames = []
    for i, (day_start, day_end) in enumerate(pairwise(midnights)):
        index = pd.date_range(
            day_start.tz_convert("UTC"), day_end.tz_convert("UTC"),
            freq="h", inclusive="left",
        )
        cutoff = index[0] - pd.Timedelta(hours=gap_hours)
        history = y[y.index < cutoff]
        if i % refit_every_days == 0:
            train = history[history.index >= cutoff - pd.Timedelta(days=train_window_days)]
            logger.info("Refitting %d models with data before %s", len(models), cutoff)
            for model in models:
                model.fit(train)

        actual = y.reindex(index)
        if actual.isna().any():
            raise DataError(f"actual load missing for test day {day_start:%Y-%m-%d}")

        frame = pd.DataFrame(
            {"target_day": f"{day_start:%Y-%m-%d}", "actual": actual}, index=index
        )
        for model in models:
            frame[model.name] = model.predict(history, index)
        frames.append(frame)
    return pd.concat(frames)


def _model_columns(predictions: pd.DataFrame) -> list[str]:
    return [c for c in predictions.columns if c not in ("target_day", "actual")]


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    """Overall metrics per model (rows) over all test hours."""
    rows = {
        n: compute_all(predictions["actual"], predictions[n])
        for n in _model_columns(predictions)
    }
    return pd.DataFrame(rows).T.sort_values("mae")


def daily_mae(predictions: pd.DataFrame) -> pd.DataFrame:
    """MAE per target day (rows) and model (columns), to spot hard days."""
    names = _model_columns(predictions)
    errors = predictions[names].sub(predictions["actual"], axis=0).abs()
    return errors.groupby(predictions["target_day"]).mean()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zone", default=ZONE, help="ENTSO-E bidding zone")
    parser.add_argument("--tz", default=TZ, help="local time zone of the target day")
    parser.add_argument("--data-start", default=DATA_START, help="UTC, inclusive")
    parser.add_argument("--data-end", default=DATA_END, help="UTC, exclusive")
    parser.add_argument("--test-start", default=TEST_START, help="first target day")
    parser.add_argument("--test-end", default=TEST_END, help="exclusive")
    parser.add_argument("--gap-hours", type=int, default=GAP_HOURS)
    parser.add_argument("--refit-days", type=int, default=REFIT_EVERY_DAYS)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    start = pd.Timestamp(args.data_start, tz="UTC")
    end = pd.Timestamp(args.data_end, tz="UTC")
    y = fetch_load(args.zone, start, end)
    entsoe_forecast = fetch_day_ahead_forecast(args.zone, start, end)

    models: list[Forecaster] = [
        EntsoeForecast(entsoe_forecast),
        SeasonalNaive(168),
        LightGBMForecaster(tz=args.tz),
    ]
    predictions = rolling_origin_backtest(
        y, models, args.test_start, args.test_end, args.tz,
        gap_hours=args.gap_hours, refit_every_days=args.refit_days,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.out / "backtest_predictions.csv", index_label="timestamp_utc")
    daily_mae(predictions).to_csv(args.out / "backtest_daily_mae.csv")
    summary = summarize(predictions)
    summary.to_csv(args.out / "backtest_summary.csv")
    print(
        f"\nBacktest {args.zone}, target days {args.test_start} -> {args.test_end} "
        f"(exclusive, {args.tz}), cutoff {args.gap_hours} h before each day"
    )
    print(summary.round(2).to_string())


if __name__ == "__main__":
    main()
