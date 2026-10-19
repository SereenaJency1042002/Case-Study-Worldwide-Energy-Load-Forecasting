# Worldwide Energy-Load Forecasting (OTS Case Study 7)

Day-ahead hourly electricity load forecasting, starting with Germany/Luxembourg
(`DE_LU`) from the ENTSO-E Transparency Platform. This is the base setup: data loader,
reference forecasts (ENTSO-E day-ahead forecast, seasonal naive), a LightGBM base
model and a rolling-origin backtest that scores every model on the same days with
MAE as the primary metric.

spotforecast2 / spotforecast2-safe will be integrated next. Every model implements
the same `fit(y)` / `predict(history, index)` interface (`models.Forecaster`), so a
spotforecast2 forecaster can be added as one more model in the backtest.

## Layout

```
src/energy_forecast/
├── data/loader.py   # ENTSO-E actual load + day-ahead forecast, CSV cache, validation
├── features.py      # local-time calendar features (DST-safe), lags >= 48 h only
├── baselines.py     # ENTSO-E day-ahead forecast, seasonal naive (168 h)
├── models.py        # common Forecaster interface + LightGBM base model
├── backtest.py      # challenge-timed rolling-origin backtest, summary, per-day MAE
└── metrics.py       # MAE, RMSE, MAPE, bias
tests/               # metrics, DST days, no-leak checks, timing, loader
```

## Setup (Windows PowerShell, Python 3.13)

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
```

## API key

The ENTSO-E key is read from the `ENTSOE_API_KEY` environment variable. It is never
stored in this repository. It is only needed when data is not already cached in
`data/raw/`.

## Backtest timing (follows the challenge rules)

- **Target day D** = 00:00-23:00 local time (`Europe/Berlin`): 24 hours, or 23/25 on
  daylight-saving switch days.
- **Cutoff**: models only see actual load from before `--gap-hours` (default 12) ahead
  of D, i.e. D-1 12:00 local. This mimics a submission on D-1 when today's actuals
  are only partly published. All lag features are >= 48 h, so they are known then.
- **Refit** every 7 days on the trailing 2 years.
- **Pinned windows** (reproducible): data 2023-01-01 to 2026-10-19 (UTC), target days
  2026-09-21 to 2026-10-18.

## Run the backtest

```powershell
python -m energy_forecast.backtest
python -m energy_forecast.backtest --test-start 2026-06-01 --test-end 2026-07-01
```

Writes `results/backtest_summary.csv`, `backtest_daily_mae.csv` and
`backtest_predictions.csv`.

## Known limitations / open questions

- The challenge scores `DE` "Actual Load"; this code uses the `DE_LU` bidding zone.
  To be confirmed.
- The real data-availability cutoff on D-1 is assumed (12 h); to be confirmed.
- No holiday or weather features yet (spotforecast2-safe `ExogBuilder`,
  `WeatherService`).
- Short gaps (<= 2 h) are interpolated; the course prefers annotation with zero
  weights (spotforecast2-safe `get_missing_weights`).
