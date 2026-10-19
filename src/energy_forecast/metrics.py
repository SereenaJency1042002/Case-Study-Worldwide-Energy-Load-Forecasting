"""Point-forecast error metrics. MAE is the case study's primary metric."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

ArrayLike = npt.ArrayLike


def _pair(actual: ArrayLike, forecast: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    if a.shape != f.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {f.shape}")
    if a.size == 0 or np.isnan(a).any() or np.isnan(f).any():
        raise ValueError("inputs must be non-empty and contain no NaN")
    return a, f


def mae(actual: ArrayLike, forecast: ArrayLike) -> float:
    a, f = _pair(actual, forecast)
    return float(np.mean(np.abs(f - a)))


def rmse(actual: ArrayLike, forecast: ArrayLike) -> float:
    a, f = _pair(actual, forecast)
    return float(np.sqrt(np.mean((f - a) ** 2)))


def mape(actual: ArrayLike, forecast: ArrayLike) -> float:
    """Mean absolute percentage error in percent."""
    a, f = _pair(actual, forecast)
    if (a == 0).any():
        raise ValueError("MAPE is undefined when actual contains zeros")
    return float(100 * np.mean(np.abs((f - a) / a)))


def bias(actual: ArrayLike, forecast: ArrayLike) -> float:
    """Mean error; positive means over-forecasting."""
    a, f = _pair(actual, forecast)
    return float(np.mean(f - a))


def compute_all(actual: ArrayLike, forecast: ArrayLike) -> dict[str, float]:
    return {
        "mae": mae(actual, forecast),
        "rmse": rmse(actual, forecast),
        "mape": mape(actual, forecast),
        "bias": bias(actual, forecast),
    }
