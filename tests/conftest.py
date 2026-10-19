import numpy as np
import pandas as pd
import pytest


def _weekly(start: str, days: int) -> pd.Series:
    """Hourly UTC load with an exact weekly pattern (no noise)."""
    index = pd.date_range(start, periods=days * 24, freq="h", tz="UTC")
    hour_of_week = np.arange(len(index)) % 168
    values = 50_000 + 10_000 * np.sin(2 * np.pi * hour_of_week / 24) + 50 * hour_of_week
    return pd.Series(values, index=index, name="load_mw")


@pytest.fixture
def weekly_load() -> pd.Series:
    """60 days from 2024-01-01."""
    return _weekly("2024-01-01", 60)


@pytest.fixture
def weekly_load_spring() -> pd.Series:
    """45 days from 2024-03-01, covering the 2024-03-31 DST switch."""
    return _weekly("2024-03-01", 45)
