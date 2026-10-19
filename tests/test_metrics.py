import pytest

from energy_forecast.metrics import bias, compute_all, mae, mape, rmse


def test_known_values():
    actual = [100.0, 200.0]
    forecast = [110.0, 180.0]
    assert mae(actual, forecast) == 15.0
    assert rmse(actual, forecast) == pytest.approx((250.0) ** 0.5)
    assert mape(actual, forecast) == pytest.approx(10.0)
    assert bias(actual, forecast) == -5.0
    assert set(compute_all(actual, forecast)) == {"mae", "rmse", "mape", "bias"}


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        mae([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        mae([1.0, float("nan")], [1.0, 2.0])
    with pytest.raises(ValueError):
        mape([0.0, 1.0], [1.0, 1.0])
