"""Sanity checks that the pipeline behaves the way Carver's framework intends."""

import numpy as np
import pandas as pd
import pytest

from carver import CarverSystem, SystemConfig
from carver.rules import ewmac_forecast, FORECAST_CAP
from carver.vol import daily_price_vol
from data_loader import load_qqq


@pytest.fixture(scope="module")
def price():
    return load_qqq("csv")


def test_data_loads(price):
    assert len(price) > 5000
    assert price.index.is_monotonic_increasing
    assert (price > 0).all()


def test_forecast_is_capped(price):
    vol = daily_price_vol(price)
    f = ewmac_forecast(price, vol, 16, 64)
    assert f.abs().max() <= FORECAST_CAP + 1e-9


def test_scaled_forecast_averages_about_ten(price):
    """Carver scales forecasts so |forecast| averages ~10 (loose 6-16 band)."""
    vol = daily_price_vol(price)
    f = ewmac_forecast(price, vol, 16, 64).dropna()
    avg_abs = f.abs().mean()
    assert 6.0 < avg_abs < 16.0, avg_abs


def test_volatility_targeting_is_in_the_ballpark(price):
    """Realised vol should land near the target (single instrument, so loose)."""
    cfg = SystemConfig(vol_target_pct=0.20)
    sys = CarverSystem(price, cfg)
    realised = sys.returns.std() * np.sqrt(256)
    assert 0.10 < realised < 0.35, realised


def test_position_scales_with_forecast_sign(price):
    sys = CarverSystem(price)
    # where forecast is long, position should be long; where short, short
    aligned = np.sign(sys.position) == np.sign(sys.combined_forecast)
    assert aligned[sys.combined_forecast.abs() > 1].mean() > 0.99


def test_fdm_reasonable(price):
    sys = CarverSystem(price)
    assert 1.0 <= sys.fdm <= 2.5
