"""Stage 5 - volatility targeting / position sizing.

Turns the combined forecast into a position (shares) that targets a *constant
risk* rather than a constant size. When markets get more volatile, the
volatility scalar shrinks and the position shrinks with it.
"""

from __future__ import annotations

import pandas as pd

from . import ROOT_BDAYS_INYEAR


def volatility_scalar(price: pd.Series, price_vol: pd.Series, capital: float,
                      vol_target_pct: float) -> pd.Series:
    """Number of shares that corresponds to an *average* forecast (=10).

    daily_cash_vol_target = capital * vol_target / sqrt(256)
    instrument_value_vol  = daily $ std of one share = price_vol (price units)
    vol_scalar            = daily_cash_vol_target / instrument_value_vol
    """
    annual_cash_vol_target = capital * vol_target_pct
    daily_cash_vol_target = annual_cash_vol_target / ROOT_BDAYS_INYEAR
    instrument_value_vol = price_vol  # already $/share per day
    return daily_cash_vol_target / instrument_value_vol


def position_from_forecast(combined_forecast: pd.Series, vol_scalar: pd.Series,
                           idm: float = 1.0, instrument_weight: float = 1.0
                           ) -> pd.Series:
    """Position in shares.

    subsystem position = vol_scalar * forecast / 10
    portfolio position  = subsystem position * instrument_weight * IDM
    (both 1.0 for a single instrument).
    """
    subsystem = vol_scalar * combined_forecast / 10.0
    return subsystem * instrument_weight * idm
