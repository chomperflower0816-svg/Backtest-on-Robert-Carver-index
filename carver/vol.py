"""Stage 1 - volatility estimation.

Carver uses one volatility number to normalise both the forecast and the
position size. It is an exponentially weighted standard deviation of *daily
price changes* (price units, e.g. $/share), with a floor so it never collapses
to zero in unusually quiet markets (pysystemtrade's ``robust_vol_calc``).
"""

from __future__ import annotations

import pandas as pd


def daily_price_vol(price: pd.Series, span: int = 35, min_periods: int = 10,
                    vol_floor_quantile: float = 0.05,
                    vol_floor_window: int = 500) -> pd.Series:
    """Exponentially weighted std of daily price changes, with a floor.

    Parameters
    ----------
    price : daily price series.
    span  : EW span in days (Carver/pysystemtrade default ~35).
    vol_floor_quantile / vol_floor_window :
        the volatility is floored at the rolling ``quantile`` of its own recent
        history, so a freakishly calm patch can't blow up position size.

    Returns a series in the same units as ``price`` (price changes).
    """
    price_change = price.diff()
    vol = price_change.ewm(span=span, min_periods=min_periods).std()

    # robust floor: never let vol fall below a low quantile of its recent self
    vol_min = vol.rolling(vol_floor_window, min_periods=1).quantile(vol_floor_quantile)
    vol = vol.clip(lower=vol_min)
    return vol


def percent_vol(price: pd.Series, **kwargs) -> pd.Series:
    """Daily volatility expressed as a fraction of price (return vol)."""
    return daily_price_vol(price, **kwargs) / price.abs()
