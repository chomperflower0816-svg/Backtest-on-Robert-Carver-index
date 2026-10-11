"""Stage 2-3 - trading rules, forecast scaling and capping.

The core (and, for a single cash ETF, the only sensible) rule is EWMAC:
the difference of a fast and a slow exponentially weighted moving average,
vol-normalised, scaled so |forecast| averages ~10, and capped at +/-20.
"""

from __future__ import annotations

import pandas as pd

FORECAST_CAP = 20.0

# Carver's published forecast scalars (Systematic Trading App. B / pysystemtrade).
# Key = (Lfast, Lslow). Each halving of speed divides the scalar by ~sqrt(2).
EWMAC_FORECAST_SCALARS = {
    (2, 8): 10.6,
    (4, 16): 7.5,
    (8, 32): 5.3,
    (16, 64): 3.75,
    (32, 128): 2.65,
    (64, 256): 1.91,
}


def ewmac_forecast_scalar(Lfast: int, Lslow: int) -> float:
    """Look up Carver's forecast scalar, interpolating on a log2 grid if needed."""
    if (Lfast, Lslow) in EWMAC_FORECAST_SCALARS:
        return EWMAC_FORECAST_SCALARS[(Lfast, Lslow)]
    # fall back to the sqrt(2)-per-doubling relationship anchored at (8,32)=5.3
    import math
    steps = math.log2(Lfast / 8.0)
    return 5.3 / (2 ** (0.5 * steps))


def raw_ewmac(price: pd.Series, Lfast: int, Lslow: int) -> pd.Series:
    """Fast EWMA minus slow EWMA, in price units."""
    fast = price.ewm(span=Lfast, min_periods=Lfast).mean()
    slow = price.ewm(span=Lslow, min_periods=Lslow).mean()
    return fast - slow


def cap_forecast(forecast: pd.Series, cap: float = FORECAST_CAP) -> pd.Series:
    return forecast.clip(lower=-cap, upper=cap)


def ewmac_forecast(price: pd.Series, vol: pd.Series, Lfast: int, Lslow: int,
                   scalar: float | None = None, cap: float = FORECAST_CAP) -> pd.Series:
    """Full EWMAC forecast: raw -> vol-adjust -> scale -> cap.

    ``vol`` is the daily price-change volatility (same units as ``price``), so
    ``raw / vol`` is dimensionless (in standard deviations of daily move).
    """
    raw = raw_ewmac(price, Lfast, Lslow)
    vol_adj = raw / vol
    if scalar is None:
        scalar = ewmac_forecast_scalar(Lfast, Lslow)
    return cap_forecast(vol_adj * scalar, cap)
