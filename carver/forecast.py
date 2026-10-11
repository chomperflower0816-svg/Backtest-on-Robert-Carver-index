"""Stage 4 - combine rule variations into one forecast.

Combined forecast = (weighted average of capped variation forecasts) x FDM,
re-capped at +/-20. The Forecast Diversification Multiplier (FDM) restores the
average back to ~10, because averaging positively-correlated forecasts shrinks
their magnitude.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .rules import FORECAST_CAP, cap_forecast


def forecast_diversification_multiplier(forecasts: pd.DataFrame,
                                        weights: np.ndarray,
                                        cap: float = 2.5) -> float:
    """FDM = 1 / sqrt(w' C w), capped.

    ``C`` is the correlation matrix of the variation forecasts. Negative
    correlations are floored at 0 (Carver: don't reward spurious diversification).
    """
    corr = forecasts.corr().values
    corr = np.nan_to_num(corr, nan=0.0)
    corr = np.clip(corr, 0.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    w = np.asarray(weights, dtype=float)
    variance = float(w @ corr @ w)
    if variance <= 0:
        return 1.0
    fdm = 1.0 / np.sqrt(variance)
    return float(min(fdm, cap))


def combine_forecasts(forecasts: pd.DataFrame,
                      weights: np.ndarray | None = None,
                      fdm: float | None = None,
                      cap: float = FORECAST_CAP):
    """Weighted-average the variation forecasts and apply the FDM.

    Returns ``(combined_forecast, fdm_used)``.
    """
    n = forecasts.shape[1]
    if weights is None:
        weights = np.full(n, 1.0 / n)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()

    if fdm is None:
        fdm = forecast_diversification_multiplier(forecasts, weights)

    weighted = forecasts.mul(weights, axis=1).sum(axis=1, min_count=1)
    combined = cap_forecast(weighted * fdm, cap)
    return combined, fdm
