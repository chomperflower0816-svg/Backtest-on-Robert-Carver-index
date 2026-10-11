"""The full Carver system wired together, plus a simple backtest accountant.

    price -> vol -> EWMAC variations -> combine(+FDM) -> position -> P&L
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import BUSINESS_DAYS_IN_YEAR, ROOT_BDAYS_INYEAR
from .forecast import combine_forecasts
from .position import position_from_forecast, volatility_scalar
from .rules import ewmac_forecast
from .vol import daily_price_vol


@dataclass
class SystemConfig:
    # EWMAC variations (Lfast, Lslow). Default = slower, ETF-appropriate set.
    ewmac_variations: list[tuple[int, int]] = field(
        default_factory=lambda: [(8, 32), (16, 64), (32, 128), (64, 256)])
    forecast_weights: list[float] | None = None      # None => equal weights
    vol_span: int = 35                                # vol lookback (days)
    capital: float = 100_000.0
    vol_target_pct: float = 0.20                      # 20% annualised risk target
    cost_per_trade_bps: float = 1.0                   # round-turn cost, bps of traded notional

    @classmethod
    def from_yaml(cls, path: str = "config.yaml") -> "SystemConfig":
        """Build a config from the friendly, plain-English ``config.yaml`` box.

        Translates the everyday knobs (``risk_level``, ``reaction_speed`` ...)
        into the engine parameters, so a user never has to touch the jargon.
        """
        import yaml

        with open(path) as fh:
            raw = yaml.safe_load(fh)

        adv = raw.get("advanced", {}) or {}
        presets = raw.get("_speed_presets", {})

        # decide the EWMAC speed pairs
        if adv.get("use_custom_speeds"):
            speeds = adv.get("custom_speeds")
        else:
            word = str(raw.get("reaction_speed", "balanced")).lower()
            if word not in presets:
                raise ValueError(
                    f"reaction_speed must be one of {list(presets)}, got {word!r}")
            speeds = presets[word]
        speeds = [tuple(p) for p in speeds]

        return cls(
            ewmac_variations=speeds,
            vol_span=int(adv.get("volatility_lookback_days", 35)),
            capital=float(raw.get("capital", 100_000.0)),
            vol_target_pct=float(raw.get("risk_level", 0.20)),
            cost_per_trade_bps=float(raw.get("cost_bps", 1.0)),
        )


class CarverSystem:
    """Runs the staged pipeline and books a backtest on a single instrument."""

    def __init__(self, price: pd.Series, config: SystemConfig | None = None):
        self.price = price.astype(float).dropna()
        self.config = config or SystemConfig()
        self._run()

    def _run(self) -> None:
        cfg = self.config
        price = self.price

        # Stage 1: volatility (in price units, $/share/day)
        self.price_vol = daily_price_vol(price, span=cfg.vol_span)

        # Stage 2-3: EWMAC forecast per variation
        forecasts = {}
        for (lf, ls) in cfg.ewmac_variations:
            forecasts[f"ewmac{lf}_{ls}"] = ewmac_forecast(price, self.price_vol, lf, ls)
        self.forecasts = pd.DataFrame(forecasts)

        # Stage 4: combine variations (+ FDM)
        weights = None if cfg.forecast_weights is None else np.array(cfg.forecast_weights)
        self.combined_forecast, self.fdm = combine_forecasts(self.forecasts, weights)

        # Stage 5: volatility targeting -> position in shares
        self.vol_scalar = volatility_scalar(price, self.price_vol, cfg.capital,
                                            cfg.vol_target_pct)
        self.position = position_from_forecast(self.combined_forecast, self.vol_scalar)

        # ---- accounting ----------------------------------------------------
        # position decided at close t is held over t -> t+1 (lag to avoid look-ahead)
        held = self.position.shift(1)
        gross_pnl = held * price.diff()

        traded_shares = self.position.diff().abs().fillna(0.0)
        cost = traded_shares * price * (cfg.cost_per_trade_bps / 1e4)
        net_pnl = (gross_pnl - cost).fillna(0.0)

        self.gross_pnl = gross_pnl
        self.cost = cost
        self.net_pnl = net_pnl
        self.returns = net_pnl / cfg.capital                      # daily % return on capital
        self.account_curve = cfg.capital + net_pnl.cumsum()

    # ------------------------------------------------------------------ stats
    def buy_and_hold_returns(self) -> pd.Series:
        return self.price.pct_change().reindex(self.returns.index).fillna(0.0)

    def stats(self) -> dict:
        return performance_stats(self.returns, label="Carver EWMAC")

    def summary(self) -> pd.DataFrame:
        rows = {
            "Carver EWMAC": performance_stats(self.returns),
            "Buy & Hold QQQ": performance_stats(self.buy_and_hold_returns()),
        }
        return pd.DataFrame(rows).T


def performance_stats(returns: pd.Series, label: str | None = None) -> dict:
    """Standard annualised performance stats from a daily return series."""
    r = returns.dropna()
    if len(r) == 0:
        return {}
    ann_return = r.mean() * BUSINESS_DAYS_IN_YEAR
    ann_vol = r.std() * ROOT_BDAYS_INYEAR
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

    curve = (1 + r).cumprod()
    running_max = curve.cummax()
    drawdown = curve / running_max - 1.0
    max_dd = drawdown.min()

    years = len(r) / BUSINESS_DAYS_IN_YEAR
    cagr = curve.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan

    stats = {
        "CAGR": cagr,
        "Ann.Return": ann_return,
        "Ann.Vol": ann_vol,
        "Sharpe": sharpe,
        "MaxDrawdown": max_dd,
        "Skew": r.skew(),
        "% pos days": (r > 0).mean(),
    }
    return stats
