"""Run Robert Carver's EWMAC trend system on QQQ and report results.

Usage:
    python run_qqq.py                      # bundled CSV, settings from config.yaml
    python run_qqq.py --source yfinance    # fresh data (needs network)
    python run_qqq.py --vol-target 0.25 --capital 250000
    python run_qqq.py --no-plot
"""

from __future__ import annotations

import argparse

import pandas as pd

from carver import CarverSystem, SystemConfig
from data_loader import load_qqq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Carver EWMAC CTA system on QQQ")
    p.add_argument("--config", default="config.yaml",
                   help="the friendly settings file (your 'control box')")
    p.add_argument("--source", default="csv",
                   help="csv | yfinance | nasdaq | path/to.csv")
    # the three below override config.yaml when given; otherwise the file wins
    p.add_argument("--capital", type=float, default=None)
    p.add_argument("--vol-target", type=float, default=None,
                   help="override risk_level (e.g. 0.20 = 20%%)")
    p.add_argument("--cost-bps", type=float, default=None)
    p.add_argument("--start", default=None, help="optional backtest start date")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--plot-file", default="qqq_account_curve.svg")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")

    price = load_qqq(source=args.source)
    if args.start:
        price = price[price.index >= args.start]
    print(f"Loaded QQQ: {len(price)} days, "
          f"{price.index.min().date()} -> {price.index.max().date()}\n")

    # load the friendly control box, then apply any CLI overrides
    try:
        cfg = SystemConfig.from_yaml(args.config)
        print(f"Settings from {args.config}")
    except FileNotFoundError:
        cfg = SystemConfig()
        print(f"(no {args.config} found - using built-in defaults)")
    if args.capital is not None:
        cfg.capital = args.capital
    if args.vol_target is not None:
        cfg.vol_target_pct = args.vol_target
    if args.cost_bps is not None:
        cfg.cost_per_trade_bps = args.cost_bps

    system = CarverSystem(price, cfg)

    print(f"EWMAC variations : {cfg.ewmac_variations}")
    print(f"Forecast Div Mult: {system.fdm:.3f}")
    print(f"Vol target       : {cfg.vol_target_pct:.0%} on ${cfg.capital:,.0f}\n")

    print("Latest state:")
    last = price.index[-1].date()
    print(f"  date            : {last}")
    print(f"  price           : {price.iloc[-1]:.2f}")
    print(f"  combined forecast: {system.combined_forecast.iloc[-1]:+.1f}  "
          f"(+10 = avg long, -10 = avg short)")
    print(f"  position        : {system.position.iloc[-1]:,.1f} shares "
          f"(${system.position.iloc[-1]*price.iloc[-1]:,.0f} notional)\n")

    print("Performance (net of costs):")
    print(system.summary().to_string())

    if not args.no_plot:
        try:
            _plot(system, args.plot_file)
            print(f"\nSaved account curve -> {args.plot_file}")
        except Exception as exc:  # plotting is optional
            print(f"\n(plot skipped: {exc})")


def _plot(system: CarverSystem, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bh = (1 + system.buy_and_hold_returns()).cumprod() * system.config.capital
    strat = (1 + system.returns).cumprod() * system.config.capital

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(strat.index, strat, label="Carver EWMAC (vol-targeted)", lw=1.3)
    ax1.plot(bh.index, bh, label="Buy & Hold QQQ", lw=1.3, alpha=0.8)
    ax1.set_yscale("log")
    ax1.set_ylabel("Capital ($, log)")
    ax1.set_title("Robert Carver EWMAC trend system vs Buy & Hold - QQQ")
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.3)

    ax2.plot(system.combined_forecast.index, system.combined_forecast,
             color="purple", lw=0.7)
    ax2.axhline(0, color="k", lw=0.6)
    ax2.axhline(10, color="g", lw=0.5, ls="--")
    ax2.axhline(-10, color="r", lw=0.5, ls="--")
    ax2.set_ylabel("Combined\nforecast")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=110)


if __name__ == "__main__":
    main()
