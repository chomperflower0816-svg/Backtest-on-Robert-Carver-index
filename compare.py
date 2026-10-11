"""
Compare the strategies on QQQ, side by side, with drawdown TIMING:
  - how deep the worst drawdown was
  - how long it took to fall from the previous high to the bottom
  - how long it then took to recover back to that high

Run:  python compare.py
"""

import pandas as pd

from carver import CarverSystem, SystemConfig


def load_price():
    return (pd.read_csv("data/QQQ.csv", parse_dates=["date"])
              .set_index("date")["close"])


# ── the strategies, each returns a daily-return series ───────────────────
def buy_and_hold(price):
    return price.pct_change().dropna()


def ema_cross(price, fast=50, slow=200, long_short=False):
    f, s = price.ewm(span=fast).mean(), price.ewm(span=slow).mean()
    down = -1 if long_short else 0
    signal = pd.Series(1, index=price.index).where(f > s, down)
    return (signal.shift(1) * price.pct_change()).dropna()


def carver_trend(price):
    sys = CarverSystem(price, SystemConfig())     # vol-targeted EWMAC ensemble
    return sys.returns.dropna()


# ── metrics ──────────────────────────────────────────────────────────────
def perf(r):
    curve = (1 + r).cumprod()
    yrs = len(r) / 252
    return {
        "Total return": curve.iloc[-1] - 1,
        "Yearly (CAGR)": curve.iloc[-1] ** (1 / yrs) - 1,
        "Yearly vol": r.std() * (252 ** 0.5),
        "Sharpe": (r.mean() * 252) / (r.std() * (252 ** 0.5)),
        "Max drawdown": (curve / curve.cummax() - 1).min(),
    }


def drawdown_timing(r):
    """Find the worst drawdown and time it."""
    curve = (1 + r).cumprod()
    dd = curve / curve.cummax() - 1
    trough = dd.idxmin()                       # date of the deepest point
    peak = curve.loc[:trough].idxmax()         # previous high-water mark
    peak_level = curve.loc[peak]

    after = curve.loc[trough:]
    recovered = after[after >= peak_level]
    recovery = recovered.index[0] if len(recovered) else None

    def yrs(d0, d1):
        return (d1 - d0).days / 365.25

    return {
        "peak": peak.date(),
        "trough": trough.date(),
        "fall_yrs": yrs(peak, trough),                       # HWM -> bottom
        "recover_yrs": yrs(trough, recovery) if recovery is not None else None,
        "underwater_yrs": yrs(peak, recovery) if recovery is not None else None,
        "recovered": recovery is not None,
    }


def main():
    price = load_price()
    strategies = {
        "Buy & Hold":          buy_and_hold(price),
        "EMA 50/200 (long)":   ema_cross(price, long_short=False),
        "EMA 50/200 (L/S)":    ema_cross(price, long_short=True),
        "Carver trend (vol)":  carver_trend(price),
    }

    print(f"QQQ  {price.index[0].date()} -> {price.index[-1].date()}\n")

    # ---- performance table ----
    rows = {name: perf(r) for name, r in strategies.items()}
    table = pd.DataFrame(rows).T
    pct = ["Total return", "Yearly (CAGR)", "Yearly vol", "Max drawdown"]
    fmt = table.copy()
    for c in pct:
        fmt[c] = fmt[c].map(lambda x: f"{x:.1%}")
    fmt["Sharpe"] = table["Sharpe"].map(lambda x: f"{x:.2f}")
    print("PERFORMANCE")
    print(fmt.to_string())

    # ---- worst-drawdown timing ----
    print("\nWORST DRAWDOWN — timing")
    print(f"{'strategy':22}{'depth':>8}{'HWM->bottom':>14}"
          f"{'bottom->recover':>17}{'total underwater':>18}")
    for name, r in strategies.items():
        d = drawdown_timing(r)
        depth = f"{rows[name]['Max drawdown']:.0%}"
        fall = f"{d['fall_yrs']:.1f} yr"
        if d["recovered"]:
            rec = f"{d['recover_yrs']:.1f} yr"
            uw = f"{d['underwater_yrs']:.1f} yr"
        else:
            rec = uw = "not yet"
        print(f"{name:22}{depth:>8}{fall:>14}{rec:>17}{uw:>18}")
        print(f"{'':22}(peak {d['peak']} -> bottom {d['trough']})")


if __name__ == "__main__":
    main()
