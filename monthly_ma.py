"""
Monthly moving-average trend filter on QQQ (the strongest simple rule found).

At each MONTH-END: if the monthly close is above its N-month moving average,
hold QQQ next month; otherwise sit in cash. You only ever look at month-end
closes (~1-2 trades a year) -- but performance, and especially DRAWDOWN, are
measured on the DAILY NAV, because while you hold QQQ your equity moves every
day and can dip intra-month well below the month-end marks.

This is the classic long-horizon time-series-momentum rule (cf. Faber's
10-month tactical filter). N in the 10-14 month range all work; 11 was the
in-sample best, 12 is the less cherry-picked literature standard.

Run:  python monthly_ma.py
"""

import numpy as np
import pandas as pd

N_MONTHS = 11      # length of the monthly moving average (try 10-14)


def load_daily():
    return pd.read_csv("data/QQQ.csv", parse_dates=["date"]).set_index("date")["close"]


def monthly_signal(daily, n):
    """Long/cash decision made at each month-end (1 = hold, 0 = cash)."""
    monthly = daily.resample("ME").last()
    ma = monthly.rolling(n).mean()
    return (monthly > ma).astype(float)


def daily_returns(daily, signal):
    """Map the month-end decision onto a DAILY position (no lookahead): the
    signal set at month-end M takes effect the next trading day and is held
    until the next month-end. Returns the daily strategy return series."""
    pos = signal.reindex(daily.index, method="ffill").shift(1)
    return (pos * daily.pct_change()).dropna()


def metrics(r):
    """All measured on the DAILY NAV (252/yr)."""
    c = (1 + r).cumprod()
    hwm = c.cummax()
    dd = c / hwm - 1
    trough = dd.idxmin()
    peak = c.loc[:trough].idxmax()
    after = c.loc[trough:]
    up = after[after >= c.loc[peak]]
    rec = up.index[0] if len(up) else None
    b2h = (rec - trough).days / 365.25 if rec is not None else np.nan
    hd = c.index[c >= hwm * (1 - 1e-12)]
    luw = (np.diff(hd.values).astype("timedelta64[D]").astype(int).max() / 365.25
           if len(hd) > 1 else np.nan)
    return {
        "CAGR": (1 + r).prod() ** (252 / len(r)) - 1,
        "Sharpe": r.mean() / r.std() * np.sqrt(252),
        "MaxDD": dd.min(),
        "bottom_to_hwm": b2h,
        "longest_uw": luw,
    }


def show(name, m, extra=""):
    print(f"{name:>16}{m['CAGR']:>8.1%}{m['Sharpe']:>8.2f}{m['MaxDD']:>8.0%}"
          f"{m['bottom_to_hwm']:>10.1f}yr{m['longest_uw']:>8.1f}yr  {extra}")


def main():
    daily = load_daily()
    print(f"QQQ  {daily.index[0].date()} -> {daily.index[-1].date()}   "
          f"(drawdown on DAILY NAV)\n")
    print(f"{'':16}{'CAGR':>8}{'Sharpe':>8}{'MaxDD':>8}{'bot->HWM':>12}{'longUW':>8}")

    sig = monthly_signal(daily, N_MONTHS)
    r = daily_returns(daily, sig)
    trades = int((sig.diff().abs() > 0).sum())
    yrs = len(r) / 252
    show(f"MA{N_MONTHS} monthly", metrics(r), f"{trades/yrs:.1f} trades/yr")
    show("Buy & Hold", metrics(daily.pct_change().dropna()))

    print("\nRobustness across nearby lengths (daily-NAV drawdown):")
    for n in (8, 10, 11, 12, 14):
        rr = daily_returns(daily, monthly_signal(daily, n))
        show(f"MA{n}", metrics(rr))


if __name__ == "__main__":
    main()
