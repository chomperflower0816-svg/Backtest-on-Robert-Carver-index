"""
Sweep many EMA-crossover settings on QQQ, ranked by TIME UNDERWATER
(how long you're stuck below a previous high) — not just return.

For each fast/slow pair (long-only) it reports:
  CAGR, Sharpe, Max drawdown,
  bottom->HWM  = recovery time of the WORST drawdown,
  longest UW   = the single longest stretch ever spent below a prior high.

Run:  python ema_sweep.py
"""

import numpy as np
import pandas as pd

DAYS = 252


def load_price():
    return (pd.read_csv("data/QQQ.csv", parse_dates=["date"])
              .set_index("date")["close"])


def ema_returns(price, fast, slow):
    f, s = price.ewm(span=fast).mean(), price.ewm(span=slow).mean()
    sig = (f > s).astype(float)
    return (sig.shift(1) * price.pct_change()).dropna()


def metrics(r):
    c = (1 + r).cumprod()
    hwm = c.cummax()
    dd = c / hwm - 1

    # worst drawdown: bottom -> recovery to a new high
    trough = dd.idxmin()
    peak = c.loc[:trough].idxmax()
    lvl = c.loc[peak]
    after = c.loc[trough:]
    up = after[after >= lvl]
    recover = up.index[0] if len(up) else None
    bottom_to_hwm = (recover - trough).days / 365.25 if recover is not None else np.nan

    # longest underwater stretch anywhere: max gap between new highs
    high_days = c.index[c >= hwm * (1 - 1e-12)]
    if len(high_days) > 1:
        gaps = np.diff(high_days.values).astype("timedelta64[D]").astype(int)
        longest_uw = gaps.max() / 365.25
    else:
        longest_uw = np.nan

    return {
        "CAGR": (1 + r).prod() ** (DAYS / len(r)) - 1,
        "Sharpe": r.mean() / r.std() * np.sqrt(DAYS),
        "MaxDD": dd.min(),
        "bottom_to_hwm": bottom_to_hwm,
        "longest_uw": longest_uw,
    }


def main():
    price = load_price()
    fasts = [5, 10, 20, 30, 50]
    slows = [40, 60, 80, 100, 150, 200]
    rows = []
    for fa in fasts:
        for sl in slows:
            if fa >= sl:
                continue
            m = metrics(ema_returns(price, fa, sl))
            rows.append({"fast": fa, "slow": sl, **m})

    df = pd.DataFrame(rows)

    # benchmark
    bh = metrics(price.pct_change().dropna())

    df = df.sort_values("longest_uw")  # best = shortest time stuck underwater

    print(f"QQQ  {price.index[0].date()} -> {price.index[-1].date()}")
    print("Sorted by shortest 'longest underwater' stretch (best at top)\n")
    print(f"{'fast':>4}{'slow':>5}{'CAGR':>8}{'Sharpe':>8}{'MaxDD':>8}"
          f"{'bottom->HWM':>13}{'longest UW':>12}")
    for _, x in df.iterrows():
        print(f"{int(x['fast']):>4}{int(x['slow']):>5}{x['CAGR']:>8.1%}"
              f"{x['Sharpe']:>8.2f}{x['MaxDD']:>8.0%}"
              f"{x['bottom_to_hwm']:>11.1f}yr{x['longest_uw']:>10.1f}yr")
    print(f"\n{'Buy & Hold':>9}{bh['CAGR']:>8.1%}{bh['Sharpe']:>8.2f}"
          f"{bh['MaxDD']:>8.0%}{bh['bottom_to_hwm']:>11.1f}yr{bh['longest_uw']:>10.1f}yr")


if __name__ == "__main__":
    main()
