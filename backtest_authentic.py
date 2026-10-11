"""
Authentic event-driven backtest of the monthly MA(N) rule on QQQ, with real
costs and an in-sample / out-of-sample split.

What "authentic" means here:
  - a real cash ledger in USD, buying WHOLE shares with the cash available
  - a flat $3 commission deducted at each actual buy and each actual sell
  - NAV marked to market every day (so drawdown is true daily-NAV drawdown)

On costs, deliberately:
  - QQQ's 0.20% expense ratio is NOT added, because it is already embedded in
    QQQ's traded price (the fund deducts it from its own NAV daily). Adding it
    again would double-count.
  - Dividends (~0.6%/yr) are NOT added, which makes results conservative.
  - One-time HKD->USD FX conversion (~0.1-0.5%) is not modelled.

Rule (version A): on the first trading day of each month, go long if the day's
OPEN is above the average of the last N month-end closes; else hold cash.

Run:  python backtest_authentic.py
"""

import numpy as np
import pandas as pd

# ── settings ─────────────────────────────────────────────────────────────
START_HKD   = 100_000.0
HKD_PER_USD = 7.8          # HKD is pegged to USD
COMMISSION  = 3.0          # USD per trade (each buy and each sell)
OHLC_CSV    = "data/QQQ.csv"        # bundled: date,open,close
SPLIT       = "2012-03-10"          # in-sample before, out-of-sample after
# ─────────────────────────────────────────────────────────────────────────


def load():
    df = pd.read_csv(OHLC_CSV, parse_dates=["date"]).set_index("date")
    return df.index, df["open"].values, df["close"].values, df["close"].resample("ME").last()


def month_start_mask(idx):
    pid = idx.to_period("M")
    return np.array([True] + [pid[i] != pid[i - 1] for i in range(1, len(idx))])


def bench_for(idx, month_end_close, n):
    """Average of the last N month-end closes, known before the month starts."""
    ma = month_end_close.rolling(n).mean()
    return ma.reindex(idx, method="ffill").shift(1).values


def run(idx, op, cl, bench, is_first, start_cash, commission):
    cash, shares, trades = start_cash, 0, 0
    nav = np.empty(len(idx))
    for t in range(len(idx)):
        if is_first[t] and not np.isnan(bench[t]):
            go_long = op[t] > bench[t]
            if go_long and shares == 0:
                qty = int((cash - commission) // op[t])
                if qty > 0:
                    cash -= qty * op[t] + commission
                    shares += qty
                    trades += 1
            elif (not go_long) and shares > 0:
                cash += shares * op[t] - commission
                shares = 0
                trades += 1
        nav[t] = cash + shares * cl[t]
    return pd.Series(nav, index=idx), trades


def buy_hold(idx, op, cl, start_cash, commission):
    shares = int((start_cash - commission) // op[0])
    cash = start_cash - shares * op[0] - commission
    return pd.Series(cash + shares * cl, index=idx)


def stats(nav, mask):
    n = nav[mask]
    r = n.pct_change().dropna()
    return {
        "CAGR": (n.iloc[-1] / n.iloc[0]) ** (252 / len(n)) - 1,
        "Sharpe": r.mean() / r.std() * np.sqrt(252),
        "MaxDD": (n / n.cummax() - 1).min(),
    }


def main():
    idx, op, cl, me = load()
    start = START_HKD / HKD_PER_USD
    is_first = month_start_mask(idx)
    split = pd.Timestamp(SPLIT)
    IS, OOS, ALL = idx < split, idx >= split, np.ones(len(idx), bool)

    # choose N on IN-SAMPLE only, then judge out-of-sample (no lookahead in selection)
    best, best_sh = None, -9
    for n in range(6, 16):
        nav, _ = run(idx, op, cl, bench_for(idx, me, n), is_first, start, COMMISSION)
        s = stats(nav, IS)["Sharpe"]
        if s > best_sh:
            best, best_sh = n, s

    print(f"Start {START_HKD:,.0f} HKD / {HKD_PER_USD} = ${start:,.2f}   "
          f"({idx[0].date()} -> {idx[-1].date()})")
    print(f"In-sample: ..{split.date()} | Out-of-sample: {split.date()}..   "
          f"| N picked on in-sample = MA{best}\n")

    hdr = f"{'CAGR':>7}{'Shrp':>7}{'MaxDD':>7}"
    print(f"{'':18}{'IN-SAMPLE':>21}|{'OUT-OF-SAMPLE':>22}|{'FULL 26y':>21}{'trds':>6}")
    print(f"{'':18}{hdr:>21}|{hdr:>22}|{hdr:>21}")
    for n in sorted({best, 11, 12}):
        nav, tr = run(idx, op, cl, bench_for(idx, me, n), is_first, start, COMMISSION)
        a, b, c = stats(nav, IS), stats(nav, OOS), stats(nav, ALL)
        print(f"{'MA'+str(n):18}"
              f"{a['CAGR']:>7.1%}{a['Sharpe']:>7.2f}{a['MaxDD']:>7.0%}|"
              f"{b['CAGR']:>7.1%}{b['Sharpe']:>7.2f}{b['MaxDD']:>7.0%}|"
              f"{c['CAGR']:>7.1%}{c['Sharpe']:>7.2f}{c['MaxDD']:>7.0%}{tr:>6}")
    bh = buy_hold(idx, op, cl, start, COMMISSION)
    a, b, c = stats(bh, IS), stats(bh, OOS), stats(bh, ALL)
    print(f"{'Buy & Hold':18}"
          f"{a['CAGR']:>7.1%}{a['Sharpe']:>7.2f}{a['MaxDD']:>7.0%}|"
          f"{b['CAGR']:>7.1%}{b['Sharpe']:>7.2f}{b['MaxDD']:>7.0%}|"
          f"{c['CAGR']:>7.1%}{c['Sharpe']:>7.2f}{c['MaxDD']:>7.0%}{'-':>6}")

    nav11, tr11 = run(idx, op, cl, bench_for(idx, me, 11), is_first, start, COMMISSION)
    print(f"\nMA11 final: ${nav11.iloc[-1]:,.0f} = {nav11.iloc[-1]*HKD_PER_USD:,.0f} HKD "
          f"({tr11} trades, ${tr11*COMMISSION:.0f} commissions)")


if __name__ == "__main__":
    main()
