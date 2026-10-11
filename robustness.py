"""
Overfitting checks for the EMA-crossover backtest on QQQ.

Three tests:
  1. PARAMETER GRID   - does a WHOLE REGION of settings work, or just one lucky
                        cell? (a single magic number = overfit)
  2. OUT-OF-SAMPLE    - pick the "best" settings on the first half, then see how
                        they do on the second half they never saw.
  3. BOOTSTRAP        - resample history to get a confidence range for the
                        Sharpe ratio: is the edge real or just luck/noise?

Run:  python robustness.py
"""

import numpy as np
import pandas as pd

np.random.seed(0)
DAYS = 252


def load_price():
    return (pd.read_csv("data/QQQ.csv", parse_dates=["date"])
              .set_index("date")["close"])


def ema_cross_returns(price, fast, slow):
    """Long-only EMA crossover daily returns (signal uses only past data)."""
    f = price.ewm(span=fast).mean()
    s = price.ewm(span=slow).mean()
    signal = (f > s).astype(float)
    return (signal.shift(1) * price.pct_change()).dropna()


def sharpe(r):
    return r.mean() / r.std() * np.sqrt(DAYS) if r.std() > 0 else np.nan

def cagr(r):
    return (1 + r).prod() ** (DAYS / len(r)) - 1

def maxdd(r):
    c = (1 + r).cumprod()
    return (c / c.cummax() - 1).min()


# ── Test 1: parameter grid ────────────────────────────────────────────────
def test_grid(price, bh_sharpe):
    fasts = [10, 20, 30, 50, 75, 100]
    slows = [100, 150, 200, 250, 300]
    print("TEST 1 — PARAMETER GRID  (Sharpe ratio for each fast/slow EMA pair)")
    print("A robust edge = most of the grid works. One lucky cell = overfit.\n")

    header = "fast\\slow" + "".join(f"{s:>8}" for s in slows)
    print(header)
    sharpes = []
    for fa in fasts:
        row = f"{fa:>9}"
        for sl in slows:
            if fa >= sl:
                row += f"{'-':>8}"
                continue
            sh = sharpe(ema_cross_returns(price, fa, sl))
            sharpes.append(sh)
            row += f"{sh:>8.2f}"
        print(row)

    sharpes = np.array(sharpes)
    beat = (sharpes > bh_sharpe).mean()
    print(f"\nBuy & Hold Sharpe = {bh_sharpe:.2f}")
    print(f"Grid Sharpe: best {sharpes.max():.2f}, median {np.median(sharpes):.2f}, "
          f"worst {sharpes.min():.2f}")
    print(f"Share of settings that BEAT buy & hold: {beat:.0%}")
    verdict = ("ROBUST — works across the board, not a single magic number."
               if beat > 0.7 and np.median(sharpes) > bh_sharpe
               else "FRAGILE — only some settings work; treat with caution.")
    print(f"Verdict: {verdict}")


# ── Test 2: out-of-sample split ───────────────────────────────────────────
def test_oos(price):
    mid = price.index[len(price) // 2]
    print("\n\nTEST 2 — OUT-OF-SAMPLE  (fit on 1st half, judge on 2nd half)")
    print(f"In-sample:  {price.index[0].date()} .. {mid.date()}")
    print(f"Out-sample: {mid.date()} .. {price.index[-1].date()}\n")

    grid = [(fa, sl) for fa in (10, 20, 30, 50, 75, 100)
            for sl in (100, 150, 200, 250, 300) if fa < sl]

    # pick the best pair using ONLY the in-sample half
    best, best_is = None, -np.inf
    for fa, sl in grid:
        r = ema_cross_returns(price, fa, sl)
        s_is = sharpe(r[r.index < mid])
        if s_is > best_is:
            best, best_is = (fa, sl), s_is

    def split_sharpe(fa, sl):
        r = ema_cross_returns(price, fa, sl)
        return sharpe(r[r.index < mid]), sharpe(r[r.index >= mid])

    b_is, b_oos = split_sharpe(*best)
    d_is, d_oos = split_sharpe(50, 200)

    print(f"{'':28}{'in-sample':>12}{'out-of-sample':>15}")
    print(f"{'Best-fit ' + str(best):28}{b_is:>12.2f}{b_oos:>15.2f}")
    print(f"{'Robust default (50/200)':28}{d_is:>12.2f}{d_oos:>15.2f}")
    drop = b_is - b_oos
    print(f"\nOverfitting penalty (best-fit IS->OOS drop): {drop:+.2f} Sharpe")
    if b_oos > 0 and abs(d_oos - b_oos) < 0.3:
        print("Good sign: the cherry-picked best behaves like the plain default "
              "out-of-sample -> little overfitting, the edge is structural.")
    else:
        print("Caution: the in-sample winner fades out-of-sample -> some overfit.")


# ── Test 3: bootstrap Sharpe confidence interval ──────────────────────────
def test_bootstrap(price, n=2000, block=20):
    r = ema_cross_returns(price, 50, 200).values
    print("\n\nTEST 3 — BOOTSTRAP  (is the Sharpe real, or could it be luck?)")
    print(f"Resampling history {n} times in {block}-day blocks...\n")

    nblocks = len(r) // block
    out = []
    for _ in range(n):
        starts = np.random.randint(0, len(r) - block, size=nblocks)
        sample = np.concatenate([r[s:s + block] for s in starts])
        sd = sample.std()
        out.append(sample.mean() / sd * np.sqrt(DAYS) if sd > 0 else 0)
    out = np.array(out)
    lo, hi = np.percentile(out, [2.5, 97.5])
    print(f"EMA 50/200 Sharpe: {sharpe(ema_cross_returns(price,50,200)):.2f}")
    print(f"95% confidence range: {lo:.2f} to {hi:.2f}")
    print(f"Chance the true Sharpe is below 0: {(out < 0).mean():.1%}")
    if lo > 0:
        print("The whole range is above zero -> the edge is unlikely to be pure luck.")
    else:
        print("The range includes zero -> can't rule out luck on this single market.")


def main():
    price = load_price()
    bh = price.pct_change().dropna()
    print(f"QQQ  {price.index[0].date()} -> {price.index[-1].date()}\n")
    test_grid(price, sharpe(bh))
    test_oos(price)
    test_bootstrap(price)

    print("\n\nWHAT THIS CAN'T FIX (honest limits):")
    print("- Only ONE instrument, and QQQ is itself a known winner (selection bias).")
    print("  The real cure is testing the SAME rule on many markets/asset classes,")
    print("  which is exactly why Carver runs 100+ futures, not one index.")
    print("- One historical path. Bootstrap helps but isn't truly new data.")


if __name__ == "__main__":
    main()
