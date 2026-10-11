"""
In-sample / out-of-sample RATIO analysis for the monthly MA(N) rule, over the
last ~30 years of the Nasdaq-100.

Data: data/QQQ_NDX_spliced.csv -- the Nasdaq-100 index (^NDX) for the pre-QQQ
years, scaled to QQQ at the 1999 junction, then real QQQ 1999->2025. This is a
PROXY-extended series (the index is not directly tradable and excludes fees /
dividends) used only to get a longer history for validation.

What it shows: how the out-of-sample conclusion depends on WHERE you split.
Key methodological point -- you must NOT pick the split ratio that flatters the
result (that is snooping the split itself). Choose it a priori so the
out-of-sample window is long enough to contain at least one full market cycle.

Run:  python is_oos_ratio.py
"""

import numpy as np
import pandas as pd

START = "1995-03-11"     # last ~30 years
N_GRID = range(8, 15)    # candidate MA lengths to "optimize" in-sample


def load():
    df = pd.read_csv("data/QQQ_NDX_spliced.csv", parse_dates=["date"]).set_index("date")
    return df[df.index >= START]


def signal(df, n):
    op, cl = df["open"], df["close"]
    me = cl.resample("ME").last()
    bench = me.rolling(n).mean().reindex(df.index, method="ffill").shift(1)
    pid = df.index.to_period("M")
    first = pd.Series(pid != pd.Series(pid, index=df.index).shift(), index=df.index).fillna(True)
    s = pd.Series(np.nan, index=df.index)
    s[first.values] = (op > bench).astype(float)[first.values]
    return s.ffill().fillna(0)


def rets(df, n):
    return (signal(df, n).shift(1) * df["close"].pct_change()).dropna()


def sharpe(r):
    return r.mean() / r.std() * np.sqrt(252)


def main():
    df = load()
    print(f"Nasdaq-100 (spliced), {df.index[0].date()} -> {df.index[-1].date()} "
          f"({(df.index[-1]-df.index[0]).days/365.25:.0f} years)\n")

    print("IN-SAMPLE / OUT-OF-SAMPLE RATIO SWEEP")
    print("Pick best N on in-sample, then test it out-of-sample. Fixed MA12 for reference.")
    print(f"{'IS:OOS':>8}{'IS yr':>6}{'OOS yr':>7}{'bestN':>6}"
          f"{'IS Shrp':>8}{'OOS Shrp':>9}{'gap':>6}{'MA12 OOS':>9}")
    total = (df.index[-1] - df.index[0]).days
    for r in (0.5, 0.6, 0.667, 0.7, 0.75, 0.8):
        split = df.index[0] + pd.Timedelta(days=int(total * r))
        best, bsh = None, -9
        for n in N_GRID:
            rr = rets(df, n)
            s = sharpe(rr[rr.index < split])
            if s > bsh:
                best, bsh = n, s
        rb = rets(df, best)
        oos_b = sharpe(rb[rb.index >= split])
        r12 = rets(df, 12)
        oos12 = sharpe(r12[r12.index >= split])
        isy = (split - df.index[0]).days / 365.25
        oosy = (df.index[-1] - split).days / 365.25
        print(f"{int(r*100)}:{100-int(r*100):<4}{isy:>6.0f}{oosy:>7.0f}{best:>6}"
              f"{bsh:>8.2f}{oos_b:>9.2f}{bsh-oos_b:>+6.2f}{oos12:>9.2f}")

    print("\nGuidance: choose the ratio a priori, not by best result.")
    print("For ~30 years, ~70:30 (21y in / 9y out) keeps enough cycles to fit AND")
    print("leaves an out-of-sample window long enough for a full bull+bear cycle.")
    print("Walk-forward / expanding-window validation is more robust than any single split.")


if __name__ == "__main__":
    main()
