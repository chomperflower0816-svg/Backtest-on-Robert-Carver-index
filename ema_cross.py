"""
The simplest CTA: a moving-average crossover on QQQ.

Rule:  fast EMA above slow EMA -> hold QQQ (uptrend)
       fast EMA below slow EMA -> out / short (downtrend)

Edit the settings, then run:   python ema_cross.py
"""

import pandas as pd

# ─── settings ────────────────────────────────────────────────────
FAST = 50          # fast moving-average length (days)
SLOW = 200         # slow moving-average length (days)
KIND = "EMA"       # "EMA" (exponential) or "SMA" (simple)
MODE = "long_only" # "long_only" (hold or cash) or "long_short" (hold or short)
# ─────────────────────────────────────────────────────────────────


def run():
    price = (pd.read_csv("data/QQQ.csv", parse_dates=["date"])
               .set_index("date")["close"])

    if KIND == "EMA":
        fast = price.ewm(span=FAST).mean()
        slow = price.ewm(span=SLOW).mean()
    else:
        fast = price.rolling(FAST).mean()
        slow = price.rolling(SLOW).mean()

    # signal: +1 when fast above slow; 0 (cash) or -1 (short) when below
    uptrend = fast > slow
    down = 0 if MODE == "long_only" else -1
    signal = pd.Series(1, index=price.index).where(uptrend, down)

    # today's return uses YESTERDAY's signal (no peeking ahead)
    mkt = price.pct_change()
    strat = signal.shift(1) * mkt
    strat = strat.dropna()

    # count how often we flip in/out
    flips = int((signal.diff().abs() > 0).sum())

    print(f"QQQ  {price.index[0].date()} -> {price.index[-1].date()}")
    print(f"Rule: {KIND} {FAST}/{SLOW} cross, {MODE}   (flipped {flips} times)\n")
    print(f"{'':16}{'EMA cross':>13}{'Buy & Hold':>13}")
    line("Total return", total(strat), total(mkt.loc[strat.index]))
    line("Yearly return", cagr(strat), cagr(mkt.loc[strat.index]))
    line("Worst drop", maxdd(strat), maxdd(mkt.loc[strat.index]))
    print(f"\nIn the market {uptrend.mean():.0%} of the time.")

    try:
        plot(strat, mkt)
    except Exception as e:
        print(f"(plot skipped: {e})")


def total(r):
    return (1 + r).prod() - 1

def cagr(r):
    return (1 + r).prod() ** (1 / (len(r) / 252)) - 1

def maxdd(r):
    c = (1 + r).cumprod()
    return (c / c.cummax() - 1).min()

def line(label, a, b):
    print(f"{label:16}{a:>12.1%}{b:>13.1%}")


def plot(strat, mkt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a = (1 + strat).cumprod()
    b = (1 + mkt.loc[strat.index]).cumprod()
    ax = a.plot(figsize=(10, 5), logy=True, label=f"{KIND} {FAST}/{SLOW} cross")
    b.plot(ax=ax, label="Buy & Hold", alpha=0.8)
    ax.set_title(f"Simplest CTA: {KIND} {FAST}/{SLOW} crossover on QQQ")
    ax.set_ylabel("Growth of $1 (log)"); ax.legend(); ax.grid(alpha=0.3)
    ax.figure.tight_layout(); ax.figure.savefig("ema_cross.png", dpi=110)
    print("Saved chart -> ema_cross.png")


if __name__ == "__main__":
    run()
