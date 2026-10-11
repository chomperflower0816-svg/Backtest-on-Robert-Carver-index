"""
Robert Carver's trend system — the SIMPLE single-index version.

Just one trend signal, sized by volatility. Edit the settings below,
then run:   python simple.py
"""

import pandas as pd

# ════════════════════════════════════════════════════════════════
#  THE ONLY SETTINGS YOU CHANGE
# ════════════════════════════════════════════════════════════════

CAPITAL     = 100_000      # how much (pretend) money to trade

RISK_LEVEL  = 0.20         # how wild the ride is:
                           #   0.10 calm · 0.20 balanced · 0.30+ aggressive

TREND_SPEED = "medium"     # how fast it reacts:  "fast" · "medium" · "slow"

BUFFER      = 0.10         # "don't sweat small moves" zone (Carver's buffering).
                           #   0.10 = only trade when the target drifts >10% of a
                           #   full position away. Bigger = fewer trades / lower
                           #   costs. Set 0 to trade to the exact target every day.

# ════════════════════════════════════════════════════════════════


# Each speed = one moving-average pair + Carver's forecast scalar for it.
SPEEDS = {        # (fast_days, slow_days, forecast_scalar)
    "fast":   (16, 64,  3.75),
    "medium": (32, 128, 2.65),
    "slow":   (64, 256, 1.91),
}


def run():
    fast_days, slow_days, scalar = SPEEDS[TREND_SPEED]

    # 1. price (split-adjusted QQQ close)
    price = (pd.read_csv("data/QQQ.csv", parse_dates=["date"])
               .set_index("date")["close"])

    # 2. volatility = how big a typical daily move is, in $ (Carver uses ~35 days)
    vol = price.diff().ewm(span=35).std()

    # 3. the trend signal: fast average minus slow average.
    #    Positive = uptrend (go long), negative = downtrend (go short).
    trend = price.ewm(span=fast_days).mean() - price.ewm(span=slow_days).mean()

    # 4. turn it into a "forecast": +10 = average buy, -10 = average sell,
    #    capped at +/-20 so one crazy day can't dominate.
    forecast = (trend / vol * scalar).clip(-20, 20)

    # 5. size the position by volatility (the whole point of the system):
    #    when the market gets choppier, hold fewer shares -> steady risk.
    daily_risk_budget = CAPITAL * RISK_LEVEL / 16        # 16 = sqrt(256 days)
    shares_at_full_signal = daily_risk_budget / vol
    target = shares_at_full_signal * forecast / 10       # the ideal position (shares)

    # 6. BUFFERING: keep yesterday's position unless the target has drifted
    #    outside a no-trade band. This is what cuts the trading down.
    position = apply_buffer(target, BUFFER * shares_at_full_signal)

    # 7. profit: yesterday's position earns today's price move (no peeking).
    pnl = position.shift(1) * price.diff()
    returns = (pnl / CAPITAL).dropna()
    trades = int((position.diff().abs() > 1e-9).sum())

    # ---- report -----------------------------------------------------------
    print(f"QQQ  {price.index[0].date()} -> {price.index[-1].date()}")
    print(f"Settings: capital ${CAPITAL:,}  risk {RISK_LEVEL:.0%}  "
          f"speed '{TREND_SPEED}' ({fast_days}/{slow_days})\n")
    print(f"Today's signal: {forecast.iloc[-1]:+.0f}   "
          f"({'LONG' if forecast.iloc[-1] > 0 else 'SHORT'} "
          f"{abs(position.iloc[-1]):,.0f} shares)")
    yrs = len(returns) / 256
    print(f"Trades made: {trades}  (~{trades / yrs:.0f}/year"
          f"{', buffer off' if BUFFER == 0 else f', buffer {BUFFER:.0%}'})\n")

    print(f"{'':16}{'Trend system':>14}{'Buy & Hold':>14}")
    show("Yearly return", stat(returns, 'cagr'),
         stat(price.pct_change().dropna(), 'cagr'))
    show("Yearly risk",   stat(returns, 'vol'),
         stat(price.pct_change().dropna(), 'vol'))
    show("Worst drop",    stat(returns, 'dd'),
         stat(price.pct_change().dropna(), 'dd'))

    try:
        plot(returns, price)
    except Exception as e:
        print(f"\n(plot skipped: {e})")


def apply_buffer(target, buffer_width):
    """Carver's buffering: only move the position when the target drifts
    outside a band of +/- buffer_width around the position we already hold.
    Returns the actually-held position each day."""
    tgt = target.values
    buf = buffer_width.values
    held = [0.0] * len(tgt)
    current = 0.0
    for i in range(len(tgt)):
        if pd.isna(tgt[i]):
            held[i] = current
            continue
        lower, upper = tgt[i] - buf[i], tgt[i] + buf[i]
        if current < lower:
            current = lower            # drifted below band -> trade up to it
        elif current > upper:
            current = upper            # drifted above band -> trade down to it
        held[i] = current              # otherwise: hold, no trade
    return pd.Series(held, index=target.index)


def stat(r, kind):
    curve = (1 + r).cumprod()
    if kind == 'cagr':
        return curve.iloc[-1] ** (1 / (len(r) / 256)) - 1
    if kind == 'vol':
        return r.std() * (256 ** 0.5)
    if kind == 'dd':
        return (curve / curve.cummax() - 1).min()


def show(label, a, b):
    print(f"{label:16}{a:>13.1%}{b:>14.1%}")


def plot(returns, price):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    strat = (1 + returns).cumprod() * CAPITAL
    bh = (1 + price.pct_change().reindex(returns.index).fillna(0)).cumprod() * CAPITAL
    ax = strat.plot(figsize=(10, 5), logy=True, label="Trend system")
    bh.plot(ax=ax, label="Buy & Hold", alpha=0.8)
    ax.set_title(f"Carver trend system on QQQ ({TREND_SPEED})")
    ax.set_ylabel("Capital ($, log)"); ax.legend(); ax.grid(alpha=0.3)
    ax.figure.tight_layout(); ax.figure.savefig("simple_result.png", dpi=110)
    print("\nSaved chart -> simple_result.png")


if __name__ == "__main__":
    run()
