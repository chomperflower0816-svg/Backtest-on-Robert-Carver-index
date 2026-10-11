"""
TrendVol: trend filter + volatility-targeted sizing on QQQ.

Literature basis (both applied to *equity* index exposure):
  - Moreira & Muir (2017, J. Finance) "Volatility-Managed Portfolios":
    scaling exposure by inverse variance raises Sharpe and produces alpha,
    because volatility spikes are not matched by higher expected returns.
  - Harvey, Hoyle, Korgaonkar, Rattray, Sargaison, van Hemert (2018)
    "The Impact of Volatility Targeting": for risk assets, vol targeting
    raises Sharpe, cuts left-tail severity and max drawdown.
  - Trend leg is the validated monthly MA11 regime filter (time-series
    momentum, Moskowitz-Ooi-Pedersen 2012 / Faber's monthly rule).

Rule (decided ONCE A MONTH, first trading day, at the open):
  regime  = open > average of last 11 month-end closes          (trend)
  weight  = min(L_MAX, SIGMA_TARGET / realized_vol_21d)  if regime else 0
  rebalance to weight only if the move exceeds a 10% NAV buffer (cost control)

Authentic ledger: integer shares, $3 per rebalance, margin interest charged
daily at MARGIN_RATE on any borrowed balance, NAV marked daily.
Parameters are A-PRIORI, not grid-picked: sigma_target = the asset's own
in-sample unconditional vol (Moreira-Muir normalization), L_max = 1.5.

Run:  python trend_vol.py
"""

import numpy as np
import pandas as pd

START_HKD, FX, COMM = 100_000.0, 7.8, 3.0
MARGIN_RATE = 0.04           # annual interest on borrowed cash (conservative)
BUFFER = 0.10                # rebalance only if trade size > 10% of NAV
SPLIT = pd.Timestamp("2012-03-10")
VOL_WINDOW = 21              # days of realized vol used for sizing


def load():
    df = pd.read_csv("data/QQQ.csv", parse_dates=["date"]).set_index("date")
    return df.index, df["open"].values, df["close"].values, df["close"]


def precompute(idx, close_series):
    me = close_series.resample("ME").last()
    ma11 = me.rolling(11).mean().reindex(idx, method="ffill").shift(1).values
    ret = close_series.pct_change()
    vol = (ret.rolling(VOL_WINDOW).std() * np.sqrt(252)).shift(1).values  # through yesterday
    pid = idx.to_period("M")
    first = np.array([True] + [pid[i] != pid[i - 1] for i in range(1, len(idx))])
    return ma11, vol, first


def run(idx, op, cl, ma11, vol, first, sigma_target, l_max,
        start=START_HKD / FX, comm=COMM):
    cash, shares, trades = start, 0, 0
    nav = np.empty(len(idx))
    for t in range(len(idx)):
        nav_now = cash + shares * op[t]
        if first[t] and not np.isnan(ma11[t]) and not np.isnan(vol[t]) and nav_now > 0:
            regime = op[t] > ma11[t]
            w = min(l_max, sigma_target / vol[t]) if regime else 0.0
            target = int(w * nav_now // op[t])
            if abs(target - shares) * op[t] > BUFFER * nav_now or (w == 0 and shares > 0):
                cash -= (target - shares) * op[t] + comm
                shares = target
                trades += 1
        if cash < 0:                                # margin interest, daily
            cash *= 1 + MARGIN_RATE / 252
        nav[t] = cash + shares * cl[t]
    return pd.Series(nav, index=idx), trades


def stats(nav, mask):
    n = nav[mask]
    r = n.pct_change().dropna()
    return {"CAGR": (n.iloc[-1] / n.iloc[0]) ** (252 / len(n)) - 1,
            "Sharpe": r.mean() / r.std() * np.sqrt(252),
            "MaxDD": (n / n.cummax() - 1).min()}


def main():
    idx, op, cl, cs = load()
    ma11, vol, first = precompute(idx, cs)
    IS, OOS, ALL = idx < SPLIT, idx >= SPLIT, np.ones(len(idx), bool)

    # ---- a-priori parameters (no grid picking) ----
    # sigma_target = the asset's own IN-SAMPLE unconditional vol
    # (Moreira-Muir normalization); L_max = 1.5, a moderate cap below Reg-T.
    is_vol = cs.pct_change()[IS].std() * np.sqrt(252)
    s_t, l_m = round(float(is_vol), 3), 1.5

    print(f"QQQ {idx[0].date()} -> {idx[-1].date()}, start "
          f"{START_HKD:,.0f} HKD = ${START_HKD/FX:,.2f}")
    print(f"In-sample ..{SPLIT.date()} | a-priori params: "
          f"sigma_target = IS uncond vol = {s_t:.1%}, L_max = {l_m}\n")

    out = []
    shares0 = int((START_HKD / FX - COMM) // op[0])
    bh = pd.Series((START_HKD / FX - shares0 * op[0] - COMM) + shares0 * cl, index=idx)
    out.append(("Buy & Hold", bh, 1))
    nav, tr = run(idx, op, cl, ma11, vol, first, 9.9, 1.0)
    out.append(("Plain MA11 (1x)", nav, tr))
    nav, tr = run(idx, op, cl, ma11, vol, first, s_t, 1.0)
    out.append((f"TrendVol (L=1)", nav, tr))
    nav, tr = run(idx, op, cl, ma11, vol, first, s_t, l_m)
    out.append((f"TrendVol (L={l_m})", nav, tr))

    hdr = f"{'CAGR':>7}{'Shrp':>7}{'MaxDD':>7}"
    print(f"{'':24}{'IN-SAMPLE':>21}|{'OUT-OF-SAMPLE':>21}|{'FULL':>21}{'trds':>6}{'endHKD':>12}")
    print(f"{'':24}{hdr:>21}|{hdr:>21}|{hdr:>21}")
    for name, nav, tr in out:
        a, b, c = stats(nav, IS), stats(nav, OOS), stats(nav, ALL)
        print(f"{name:24}"
              f"{a['CAGR']:>7.1%}{a['Sharpe']:>7.2f}{a['MaxDD']:>7.0%}|"
              f"{b['CAGR']:>7.1%}{b['Sharpe']:>7.2f}{b['MaxDD']:>7.0%}|"
              f"{c['CAGR']:>7.1%}{c['Sharpe']:>7.2f}{c['MaxDD']:>7.0%}"
              f"{tr:>6}{nav.iloc[-1]*FX:>12,.0f}")

    print("\nRobustness (neighboring params, FULL-sample CAGR / MaxDD / endHKD):")
    for s in (0.25, 0.30, 0.35):
        for l in (1.25, 1.5, 2.0):
            nav, _ = run(idx, op, cl, ma11, vol, first, s, l)
            c = stats(nav, ALL)
            print(f"  sigma={s:.0%} L={l:<5}: {c['CAGR']:.1%} / {c['MaxDD']:.0%} / "
                  f"{nav.iloc[-1]*FX:,.0f}")


if __name__ == "__main__":
    main()
