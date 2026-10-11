"""
QQQAI-style options strategy on QQQ, backtested with REAL implied vol (VXN).

Structure (monthly cycle, first trading day, at the open):
  LEG 1 (income / "safety cushion"): sell a 1-month cash-secured put.
    Strike = S * exp(-CUSHION * sigma * sqrt(T)), sigma = that day's VXN.
    Higher VXN -> strike automatically further OTM (the VIX-timed cushion).
    Premium priced Black-Scholes at FLAT VXN (real OTM puts trade above ATM
    vol, so we UNDER-collect -> conservative); FAIR variants add +3 vol pts.
  LEG 2 (tail hedge): long 6-month 25%-OTM puts, notional = HEDGE * NAV,
    rolled monthly. Priced at VXN + 4 vol pts (skew: we overpay ->
    conservative); sold back at +3 pts (1-pt bid haircut).
  GATES tested: none / MA11 trend regime / regime + VXN storm filter
    (skip new short puts when VXN > its rolling 1y 80th percentile).

Pricing data: data/VXN_spliced.csv -- real CBOE VXN 2001-2025 (two
independent sources cross-validated, mean abs diff 0.002), pre-Feb-2001
proxied by VIX*1.17. r = 2% flat, cash collateral earns 2%, no dividends.
European settlement at the next cycle's open. All marks daily via BS at that
day's VXN -> daily NAV.

Authentic ledger: integer contracts (100 shares each), $3 per option trade,
cash-secured sizing (no leverage), start 100,000 HKD.

Run:  python putwrite_vxn.py
"""

import math

import numpy as np
import pandas as pd

START_HKD, FX, COMM = 100_000.0, 7.8, 3.0
CUSHION = 0.75            # strike distance in units of sigma*sqrt(T)
TAIL_OTM = 0.75           # tail strike = 75% of spot (25% OTM)
TAIL_MONTHS = 6
R = 0.02                  # flat risk-free for BS
R_CASH = 0.02             # interest earned on cash collateral (T-bill proxy)
SKEW_BUY, SKEW_SELL = 0.04, 0.03
SPLIT = pd.Timestamp("2012-03-10")


def bs_put(S, K, sig, T, r=R):
    if T <= 0:
        return max(K - S, 0.0)
    sq = sig * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / sq
    d2 = d1 - sq
    N = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    return K * math.exp(-r * T) * N(-d2) - S * N(-d1)


def load():
    px = pd.read_csv("data/QQQ.csv", parse_dates=["date"]).set_index("date")
    vxn = pd.read_csv("data/VXN_spliced.csv", parse_dates=["date"]).set_index("date")["VXN"]
    df = px.join(vxn / 100.0, how="inner").dropna()
    return df.index, df["open"].values, df["close"].values, df


def precompute(idx, df):
    cs = df["close"]
    me = cs.resample("ME").last()
    ma11 = me.rolling(11).mean().reindex(idx, method="ffill").shift(1).values
    vxn = df["VXN"].values
    vxn_prev = df["VXN"].shift(1).values                 # known at the open
    storm = (df["VXN"].shift(1) >
             df["VXN"].shift(1).rolling(252).quantile(0.80)).values
    pid = idx.to_period("M")
    first = np.array([True] + [pid[i] != pid[i - 1] for i in range(1, len(idx))])
    return ma11, vxn, vxn_prev, storm, first


def run(idx, op, cl, ma11, vxn, vxn_prev, storm, first,
        gate="none", hedge=0.0, skew_sold=0.0):
    """gate: 'none' | 'trend' | 'trend+storm'. hedge: tail notional / NAV."""
    cash = START_HKD / FX
    sp = None                    # short put: dict(K, n, t_expiry_index)
    tp = None                    # tail put:  dict(K, n, expiry_index)
    firsts = np.where(first)[0]
    next_first = {}              # day index -> next cycle's day index
    for a, b in zip(firsts, firsts[1:]):
        next_first[a] = b
    trades = 0
    nav = np.empty(len(idx))
    for t in range(len(idx)):
        S = op[t]
        if first[t] and not np.isnan(ma11[t]) and not np.isnan(vxn_prev[t]):
            sig = vxn_prev[t]
            # settle expiring short put at this open
            if sp is not None and sp["exp"] == t:
                cash -= max(sp["K"] - S, 0.0) * 100 * sp["n"]
                sp = None
            # roll tail: sell back existing
            if tp is not None:
                Trem = max((tp["exp"] - t) / 252, 0.0)
                cash += bs_put(S, tp["K"], sig + SKEW_SELL, Trem) * 100 * tp["n"] - COMM
                trades += 1
                tp = None
            # decide regime
            on = True
            if gate in ("trend", "trend+storm"):
                on = S > ma11[t]
            sell_ok = on and not (gate == "trend+storm" and storm[t])
            nav_now = cash  # flat at this instant (options just closed/settled)
            if sell_ok and t in next_first and nav_now > 0:
                # tail first (protection in place before selling)
                if hedge > 0:
                    Kt = TAIL_OTM * S
                    n_t = max(int(round(hedge * nav_now / (100 * S))), 1)
                    cost = bs_put(S, Kt, sig + SKEW_BUY, TAIL_MONTHS / 12) * 100 * n_t
                    cash -= cost + COMM
                    trades += 1
                    tp = {"K": Kt, "n": n_t, "exp": t + int(TAIL_MONTHS * 21)}
                # short put, cash-secured
                T = (next_first[t] - t) / 252
                Ks = S * math.exp(-CUSHION * sig * math.sqrt(T))
                n_s = int(cash // (Ks * 100))
                if n_s > 0:
                    cash += bs_put(S, Ks, sig + skew_sold, T) * 100 * n_s - COMM
                    trades += 1
                    sp = {"K": Ks, "n": n_s, "exp": next_first[t]}
        if cash > 0:                                 # collateral earns T-bills
            cash *= 1 + R_CASH / 252
        # daily mark-to-market at close
        v = vxn[t] if not np.isnan(vxn[t]) else 0.3
        m = cash
        if sp is not None:
            m -= bs_put(cl[t], sp["K"], v + skew_sold, max((sp["exp"] - t) / 252, 0.0)) * 100 * sp["n"]
        if tp is not None:
            m += bs_put(cl[t], tp["K"], v + SKEW_SELL, max((tp["exp"] - t) / 252, 0.0)) * 100 * tp["n"]
        nav[t] = m
    return pd.Series(nav, index=idx), trades


def stats(nav, mask):
    n = nav[mask]
    r = n.pct_change().dropna()
    return {"CAGR": (n.iloc[-1] / n.iloc[0]) ** (252 / len(n)) - 1,
            "Sharpe": r.mean() / r.std() * np.sqrt(252),
            "MaxDD": (n / n.cummax() - 1).min()}


def crash_window(nav, a, b):
    w = nav[(nav.index >= a) & (nav.index <= b)]
    return w.iloc[-1] / w.iloc[0] - 1 if len(w) > 1 else np.nan


def main():
    idx, op, cl, df = load()
    ma11, vxn, vxn_prev, storm, first = precompute(idx, df)
    IS, OOS, ALL = idx < SPLIT, idx >= SPLIT, np.ones(len(idx), bool)
    print(f"QQQ + real VXN, {idx[0].date()} -> {idx[-1].date()}, "
          f"start {START_HKD:,.0f} HKD\n")

    variants = [
        ("PutWrite always", dict(gate="none", hedge=0.0)),
        ("PutWrite + MA11 gate", dict(gate="trend", hedge=0.0)),
        ("  + storm filter", dict(gate="trend+storm", hedge=0.0)),
        ("  + tail h=1", dict(gate="trend", hedge=1.0)),
        ("  + tail h=2 (QQQAI)", dict(gate="trend", hedge=2.0)),
        ("FAIR gate, no tail", dict(gate="trend", hedge=0.0, skew_sold=0.03)),
        ("FAIR gate + tail h=1", dict(gate="trend", hedge=1.0, skew_sold=0.03)),
        ("FAIR gate + tail h=2", dict(gate="trend", hedge=2.0, skew_sold=0.03)),
    ]
    results = []
    sh0 = int((START_HKD / FX - COMM) // op[0])
    bh = pd.Series((START_HKD / FX - sh0 * op[0] - COMM) + sh0 * cl, index=idx)
    results.append(("Buy & Hold QQQ", bh, 1))
    for name, kw in variants:
        nav, tr = run(idx, op, cl, ma11, vxn, vxn_prev, storm, first, **kw)
        results.append((name, nav, tr))

    hdr = f"{'CAGR':>7}{'Shrp':>6}{'MaxDD':>7}"
    print(f"{'':22}{'IN-SAMPLE':>20}|{'OUT-OF-SAMPLE':>20}|{'FULL':>20}{'trds':>6}{'endHKD':>12}")
    print(f"{'':22}{hdr:>20}|{hdr:>20}|{hdr:>20}")
    for name, nav, tr in results:
        a, b, c = stats(nav, IS), stats(nav, OOS), stats(nav, ALL)
        print(f"{name:22}"
              f"{a['CAGR']:>7.1%}{a['Sharpe']:>6.2f}{a['MaxDD']:>7.0%}|"
              f"{b['CAGR']:>7.1%}{b['Sharpe']:>6.2f}{b['MaxDD']:>7.0%}|"
              f"{c['CAGR']:>7.1%}{c['Sharpe']:>6.2f}{c['MaxDD']:>7.0%}"
              f"{tr:>6}{nav.iloc[-1]*FX:>12,.0f}")

    print("\nCrash windows (total return of NAV):")
    wins = [("dot-com", "2000-03-01", "2002-10-09"),
            ("GFC", "2008-09-01", "2009-03-09"),
            ("COVID", "2020-02-19", "2020-03-23"),
            ("2022 bear", "2022-01-03", "2022-12-28")]
    print(f"{'':22}" + "".join(f"{w[0]:>10}" for w in wins))
    for name, nav, _ in results:
        print(f"{name:22}" + "".join(f"{crash_window(nav, w[1], w[2]):>10.0%}" for w in wins))


if __name__ == "__main__":
    main()
