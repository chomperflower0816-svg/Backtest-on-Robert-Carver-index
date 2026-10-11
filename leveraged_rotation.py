"""
Leveraged-ETF trend rotation on QQQ -- leverage WITHOUT margin or options.

Idea (Gayed 2016, "Leverage for the Long Run", Dow Award): hold a leveraged
ETF (QLD 2x / TQQQ 3x) only when the trend filter is ON, cash when OFF.
Daily-reset leveraged ETFs suffer volatility decay, but volatility clusters
in downtrends -- exactly when the trend filter has you out. So the filter is
what makes leveraged ETFs survivable.

Leveraged ETF prices are SYNTHESIZED from QQQ daily returns:
    r_letf = L * r_qqq - drag/252
calibrated against REAL TQQQ 2010-2021 (data/QQQ_TQQQ_2010_2021.csv):
beta 2.95, corr 0.9985, implied drag 3.23%/yr (synthetic ends ~6% BELOW real
over 11.6y, i.e. slightly conservative). Because 2010-2021 was a zero-rate
era, a HIGH-COST scenario (financing +2%/yr) is also reported.

Strategies (all: monthly decision at first open, MA11 regime, authentic
ledger with integer shares, $3 per trade, 10% NAV rebalance buffer):
  A. Buy & Hold QQQ
  B. MA11 -> QQQ (1x timing, reference)
  C. MA11 -> QLD (2x rotation)
  D. MA11 -> TQQQ (3x rotation)
  E. TrendVol-LETF: w = min(2, 32.5%/vol21), built as a QQQ+QLD mix
     (the a-priori TrendVol rule, executed without margin)

Run:  python leveraged_rotation.py
"""

import numpy as np
import pandas as pd

START_HKD, FX, COMM = 100_000.0, 7.8, 3.0
BUFFER = 0.10
SPLIT = pd.Timestamp("2012-03-10")
SIGMA_T, W_CAP = 0.325, 2.0          # a-priori TrendVol params (IS uncond vol)
DRAG = {2: 0.0209, 3: 0.0323}        # calibrated: ER 0.95% + (L-1)*1.14% financing
DRAG_HI = {2: 0.0409, 3: 0.0723}     # high-cost scenario: financing +2%/yr


def load():
    df = pd.read_csv("data/QQQ.csv", parse_dates=["date"]).set_index("date")
    return df.index, df["open"].values, df["close"].values, df["close"]


def synth_letf(op, cl, L, drag):
    """Daily-reset leveraged ETF open/close series from QQQ prices."""
    n = len(cl)
    sc, so = np.empty(n), np.empty(n)
    sc[0] = so[0] = 100.0
    for t in range(1, n):
        so[t] = sc[t - 1] * (1 + L * (op[t] / cl[t - 1] - 1))
        sc[t] = sc[t - 1] * (1 + L * (cl[t] / cl[t - 1] - 1) - drag / 252)
    return so, sc


def precompute(idx, close_series):
    me = close_series.resample("ME").last()
    ma11 = me.rolling(11).mean().reindex(idx, method="ffill").shift(1).values
    vol = (close_series.pct_change().rolling(21).std() * np.sqrt(252)).shift(1).values
    pid = idx.to_period("M")
    first = np.array([True] + [pid[i] != pid[i - 1] for i in range(1, len(idx))])
    return ma11, vol, first


def run(idx, opens, closes, weights_fn, ma11, vol, first):
    """Generic multi-asset monthly ledger.
    opens/closes: dict name -> price array. weights_fn(t, regime, vol) -> dict
    name -> portfolio weight (rest is cash)."""
    names = list(opens)
    cash, sh = START_HKD / FX, {k: 0 for k in names}
    trades = 0
    nav = np.empty(len(idx))
    for t in range(len(idx)):
        nav_now = cash + sum(sh[k] * opens[k][t] for k in names)
        if first[t] and not np.isnan(ma11[t]) and not np.isnan(vol[t]) and nav_now > 0:
            regime = opens["QQQ"][t] > ma11[t] if "QQQ" in opens else None
            w = weights_fn(t, regime, vol[t])
            for k in names:
                target = int(w.get(k, 0.0) * nav_now // opens[k][t])
                if abs(target - sh[k]) * opens[k][t] > BUFFER * nav_now \
                        or (w.get(k, 0.0) == 0 and sh[k] > 0):
                    cash -= (target - sh[k]) * opens[k][t] + COMM
                    sh[k] = target
                    trades += 1
        nav[t] = cash + sum(sh[k] * closes[k][t] for k in names)
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

    def scenarios(drags):
        qld_o, qld_c = synth_letf(op, cl, 2, drags[2])
        tqq_o, tqq_c = synth_letf(op, cl, 3, drags[3])
        base = {"QQQ": (op, cl), "QLD": (qld_o, qld_c), "TQQQ": (tqq_o, tqq_c)}
        O = {k: v[0] for k, v in base.items()}
        C = {k: v[1] for k, v in base.items()}
        out = []
        # A. buy & hold
        sh0 = int((START_HKD / FX - COMM) // op[0])
        bh = pd.Series((START_HKD / FX - sh0 * op[0] - COMM) + sh0 * cl, index=idx)
        out.append(("Buy & Hold QQQ", bh, 1))
        # B/C/D single-asset rotations
        for name, asset in [("MA11 -> QQQ (1x)", "QQQ"),
                            ("MA11 -> QLD (2x)", "QLD"),
                            ("MA11 -> TQQQ (3x)", "TQQQ")]:
            opens = {a: O[a] for a in ("QQQ", asset)} if asset != "QQQ" else {"QQQ": O["QQQ"]}
            closes = {a: C[a] for a in ("QQQ", asset)} if asset != "QQQ" else {"QQQ": C["QQQ"]}
            nav, tr = run(idx, opens, closes,
                          lambda t, reg, v, a=asset: {a: 1.0} if reg else {},
                          ma11, vol, first)
            out.append((name, nav, tr))
        # E. TrendVol via QQQ+QLD mix
        def tv_weights(t, reg, v):
            if not reg:
                return {}
            w = min(W_CAP, SIGMA_T / v)
            if w <= 1:
                return {"QQQ": w}
            return {"QQQ": 2 - w, "QLD": w - 1}
        nav, tr = run(idx, {k: O[k] for k in ("QQQ", "QLD")},
                      {k: C[k] for k in ("QQQ", "QLD")}, tv_weights, ma11, vol, first)
        out.append(("TrendVol QQQ+QLD mix", nav, tr))
        return out

    for label, drags in [("CALIBRATED costs (2010-21 rates)", DRAG),
                         ("HIGH-COST scenario (financing +2%)", DRAG_HI)]:
        print(f"\n=== {label}: QLD drag {drags[2]:.1%}/yr, TQQQ drag {drags[3]:.1%}/yr ===")
        hdr = f"{'CAGR':>7}{'Shrp':>6}{'MaxDD':>7}"
        print(f"{'':22}{'IN-SAMPLE':>20}|{'OUT-OF-SAMPLE':>20}|{'FULL':>20}{'trds':>6}{'endHKD':>12}")
        print(f"{'':22}{hdr:>20}|{hdr:>20}|{hdr:>20}")
        for name, nav, tr in scenarios(drags):
            a, b, c = stats(nav, IS), stats(nav, OOS), stats(nav, ALL)
            print(f"{name:22}"
                  f"{a['CAGR']:>7.1%}{a['Sharpe']:>6.2f}{a['MaxDD']:>7.0%}|"
                  f"{b['CAGR']:>7.1%}{b['Sharpe']:>6.2f}{b['MaxDD']:>7.0%}|"
                  f"{c['CAGR']:>7.1%}{c['Sharpe']:>6.2f}{c['MaxDD']:>7.0%}"
                  f"{tr:>6}{nav.iloc[-1]*FX:>12,.0f}")


if __name__ == "__main__":
    main()
