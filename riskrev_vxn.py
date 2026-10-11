"""
Full QQQAI replica: sell put -> finance OTM calls ("never miss the rally"
leg) + optional tail puts. Priced with real VXN like putwrite_vxn.py.

Monthly cycle (first trading day, open), MA11 gate optional:
  1. sell 1-month cash-secured put, strike S*exp(-0.75*sigma*sqrt(T)),
     priced at VXN + 3pts (fair skew)
  2. spend CALL_BUDGET x (premium collected) buying 1-month OTM calls,
     strike S*exp(+CALL_OTM*sigma*sqrt(T)), priced at FLAT VXN
     (real index OTM calls trade BELOW ATM vol -> we overpay -> conservative).
     Premium buys >1x notional of calls: the convexity that can BEAT the
     market in strong rallies.
  3. optional 6-month 25%-OTM tail puts (h x NAV), rolled monthly.
Cash earns 2%. Integer contracts, $3/trade, daily BS mark-to-market.

Run:  python riskrev_vxn.py
"""

import math

import numpy as np
import pandas as pd

START_HKD, FX, COMM = 100_000.0, 7.8, 3.0
CUSHION = 0.75
CALL_OTM = 0.50           # call strike distance in sigma*sqrt(T) units
TAIL_OTM, TAIL_MONTHS = 0.75, 6
R, R_CASH = 0.02, 0.02
SKEW_PUT, SKEW_BUY, SKEW_SELL = 0.03, 0.04, 0.03
SPLIT = pd.Timestamp("2012-03-10")


def _N(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_put(S, K, sig, T, r=R):
    if T <= 0:
        return max(K - S, 0.0)
    sq = sig * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / sq
    return K * math.exp(-r * T) * _N(-(d1 - sq)) - S * _N(-d1)


def bs_call(S, K, sig, T, r=R):
    if T <= 0:
        return max(S - K, 0.0)
    sq = sig * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / sq
    return S * _N(d1) - K * math.exp(-r * T) * _N(d1 - sq)


def load():
    px = pd.read_csv("data/QQQ.csv", parse_dates=["date"]).set_index("date")
    vxn = pd.read_csv("data/VXN_spliced.csv", parse_dates=["date"]).set_index("date")["VXN"]
    df = px.join(vxn / 100.0, how="inner").dropna()
    return df.index, df["open"].values, df["close"].values, df


def precompute(idx, df):
    me = df["close"].resample("ME").last()
    ma11 = me.rolling(11).mean().reindex(idx, method="ffill").shift(1).values
    vxn = df["VXN"].values
    vxn_prev = df["VXN"].shift(1).values
    pid = idx.to_period("M")
    first = np.array([True] + [pid[i] != pid[i - 1] for i in range(1, len(idx))])
    return ma11, vxn, vxn_prev, first


def run(idx, op, cl, ma11, vxn, vxn_prev, first,
        gate=True, call_budget=1.0, call_otm=CALL_OTM, hedge=0.0):
    cash = START_HKD / FX
    sp = cp = tp = None
    firsts = np.where(first)[0]
    nxt = dict(zip(firsts, firsts[1:]))
    trades = 0
    nav = np.empty(len(idx))
    for t in range(len(idx)):
        S = op[t]
        if first[t] and not np.isnan(ma11[t]) and not np.isnan(vxn_prev[t]):
            sig = vxn_prev[t]
            if sp is not None and sp["exp"] == t:      # settle short put
                cash -= max(sp["K"] - S, 0.0) * 100 * sp["n"]
                sp = None
            if cp is not None and cp["exp"] == t:      # settle long calls
                cash += max(S - cp["K"], 0.0) * 100 * cp["n"]
                cp = None
            if tp is not None:                          # roll tail
                Trem = max((tp["exp"] - t) / 252, 0.0)
                cash += bs_put(S, tp["K"], sig + SKEW_SELL, Trem) * 100 * tp["n"] - COMM
                trades += 1
                tp = None
            on = (S > ma11[t]) if gate else True
            if on and t in nxt and cash > 0:
                T = (nxt[t] - t) / 252
                if hedge > 0:
                    Kt = TAIL_OTM * S
                    n_t = max(int(round(hedge * cash / (100 * S))), 1)
                    cash -= bs_put(S, Kt, sig + SKEW_BUY, TAIL_MONTHS / 12) * 100 * n_t + COMM
                    trades += 1
                    tp = {"K": Kt, "n": n_t, "exp": t + int(TAIL_MONTHS * 21)}
                Ks = S * math.exp(-CUSHION * sig * math.sqrt(T))
                n_s = int(cash // (Ks * 100))
                if n_s > 0:
                    prem = bs_put(S, Ks, sig + SKEW_PUT, T) * 100 * n_s
                    cash += prem - COMM
                    trades += 1
                    sp = {"K": Ks, "n": n_s, "exp": nxt[t]}
                    if call_budget > 0:
                        Kc = S * math.exp(call_otm * sig * math.sqrt(T))
                        c1 = bs_call(S, Kc, sig, T) * 100          # flat vol: overpay
                        n_c = int(call_budget * prem // c1)
                        if n_c > 0:
                            cash -= c1 * n_c + COMM
                            trades += 1
                            cp = {"K": Kc, "n": n_c, "exp": nxt[t]}
        if cash > 0:
            cash *= 1 + R_CASH / 252
        v = vxn[t] if not np.isnan(vxn[t]) else 0.3
        m = cash
        if sp is not None:
            m -= bs_put(cl[t], sp["K"], v + SKEW_PUT, max((sp["exp"] - t) / 252, 0.0)) * 100 * sp["n"]
        if cp is not None:
            m += bs_call(cl[t], cp["K"], v, max((cp["exp"] - t) / 252, 0.0)) * 100 * cp["n"]
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


def capture(nav, qqq_close, idx, up=True):
    """Average strategy monthly return in months where QQQ moved > +3% (up)
    or < -3% (down)."""
    n = nav.resample("ME").last().pct_change().dropna()
    q = pd.Series(qqq_close, index=idx).resample("ME").last().pct_change().dropna()
    n, q = n.align(q, join="inner")
    sel = q > 0.03 if up else q < -0.03
    return n[sel].mean(), q[sel].mean(), int(sel.sum())


def main():
    idx, op, cl, df = load()
    ma11, vxn, vxn_prev, first = precompute(idx, df)
    IS, OOS, ALL = idx < SPLIT, idx >= SPLIT, np.ones(len(idx), bool)
    print(f"QQQ + real VXN, {idx[0].date()} -> {idx[-1].date()}\n")

    sh0 = int((START_HKD / FX - COMM) // op[0])
    bh = pd.Series((START_HKD / FX - sh0 * op[0] - COMM) + sh0 * cl, index=idx)
    results = [("Buy & Hold QQQ", bh, 1)]
    for name, kw in [
        ("RR100 gate (put->calls)", dict(gate=True, call_budget=1.0)),
        ("RR50 gate", dict(gate=True, call_budget=0.5)),
        ("RR100 gate, call@.25s", dict(gate=True, call_budget=1.0, call_otm=0.25)),
        ("RR100 gate + tail h=1", dict(gate=True, call_budget=1.0, hedge=1.0)),
        ("RR100 always (no gate)", dict(gate=False, call_budget=1.0)),
    ]:
        nav, tr = run(idx, op, cl, ma11, vxn, vxn_prev, first, **kw)
        results.append((name, nav, tr))

    hdr = f"{'CAGR':>7}{'Shrp':>6}{'MaxDD':>7}"
    print(f"{'':24}{'IN-SAMPLE':>20}|{'OUT-OF-SAMPLE':>20}|{'FULL':>20}{'trds':>6}{'endHKD':>12}")
    print(f"{'':24}{hdr:>20}|{hdr:>20}|{hdr:>20}")
    for name, nav, tr in results:
        a, b, c = stats(nav, IS), stats(nav, OOS), stats(nav, ALL)
        print(f"{name:24}"
              f"{a['CAGR']:>7.1%}{a['Sharpe']:>6.2f}{a['MaxDD']:>7.0%}|"
              f"{b['CAGR']:>7.1%}{b['Sharpe']:>6.2f}{b['MaxDD']:>7.0%}|"
              f"{c['CAGR']:>7.1%}{c['Sharpe']:>6.2f}{c['MaxDD']:>7.0%}"
              f"{tr:>6}{nav.iloc[-1]*FX:>12,.0f}")

    print("\nRally capture (months QQQ > +3%) and crash months (QQQ < -3%):")
    print(f"{'':24}{'strat avg':>10}{'QQQ avg':>9}{'months':>8}   |"
          f"{'strat avg':>10}{'QQQ avg':>9}{'months':>8}")
    for name, nav, _ in results:
        u = capture(nav, cl, idx, up=True)
        d = capture(nav, cl, idx, up=False)
        print(f"{name:24}{u[0]:>10.1%}{u[1]:>9.1%}{u[2]:>8}   |"
              f"{d[0]:>10.1%}{d[1]:>9.1%}{d[2]:>8}")


if __name__ == "__main__":
    main()
