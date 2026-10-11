# Robert Carver's CTA framework — study notes

Sources: Robert Carver's blog *qoppac.blogspot.com* ("This Blog is Systematic"),
his books *Systematic Trading* (2015) and *Advanced Futures Trading Strategies*
(2023), and his open-source `pysystemtrade` / `systematictradingexamples` code.

Carver is a former AHL portfolio manager. His whole approach is a **staged
pipeline** that turns raw prices into a position, where every stage is
*volatility-normalised* so that the same machinery works on any instrument and
any speed of signal. This file is the reference the code in `carver/` implements.

---

## The big idea: forecasts, not signals

A classic system gives a binary buy/sell. Carver instead produces a **continuous
forecast**, scaled so that:

- `0`   = no opinion / flat
- `+10` = an *average-strength* long ("a regular buy")
- `-10` = an average-strength short
- capped at `±20` (twice average) so a single crazy reading can't dominate

Everything downstream consumes that `-20…+20` number. The position is literally
proportional to the forecast.

---

## The pipeline (one instrument)

```
price
  │
  ├─► volatility estimate σ  (used everywhere for normalisation)
  │
  ├─► trading rule(s) ──► raw forecast
  │        e.g. EWMAC = fast_EWMA − slow_EWMA
  │
  ├─► forecast / σ           → "vol-adjusted" forecast (dimensionless)
  │
  ├─► × forecast scalar      → scaled so |forecast| averages 10
  │
  ├─► cap at ±20             → capped forecast
  │
  ├─► weighted average over rule variations × FDM → combined forecast
  │
  └─► position sizing (volatility targeting) ──► position in shares/contracts
```

### 1. Volatility estimation
A single volatility number drives both the forecast normalisation and the
position sizing. Carver uses an **exponentially weighted standard deviation of
daily price changes** (`pysystemtrade` default span ≈ 35 days), with a floor so
it never collapses to zero in quiet markets. Annualise with √256 (he uses 256
business days/year).

### 2. Trading rule — EWMAC (the core trend rule)
Exponentially Weighted Moving Average Crossover:

```
raw_ewmac(Lfast, Lslow) = EWMA(price, Lfast) − EWMA(price, Lslow)
```

with `Lslow = 4 × Lfast`. Positive ⇒ price trending up ⇒ go long. He runs it at
several speeds simultaneously (each a "variation"):

| EWMAC (Lfast, Lslow) | character        |
|----------------------|------------------|
| 2, 8                 | very fast        |
| 4, 16                | fast             |
| 8, 32                | medium           |
| 16, 64               | medium-slow      |
| 32, 128              | slow             |
| 64, 256              | very slow        |

### 3. Vol-adjust and scale
`raw_ewmac` is in price units, so divide by the daily price-vol σ to make it
comparable across instruments and time. Then multiply by a **forecast scalar**
chosen so the long-run average of `|forecast|` is 10. Carver's published scalars
(from *Systematic Trading* Appendix B / `pysystemtrade`):

| EWMAC  | scalar |
|--------|--------|
| 2,8    | 10.6   |
| 4,16   | 7.5    |
| 8,32   | 5.3    |
| 16,64  | 3.75   |
| 32,128 | 2.65   |
| 64,256 | 1.91   |

(Each step halving the speed divides the scalar by ≈√2.) Then **cap at ±20**.

### 4. Combine variations
Combined forecast = weighted average of the capped variation forecasts,
multiplied by a **Forecast Diversification Multiplier (FDM)** that restores the
average back to 10 (averaging correlated forecasts shrinks them). FDM is
`1 / √(wᵀ · C · w)` where `C` is the forecast correlation matrix and `w` the
forecast weights; capped (Carver caps multipliers at ≈2.5). Re-cap at ±20.

### 5. Position sizing — volatility targeting
The point of the whole system: **target a constant risk**, not a constant
position. Steps (Carver's notation):

```
annual_cash_vol_target = capital × volatility_target%        # e.g. 100k × 20%
daily_cash_vol_target  = annual_cash_vol_target / √256        # /16
instrument_value_vol   = price × 1 × daily_return_vol%        # $ risk of 1 share
                       = daily std of price changes ($/share)
volatility_scalar      = daily_cash_vol_target / instrument_value_vol  # shares @ forecast 10
position (shares)      = volatility_scalar × combined_forecast / 10
```

So when markets get more volatile, σ rises, the vol scalar falls, and the
position shrinks automatically — risk stays roughly constant.

### 6. Portfolio level (multiple instruments)
For a portfolio Carver adds **instrument weights** and an **Instrument
Diversification Multiplier (IDM)**, same maths as the FDM but across
instruments. For a single instrument like QQQ both collapse to 1, so this
project focuses on stages 1–5.

---

## Applying this to QQQ (a cash equity ETF)

- **No carry rule.** Carver's second core rule, *carry*, needs a futures term
  structure / financing spread. A plain ETF has no clean carry signal, so this
  project uses the **EWMAC trend ensemble only** — the honest core of the system
  for a single ETF. (Dividend yield ≈ 0.5%/yr is ignored; the data is
  split-adjusted close.)
- **Speed selection.** Very fast variations (2,8 / 4,16) trade too often to
  survive realistic retail costs on an ETF, so the default ensemble is the
  slower, cheaper set **{8,32 · 16,64 · 32,128 · 64,256}** with equal weights.
- **Long/short.** The system is symmetric — it will go short QQQ when the trend
  is down (e.g. 2000–02, 2008, 2022). Compared against simple buy-&-hold.
- **Volatility target** defaults to 20% annualised on the trading capital.
