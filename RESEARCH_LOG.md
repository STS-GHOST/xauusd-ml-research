# Research log

**English** | [日本語](RESEARCH_LOG.ja.md)

A record of rebuilding the measurement machinery for a CatBoost direction model on
XAUUSD H1, through to a conclusion. Organised by **what was learned / what was wrong /
what is open**, not chronologically.

---

## 1. Conclusion

Purge/embargo walk-forward, 2021-01 to 2026-07, fully out-of-sample:

| Regime | n | Partial IC | 95% CI |
|---|---|---|---|
| Full period | 32,941 | **−0.0002** | [−0.0195, +0.0193] |
| 2021-22 flat | 11,763 | −0.0147 | [−0.0516, +0.0138] |
| 2023-26 uptrend | 21,178 | +0.0099 | [−0.0139, +0.0344] |

Converted to EV (gate pass rate 12%, 6-bar return sd = 1.85 ATR, multiplier 3.084):

| | IC | EV (ATR/trade) |
|---|---|---|
| Point estimate | −0.0002 | −0.001 |
| **95% upper bound** | **+0.0193** | **+0.059** |
| Required | 0.029 | 0.090 |
| Friction (spread + stop only) | — | **0.053** |

**The 95% upper bound of the signal barely exceeds the most generous estimate of
friction.** Excluded, not undecidable.

---

## 2. Measurement design

### 2-1 Why bar-level instead of trade-level

The original evaluation compared EV across gated trades (n ≈ 200–500). With
sd ≈ 1.7–2.0 ATR and a target effect of 0.09 ATR, the required sample is

```
n ≈ (2.5 · sd / 0.09)² ≈ 3,000 trades  →  17 years at 14.5 trades/month
```

**Within that frame the question is undecidable in principle.** The gate discards 88%
of the measurable population — correct as trading, self-harm as measurement.

| Measured as | Effective n | SE of EV (ATR) |
|---|---|---|
| Gated trades | 192 | 0.123 |
| Bar-level IC | 3,190–6,500 | **0.044** |

A 2.8× gain in precision. On that footing, "is there signal" (a test) was separated
from "can the trading wrapper capture it" (a design judgement).

### 2-2 Why partial IC

Targeting excess return `fwd − h·drift(N)` injects, for a score correlated ρ with drift:

```
mechanical term = −ρ · h/√(N·h) = −ρ × 0.110     (N=500, h=6)
```

Measured on a positive control (a pure drift-tracking score): −0.083 on one draw,
**−0.0615** averaged over 8 independent draws. A spurious negative more than twice the
size of the effect being sought.

→ Switched to a partial correlation that removes drift from **both** the score and the
target. Over the same 8 draws: mean **+0.0015**, sd 0.0085. Unbiased.

**The same arithmetic applies to training labels.** Putting `−h·drift(500)` into the
label makes that term 11% of label sd and almost perfectly reconstructible from
features such as `dist_ma200`. Gradient boosting eats it first and learns to fade
recent drift. → **Train on raw returns; use partial IC only for evaluation.**

### 2-3 Block length

The SE of IC depends on the serial correlation of the score and cannot be assumed.
Measured on synthetic data:

| Score half-life | SE(IC) | Effective n |
|---|---|---|
| None | 0.0050 | 39,000 |
| 7 bars | 0.0108 | 8,600 |
| 23 bars | 0.0123 | 6,600 |

The real score has a half-life of **4 bars**. Block length is derived from the measured
value on every run.

---

## 3. Stagewise decomposition

From raw signal to realised P&L (2021-22, ATR units). S2→S3→S4 are transforms on an
identical trade set, so they admit a paired comparison with much lower variance.

| Stage | Contribution | CI | Verdict |
|---|---|---|---|
| `atr_ratio ≥ 1.10` | −0.091 | — | See §4 |
| 3-ATR stop-loss | −0.027 | [−0.071, +0.007] | Undecidable |
| Spread | **−0.026** | [−0.028, −0.025] | **Determined** |
| Single-position constraint | −0.106 / +0.018 | — | **Unstable (open)** |

### The cost assumption was too high

Measured spread, median **0.0156–0.0245 ATR**. The research code's `COST_ATR = 0.0287`
overstated it by roughly 1.3× in 2021-22. The stop-loss hit rate was 3.0–11.3% and its
contribution stayed within the undecidable range.

---

## 4. The `atr_ratio` filter — disabled

Three thresholds × two regimes, final-stage EV (ATR):

| Filter | 2020 | n | 2021-22 | n |
|---|---|---|---|---|
| None | −0.026 | 317 | **−0.001** | 714 |
| ≥ 0.90 | −0.019 | 248 | −0.003 | 590 |
| ≥ 1.10 | −0.094 | 157 | −0.038 | 329 |

**Tighter is worse, with consistent sign across all 6 cells.** No cell is individually
decidable, but there is no supporting evidence and it discards 54% of the sample.
**Disabled as a design judgement, not as a statistical win.**

Note: at the intermediate stage (on `fwd_h`), the 1.10 filter looked **favourable in
2020 (+0.069)** and **unfavourable in 2021-22 (−0.091)** — opposite signs. At the final
stage both are unfavourable. The filter affects selectivity, stop-loss hit rate and the
spread/ATR ratio simultaneously, so **a single stage does not extrapolate to the final
outcome.**

---

## 5. How the original conclusion went wrong

The original evaluation reported: losses on all 5 seeds in the flat regime (2021-22),
gains on all 5 seeds in the uptrend (2023-26). The interpretation was rewritten three
times.

### 5-1 "It was all drift" → rejected

Measured drift per 6-bar holding period: **0.055–0.089 ATR**. That accounts for roughly
40% of the EV gap (0.203), but **only 1.9pp of the 13pp win-rate gap.**

### 5-2 "It was all the intercept" → rejected

Score median 0.539 (log-odds +0.156). Hypothesis: the absolute threshold
`conf ≥ 0.60` inherits an intercept learned in an uptrend regime.

| | log-odds | Absolute gate | Rolling-rank gate |
|---|---|---|---|
| 2020 | 0.1585 | 16.66% | 17.36% |
| 2021-22 | 0.1653 | 17.91% | 17.51% |

**The intercept exists but does not move between regimes.** The two gates overlap 91%.
Replacing the absolute gate with a rolling-rank gate was effectively a no-op.

Separately, `IC_raw ≈ IC_par` (difference 0.0002–0.0015) holds in every regime,
including the strong 2020 uptrend. So **the score's *variation* does not track drift;
the drift-tracking lives in the *intercept*.** Rank correlation is invariant to the
intercept, so IC never saw it.

### 5-3 "The uptrend was in-sample" → confirmed

```python
XAU_M1_PATH = "XAUUSD_M1_202301030100_202607312354.csv"   # starts 2023-01-03
model.fit(X, y)                                           # no holdout
```

**The entire 2023-26 evaluation was in-sample.** Under walk-forward, the regime
difference disappeared.

Comparing the same 2021-22 bars, changing only the training procedure, flips the sign:

| | Frozen model | Walk-forward |
|---|---|---|
| Partial IC | **+0.0147** | **−0.0147** |
| Gated EV | +0.0637 | −0.0504 |
| Final EV | −0.038 | **−0.209** [−0.323, −0.074] |

The CIs overlap, so this is not a statistical contradiction — but **the estimate is not
robust to the training procedure.** That final −0.209 is the first result in the whole
exercise whose CI excludes zero, and it points down.

---

## 6. Defects found and fixed

| # | Defect | Consequence |
|---|---|---|
| 1 | Training data started 2023, no holdout | Invalidated the entire regime comparison |
| 2 | No purge/embargo in walk-forward | 6 bars leaked at every fold boundary |
| 3 | Validation code entered one bar early | Two conclusions reversed (below) |
| 4 | Excess return used as evaluation target | Spurious IC of −0.06 |
| 5 | ATR: SMA in Python vs Wilder in MQL5 | 10.17% mean abs diff, 65.54% max |
| 6 | Index mismatch between `score` and `target` | IC silently returned NaN |

### Detail on #3

The correct convention is **signal = bar t (on its close) / entry = bar t+1 open /
stop checked over bars t+1..t+6 / exit at the stop price or bar t+6 close.**

The faulty implementation entered at bar *t*'s open and scanned a 7-hour window from
bar *t*. What it produced:

| Claim | Faulty value | After fix |
|---|---|---|
| The 3-ATR stop destroys value | −0.075 (−1.78 ATR per hit) | **−0.007** |
| Single-position constraint is the biggest loss, via length bias | −0.116 | **+0.018** |

**Both retracted.** The stop-loss hit rate also fell from 4.2% to 3.0%.

### Why #4 escaped the null test

A meaningless score reads 0 under both statistics. The defect only surfaced on asking
**"does a known input return the known answer?"**

---

## 7. Lessons

**1. Use positive controls, not only null tests.**
Without an input of known character (a pure drift tracker), #4 was undetectable.

**2. Pin the timing convention in a test.**
Kept in `tests/test_timing_convention.py`. "Corrupting bar *t* must not change P&L" is
not enough on its own — it must be paired with "corrupting bar *t+1* must change P&L",
otherwise an implementation that uses nothing at all passes.

**3. Do not extrapolate a single stage to the final outcome.**
A filter changes selectivity, stop-loss hit rate and spread/ATR ratio at once (§4).

**4. Do not take logs of quantities near zero. Gate derived statistics.**
Fitting `log|IC| ~ log h` to synthetic noise produced slopes of **+1.24** and **−1.06**.
`horizon_sweep.py` now refuses to report a slope unless at least half the horizons have
a CI excluding zero.

**5. A frozen model's "OOS" is defined by its training data start date.**
One line of a filename determined the meaning of every conclusion (§5-3).

**6. When observation contradicts theory, suspect the implementation first.**
When the prediction "the single-position constraint should not move EV" appeared to
fail, a mechanism (length bias) was constructed to explain it. It was a bug (#3).

**7. Test that the test can fail.**
`test_unpurged_split_would_leak` verifies that the purge check is not a vacuous assertion.

---

## 8. Open questions

1. **The single-position constraint changes sign with conditions.**
   +0.018 with the frozen model, −0.106 under walk-forward. The stop-loss hit rate also
   moved from 3.0% to 11.3%, which is consistent with a length-bias hypothesis: greater
   dispersion in holding time oversamples short-lived (stopped-out) trades.
   **Test:** sweep the stop width and look for correlation between the degradation and
   the hit rate. Currently a hypothesis, not a finding.

2. **The holding-period sweep results are not adopted.**
   11 of 12 cells undecidable. Also: (a) rows across horizons are nested within one
   series and are not independent, (b) drift control is fixed at 500 bars, so longer
   horizons are less well controlled, (c) the two regimes have opposite signs.
   "IC increases with h" is plausibly just picking up each period's medium-term
   component. **Any horizon work must restart from the control design.**

3. **Train/serve distribution mismatch.**
   Training drops the neutral band (|move| ≤ 0.5 ATR, 30% of bars) but inference runs on
   every bar. IC is a rank correlation and is unaffected, but `conf` calibration is not —
   which matters as long as an absolute threshold is used.

4. **2020 is never out-of-sample.**
   Consumed by `min_train = 6000`. Behaviour in the high-volatility, high-spread period
   remains unmeasured.

5. **Correction to an earlier record on triple-barrier labels.**
   A previous note recorded "triple barrier (±3 ATR) was tried and failed." That is
   wrong. With a 6-bar return sd of 1.7 ATR, almost every bar times out; the negative
   class reaches 90.6% and the model collapses to a near-constant output. Identical
   results across seeds were a consequence of the label definition, not a bug.
   **The correct record is "±3 ATR was never a valid test."** Retrying would require
   widths of ±1.0–1.5 ATR, or splitting timeout into a third class.

---

## 9. Where this could go

Detection limit: 38,947 bars is all the data there is. Effective n ≈ 6,500,
SE(IC) ≈ 0.010. **Within this frame (single instrument / H1 / 6-bar holding /
9 price-derived features), swapping features alone still only resolves effects of
IC ≳ 0.025.**

1. **Change the features.** Currently all self-price derived. Cross-asset inputs,
   session structure (Tokyo/London/NY handovers), spread itself as a feature.

2. **Change the target.** Volatility or range rather than direction. Easier to predict,
   and it acts directly on the largest friction term (spread, 0.026).

3. **Leave the frame.** At a signal ceiling of 0.059 against friction of 0.053, nothing
   measurable on this timeframe and instrument is large enough to trade.

Direction prediction is 0/9 across this project. **Option 2 is the recommendation.**
