# xauusd-ml-research

**English** | [日本語](README.ja.md)

A record of validating a CatBoost direction model on XAUUSD (gold) hourly bars, through to an Expert Advisor implementation for MetaTrader 5.

## Result: no edge was found

Evaluated with purge/embargo walk-forward over 2021-01 to 2026-07, fully out-of-sample:

| | Value |
|---|---|
| Drift-conditioned partial IC (full period) | **−0.0002** |
| 95% confidence interval | **[−0.0195, +0.0193]** |
| Sample | 32,947 bars |
| Required for viability | IC ≥ 0.029 — **outside the interval** |

Converted to expected value per trade:

```
Signal, 95% upper bound   +0.059 ATR
Friction floor            +0.053 ATR   (spread 0.027 + stop-loss 0.027)
                          ─────────────
Break-even at best. Not tradeable.
```

This is not "undecidable yet." It is "an effect of the size being sought is not present in this feature set."

The repository exists to preserve the measurement machinery that produced that negative result, and the mistakes of my own that I found and corrected on the way. There is no upward-sloping equity curve here.

---

## Why publish a negative result

The real work on this project went into the design of the measurement, not into improving the model.

The original evaluation showed a clean story: losses in the flat regime (2021-22), profits in the uptrend (2023-26). It was wrong, and the cause was one line:

```python
XAU_M1_PATH = "XAUUSD_M1_202301030100_202607312354.csv"   # starts 2023-01-03
...
model.fit(X, y)                                           # no holdout
```

**Training data began in 2023, so the entire "profitable uptrend" evaluation window was in-sample.** Under a correct walk-forward, the regime difference disappeared.

Other defects of the same family:

- The walk-forward had no purge/embargo. Labels are `shift(-6)`, so every fold boundary leaked 6 bars.
- My own validation code entered one bar early, introducing look-ahead. Two conclusions reversed after the fix.
- Using excess return as the evaluation target produced a spurious negative IC from a mechanical term in its own definition.
- ATR was computed with SMA in Python and Wilder smoothing in MQL5 (10.17% mean absolute difference).

**Every one of these produced plausible-looking numbers.** They surfaced only after adding *positive controls* rather than null tests alone.

---

## Layout

```
src/
  excess_return.py      Drift removal, partial IC, block bootstrap
  ic_report.py          IC diagnostics by regime
  walk_forward.py       Walk-forward with purge/embargo + seed ensembling
  wrapper_decomp.py     Stagewise decomposition from raw signal to realised P&L
                        (stop-loss resolved on M1 bars)
  gate_analysis.py      Absolute vs rolling-rank gate, paired comparison
  make_scores.py        OOS scores from a frozen ONNX model (auto-excludes in-sample)
  horizon_sweep.py      Holding-period sweep (results NOT adopted — see research log)
  train_export_onnx.py  Production training and ONNX export (cannot be used to evaluate)

mql5/
  XAU_CatBoost_EA.mq5   MT5 Expert Advisor: ONNX inference, gating, timeout exit,
                        full decision logging

tests/
  test_excess_return.py      Regression tests on measurement definitions
  test_timing_convention.py  Look-ahead checks on entry timing
  test_walk_forward.py       Leakage checks on purge/embargo
```

### About the tests

Every test maps one-to-one onto a mistake that was actually made here.

```bash
pip install -r requirements.txt pytest
pytest tests/ -v
```

Two are worth pointing at.

**`test_excess_target_is_biased_but_partial_ic_is_not`**
Runs a positive control: a score that does nothing but track drift. It should measure ~0 on any statistic claiming to capture "signal beyond drift." With excess return as the target it systematically reads **−0.06** — a spurious negative twice the size of the effect being sought. **A null test cannot catch this defect**, because a meaningless score reads 0 under either statistic.

**`test_signal_bar_data_cannot_affect_pnl` / `test_entry_bar_open_does_affect_pnl`**
Corrupting bar *t* must not change any P&L; corrupting bar *t+1* must. Without the second test, the first would also pass for an implementation that uses nothing at all.

---

## Method

| | |
|---|---|
| Instrument | XAUUSD H1, 2020-01 to 2026-07 (38,947 bars) |
| Features | `ret_1/3/6`, `dist_ma50/200`, `rsi`, `atr_ratio`, `hour`, `dayofweek` |
| Label | 6-bar forward return / ATR, binary, ±0.5 ATR neutral band removed |
| Model | CatBoost (depth=4, l2=30, 300 iterations, lr=0.05), averaged over 5 seeds |
| Evaluation | 22 expanding folds, purge = 6 bars, embargo = 24 bars |
| Primary statistic | Partial IC, conditioned on trailing 500-bar drift |
| Uncertainty | Block bootstrap; block length set from the measured half-life of the score |
| Costs | Real spread from tick data — no fixed-cost assumption |

### Why partial IC

Targeting excess return `fwd − h·drift(N)` injects, for any score correlated ρ with drift:

```
mechanical term = −ρ · h/√(N·h) = −ρ × 0.110     (N=500, h=6)
```

So drift is removed from **both** the score and the target via partial correlation. Verified unbiased across 8 independent draws: mean +0.0015, sd 0.0085.

The same arithmetic applies to training labels. Putting `−h·drift` into the label makes that term 11% of label standard deviation and almost perfectly reconstructible from features such as `dist_ma200`. Gradient boosting consumes it first and learns to fade recent drift — a prior that is not in the data. **Hence: train on raw returns, evaluate with partial IC.**

---

## What mattered and what did not

Stagewise decomposition from raw signal to realised P&L (2021-22, ATR units):

| Stage | Contribution | Verdict |
|---|---|---|
| Spread | **−0.026** | Determined (CI width 0.002) |
| 3-ATR stop-loss | −0.027 | Undecidable |
| Single-position constraint | −0.106 / +0.018 | **Unstable** (unresolved) |
| `atr_ratio ≥ 1.10` filter | **−0.091** | See below |

The `atr_ratio` filter degraded final EV in all 6 cells of a 3-threshold × 2-regime grid, with consistent sign. No individual cell is decidable, but there is no supporting evidence and it discards 54% of the sample, so it was **disabled as a design judgement rather than a statistical win** (EA default changed from 1.10 to 0.00).

Measured cost was 0.0156–0.0245 ATR, against the `COST_ATR = 0.0287` the research code had assumed — roughly 1.3× overstated.

---

## Data

**Price data is not included.** The M1 history is licensed from a broker and cannot be redistributed.

To run anything here, export XAUUSD M1 data from MT5 as tab-separated CSV with the columns `<DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE> <TICKVOL> <VOL> <SPREAD>` and place it at the repository root.

```bash
python3 src/walk_forward.py   --m1 XAUUSD_M1_<range>.csv --out model_scores_wf.csv
python3 src/ic_report.py      --m1 XAUUSD_M1_<range>.csv --score model_scores_wf.csv
python3 src/wrapper_decomp.py --m1 XAUUSD_M1_<range>.csv --score model_scores_wf.csv
```

---

## Detail

The full experiment record, the list of corrected errors, and open questions are in [RESEARCH_LOG.md](RESEARCH_LOG.md).

## Disclaimer

**Not investment advice.** The code here is a research record and is not intended for live trading. Its conclusion is that no edge was detected. Leveraged trading can lose more than the amount deposited. Use at your own risk.

## License

MIT License — see [LICENSE](LICENSE).
