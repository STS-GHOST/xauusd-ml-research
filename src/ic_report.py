"""
ic_report.py — バー単位 IC 診断

目的
----
既存 CatBoost のスコアが「ドリフト追随」なのか「ドリフト以外の信号」なのかを、
再学習なしで分離する。

  IC(score, fwd_h)          … 生リターンへの当てはまり
  IC(score, excess_causal)  … ドリフト除去後への当てはまり

  前者 > 0 かつ 後者 ≈ 0  → ドリフト追随器（BUY比率98-100% の正体）
  両方 > 0                → ドリフト以外の信号がある

レジームは明示的に固定する（教訓45: 記憶ではなく構造で担保）。

使い方
------
  python ic_report.py --m1 XAUUSD_M1_202001020100_202607312354.csv \
                      --score model_scores.csv

  score CSV は 2 列: datetime, score （H1 の足時刻に合わせる）

  --demo を付けると合成データで配管とポジコンだけ検証する（実データ不要）
"""

from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from excess_return import build_targets, information_coefficient, partial_ic, _synth

HORIZON = 6
DRIFT_WINDOW = 500          # 約1か月（H1）
ATR_PERIOD = 14

REGIMES = [
    ("2020 高ボラ",   "2020-01-01", "2020-12-31"),
    ("2021-22 横ばい", "2021-01-01", "2022-12-31"),
    ("2023-26 上昇",   "2023-01-01", "2026-12-31"),
]


# ------------------------------------------------------------------ データ

def load_m1_to_h1(path: str) -> pd.DataFrame:
    """MT5 のタブ区切り M1 エクスポートを H1 に集約。"""
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.strip("<>").lower() for c in df.columns]
    if "date" in df.columns and "time" in df.columns:
        ts = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
    else:
        ts = pd.to_datetime(df.iloc[:, 0])
    df.index = ts

    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "spread" in df.columns:
        agg["spread"] = "median"
    if "tickvol" in df.columns:
        agg["tickvol"] = "sum"

    h1 = df.resample("1h").agg(agg).dropna(subset=["close"])
    return h1


def autocorr_half_life(x: np.ndarray, max_lag: int = 400) -> int:
    """スコアの持続性。ブロック長の根拠にする（ブロック長を勘で決めない）。"""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    x = x - x.mean()
    if x.std() == 0:
        return 1
    for lag in range(1, min(max_lag, len(x) // 4)):
        r = np.corrcoef(x[:-lag], x[lag:])[0, 1]
        if r < 0.5:
            return lag
    return max_lag


# ------------------------------------------------------------------ 集計

def ic_table(h1: pd.DataFrame, score: pd.Series, label: str = "") -> pd.DataFrame:
    joined = h1.join(score.rename("score"), how="inner")
    hl = autocorr_half_life(joined["score"].to_numpy())
    block = max(4 * HORIZON, 4 * hl)

    rows = []
    spans = [("全期間", None, None)] + REGIMES
    for name, lo, hi in spans:
        sub = joined if lo is None else joined.loc[lo:hi]
        sub = sub.dropna(subset=["score", "fwd_h", "excess_causal"])
        if len(sub) < 200:
            rows.append(dict(regime=name, n=len(sub)))
            continue
        raw = information_coefficient(sub["score"].to_numpy(),
                                      sub["fwd_h"].to_numpy(),
                                      HORIZON, n_boot=1000, block=block)
        exc = information_coefficient(sub["score"].to_numpy(),
                                      sub["excess_causal"].to_numpy(),
                                      HORIZON, n_boot=1000, block=block)
        par = partial_ic(sub["score"].to_numpy(), sub["fwd_h"].to_numpy(),
                         [sub["drift_causal"].to_numpy()],
                         HORIZON, n_boot=1000, block=block)
        rows.append(dict(
            regime=name, n=len(sub),
            IC_raw=raw["ic"], raw_lo=raw["lo"], raw_hi=raw["hi"],
            IC_par=par["ic"], par_lo=par["lo"], par_hi=par["hi"],   # ← 主判定
            IC_exc=exc["ic"],                                        # 参考のみ（偏りあり）
        ))
    out = pd.DataFrame(rows)
    print(f"\n=== {label} ===")
    print(f"スコア半減期 {hl} 本 / ブロック長 {block} 本")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(out.round(4).to_string(index=False))
    return out


# ------------------------------------------------------------------ デモ

def demo():
    """実データなしで配管と検出力を検証する。"""
    n = 40000
    df = _synth(n, mu=0.0, sigma=1.0, seed=11)
    # 前半 15000 本はドリフトなし、後半は正のドリフト（レジームを人工的に作る）
    step = np.diff(df["close"].to_numpy(), prepend=df["close"].iloc[0])
    step[15000:] += 0.03
    close = df["close"].iloc[0] + np.cumsum(step)
    rng = np.random.default_rng(3)
    df = pd.DataFrame(dict(open=close,
                           high=close + np.abs(rng.normal(0, .5, n)),
                           low=close - np.abs(rng.normal(0, .5, n)),
                           close=close), index=df.index)

    h1 = build_targets(df, HORIZON, DRIFT_WINDOW, ATR_PERIOD)

    # --- ポジコン1: 純粋なドリフト追随器（過去200本の平均リターン）
    r = h1["r_bar"].rolling(200, min_periods=200).mean()
    ic_table(h1, r, "ポジコン1: ドリフト追随器（excess IC が消えるはず）")

    # --- ポジコン2: 本物の信号（未来の excess を薄く混ぜる）
    tgt = h1["excess_causal"]
    sig = 0.05 * tgt + rng.normal(0, tgt.std(), len(tgt))
    sig = pd.Series(sig, index=h1.index).rolling(20, min_periods=1).mean()  # 現実的に持続させる
    ic_table(h1, sig, "ポジコン2: 本物の信号 IC≈0.05（両方で残るはず）")

    # --- 帰無: 持続する無意味なスコア
    e = rng.normal(size=len(h1))
    noise = pd.Series(e, index=h1.index).ewm(span=40).mean()
    ic_table(h1, noise, "帰無: 持続するノイズ（両方 CI が 0 を跨ぐはず）")


# ------------------------------------------------------------------ main

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--m1")
    ap.add_argument("--score")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()

    if a.demo or not a.m1:
        demo()
    else:
        h1 = load_m1_to_h1(a.m1)
        print(f"H1 {len(h1)} 本  {h1.index[0]} .. {h1.index[-1]}")
        h1 = build_targets(h1, HORIZON, DRIFT_WINDOW, ATR_PERIOD)
        sc = pd.read_csv(a.score)
        sc.index = pd.to_datetime(sc.iloc[:, 0])
        score = sc.iloc[:, 1]
        print(f"score {len(score)} 件  重なり {len(h1.index.intersection(score.index))} 件")
        ic_table(h1, score, "実データ")
