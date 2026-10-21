"""
gate_analysis.py — 切片 vs 変動の分離、および絶対ゲート／相対ゲートの対照比較

背景
----
偏IC の結果から:
  ・スコアの「変動」はドリフト追随ではない（半減期 4 本、IC_raw ≈ IC_par）
  ・スコアの「切片」が上昇レジーム由来（中央値 0.539、対数オッズ +0.156）
  ・順位相関は切片に不変 → IC は切片を見ていない
  ・conf>=0.60 という絶対閾値は切片をまるごと受け継ぐ

したがって絶対閾値を「過去 N 本の中の上位 X%」に置き換えると、切片が自動的に
落ちる。その効果を測る。

出力
----
  A. レジーム別スコア分布とゲート通過率（比率なので判定可能）
  B. 絶対ゲート／相対ゲートそれぞれのゲート後リターン
  C. 両者の差のペア比較（同一バー上なので差の分散は個別より小さい）
  D. ローリング順位スコアの偏IC

使い方
------
  python3 gate_analysis.py --m1 XAUUSD_M1_...csv --score model_scores.csv
"""

from __future__ import annotations
import argparse
import numpy as np
import pandas as pd

from excess_return import build_targets, partial_ic
from ic_report import load_m1_to_h1, autocorr_half_life, REGIMES, \
    HORIZON, DRIFT_WINDOW, ATR_PERIOD

RANK_WINDOW = 500      # 相対ゲートの参照窓（因果的）
GATE_FRAC = 0.174      # conf>=0.60 の全期間通過率に合わせる


# ------------------------------------------------------------- ブートストラップ

def block_boot(values: np.ndarray, block: int, n_boot: int = 2000, seed: int = 0):
    """平均のブロックブートストラップ CI。values は時系列順であること。"""
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    n = len(v)
    if n < 30:
        return (v.mean() if n else np.nan), np.nan, np.nan
    if n < block * 3:
        # 小標本ではブロックを縮める（NaN で黙って消さない）
        block = max(1, n // 10)
    rng = np.random.default_rng(seed)
    nb = n // block
    pool = np.arange(0, n - block)
    out = np.empty(n_boot)
    ar = np.arange(block)
    for b in range(n_boot):
        st = rng.choice(pool, size=nb, replace=True)
        out[b] = v[(st[:, None] + ar[None, :]).ravel()].mean()
    lo, hi = np.percentile(out, [2.5, 97.5])
    return v.mean(), lo, hi


def paired_boot(diff: np.ndarray, mask: np.ndarray, block: int,
                n_boot: int = 2000, seed: int = 0):
    """
    ペア比較。同一バー上で「絶対ゲートの寄与 − 相対ゲートの寄与」を
    バー単位の系列として持ち、その平均をブロックブートストラップする。
    片方だけが通ったバーの情報も落とさない。
    """
    return block_boot(np.where(mask, diff, np.nan), block, n_boot, seed)


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m1", required=True)
    ap.add_argument("--score", required=True)
    a = ap.parse_args()

    h1 = build_targets(load_m1_to_h1(a.m1), HORIZON, DRIFT_WINDOW, ATR_PERIOD)
    sc = pd.read_csv(a.score)
    sc.index = pd.to_datetime(sc.iloc[:, 0])
    d = h1.join(sc["score"], how="inner")
    print(f"H1 {len(h1)} 本 / スコア {len(sc)} 件 / 重なり {len(d)} 本")

    # --- 相対スコア: 過去 RANK_WINDOW 本の中での順位（因果的、当該バーを含む）
    d["score_rank"] = (d["score"]
                       .rolling(RANK_WINDOW, min_periods=RANK_WINDOW)
                       .rank(pct=True))

    d["gate_abs"] = d["score"] >= 0.60
    d["gate_rel"] = d["score_rank"] >= (1 - GATE_FRAC)

    hl = autocorr_half_life(d["score"].to_numpy())
    block = max(4 * HORIZON, 4 * hl)
    print(f"スコア半減期 {hl} 本 / ブロック長 {block} 本 / 相対窓 {RANK_WINDOW} 本\n")

    spans = [("全期間", None, None)] + REGIMES
    rows_a, rows_b, rows_c, rows_d = [], [], [], []

    for name, lo, hi in spans:
        s = d if lo is None else d.loc[lo:hi]
        s = s.dropna(subset=["score", "fwd_h", "drift_causal"])
        if len(s) < 500:
            continue

        # ---- A 分布と通過率
        q = np.percentile(s["score"], [25, 50, 75])
        rows_a.append(dict(regime=name, n=len(s),
                           p25=q[0], p50=q[1], p75=q[2],
                           logodds=np.log(q[1] / (1 - q[1])),
                           pass_abs=s["gate_abs"].mean(),
                           pass_rel=s["gate_rel"].mean()))

        sr = s.dropna(subset=["score_rank"])
        if len(sr) < 500:
            continue

        # ---- B ゲート後リターン（生 6 バーリターン、コスト除く）
        for gname, col in (("絶対 conf>=0.60", "gate_abs"),
                           ("相対 上位17.4%", "gate_rel")):
            v = sr.loc[sr[col], "fwd_h"].to_numpy()
            m, l, h = block_boot(v, block)
            rows_b.append(dict(regime=name, gate=gname, n=len(v),
                               EV=m, lo=l, hi=h,
                               winrate=float(np.mean(v > 0)) if len(v) else np.nan))

        # ---- C ペア比較（同一バー上の差）
        # 各バーの寄与 = (通過なら fwd_h、通過しなければ 0) を通過率で正規化
        ca = np.where(sr["gate_abs"], sr["fwd_h"], 0.0) / max(sr["gate_abs"].mean(), 1e-9)
        cr = np.where(sr["gate_rel"], sr["fwd_h"], 0.0) / max(sr["gate_rel"].mean(), 1e-9)
        m, l, h = block_boot(ca - cr, block)
        rows_c.append(dict(regime=name, n=len(sr),
                           diff_abs_minus_rel=m, lo=l, hi=h,
                           overlap=float(np.mean(sr["gate_abs"] & sr["gate_rel"]))))

        # ---- D 相対スコアの偏IC（変動だけを見る）
        p_raw = partial_ic(sr["score"], sr["fwd_h"], [sr["drift_causal"]],
                           HORIZON, n_boot=1000, block=block)
        p_rel = partial_ic(sr["score_rank"], sr["fwd_h"], [sr["drift_causal"]],
                           HORIZON, n_boot=1000, block=block)
        rows_d.append(dict(regime=name, n=len(sr),
                           IC_score=p_raw["ic"], s_lo=p_raw["lo"], s_hi=p_raw["hi"],
                           IC_rank=p_rel["ic"], r_lo=p_rel["lo"], r_hi=p_rel["hi"]))

    opt = dict(index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print("=== A. スコア分布とゲート通過率 ===")
        print(pd.DataFrame(rows_a).round(4).to_string(**opt))
        print("\n=== B. ゲート後の 6 バーリターン（ATR単位・コスト除く）===")
        print(pd.DataFrame(rows_b).round(4).to_string(**opt))
        print("\n=== C. 絶対 − 相対 のペア比較（負なら相対ゲートが優る）===")
        print(pd.DataFrame(rows_c).round(4).to_string(**opt))
        print("\n=== D. 偏IC: 生スコア vs ローリング順位スコア ===")
        print(pd.DataFrame(rows_d).round(4).to_string(**opt))

    print("\n注: B の EV はコストを含まない。判定には C（差）を使うこと。")
    print("    C は同一バー上のペア比較なので、B の各列より分散が小さい。")


if __name__ == "__main__":
    main()
