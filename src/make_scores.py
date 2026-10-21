"""
make_scores.py — 凍結 ONNX から OOS スコアを吐く

重要な前提
----------
final_model_export_wilder_d4l30.py の学習データは

    XAU_M1_PATH = "XAUUSD_M1_202301030100_202607312354.csv"   ← 2023-01-03 開始

であり、かつ「全データでモデル学習」（ホールドアウトなし）。したがって

    2020        … OOS   ○
    2021-22     … OOS   ○   ← §0 の横ばい期。ここは完全にクリーン
    2023-26     … in-sample ✕  この ONNX では測定不能

既定では 2023-01-03 以降を除外して出力する。--include-insample で残せるが、
その場合 IC は水増しされる。レジーム比較には使えない。

特徴量は既存スクリプトからそのまま import する（定義のずれを持ち込まないため）。

使い方
------
  python3 make_scores.py \
      --m1 XAUUSD_M1_202001020100_202607312354.csv \
      --onnx xauusd_catboost_wilder_d4l30.onnx \
      --out model_scores.csv
"""

from __future__ import annotations
import argparse
import sys
import numpy as np
import pandas as pd

# 既存スクリプトの特徴量定義をそのまま使う（main() は __main__ 下なので実行されない）
try:
    from train_export_onnx import (
        load_m1, resample_to_h1, add_base_features_corrected,
    )
except ImportError:  # pragma: no cover
    sys.exit("train_export_onnx.py を同じフォルダに置いてください")

FEATURE_COLS = [
    "ret_1", "ret_3", "ret_6",
    "dist_ma50", "dist_ma200",
    "rsi", "atr_ratio",
    "hour", "dayofweek",
]

TRAIN_START = "2023-01-03"   # 学習データの開始 = in-sample の始まり


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m1", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", default="model_scores.csv")
    ap.add_argument("--include-insample", action="store_true")
    a = ap.parse_args()

    import onnxruntime as ort

    print("1. M1 読み込み → H1 集約 → 特徴量生成")
    h1 = resample_to_h1(load_m1(a.m1))
    h1 = add_base_features_corrected(h1)
    print(f"   H1 {len(h1)} 本  {h1['datetime'].min()} .. {h1['datetime'].max()}")

    clean = h1.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    print(f"   特徴量が揃う行 {len(clean)} 本（ウォームアップで {len(h1)-len(clean)} 本落ち）")

    print("\n2. ONNX 推論")
    sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    print(f"   入力 {in_name} {sess.get_inputs()[0].shape}")
    print(f"   出力 {[o.name for o in sess.get_outputs()]}")

    X = clean[FEATURE_COLS].astype(np.float32).values
    out = sess.run(None, {in_name: X})[0]
    out = np.asarray(out)
    if out.ndim != 2 or out.shape[1] != 2:
        sys.exit(f"想定外の出力形状 {out.shape}（ZipMap 除去済みの [N,2] を期待）")
    score = out[:, 1]          # P(上昇)

    res = pd.DataFrame({"datetime": clean["datetime"], "score": score})

    # --- 健全性チェック（黙って壊れないように）
    print(f"\n3. スコア分布")
    q = np.percentile(score, [1, 25, 50, 75, 99])
    print(f"   p1={q[0]:.4f} p25={q[1]:.4f} p50={q[2]:.4f} p75={q[3]:.4f} p99={q[4]:.4f}")
    print(f"   ユニーク値 {len(np.unique(np.round(score,6)))} 個 / {len(score)} 行")
    if len(np.unique(np.round(score, 4))) < 20:
        print("   *** 警告: スコアがほぼ定数です。IC は測れません ***")
    print(f"   conf>=0.60 の割合 {np.mean(score>=0.60):.1%}")

    # --- in-sample 区間の扱い
    mask_oos = res["datetime"] < TRAIN_START
    n_in = (~mask_oos).sum()
    print(f"\n4. in-sample 判定（学習開始 {TRAIN_START}）")
    print(f"   OOS       {mask_oos.sum():6d} 本")
    print(f"   in-sample {n_in:6d} 本")
    if a.include_insample:
        print("   --include-insample 指定: in-sample も出力します。")
        print("   *** この出力でレジーム比較をしてはいけません ***")
    else:
        res = res[mask_oos]
        print(f"   in-sample を除外 → 出力 {len(res)} 本")

    res.to_csv(a.out, index=False)
    print(f"\n5. 出力 {a.out}  {len(res)} 行")
    if len(res):
        print(f"   {res['datetime'].min()} .. {res['datetime'].max()}")


if __name__ == "__main__":
    main()
