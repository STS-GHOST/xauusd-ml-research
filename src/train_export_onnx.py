"""
XAUUSD H1 CatBoost モデル 最終学習 & ONNX出力(Wilder修正版)

変更点:
  ・ATR・RSIをMQL5の iATR()/iRSI() と同じWilderの平滑化に修正
    (旧SMA版との平均絶対差10.17%、最大65.54% - 無視できない差だった)
  ・ハイパーパラメータを depth=4, l2_leaf_reg=30 に変更
    (vol_ratio>=1.10フィルター適用・確信度0.60=EAデフォルトでの比較で、
     EV・勝率・負けフォールド数の3指標が同時に最良だった設定)

特徴量セット・順序は元のスクリプトと同一:
  ret_1, ret_3, ret_6, dist_ma50, dist_ma200, rsi, atr_ratio, hour, dayofweek

使い方:
  MT5 から書き出した XAUUSD の M1 CSV(タブ区切り)を同じフォルダに置いて実行。
  出力: xauusd_catboost_wilder_d4l30.onnx
  (既存の xauusd_catboost.onnx を上書きする前に、必ずバックアップを取ること)

【重要・2026-08-21 追記】
  このスクリプトは **全データで学習し、ホールドアウトを取らない**。
  本番モデルの書き出し専用であり、**評価には使えない**。
  XAU_M1_PATH が 2023-01-03 開始だったため、2023-26 の評価は全て in-sample
  だった。この取り違えが過去の結論を大きく誤らせている。
  OOS 評価は src/walk_forward.py (purge/embargo 付き) を使うこと。

  価格データ本体はライセンス上リポジトリに含めていない。各自で用意すること。
"""

import os
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier


def load_m1(path):
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.strip("<>").lower() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def resample_to_h1(df):
    df = df.set_index("datetime")
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "tickvol": "sum"}
    h1 = df.resample("1h").agg(agg)
    h1 = h1.dropna(subset=["open", "close"]).reset_index()
    return h1


def true_range(high, low, close):
    prev_close = close.shift(1)
    return pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)


def wilder_atr(high, low, close, period=14):
    """MQL5 iATR() と同じWilderの平滑化"""
    tr = true_range(high, low, close)
    tr_vals = tr.values
    n = len(tr_vals)
    atr_vals = np.full(n, np.nan)
    if n > period:
        atr_vals[period - 1] = np.mean(tr_vals[0:period])
        for i in range(period, n):
            atr_vals[i] = (atr_vals[i - 1] * (period - 1) + tr_vals[i]) / period
    return pd.Series(atr_vals, index=close.index)


def wilder_rsi(close, period=14):
    """MQL5 iRSI() と同じWilderの平滑化"""
    delta = close.diff()
    gain = delta.clip(lower=0).values
    loss = (-delta.clip(upper=0)).values
    n = len(close)
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    if n > period:
        avg_gain[period] = np.nanmean(gain[1:period + 1])
        avg_loss[period] = np.nanmean(loss[1:period + 1])
        for i in range(period + 1, n):
            avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
    return pd.Series(rsi, index=close.index)


def add_base_features_corrected(df):
    high, low, close = df["high"], df["low"], df["close"]
    df["atr"] = wilder_atr(high, low, close, period=14)
    df["rsi"] = wilder_rsi(close, period=14)

    df["ret_1"] = (close - close.shift(1)) / df["atr"]
    df["ret_3"] = (close - close.shift(3)) / df["atr"]
    df["ret_6"] = (close - close.shift(6)) / df["atr"]

    ma_50 = close.rolling(50).mean()
    ma_200 = close.rolling(200).mean()
    df["dist_ma50"] = (close - ma_50) / df["atr"]
    df["dist_ma200"] = (close - ma_200) / df["atr"]

    df["atr_ratio"] = df["atr"] / df["atr"].rolling(100).mean()

    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek
    return df


def add_label(df, horizon=6, threshold=0.5):
    future_close = df["close"].shift(-horizon)
    move = (future_close - df["close"]) / df["atr"]
    df["label"] = np.select(
        [move > threshold, move < -threshold],
        [2, 0],
        default=1,
    )
    return df


def main():
    print("=" * 60)
    print("XAUUSD H1 CatBoost 最終学習 & ONNX出力(Wilder修正版)")
    print("=" * 60)

    # ==== ここを書き換えて調整 ====
    XAU_M1_PATH = "XAUUSD_M1_202301030100_202607312354.csv"
    HORIZON = 6
    DEPTH = 4
    L2_LEAF_REG = 30
    ITERATIONS = 300
    LEARNING_RATE = 0.05
    ONNX_FILENAME = "xauusd_catboost_wilder_d4l30.onnx"
    # ================================

    print("\n1. データ読み込み・H1集約・特徴量生成中(Wilder ATR/RSI)...")
    df_m1 = load_m1(XAU_M1_PATH)
    df = resample_to_h1(df_m1)
    df = add_base_features_corrected(df)

    df["future_ret_atr"] = (df["close"].shift(-HORIZON) - df["close"]) / df["atr"]
    df = add_label(df, horizon=HORIZON)

    feature_cols = [
        "ret_1", "ret_3", "ret_6",
        "dist_ma50", "dist_ma200",
        "rsi", "atr_ratio",
        "hour", "dayofweek",
    ]

    clean = df.dropna(subset=feature_cols + ["label", "future_ret_atr", "atr"]).reset_index(drop=True)
    clean = clean[clean["label"] != 1].reset_index(drop=True)
    clean["label"] = (clean["label"] == 2).astype(int)

    X = clean[feature_cols].astype(np.float32)
    y = clean["label"]

    print(f"   期間: {clean['datetime'].min()} 〜 {clean['datetime'].max()}")
    print(f"   学習データ: {len(X)}件")
    print(f"   上昇ラベル: {y.sum()}件 ({y.mean()*100:.1f}%)")
    print(f"   下降ラベル: {(1-y).sum()}件 ({(1-y.mean())*100:.1f}%)")
    print(f"   特徴量: {feature_cols}")
    print(f"   ハイパーパラメータ: depth={DEPTH}, l2_leaf_reg={L2_LEAF_REG}, "
          f"iterations={ITERATIONS}, learning_rate={LEARNING_RATE}")
    print("   ※ このモデルは全データ(全レジーム)で学習。エントリー時のレジーム")
    print("     フィルター(vol_ratio>=1.10)はEA側のInpVolThresholdで従来通り適用される。")

    print("\n2. 全データでモデル学習中...")
    model = CatBoostClassifier(
        iterations=ITERATIONS,
        depth=DEPTH,
        learning_rate=LEARNING_RATE,
        l2_leaf_reg=L2_LEAF_REG,
        loss_function="Logloss",
        verbose=0,
        random_seed=42,
    )
    model.fit(X, y)
    print("   学習完了")

    print("\n   特徴量重要度:")
    importances = model.get_feature_importance()
    for col, imp in sorted(zip(feature_cols, importances), key=lambda x: -x[1]):
        print(f"   {col:<15} {imp:.2f}")

    print(f"\n3. ONNX出力中: {ONNX_FILENAME}")
    model.save_model(
        ONNX_FILENAME,
        format="onnx",
        export_parameters={
            "onnx_domain": "ai.catboost",
            "onnx_model_version": 1,
            "onnx_doc_string": f"XAUUSD H1 Direction Model horizon={HORIZON} depth={DEPTH} "
                               f"l2={L2_LEAF_REG} wilder_atr_rsi",
            "onnx_zipmap": "false",
        },
    )

    print("4. ONNX後処理中(ZipMap除去・label削除)...")
    import onnx
    from onnx import helper, TensorProto

    model_onnx = onnx.load(ONNX_FILENAME)
    g = model_onnx.graph

    zm_nodes = [n for n in g.node if n.op_type == "ZipMap"]
    if zm_nodes:
        zm = zm_nodes[0]
        raw = zm.input[0]
        g.node.remove(zm)
        new_out = []
        for o in g.output:
            if o.name == "probabilities":
                new_out.append(helper.make_tensor_value_info(raw, TensorProto.FLOAT, [None, 2]))
            else:
                new_out.append(o)
        del g.output[:]
        g.output.extend(new_out)
        print("   ZipMap除去: OK")
    else:
        print("   ZipMap: なし(スキップ)")

    prob_out = [o for o in g.output if "label" not in o.name]
    del g.output[:]
    g.output.extend(prob_out)
    print("   label出力削除: OK")

    onnx.save(model_onnx, ONNX_FILENAME)

    print("\n5. ONNX構造確認...")
    import onnxruntime as ort
    sess = ort.InferenceSession(ONNX_FILENAME, providers=["CPUExecutionProvider"])
    print("   入力:")
    for i in sess.get_inputs():
        print(f"     {i.name}  {i.type}  {i.shape}")
    print("   出力:")
    for o in sess.get_outputs():
        print(f"     {o.name}  {o.type}  {o.shape}")

    test_input = X.iloc[:1].values.astype(np.float32)
    result = sess.run(None, {"features": test_input})
    print(f"\n   推論テスト(1件): P(下降)={result[0][0][0]:.4f}  P(上昇)={result[0][0][1]:.4f}")

    file_size = os.path.getsize(ONNX_FILENAME) / 1024
    print(f"\n{'='*60}")
    print(f"完了: {ONNX_FILENAME} ({file_size:.1f} KB)")
    print(f"{'='*60}")
    print("\n次のステップ(反映前に必ず既存ファイルをバックアップすること):")
    print("  MT5 のデータフォルダ配下 MQL5/Files/ に ONNX を配置する。")
    print("  場所は MT5 の [ファイル]→[データフォルダを開く] で確認できる。")
    print("  例:")
    print("    MT5_FILES=\"$MT5_DATA_DIR/MQL5/Files\"")
    print("    cp \"$MT5_FILES/xauusd_catboost.onnx\" \\")
    print("       \"$MT5_FILES/xauusd_catboost_backup_$(date +%Y%m%d).onnx\"")
    print(f"    cp {ONNX_FILENAME} \"$MT5_FILES/xauusd_catboost.onnx\"")
    print("  ストラテジーテスターを使う場合はテスターエージェント配下にも同じ配置が要る。")
    print("\n※ 注意: このスクリプトはホールドアウトを取らず全データで学習する。")
    print("  出力モデルの評価に学習区間を含めてはならない(RESEARCH_LOG 参照)。")


if __name__ == "__main__":
    main()