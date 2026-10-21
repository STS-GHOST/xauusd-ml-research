# MQL5 side

**English** | [日本語](#mql5-側)

## Files

| File | Role |
|---|---|
| `XAU_CatBoost_EA.mq5` | The Expert Advisor. Gating, timeout exit, 3-ATR stop, full decision logging. |
| `include/MLFeatures.mqh` | The 9 features, recomputed in MQL5. Wilder ATR/RSI, matched against the Python pipeline. |
| `include/MLModel.mqh` | The CatBoost trees, generated and inlined. No runtime ONNX dependency. |

To build: place the two headers in `MQL5/Include/` and the `.mq5` in `MQL5/Experts/`, then compile in MetaEditor.

## Known discrepancy between the two model artifacts

There are **two separately trained models** in this project, and they do not share hyperparameters:

| | `include/MLModel.mqh` (what the EA runs) | `src/train_export_onnx.py` (ONNX output) |
|---|---|---|
| Iterations | 60 | 300 |
| Learning rate | 0.1 | 0.05 |
| Depth / L2 | 4 / 30 | 4 / 30 |
| Training span | 2023-01-13 – 2026-07-31 (14,908 rows) | 2023-01-03 – 2026-07-31 |

The walk-forward evaluation in `src/walk_forward.py` defaults to the **300-iteration / lr 0.05** configuration, i.e. it matches the ONNX script rather than the model the EA actually executes.

This is left documented rather than silently reconciled. It does not change the headline conclusion — the confidence interval on IC is [−0.0195, +0.0193] and a hyperparameter change at this scale does not move an effect from 0 to 0.029 — but it is exactly the class of mismatch that produced the larger errors recorded in [RESEARCH_LOG.md](../RESEARCH_LOG.md), so it is written down instead of tidied away.

The generator for `MLModel.mqh` (`build_mql5_model.py`) is not included in this repository.

## Both models are superseded by the result

Neither configuration showed an out-of-sample edge. See the root [README.md](../README.md).

---

# MQL5 側

[English](#mql5-side) | **日本語**

## ファイル

| ファイル | 役割 |
|---|---|
| `XAU_CatBoost_EA.mq5` | EA 本体。ゲート、タイムアウト決済、3 ATR 損切り、全判定のログ出力。 |
| `include/MLFeatures.mqh` | 9 特徴量の MQL5 側実装。Wilder ATR/RSI、Python パイプラインと一致検証済み。 |
| `include/MLModel.mqh` | CatBoost の木を自動生成で埋め込んだもの。実行時に ONNX を必要としない。 |

ビルド: ヘッダ 2 本を `MQL5/Include/` に、`.mq5` を `MQL5/Experts/` に置いて MetaEditor でコンパイル。

## 2 つのモデル成果物のハイパーパラメータが一致していない

このプロジェクトには**別々に学習された 2 つのモデル**があり、設定が揃っていません。

| | `include/MLModel.mqh`（EA が実行） | `src/train_export_onnx.py`（ONNX 出力） |
|---|---|---|
| イテレーション | 60 | 300 |
| 学習率 | 0.1 | 0.05 |
| depth / L2 | 4 / 30 | 4 / 30 |
| 学習区間 | 2023-01-13〜2026-07-31（14,908 行） | 2023-01-03〜2026-07-31 |

`src/walk_forward.py` の既定値は **300 イテレーション / lr 0.05**、つまり ONNX 側に揃えてあり、**EA が実際に動かしているモデルとは異なります。**

これは黙って揃えずに記録として残しています。結論は変わりません（IC の信頼区間は [−0.0195, +0.0193] で、この程度のハイパーパラメータ差が効果を 0 から 0.029 に動かすことはない）。ただし[研究ログ](../RESEARCH_LOG.ja.md)に記録した大きな誤りと**同じ種類の食い違い**なので、整理して消すのではなく書き残します。

`MLModel.mqh` の生成スクリプト（`build_mql5_model.py`）はこのリポジトリに含めていません。

## どちらのモデルも結論の対象外

いずれの設定でもアウトオブサンプルのエッジは確認できていません。ルートの [README.ja.md](../README.ja.md) を参照。
