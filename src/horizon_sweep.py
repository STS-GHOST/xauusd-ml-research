"""
horizon_sweep.py — 保有期間 h に対する偏IC と損益分岐比のスイープ

問い
----
コストはトレードあたり固定。リターンの sd は sd_h ∝ √h で伸びる。
したがって IC が h に対して平坦なら、h を延ばすほど「信号予算 / コスト」が改善する。

  IC_h が平坦        → EV_h ∝ √h、コスト不変 → 比は √h で改善
  IC_h ∝ 1/√h       → EV_h は平坦、コスト不変 → 比は改善しない

どちらかは測らないと分からない。モデルは h=6 のラベルで学習されているので、
落ちる側に賭けるのが自然。それを確認する。

出力
----
  h ごとに:
    IC_par        ドリフトを条件付けた偏IC（全バー、高 n）
    sd_h          h バーリターンの標準偏差（ATR単位）
    EV_gate       IC から導いたゲート後 EV = IC * sd_h * E[z|上位f]
    friction      コスト + SL + 1ポジ制約（実測 or 概算）
    ratio         EV_gate / friction   … 1 を超えないと話にならない

使い方
------
  python3 horizon_sweep.py --m1 XAUUSD_M1_...csv --score model_scores.csv
"""

from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from statistics import NormalDist

from excess_return import wilder_atr, partial_ic
from ic_report import load_m1_to_h1, autocorr_half_life

DRIFT_WINDOW = 500
ATR_PERIOD = 14
CONF = 0.60
HORIZONS = [3, 6, 9, 12, 18, 24]
REGIMES = [("2020 高ボラ", "2020-01-01", "2020-12-31"),
           ("2021-22 横ばい", "2021-01-01", "2022-12-31")]
ND = NormalDist()


def build(h1: pd.DataFrame, h: int) -> pd.DataFrame:
    """h バー先リターンと、h に合わせた因果的ドリフト推定を作る。"""
    out = h1.copy()
    c = out["close"].to_numpy(float)
    atr = out["atr"].to_numpy(float)

    fwd = np.full(len(c), np.nan)
    fwd[:-h] = (c[h:] - c[:-h]) / atr[:-h]
    out["fwd"] = fwd

    r_bar = np.full(len(c), np.nan)
    r_bar[1:] = (c[1:] - c[:-1]) / atr[1:]
    out["drift"] = (pd.Series(r_bar, index=out.index)
                    .rolling(DRIFT_WINDOW, min_periods=DRIFT_WINDOW)
                    .mean().to_numpy()) * h
    return out


def e_z_top(frac: float) -> float:
    """上位 frac を選んだときの標準正規の条件付き期待値。"""
    z = ND.inv_cdf(1 - frac)
    return ND.pdf(z) / frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m1", required=True)
    ap.add_argument("--score", required=True)
    ap.add_argument("--point", type=float, default=0.01)
    ap.add_argument("--cost_atr", type=float, default=None,
                    help="トレードあたりコスト(ATR)。省略時は M1 スプレッドから実測")
    a = ap.parse_args()

    m1 = pd.read_csv(a.m1, sep="\t")
    m1.columns = [c.strip("<>").lower() for c in m1.columns]
    m1["datetime"] = pd.to_datetime(m1["date"].astype(str) + " " + m1["time"].astype(str))

    h1 = load_m1_to_h1(a.m1)
    h1["atr"] = wilder_atr(h1["high"], h1["low"], h1["close"], ATR_PERIOD)

    sc = pd.read_csv(a.score)
    sc.index = pd.to_datetime(sc.iloc[:, 0])
    base = h1.join(sc["score"], how="inner")
    print(f"H1 {len(h1)} / スコア重なり {len(base)}")

    hl = autocorr_half_life(base["score"].to_numpy())
    print(f"スコア半減期 {hl} 本\n")

    for rname, lo, hi in REGIMES:
        rows = []
        for h in HORIZONS:
            d = build(base, h).loc[lo:hi]
            d = d.dropna(subset=["score", "fwd", "drift"])
            if len(d) < 500:
                continue
            block = max(4 * h, 4 * hl)

            r = partial_ic(d["score"], d["fwd"], [d["drift"]],
                           h, n_boot=1000, block=block)

            frac = float((d["score"] >= CONF).mean())
            sd_h = float(d["fwd"].std())

            # コスト（ATR単位）: トレードあたり固定、h に依存しない
            if a.cost_atr is not None:
                cost = a.cost_atr
            else:
                sp = d.get("spread")
                cost = float(np.nanmedian(sp.to_numpy() * a.point / d["atr"].to_numpy())) \
                    if sp is not None else 0.02

            ev = r["ic"] * sd_h * e_z_top(max(frac, 1e-6))
            ev_lo = r["lo"] * sd_h * e_z_top(max(frac, 1e-6))
            ev_hi = r["hi"] * sd_h * e_z_top(max(frac, 1e-6))

            rows.append(dict(h=h, n=len(d),
                             IC=r["ic"], ic_lo=r["lo"], ic_hi=r["hi"],
                             sd_h=sd_h, gate=frac,
                             EV=ev, ev_lo=ev_lo, ev_hi=ev_hi,
                             cost=cost, ratio=ev / cost if cost else np.nan))

        df = pd.DataFrame(rows)
        print(f"{'='*78}\n{rname}\n{'='*78}")
        with pd.option_context("display.width", 220, "display.max_columns", 30):
            print(df.round(4).to_string(index=False))

        # IC の h 依存性: 平坦 なら ratio は √h で伸びる
        # 傾きは IC が 0 と区別できるときだけ意味を持つ。
        # ノイズに対して log|IC| を回すと大きな傾きが平気で出るので、必ず門番を置く。
        sig = ((df["ic_lo"] > 0) | (df["ic_hi"] < 0)).sum()
        if len(df) >= 3 and sig >= max(2, len(df) // 2):
            x = np.log(df["h"].to_numpy(float))
            y = np.log(np.abs(df["IC"].to_numpy()) + 1e-9)
            slope = np.polyfit(x, y, 1)[0]
            print(f"\n  log IC ~ log h の傾き = {slope:+.3f}  (CI が 0 を外れた h: {sig}/{len(df)})")
            print("     0 付近  → IC は h に平坦。EV は √h で伸び、コストは不変 → h を延ばす価値あり")
            print("   -0.5 付近 → EV は h に平坦。コスト償却分しか得られない")
        else:
            print(f"\n  傾きは算出しない: IC が 0 と区別できた h が {sig}/{len(df)} 件しかない。")
            print("    ノイズに log|IC| を当てると無意味な傾きが出るため。EV と CI を直接読むこと。")
        print()

    print("注: EV は偏IC から導いた推定であり、SL・1ポジ制約は含まない。")
    print("    ratio > 1 は必要条件であって十分条件ではない。")
    print("    最終判断は wrapper_decomp.py を該当 h で回して確認すること。")


if __name__ == "__main__":
    main()
