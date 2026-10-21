"""
wrapper_decomp.py — 「バー単位のリターン」と「実トレード成績」の差を段階分解する

問題
----
2021-22 横ばい期で
    conf>=0.60 の生 6 バーリターン   = +0.0637 ATR
    §0 の実トレード成績              = -0.105  ATR
差 0.17 ATR。ドリフト・切片・ゲート選択性は既に否定された。残りはラッパー側。

段階（入れ子の部分集合 + 変換）
------------------------------
  S0  全バー（スコアがある行すべて）      … 母集団
  S1  + conf >= 0.60                      … 部分集合
  S2  + atr_ratio >= 1.10                 … 部分集合
  S3  + SL 3ATR / 6本時間切れ を M1 で解決 … 同一集合・値の変換
  S4  + 実スプレッド                      … 同一集合・値の変換
  S5  + 1ポジション制約                   … 部分集合（逐次）

S2→S3→S4 は同一トレード集合なのでペア比較でき、分散が小さい。
どの段で 0.17 ATR が消えるかを見る。

SL の解決は必ず M1 で行う。H1 の高安だけでは足内の到達順序が分からず、
そこがまさに疑っている箇所だから。

前提（確認済み）: 全エントリー BUY / SL = 3ATR / TP なし / 6 本で時間切れ

使い方
------
  python3 wrapper_decomp.py --m1 XAUUSD_M1_...csv --score model_scores.csv
  （--point は 1 ポイントの価格単位。XAUUSD は通常 0.01）
"""

from __future__ import annotations
import argparse
import numpy as np
import pandas as pd

from excess_return import wilder_atr
from gate_analysis import block_boot

HORIZON = 6
ATR_PERIOD = 14
SL_ATR = 3.0
CONF = 0.60
VOL_RATIO = 1.10
# ic_report.py と同一。事後に区切りを動かさないため、ここで固定する。
REGIMES = [("2020 高ボラ", "2020-01-01", "2020-12-31"),
           ("2021-22 横ばい", "2021-01-01", "2022-12-31"),
           ("2023-26 上昇", "2023-01-01", "2026-12-31")]
BLOCK = 24


def load_m1(path):
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.strip("<>").lower() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
    return df.sort_values("datetime").reset_index(drop=True)


def to_h1(m1):
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "spread" in m1.columns:
        agg["spread"] = "median"
    h1 = m1.set_index("datetime").resample("1h").agg(agg)
    return h1.dropna(subset=["close"])


def simulate(pos, h1_o, h1_c, h1_lo, h1_atr, m1_ts, m1_l, m1_c, m1_sp,
             h1_ts, point, horizon=HORIZON, sl_atr=SL_ATR,
             cost_mode="spread", flat_cost=0.0287, sl_source="m1"):
    """
    concurrent_position_sim.py と同じ建玉規約:
        シグナル = バー t（終値で確定）
        建玉     = バー t+1 の始値      active_from = t+1
        SL 判定  = バー t+1 〜 t+6
        決済     = SL 価格、または バー t+6 の終値

    pos は H1 の「位置」（ラベル時刻ではない）。週末ギャップがあるため
    壁時計ではなくバー位置で数える。

    sl_source="h1" で §0 と同じ H1 安値判定、"m1" で分足解決。
    """
    n = len(pos)
    pnl_raw = np.full(n, np.nan)
    pnl_cost = np.full(n, np.nan)
    exit_pos = np.full(n, -1, dtype=np.int64)
    hit_sl = np.zeros(n, dtype=bool)
    nbar = len(h1_o)

    for k in range(n):
        t = pos[k]
        if t + horizon >= nbar:
            continue
        atr = h1_atr[t]
        entry = h1_o[t + 1]
        if not (np.isfinite(atr) and atr > 0 and np.isfinite(entry)):
            continue
        stop = entry - sl_atr * atr

        j_hit = -1
        if sl_source == "h1":
            for j in range(t + 1, t + horizon + 1):
                if h1_lo[j] <= stop:
                    j_hit = j
                    break
        else:
            i0 = np.searchsorted(m1_ts, h1_ts[t + 1], side="left")
            i1 = np.searchsorted(m1_ts, h1_ts[t + horizon], side="left")
            i1 = min(i1 + 60, len(m1_ts))
            seg = m1_l[i0:i1]
            hitidx = np.flatnonzero(seg <= stop)
            if hitidx.size:
                j_hit = t + 1  # バー位置は近似（枠解放の順序付けにのみ使う）
                # 到達した M1 位置から H1 位置を復元
                j_hit = int(np.searchsorted(h1_ts, m1_ts[i0 + hitidx[0]],
                                            side="right") - 1)

        if j_hit >= 0:
            exit_px = stop
            exit_pos[k] = j_hit
            hit_sl[k] = True
        else:
            exit_px = h1_c[t + horizon]
            exit_pos[k] = t + horizon

        pnl_raw[k] = (exit_px - entry) / atr
        if cost_mode == "flat":
            cost = flat_cost
        else:
            i0 = np.searchsorted(m1_ts, h1_ts[t + 1], side="left")
            cost = (m1_sp[min(i0, len(m1_sp) - 1)] * point / atr) if m1_sp is not None else 0.0
        pnl_cost[k] = pnl_raw[k] - cost

    return pnl_raw, pnl_cost, exit_pos, hit_sl


def selftest():
    """バー t 以前のデータが損益に混入していないことを検査する（今回踏んだ不具合の再発防止）。"""
    rng = np.random.default_rng(0)
    n = 3000
    c = 1800 + np.cumsum(rng.normal(0, 1.0, n))
    o = np.concatenate([[c[0]], c[:-1]])
    lo = np.minimum(o, c) - np.abs(rng.normal(0, .5, n))
    ts = pd.date_range("2021-01-01", periods=n, freq="1h").values.astype("datetime64[ns]")
    atr = np.full(n, 2.0)
    pos = np.arange(300, n - 20, 7)

    def run(o_, lo_):
        return simulate(pos, o_, c, lo_, atr, ts, lo_, c, None, ts, 0.01,
                        cost_mode="flat", sl_source="h1")[0]

    base = run(o, lo)
    o2, lo2 = o.copy(), lo.copy()
    o2[pos] -= 500.0          # バー t の始値を壊す
    lo2[pos] -= 500.0         # バー t の安値を壊す（先読み SL が出るなら発火する）
    pert = run(o2, lo2)
    t1 = np.allclose(base[np.isfinite(base)], pert[np.isfinite(pert)])
    print(f"T1 バー t のデータは損益に影響しない : {'PASS' if t1 else 'FAIL'}")

    o3 = o.copy()
    o3[pos + 1] -= 500.0      # バー t+1 の始値を壊す → 必ず変わるべき
    t2 = not np.allclose(base[np.isfinite(base)], run(o3, lo)[np.isfinite(base)])
    print(f"T2 バー t+1 の始値は損益に影響する   : {'PASS' if t2 else 'FAIL'}")
    print("ALL PASS" if (t1 and t2) else "*** FAILURE ***")
    return t1 and t2


def stage_row(name, vals, block=BLOCK):
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    m, lo, hi = block_boot(v, block, n_boot=1500)
    return dict(stage=name, n=len(v), EV=m, lo=lo, hi=hi,
                winrate=float(np.mean(v > 0)) if len(v) else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m1", required=True)
    ap.add_argument("--score", required=True)
    ap.add_argument("--point", type=float, default=0.01)
    ap.add_argument("--cost_mode", choices=["spread", "flat"], default="spread")
    ap.add_argument("--flat_cost", type=float, default=0.0287)
    ap.add_argument("--sl_source", choices=["m1", "h1"], default="m1")
    ap.add_argument("--atr_ratio_min", type=float, default=VOL_RATIO)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        selftest()
        return

    print("1. M1 読み込み")
    m1 = load_m1(a.m1)
    print(f"   M1 {len(m1)} 本  {m1.datetime.iloc[0]} .. {m1.datetime.iloc[-1]}")

    h1 = to_h1(m1)
    atr = wilder_atr(h1["high"], h1["low"], h1["close"], ATR_PERIOD)
    h1 = h1.assign(atr=atr,
                   atr_ratio=atr / pd.Series(atr, index=h1.index).rolling(100).mean())
    fwd = np.full(len(h1), np.nan)
    c = h1["close"].to_numpy()
    a_ = h1["atr"].to_numpy()
    fwd[:-HORIZON] = (c[HORIZON:] - c[:-HORIZON]) / a_[:-HORIZON]
    h1["fwd_h"] = fwd

    sc = pd.read_csv(a.score)
    sc.index = pd.to_datetime(sc.iloc[:, 0])
    d = h1.join(sc["score"], how="inner").dropna(subset=["atr", "atr_ratio", "score"])
    print(f"   H1 {len(h1)} / スコア重なり {len(d)}")

    h1_ts_all = h1.index.values.astype("datetime64[ns]")
    h1_o_all = h1["open"].to_numpy(float)
    h1_c_all = h1["close"].to_numpy(float)
    h1_lo_all = h1["low"].to_numpy(float)
    h1_atr_all = h1["atr"].to_numpy(float)

    m1_ts = m1["datetime"].values.astype("datetime64[ns]")
    m1_o = m1["open"].to_numpy(float)
    m1_h = m1["high"].to_numpy(float)
    m1_l = m1["low"].to_numpy(float)
    m1_c = m1["close"].to_numpy(float)
    m1_sp = m1["spread"].to_numpy(float) if "spread" in m1.columns else None
    if m1_sp is not None:
        print(f"   スプレッド中央値 {np.median(m1_sp):.0f}pt "
              f"(= {np.median(m1_sp)*a.point:.3f} 価格単位)")

    for rname, lo, hi in REGIMES:
        s = d.loc[lo:hi]
        if len(s) < 500:
            continue
        print(f"\n{'='*66}\n{rname}   n(bar)={len(s)}\n{'='*66}")

        rows = [stage_row("S0 全バー", s["fwd_h"])]

        m_conf = s["score"] >= CONF
        s1 = s[m_conf]
        rows.append(stage_row("S1 +conf>=0.60", s1["fwd_h"]))

        s2 = s1[s1["atr_ratio"] >= a.atr_ratio_min]
        rows.append(stage_row(f"S2 +atr_ratio>={a.atr_ratio_min}", s2["fwd_h"]))

        pos = np.searchsorted(h1_ts_all, s2.index.values.astype("datetime64[ns]"))
        praw, pcost, eidx, hsl = simulate(
            pos, h1_o_all, h1_c_all, h1_lo_all, h1_atr_all,
            m1_ts, m1_l, m1_c, m1_sp, h1_ts_all, a.point,
            cost_mode=a.cost_mode, flat_cost=a.flat_cost, sl_source=a.sl_source)
        rows.append(stage_row("S3 +SL3ATR(M1解決)", praw))
        rows.append(stage_row("S4 +スプレッド", pcost))

        # S5 1ポジション制約: 決済まで次を建てない
        # 1ポジ制約: 建玉は t+1、決済バーと同一バーの新規建ては許可（EA挙動）
        keep = np.zeros(len(pos), dtype=bool)
        free_from = -10**9
        for k in range(len(pos)):
            if eidx[k] < 0:
                continue
            if pos[k] + 1 >= free_from:
                keep[k] = True
                free_from = eidx[k]
        rows.append(stage_row("S5 +1ポジション制約", pcost[keep]))

        df = pd.DataFrame(rows)
        df["Δ"] = df["EV"].diff()
        with pd.option_context("display.width", 200):
            print(df.round(4).to_string(index=False))

        # ペア比較（同一集合なので分散が小さい）
        print("\n  同一トレード集合上のペア比較:")
        for lab, diff in (("S2→S3 (SL の寄与)", praw - s2["fwd_h"].to_numpy()),
                          ("S3→S4 (コストの寄与)", pcost - praw)):
            m, l, h = block_boot(diff, BLOCK, n_boot=1500)
            if not np.isfinite(l):
                flag = "  (n 不足で判定不能)"
            else:
                flag = "" if (l < 0 < h) else "  ***"
            print(f"    {lab:<26} {m:+.4f}  CI[{l:+.4f},{h:+.4f}]{flag}")

        print(f"\n  SL 到達率 {hsl.mean():.1%} / "
              f"1ポジ制約で残った割合 {keep.mean():.1%}")
        if m1_sp is not None and len(pos):
            ii = np.searchsorted(m1_ts, h1_ts_all[np.minimum(pos + 1, len(h1_ts_all) - 1)])
            ii = np.minimum(ii, len(m1_sp) - 1)
            cost_atr = (m1_sp[ii] * a.point) / h1_atr_all[pos]
            print(f"  スプレッド実測 中央値 {np.nanmedian(cost_atr):.4f} ATR "
                  f"(§0 の想定 {a.flat_cost} / 適用モード {a.cost_mode})")


if __name__ == "__main__":
    main()
