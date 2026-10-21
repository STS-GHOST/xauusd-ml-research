"""
excess_return.py  —  ドリフト除去した超過リターンの構築（学習ラベル / 評価目的の共通土台）

設計方針
--------
1. すべて「エントリー時 ATR で正規化した単位」で扱う。ATR は Wilder 平滑（MQL5 準拠）。
   → 教訓42 対策。ATR 水準が 5 倍変わる 2020-2026 で ATR 単位を跨いで比較するため、
     ドリフト推定も ATR 正規化した per-bar リターンの平均として持つ。

2. ドリフト推定は「その時点で入手可能な情報のみ」。バー t の決定に使うのは t 以前の
   リターンだけ。評価専用の in-sample ドリフト（先読みあり）は別関数に分け、
   名前で区別する（realized_ / causal_）。

3. 先読みは信じずに毎回テストする。self_test() が未来バーの改変に対する不変性を検査。

用語
----
  fwd_h      : (P[t+h] - P[t]) / ATR[t]          … 生の h バー先リターン（ATR単位）
  r_bar      : (P[i] - P[i-1]) / ATR[i]          … ATR 正規化した 1 バーリターン
  drift_hat  : 過去 N 本の r_bar の平均 × h      … 因果的ドリフト推定
  excess_h   : fwd_h - drift_hat                 … 超過リターン
"""

from __future__ import annotations
import numpy as np
import pandas as pd


# ---------------------------------------------------------------- ATR (Wilder)

def wilder_atr(high, low, close, period: int = 14) -> np.ndarray:
    """MQL5 の iATR と同じ Wilder 平滑。SMA は使わない（過去の既知バグ）。"""
    high = np.asarray(high, float)
    low = np.asarray(low, float)
    close = np.asarray(close, float)
    n = len(close)

    prev_close = np.concatenate([[np.nan], close[:-1]])
    tr = np.nanmax(np.vstack([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ]), axis=0)
    tr[0] = high[0] - low[0]

    atr = np.full(n, np.nan)
    if n < period:
        return atr
    atr[period - 1] = tr[:period].mean()
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


# ------------------------------------------------------- 超過リターンの構築

def build_targets(df: pd.DataFrame,
                  horizon: int = 6,
                  drift_window: int = 500,
                  atr_period: int = 14) -> pd.DataFrame:
    """
    df: DatetimeIndex, columns = open/high/low/close（H1）
    返り値に以下を追加:
        atr, r_bar, fwd_h, drift_causal, excess_causal
    すべて ATR 単位。末尾 horizon 本と先頭 (atr_period + drift_window) 本は NaN。
    """
    out = df.copy()
    c = out["close"].to_numpy(float)

    atr = wilder_atr(out["high"], out["low"], c, atr_period)
    out["atr"] = atr

    # 1 バーリターン（各バー自身の ATR で正規化）
    r_bar = np.full(len(c), np.nan)
    r_bar[1:] = (c[1:] - c[:-1]) / atr[1:]
    out["r_bar"] = r_bar

    # h バー先リターン（エントリー時 ATR で正規化）
    fwd = np.full(len(c), np.nan)
    fwd[:-horizon] = (c[horizon:] - c[:-horizon]) / atr[:-horizon]
    out["fwd_h"] = fwd

    # 因果的ドリフト: t 以前の r_bar のみ。rolling は右端を含むので shift 不要
    # （r_bar[t] は P[t]-P[t-1] なので t 時点で既知）
    drift = (pd.Series(r_bar, index=out.index)
             .rolling(drift_window, min_periods=drift_window)
             .mean()
             .to_numpy()) * horizon
    out["drift_causal"] = drift
    out["excess_causal"] = out["fwd_h"] - drift

    return out


def realized_drift_excess(fwd_h: pd.Series) -> pd.Series:
    """
    評価専用（先読みあり）。評価窓内の実現平均を引くだけ。
    学習ラベルには使わないこと。窓固定 exposure ベンチマークと等価。
    """
    return fwd_h - fwd_h.mean()


# ------------------------------------------------------------------ IC 推定

def information_coefficient(score, target, horizon: int = 6, n_boot: int = 2000,
                            block: int | None = None, seed: int = 0):
    """
    Spearman IC と、重複を考慮したブロックブートストラップ CI。
    block 既定は horizon*4（ゲートのクラスター長より長く取る）。
    """
    # index を持ち込むと align で全 NaN になるので必ず落とす
    s = pd.Series(np.asarray(score, float)).rank()
    y = pd.Series(np.asarray(target, float)).rank()
    if len(s) != len(y):
        raise ValueError(f"length mismatch: score={len(s)} target={len(y)}")
    m = s.notna() & y.notna()
    s, y = s[m].to_numpy(), y[m].to_numpy()
    n = len(s)
    if n < 50:
        return dict(ic=np.nan, lo=np.nan, hi=np.nan, n=n, n_eff=np.nan)

    ic = np.corrcoef(s, y)[0, 1]
    block = block or horizon * 4
    nb = n // block
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    starts_pool = np.arange(0, n - block)
    for b in range(n_boot):
        st = rng.choice(starts_pool, size=nb, replace=True)
        idx = (st[:, None] + np.arange(block)[None, :]).ravel()
        boots[b] = np.corrcoef(s[idx], y[idx])[0, 1]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(ic=ic, lo=lo, hi=hi, n=n, n_eff=n / horizon,
                se_boot=boots.std(ddof=1))


def partial_ic(score, target, control, horizon: int = 6, n_boot: int = 1000,
               block: int | None = None, seed: int = 0):
    """
    ドリフトを条件付けた偏 IC。

    excess を目標にすると、score が drift 推定と相関する場合に機械的な負のバイアスが
    乗る（-rho * 6/sqrt(N*h) ≈ -0.08 @ N=500,h=6）。score 側と target 側の両方から
    control（= drift 推定）を落として初めて「ドリフト以外の信号」が測れる。
    """
    s = pd.Series(np.asarray(score, float)).rank()
    y = pd.Series(np.asarray(target, float)).rank()
    ctrl = control if isinstance(control, (list, tuple)) else [control]
    Zc = [pd.Series(np.asarray(c, float)).rank() for c in ctrl]
    m = s.notna() & y.notna()
    for c in Zc:
        m &= c.notna()
    s, y = s[m].to_numpy(), y[m].to_numpy()
    z = np.vstack([c[m].to_numpy() for c in Zc]).T
    n = len(s)
    if n < 50:
        return dict(ic=np.nan, lo=np.nan, hi=np.nan, n=n)

    def _resid_corr(s_, y_, z_):
        Z = np.hstack([np.ones((len(s_), 1)), z_])
        beta_s = np.linalg.lstsq(Z, s_, rcond=None)[0]
        beta_y = np.linalg.lstsq(Z, y_, rcond=None)[0]
        return np.corrcoef(s_ - Z @ beta_s, y_ - Z @ beta_y)[0, 1]

    ic = _resid_corr(s, y, z)
    block = block or horizon * 4
    nb = n // block
    rng = np.random.default_rng(seed)
    pool = np.arange(0, n - block)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        st = rng.choice(pool, size=nb, replace=True)
        idx = (st[:, None] + np.arange(block)[None, :]).ravel()
        boots[b] = _resid_corr(s[idx], y[idx], z[idx, :])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(ic=ic, lo=lo, hi=hi, n=n, se_boot=boots.std(ddof=1))


# ------------------------------------------------------------------ self test

def _synth(n=40000, mu=0.0, sigma=1.0, seed=0):
    rng = np.random.default_rng(seed)
    step = rng.normal(mu, sigma, n)
    close = 1000 + np.cumsum(step)
    high = close + np.abs(rng.normal(0, sigma * 0.5, n))
    low = close - np.abs(rng.normal(0, sigma * 0.5, n))
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    return pd.DataFrame(dict(open=close, high=high, low=low, close=close), index=idx)


def self_test(verbose=True):
    ok = True
    H, N = 6, 500

    # T1 先読み: 未来バーを壊しても、過去の excess は 1 件も変わらないこと
    df = _synth(6000, mu=0.01, seed=1)
    a = build_targets(df, H, N)
    df2 = df.copy()
    cut = 4000
    df2.iloc[cut:, :] = df2.iloc[cut:, :] + 500.0
    b = build_targets(df2, H, N)
    lhs = a["excess_causal"].iloc[: cut - H]
    rhs = b["excess_causal"].iloc[: cut - H]
    t1 = np.allclose(lhs.dropna(), rhs.dropna())
    ok &= t1
    if verbose:
        print(f"T1 先読みなし(excess)            : {'PASS' if t1 else 'FAIL'}")

    # T2 ドリフト回収: 既知の定数ドリフトを入れて excess の平均が 0 に寄ること
    df = _synth(40000, mu=0.02, sigma=1.0, seed=2)
    a = build_targets(df, H, N)
    raw = a["fwd_h"].mean()
    exc = a["excess_causal"].mean()
    t2 = abs(exc) < abs(raw) * 0.25
    ok &= t2
    if verbose:
        print(f"T2 ドリフト回収 raw={raw:+.4f} -> excess={exc:+.4f} : {'PASS' if t2 else 'FAIL'}")

    # T3 分散の増分が理論値 1+h/N に一致すること
    df = _synth(80000, mu=0.0, sigma=1.0, seed=3)
    a = build_targets(df, H, N).dropna(subset=["fwd_h", "excess_causal"])
    ratio = a["excess_causal"].var() / a["fwd_h"].var()
    theo = 1 + H / N
    t3 = abs(ratio - theo) < 0.03
    ok &= t3
    if verbose:
        print(f"T3 分散比 実測={ratio:.4f} 理論={theo:.4f}          : {'PASS' if t3 else 'FAIL'}")

    # T4 ATR が Wilder であること（SMA との差が出ること＝取り違え検知）
    df = _synth(3000, seed=4)
    atr_w = wilder_atr(df.high, df.low, df.close, 14)
    tr = np.maximum(df.high - df.low,
                    np.maximum((df.high - df.close.shift()).abs(),
                               (df.low - df.close.shift()).abs()))
    atr_s = tr.rolling(14).mean().to_numpy()
    d = np.nanmean(np.abs(atr_w - atr_s) / atr_s)
    t4 = d > 0.005
    ok &= t4
    if verbose:
        print(f"T4 Wilder≠SMA 平均相対差={d:.4%}            : {'PASS' if t4 else 'FAIL'}")

    # T5 帰無下で IC の CI が 0 を含むこと（偽陽性チェック）
    df = _synth(40000, mu=0.0, seed=5)
    a = build_targets(df, H, N).dropna(subset=["excess_causal"])
    rng = np.random.default_rng(9)
    score = rng.normal(size=len(a))
    r = information_coefficient(score, a["excess_causal"], H, n_boot=500)
    t5 = r["lo"] < 0 < r["hi"]
    ok &= t5
    if verbose:
        print(f"T5 帰無 IC={r['ic']:+.4f} CI=[{r['lo']:+.4f},{r['hi']:+.4f}] : {'PASS' if t5 else 'FAIL'}")
        print(f"   参考: SE_boot={r['se_boot']:.4f}  n_eff={r['n_eff']:.0f}")

    # T6 ポジコン: 純粋なドリフト追随器に対し
    #    excess を目標にすると系統的な負のバイアス、偏ICなら 0 になること
    exc_ics, par_ics = [], []
    for sd in range(4):
        d = _synth(40000, mu=0.0, seed=200 + sd)
        a = build_targets(d, H, N)
        sc = a["r_bar"].rolling(200, min_periods=200).mean()
        j = a.join(sc.rename("sc")).dropna(
            subset=["fwd_h", "excess_causal", "drift_causal", "sc"])
        exc_ics.append(information_coefficient(
            j["sc"], j["excess_causal"], H, n_boot=150, block=404)["ic"])
        par_ics.append(partial_ic(
            j["sc"], j["fwd_h"], [j["drift_causal"]], H, n_boot=150, block=404)["ic"])
    me, mp = float(np.mean(exc_ics)), float(np.mean(par_ics))
    t6 = (me < -0.03) and (abs(mp) < 0.015)
    ok &= t6
    if verbose:
        print(f"T6 ポジコン IC_exc={me:+.4f}(偏り) IC_par={mp:+.4f}(無偏) : {'PASS' if t6 else 'FAIL'}")

    print("\n" + ("ALL PASS" if ok else "*** FAILURE ***"))
    return ok


if __name__ == "__main__":
    self_test()
