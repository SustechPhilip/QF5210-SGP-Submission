"""模型与回测：主循环、仓位、成本、指标。

对应 v2 §4.4 / §5 / §6。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from screen import (MIN_TRAIN_SCREEN, ann_sharpe, apply_norm, cost_series,
                    net_returns, norm_stats, rfe_ols, screen_factors,
                    set_ann, single_factor_signals)

MIN_TRAIN = 250
TARGET_VOL = 0.10      # 固定，不在验证集选
TICK = 0.05


# ---------------------------------------------------------------- 波动率目标

def vol_target_60(full: pd.DataFrame, ann: float) -> pd.Series:
    """σ̂_60,t = sd(r^full_{t-60..t-1}) × sqrt(ANN)，取自 full.shift(1)。

    r^full_s = ln(C_s / C_{s-1})，用收盘到收盘。
    """
    r = np.log(full["C"]).diff()
    s = r.shift(1).rolling(60, min_periods=20).std(ddof=0) * np.sqrt(ann)
    return s


# ---------------------------------------------------------------- 主循环

def run_model(X: pd.DataFrame, y: pd.Series, price: pd.Series,
              cols: list, model: str = "ols", alpha: float = 1.0,
              window: str = "expanding", win_len: int = 250,
              fee: float = 2e-4, target_vol: float = TARGET_VOL,
              vol_series: pd.Series | None = None,
              min_train: int = MIN_TRAIN,
              ann: float = 260.0,
              pos_prev: float = 0.0,
              start: pd.Timestamp | None = None) -> pd.DataFrame:
    """滚动重估的多因子模型。

    每个决策日 t：
      rows = 训练窗 ∩ {标签已知(位置 <= t-2)} ∩ {入选特征非 NaN}
      统计量只用 rows 内数据
      拟合 -> 预测 z_t -> 仓位 -> 成本 -> 净收益

    两种训练窗：
      * `expanding`：从样本起点到 t-2 的全部可用行；要求**可用行数** >= min_train。
      * `rolling`（win_len 行）：取 t-2 之前**最近的 win_len 个可用行**。
        注意不是"最近 win_len 个交易日"——若那 250 个交易日里有稀疏日/因子缺失行，
        可用行会少于 250，min_train=250 永远无法满足，模型将完全不出信号。
        因此按**可用样本数**定义窗口，并在文档中说明该口径。

    返回 DataFrame：pos、yhat、z、n_train 等。
    """
    dates = X.index
    n = len(dates)
    pos = pd.Series(np.nan, index=dates)
    yhat = pd.Series(np.nan, index=dates)
    zsc = pd.Series(np.nan, index=dates)
    ntr = pd.Series(np.nan, index=dates)

    Xv = X[cols].to_numpy(dtype=float)
    yv = y.reindex(dates).to_numpy(dtype=float)
    price_v = price.reindex(dates).to_numpy(dtype=float)
    vol_v = None if vol_series is None else vol_series.reindex(dates).to_numpy(dtype=float)

    col_idx = list(range(len(cols)))
    last_valid = np.full(len(cols), np.nan)   # as-of：每个特征最近一次可用值

    # 可用行（特征非 NaN 且标签非 NaN）；rolling 窗口按可用样本数切
    row_ok = np.isfinite(yv) & np.isfinite(Xv).all(axis=1)
    cum_ok = np.cumsum(row_ok.astype(np.int64))   # cum_ok[i] = 前 i+1 行中可用行数

    for t in range(n):
        if start is not None and dates[t] < start:
            continue
        hi = t - 2                      # 标签已知规则
        if hi < 0:
            continue

        if window == "expanding":
            lo = 0
            if cum_ok[hi] < min_train:
                continue
        else:
            # 最近 win_len 个**可用**行
            if cum_ok[hi] < win_len:
                continue
            threshold = cum_ok[hi] - win_len + 1
            # 第一个使 cum_ok >= threshold 的位置
            lo = int(np.searchsorted(cum_ok, threshold, side="left"))
            if lo > hi:
                continue

        seg = Xv[lo:hi + 1, :]
        ys = yv[lo:hi + 1]
        ok = np.isfinite(ys) & np.isfinite(seg).all(axis=1)
        if ok.sum() < min_train:
            continue
        df_tr = pd.DataFrame(seg[ok], columns=cols)
        ys = ys[ok]
        if np.nanstd(ys) == 0:
            continue

        # 标准化：只用训练窗内统计量
        st = norm_stats(df_tr)
        Ztr = apply_norm(df_tr, st).to_numpy()

        # 预测行：thin_day 的 B 类因子为 NaN，用 as-of 规则填前一个可用值
        row = Xv[t, :].copy()
        miss = ~np.isfinite(row)
        if miss.any():
            row[miss] = last_valid[miss]
        if not np.isfinite(row).all():
            continue
        zrow = apply_norm(pd.DataFrame(row.reshape(1, -1), columns=cols), st).to_numpy()[0]

        # 拟合
        if model in ("ols", "ridge"):
            Xc = Ztr - Ztr.mean(axis=0)
            yc = ys - ys.mean()
            if model == "ols":
                try:
                    beta, *_ = np.linalg.lstsq(Xc, yc, rcond=None)
                except np.linalg.LinAlgError:
                    continue
            else:
                A = Xc.T @ Xc + alpha * np.eye(Xc.shape[1])
                try:
                    beta = np.linalg.solve(A, Xc.T @ yc)
                except np.linalg.LinAlgError:
                    continue
            fitted = Ztr @ beta + ys.mean()
            pred = float(zrow @ beta + ys.mean())
        else:
            raise ValueError(model)

        sd_fit = float(np.std(fitted, ddof=0))
        if not np.isfinite(sd_fit) or sd_fit == 0:
            continue
        z = pred / sd_fit
        p = float(np.clip(z, -1, 1))

        # 波动率目标
        if vol_v is not None and np.isfinite(vol_v[t]) and vol_v[t] > 0:
            p = p * min(1.0, target_vol / vol_v[t])

        pos.iloc[t] = p
        yhat.iloc[t] = pred
        zsc.iloc[t] = z
        ntr.iloc[t] = ok.sum()
        # 更新 as-of 状态
        cur = Xv[t, :]
        upd = np.isfinite(cur)
        last_valid[upd] = cur[upd]
    return pd.DataFrame({"pos": pos, "yhat": yhat, "z": zsc, "n_train": ntr})


# ---------------------------------------------------------------- 基准

def run_buyhold(dates, price: pd.Series, y: pd.Series, fee: float,
                start: pd.Timestamp | None = None) -> pd.Series:
    """纯买入持有：pos ≡ 1，不做波动率缩放。测试期从 start 重新建仓起算。"""
    pos = pd.Series(1.0, index=dates)
    if start is not None:
        pos.loc[pos.index < start] = 0.0
    return pos


def run_long_vt(dates, vol_series: pd.Series, y: pd.Series, price: pd.Series,
                fee: float, target_vol: float = TARGET_VOL,
                start: pd.Timestamp | None = None) -> pd.Series:
    """常多 + 波动率目标。"""
    pos = pd.Series(np.nan, index=dates)
    v = vol_series.reindex(dates)
    for t in range(len(dates)):
        if start is not None and dates[t] < start:
            continue
        if np.isfinite(v.iloc[t]) and v.iloc[t] > 0:
            pos.iloc[t] = min(1.0, target_vol / v.iloc[t])
        else:
            pos.iloc[t] = 1.0
    return pos


def run_mom20(X: pd.DataFrame, dates, vol_series: pd.Series,
              target_vol: float = TARGET_VOL,
              start: pd.Timestamp | None = None) -> pd.Series:
    """20 日动量符号基准，经波动率目标缩放。"""
    mom = X["A1_MOM_20"].reindex(dates)
    pos = pd.Series(np.nan, index=dates)
    v = vol_series.reindex(dates)
    for t in range(len(dates)):
        if start is not None and dates[t] < start:
            continue
        m = mom.iloc[t]
        if not np.isfinite(m) or m == 0:
            continue
        p = float(np.sign(m))
        if np.isfinite(v.iloc[t]) and v.iloc[t] > 0:
            p = p * min(1.0, target_vol / v.iloc[t])
        pos.iloc[t] = p
    return pos


# ---------------------------------------------------------------- 指标

def metrics(R: pd.Series, ann: float = 260.0) -> dict:
    """由唯一净收益序列 R 计算指标。"""
    r = R.dropna()
    if len(r) == 0:
        # 空区间也要返回完整的键，避免下游 KeyError
        return dict(n=0, cagr=np.nan, ann_vol=np.nan, sharpe=np.nan, max_dd=np.nan,
                    calmar=np.nan, mean_daily=np.nan, std_daily=np.nan, total=np.nan,
                    win_rate=np.nan, profit_factor=np.nan, var95=np.nan,
                    cvar95=np.nan, sortino=np.nan, skew=np.nan, kurt=np.nan,
                    ann_turnover=np.nan)
    cum = (1 + r).cumprod()
    n = len(r)
    years = n / ann
    cagr = float(cum.iloc[-1] ** (1 / years) - 1) if years > 0 else np.nan
    vol = float(r.std(ddof=0) * np.sqrt(ann))
    sharpe = float(r.mean() / r.std(ddof=0) * np.sqrt(ann)) if r.std(ddof=0) > 0 else np.nan
    peak = cum.cummax()
    dd = cum / peak - 1
    mdd = float(dd.min())
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan
    downside = r[r < 0]
    dstd = float(np.sqrt(np.mean(np.minimum(r, 0) ** 2))) if len(downside) else np.nan
    sortino = float(r.mean() / dstd * np.sqrt(ann)) if dstd and np.isfinite(dstd) and dstd > 0 else np.nan
    win = float((r > 0).mean())
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    pf = float(gains / losses) if losses > 0 else np.nan
    q05 = float(r.quantile(0.05))
    tail = r[r <= q05]
    cvar = float(tail.mean()) if len(tail) else np.nan
    return dict(
        n=int(n), cagr=cagr, ann_vol=vol, sharpe=sharpe, max_dd=mdd, calmar=calmar,
        mean_daily=float(r.mean()), std_daily=float(r.std(ddof=0)),
        total=float(cum.iloc[-1] - 1), win_rate=win, profit_factor=pf,
        var95=q05, cvar95=cvar, sortino=sortino,
        skew=float(r.skew()), kurt=float(r.kurt()),
        ann_turnover=None,
    )


def turnover(pos: pd.Series, ann: float = 260.0) -> float:
    p = pos.fillna(0.0)
    turnover_sum = (p - p.shift(1).fillna(0.0)).abs().sum()
    n = p.notna().sum()
    if n == 0:
        return np.nan
    return float(turnover_sum / (n / ann))


def add_turnover(m: dict, pos: pd.Series, ann: float) -> dict:
    m["ann_turnover"] = turnover(pos, ann)
    return m


# ---------------------------------------------------------------- 随机对照

def circular_shift_null(pos: pd.Series, y: pd.Series, price: pd.Series,
                        fee: float, n_boot: int = 1000, ann: float = 260.0,
                        seed: int = 20261028) -> pd.Series:
    """循环平移零假设：把策略自己的仓位序列相对收益循环平移。

    保留真实换手与自相关，只打断信号与收益的对应关系。
    """
    rng = np.random.default_rng(seed)
    p = pos.fillna(0.0).to_numpy()
    yy = y.reindex(pos.index).to_numpy()
    pr = price.reindex(pos.index).to_numpy()
    mask = np.isfinite(p) & np.isfinite(yy)
    p, yy, pr = p[mask], yy[mask], pr[mask]
    n = len(p)
    if n < 10:
        return pd.Series(dtype=float)
    out = []
    for _ in range(n_boot):
        k = int(rng.integers(1, n))
        pk = np.roll(p, k)
        dp = np.abs(pk - np.concatenate([[0.0], pk[:-1]]))
        r = pk * yy - dp * (fee + 0.05 / pr)
        out.append(ann_sharpe(pd.Series(r), ann=ann))
    return pd.Series(out)
