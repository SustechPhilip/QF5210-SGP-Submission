"""因子筛选：外推单因子 Sharpe -> 贪心筛选 -> RFE。

对应 v2 §4.1 / §4.2 / §4.3。参数一律从 config.py 取，不在此硬编码。

关键点：单因子信号必须是**滚动外推**（walk-forward）的，系数只用 <=t-2 的数据拟合。
在训练窗内拟合再在同一段评估会让样本内 Sharpe 几乎必然 >0，使 loose 档的
`Sharpe > 0` 阈值形同虚设。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MIN_TRAIN_SCREEN, SCREEN_PRESETS  # noqa: E402

ANN_CACHE: dict = {"ann": 260.0}


def set_ann(ann: float) -> None:
    ANN_CACHE["ann"] = ann


# ---------------------------------------------------------------- 已知时间

def label_known_mask(labels: pd.DataFrame, decision_ts: pd.Series) -> pd.Series:
    """布尔 Series：该行的标签在**当日决策时点**之前是否已实现（行级）。

    labels.y_cc_known_ts[t] = t+1 收盘；decision_ts[t] = t 17:00。
    t-1 行的标签要到 t 日收盘才实现，决策时未知，因此不可用。

    实际筛选训练行时用位置规则更直接：决策日 t 可用 s <= t-2（交易日位置）。
    见 `usable_positions()`。
    """
    known = labels["y_cc_known_ts"]
    return known < decision_ts.reindex(known.index)


def usable_positions(t: int, window: str = "expanding",
                     win_len: int = 250) -> tuple:
    """决策日位置 t 的训练行位置区间（含端点）。

    标签已知规则：s <= t-2（因为 s+1 日收盘才实现 s 的标签，而 t 日 17:00 决策时
    t-1 日的标签尚未产生）。
    """
    hi = t - 2
    if hi < 0:
        return (0, -1)
    lo = 0 if window == "expanding" else max(0, hi - win_len + 1)
    return (lo, hi)


# ---------------------------------------------------------------- 标准化

def norm_stats(X: pd.DataFrame) -> dict:
    """训练窗内的均值/标准差/1%/99% 分位数。"""
    return dict(mu=X.mean(), sd=X.std(ddof=0),
                q01=X.quantile(0.01), q99=X.quantile(0.99))


def apply_norm(X: pd.DataFrame, st: dict) -> pd.DataFrame:
    """截尾后标准化。"""
    w = X.clip(lower=st["q01"], upper=st["q99"], axis=1)
    return (w - st["mu"]) / st["sd"].replace(0, np.nan)


# ---------------------------------------------------------------- 成本与收益

def cost_series(pos: pd.Series, price: pd.Series, fee: float) -> pd.Series:
    """cost_t = |Δpos| × (fee + 0.05/C_t)。pos 为 NaN 视为 0（无仓位）。"""
    p = pos.fillna(0.0)
    dp = (p - p.shift(1).fillna(0.0)).abs()
    return dp * (fee + 0.05 / price)


def net_returns(pos: pd.Series, y: pd.Series, price: pd.Series,
                fee: float) -> pd.Series:
    """R[t] = pos_t × y_t − cost_t。pos_t 决定于 t 日 17:00，y_t 在 t+1 收盘实现。"""
    return pos * y - cost_series(pos, price, fee)


def ann_sharpe(r: pd.Series, ann: float | None = None) -> float:
    ann = ANN_CACHE["ann"] if ann is None else ann
    r = r.dropna()
    if len(r) < 2 or r.std(ddof=0) == 0:
        return np.nan
    return float(r.mean() / r.std(ddof=0) * np.sqrt(ann))


# ---------------------------------------------------------------- 单因子外推

def _ols_stats(f: pd.Series, y: pd.Series, valid: pd.Series,
               window: str, win_len: int):
    """在有效行上做单变量 OLS 的向量化解。

    训练行位置必须 <= t-2（标签已知规则）。用 shift(2) 把窗口右移到正确位置，
    再用 rolling/cumsum 一次算完所有日期，避免逐日拟合。

    返回 (beta, alpha, n, sd_fit)，都与原索引对齐。
    """
    f = f.astype(float)
    y = y.astype(float)
    v = valid.astype(float)

    fv = f.where(valid)
    yv = y.where(valid)
    ones = v
    ff = (fv * fv).fillna(0.0)
    fy = (fv * yv).fillna(0.0)

    if window == "expanding":
        def agg(s):
            return s.shift(2).expanding(min_periods=1).sum()
    else:
        def agg(s):
            return s.shift(2).rolling(win_len, min_periods=1).sum()

    n = agg(ones)
    Sf = agg(fv.fillna(0.0))
    Sf2 = agg(ff)
    Sy = agg(yv.fillna(0.0))
    Sfy = agg(fy)

    den = n * Sf2 - Sf * Sf
    with np.errstate(divide="ignore", invalid="ignore"):
        nz = n.replace(0.0, np.nan)
        beta = (n * Sfy - Sf * Sy) / den.replace(0.0, np.nan)
        alpha = (Sy - beta * Sf) / nz
        var_f = (Sf2 - Sf * Sf / nz) / (n - 1).replace(0.0, np.nan)
        sd_f = np.sqrt(var_f.clip(lower=0))
        sd_fit = beta.abs() * sd_f
    # 训练窗内因子均值（用于「去均值信号」变体）
    fbar = Sf / nz
    return beta, alpha, n, sd_fit, fbar


def single_factor_signals(X: pd.DataFrame, y: pd.Series, price: pd.Series,
                          fee: float,
                          min_train: int = MIN_TRAIN_SCREEN,
                          window: str = "expanding",
                          win_len: int = 250,
                          sharpe_end: pd.Timestamp | None = None,
                          demean_signal: bool = False,
                          eval_start: pd.Timestamp | None = None
                          ) -> tuple:
    """对每个因子做滚动外推，返回 (signals DataFrame, sharpe Series)。

    每个日期 t：
      rows = 训练窗 ∩ {标签已知(位置 <= t-2)} ∩ {因子非 NaN} ∩ {标签非 NaN}
      若 len(rows) < min_train -> 该日不产生信号
      OLS(y ~ f) 在 rows 上拟合 -> ŷ_t = a + b·f_t
      s_t = clip(ŷ_t / sd(rows 内拟合值), -1, 1)

    `sharpe_end`：**用于计算 Sharpe 的样本截止日**。筛选必须只用训练段
    （≤2022-12-31）的外推 Sharpe，否则用全样本 Sharpe 挑因子等于把测试期
    信息泄漏进特征选择（v2 §4.2「按训练期 Sharpe_i 贪心纳入」）。
    传入 None 表示用全样本（仅用于描述性展示，不用于筛选）。

    `demean_signal`：**去均值信号**变体。默认 s_t 用拟合值 ŷ_t 本身；变体把
    训练窗内拟合值的均值 alpha + beta·fbar 减掉，即
        s_t = clip( (ŷ_t − mean_fit) / sd_fit, -1, 1 )
    含义：信号只保留「因子相对训练窗均值的偏离」这一部分，去掉水平漂移。

    `eval_start`：**统一评估窗口**变体。默认每个因子的 Sharpe 在自己的
    信号起始日之后计算（不同因子的可用起点不同，横比不公平）；变体把所有
    因子的 Sharpe 限制到同一个窗口 [eval_start, sharpe_end] 里计算。
    """
    dates = X.index
    yv = y.reindex(dates)
    base_valid = yv.notna()

    sig = pd.DataFrame(np.nan, index=dates, columns=X.columns)
    for col in X.columns:
        f_raw = X[col].reindex(dates)
        # as-of 填充：与主模型 `last_valid` 同一口径——稀疏日 / 数据中断日 /
        # t-1 数据缺失导致的 NaN，都用最近一次可用值填。
        # 历史 bug：稀疏日信号直接是 NaN，cost_series 把它当成「先平仓再开仓」，
        # 每个 B 因子在训练段平均多扣 5 次往返成本，而主模型在同样位置用的是
        # 前值（不换手）。两处口径必须一致。
        f = f_raw.ffill()
        # 拟合只用因子**真实有值**的行（坏日的填充值不进拟合）
        valid = base_valid & f_raw.notna()
        beta, alpha, n, sd_fit, fbar = _ols_stats(f_raw, yv, valid, window, win_len)
        yhat = alpha + beta * f
        if demean_signal:
            # 训练窗内拟合值的均值 = alpha + beta·fbar；减掉它即「去均值信号」
            yhat = yhat - (alpha + beta * fbar)
        s = (yhat / sd_fit.replace(0.0, np.nan)).clip(-1, 1).where(n >= min_train)
        sig[col] = s

    if sharpe_end is None:
        ev = dates
    else:
        ev = dates[dates <= sharpe_end]
    if eval_start is not None:
        ev = ev[ev >= eval_start]
    sharpes = {}
    for col in X.columns:
        s = sig[col].reindex(ev)
        r = net_returns(s, yv.reindex(ev), price.reindex(ev), fee)
        sharpes[col] = ann_sharpe(r)
    return sig, pd.Series(sharpes)


def common_signal_start(sig: pd.DataFrame) -> pd.Timestamp:
    """所有因子都已有信号的**最晚**起始日。

    用作「统一评估窗口」的左端点：窗口左端取在最后一个因子的信号起点，
    保证窗口内每个因子都有完整可评的信号。
    """
    starts = []
    for col in sig.columns:
        v = sig[col].dropna()
        if len(v):
            starts.append(v.index[0])
    if not starts:
        raise ValueError("没有任何因子产生过信号")
    return max(starts)


# ---------------------------------------------------------------- 筛选

SCREEN_PRESETS = SCREEN_PRESETS  # 来自 config.py（保持旧名兼容）


def screen_factors(X: pd.DataFrame, sharpes: pd.Series, preset: str,
                   train_end: pd.Timestamp) -> list:
    """贪心筛选：按外推 Sharpe 降序纳入；候选与所有已入选因子的 |Spearman| 都需低于阈值。

    相关性用训练期（<= train_end）的因子值计算。
    """
    cfg = SCREEN_PRESETS[preset]
    tr = X.loc[:train_end]
    cand = sharpes[(sharpes > cfg["sharpe"]) & sharpes.notna()].sort_values(ascending=False)
    selected: list = []
    for col in cand.index:
        if col in selected:
            continue
        ok = True
        for s in selected:
            pair = tr[[col, s]].dropna()
            if len(pair) < 100:
                ok = False
                break
            rho = pair[col].corr(pair[s], method="spearman")
            if not np.isfinite(rho) or abs(rho) >= cfg["corr"]:
                ok = False
                break
        if ok:
            selected.append(col)
    return selected


# ---------------------------------------------------------------- RFE

def rfe_ols(X: pd.DataFrame, y: pd.Series, cols: list,
            train_end: pd.Timestamp, n_keep: int = 10) -> list:
    """在训练集上反复做 OLS，删除 |t 统计量| 最小的特征，直到剩 n_keep 个。

    只在训练集上做一次，特征集合之后固定。
    """
    keep = list(cols)
    if len(keep) <= n_keep:
        return keep
    tr = X.loc[:train_end]
    ytr = y.reindex(tr.index)
    while len(keep) > n_keep:
        sub = tr[keep].dropna()
        yy = ytr.reindex(sub.index).dropna()
        sub = sub.loc[yy.index]
        if len(sub) < len(keep) + 5:
            break
        Xd = sub.to_numpy(dtype=float)
        yv = yy.to_numpy(dtype=float)
        Xc = Xd - Xd.mean(axis=0)
        yc = yv - yv.mean()
        try:
            beta, *_ = np.linalg.lstsq(Xc, yc, rcond=None)
        except np.linalg.LinAlgError:
            break
        resid = yc - Xc @ beta
        dof = max(len(sub) - len(keep) - 1, 1)
        sigma2 = float(resid @ resid) / dof
        try:
            XtX_inv = np.linalg.pinv(Xc.T @ Xc)
        except np.linalg.LinAlgError:
            break
        se = np.sqrt(np.maximum(sigma2 * np.diag(XtX_inv), 1e-300))
        tstat = np.abs(beta / se)
        worst = int(np.argmin(tstat))
        keep.pop(worst)
    return keep
