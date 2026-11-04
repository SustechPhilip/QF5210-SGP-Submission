"""统计推断工具：PSR、Deflated Sharpe、Sharpe 差异的自助法检验。

三件事，都来自 Bailey & López de Prado (2014) 与 Politis & Romano (1994)：

1. **PSR（Probabilistic Sharpe Ratio）**：给定日度 Sharpe、非零偏度与峰度，
   算出「真实 Sharpe 大于某个基准」的概率。单个策略的 p 值会**高估**显著性，
   因为试过很多方案之后，偶然出现一个好看的 Sharpe 是迟早的事。

2. **Deflated Sharpe Ratio（DSR）**：把 PSR 的基准从 0 换成
   **多次试验下的期望最大 Sharpe**，从而把「多重检验」折算进显著性。
   基准值由试验次数 N 与试验 Sharpe 分布的方差 V 决定。

3. **配对平稳块自助法（stationary bootstrap）**：检验两个策略的 Sharpe 差
   是否显著。两个策略必须在**同一组交易日**上配对重抽样，不能各抽各的。

**所有 Sharpe 都是日度（不年化）**，除非函数名里写了 ann。
年化会把 PSR 的分母放大 sqrt(260) 倍，是错的——PSR 的推导前提是
`sr = mean/std`（日度），且 `T` 是观测数。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

# 欧拉-马斯克罗尼常数
EULER = 0.5772156649


def psr(sr: float, T: int, skew: float, kurt: float, sr_star: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio。

    sr      : **日度** Sharpe（mean/std，不年化）
    T       : 观测数
    skew    : 收益偏度
    kurt    : 收益**原始**峰度（正态 = 3），不是超额峰度
    sr_star : 基准 Sharpe（DSR 里传期望最大 Sharpe）

    返回 P(真实 Sharpe > sr_star)。
    """
    if T < 3:
        return np.nan
    num = (sr - sr_star) * np.sqrt(T - 1)
    den = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    if not np.isfinite(den) or den == 0:
        return np.nan
    return float(norm.cdf(num / den))


def expected_max_sr(sr_trials_daily: np.ndarray) -> float:
    """N 次独立试验下，最大 Sharpe 的期望。

    sr_trials_daily : 各次试验的**日度** Sharpe（长度 N）
    """
    s = np.asarray(sr_trials_daily, dtype=float)
    s = s[np.isfinite(s)]
    N = len(s)
    if N < 2:
        return 0.0
    V = float(np.var(s, ddof=1))
    return float(np.sqrt(V) * ((1 - EULER) * norm.ppf(1 - 1 / N)
                               + EULER * norm.ppf(1 - 1 / (N * np.e))))


def deflated_sr(R: pd.Series, sr_trials_daily: np.ndarray,
                ann: float = 260.0) -> dict:
    """Deflated Sharpe Ratio。

    R               : 被评估策略的**测试期日度收益**
    sr_trials_daily : 全部试过的方案的测试期日度 Sharpe
    ann             : 年化系数（与项目其他地方同一个口径）

    返回字典，含 psr（对基准 0）、psr0（对期望最大 Sharpe）、
    dsr（= psr0）、期望最大 Sharpe、试验次数、原始 p 值。

    **PSR/DSR 的推导前提是日度 Sharpe**（`sr = mean/std`），
    年化只用于 `sr_annualized` 这个**展示字段**，不进入任何计算。
    """
    r = pd.Series(R).dropna().astype(float)
    if len(r) < 3:
        return dict(n=len(r), psr=np.nan, psr0=np.nan, dsr=np.nan,
                    sr_daily=np.nan, sr_star=np.nan, p_value=np.nan, n_trials=0)
    sr = float(r.mean() / r.std(ddof=1))
    # pandas 的 .kurt() 是**超额**峰度（正态 = 0），PSR 公式要原始峰度（正态 = 3）
    kurt = float(r.kurt()) + 3.0
    skew = float(r.skew())
    T = len(r)
    sr_star = expected_max_sr(sr_trials_daily)
    p0 = psr(sr, T, skew, kurt, sr_star=0.0)
    pd_ = psr(sr, T, skew, kurt, sr_star=sr_star)
    n_trials = int(np.sum(np.isfinite(np.asarray(sr_trials_daily, dtype=float))))
    return dict(
        n=T,
        sr_daily=sr,
        sr_annualized=sr * np.sqrt(ann),
        skew=skew,
        kurt=kurt,
        psr=p0,
        psr0=pd_,
        dsr=pd_,
        sr_star=sr_star,
        p_value=(1.0 - p0) if np.isfinite(p0) else np.nan,
        n_trials=n_trials,
    )


# ---------------------------------------------------------------------------
# 配对平稳块自助法
# ---------------------------------------------------------------------------

def stationary_bootstrap_indices(T: int, n: int, mean_block: int,
                                 rng: np.random.Generator) -> np.ndarray:
    """生成 n 组平稳块自助法的重抽样索引，每组长度 T。

    平稳块自助（Politis & Romano 1994）：每一步以概率 p = 1/mean_block
    跳到一个随机位置，否则前进一步（并环绕）。块长由几何分布的均值控制。

    相比普通 bootstrap，平稳块自助**保留时序自相关**——日度收益有自相关，
    独立重抽样会低估方差、做出过于乐观的结论。
    """
    p = 1.0 / mean_block
    out = np.empty((n, T), dtype=np.int64)
    for b in range(n):
        idx = np.empty(T, dtype=np.int64)
        idx[0] = rng.integers(T)
        for i in range(1, T):
            idx[i] = rng.integers(T) if rng.random() < p else (idx[i - 1] + 1) % T
        out[b] = idx
    return out


def sharpe_diff_bootstrap(Ra: pd.Series, Rb: pd.Series, n: int = 5000,
                           mean_block: int = 10, seed: int = 0,
                           annualize: bool = True, ann: float = 260.0) -> dict:
    """检验 Sharpe(A) − Sharpe(B) 是否显著大于 0（单侧）。

    两个策略在**同一组交易日**上配对重抽样。

    Ra, Rb : 日度净收益（Series）
    n      : 自助次数
    mean_block : 平均块长（交易日）。10 天约合两周，能容纳周内效应
    seed   : 随机种子（保证可复现）
    ann    : 年化系数（与项目其他地方同一个口径）

    注意：**p 值与 CI 的尺度是自洽的**——年化只是乘一个常数因子，
    「重抽样差值 ≥ 观测差值」的比例不受影响。
    """
    d = pd.concat([pd.Series(Ra).rename("a"), pd.Series(Rb).rename("b")],
                  axis=1).dropna()
    if len(d) < 30:
        return dict(n_obs=len(d), obs_diff=np.nan, p_value=np.nan,
                    boot_diff_mean=np.nan, boot_diff_ci=np.nan)
    arr = d.to_numpy()
    T = len(arr)
    scale = np.sqrt(ann) if annualize else 1.0

    def sr(x: np.ndarray) -> np.ndarray:
        m = x.mean(axis=0)
        s = x.std(axis=0, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return m / s * scale

    obs = float(sr(arr)[0] - sr(arr)[1])
    rng = np.random.default_rng(seed)
    idxs = stationary_bootstrap_indices(T, n, mean_block, rng)
    diffs = np.empty(n)
    for b in range(n):
        s = sr(arr[idxs[b]])
        diffs[b] = s[0] - s[1]
    # 单侧 p 值：重抽样分布中 >= 观测差值的比例
    p = float(np.mean(diffs >= obs))
    return dict(
        n_obs=T,
        n_boot=n,
        mean_block=mean_block,
        seed=seed,
        obs_diff=obs,
        p_value=p,
        boot_diff_mean=float(diffs.mean()),
        boot_diff_ci=(float(np.percentile(diffs, 2.5)),
                      float(np.percentile(diffs, 97.5))),
    )


def daily_sharpe(r: pd.Series) -> float:
    """日度 Sharpe（mean/std，不年化）。"""
    r = pd.Series(r).dropna().astype(float)
    if len(r) < 3 or r.std(ddof=1) == 0:
        return np.nan
    return float(r.mean() / r.std(ddof=1))
