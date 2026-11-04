"""仓位构造：主模型仓位 + EMA 平滑改进版 + 三个仓位规则变体。

**重要：本模块里的平滑与变体都是「事后改进版」**——它们是看过测试集结果之后
才加的，只在报告的「事后改进版」一节单独披露，**不替换主结果**
（主结果仍是不平滑的 OLS 仓位，见 backtest.py 的 `ols_exp`）。

设计要点（与改进版的说明一一对应）：

| 环节 | 做法 | 原因 |
|---|---|---|
| 平滑对象 | 模型打分 z（**截断之前**） | 先截断再平滑的话，截断已经把噪声放大成满仓翻转，平滑效果会差很多 |
| 实现方式 | 手写递归 `s = λ·s_prev + (1−λ)·z` | pandas 的 `ewm` 默认 `adjust=True`，前期权重不同；遇缺失值时它按时间衰减，不是沿用上一期。手写能保证回测与实盘逐位一致 |
| 缺失日 | 沿用上一期的 s，**不更新** | 与主模型的 as-of 规则一致，缺失日不产生换手 |
| 尺度系数 k | `std(z_训练段) / std(s_训练段)`，算一次后固定 | 平滑后波动变小，不调回来的话截断比例会变；只用训练段，避免用到测试期信息 |
| 波动率缩放 | 放在截断之后，与主结果同口径 | 不改动已有口径 |
| 半衰期 | 网格 {1,2,3,5,7,10}，按**验证期**净 Sharpe 取最大值 | 规则事先写定；见 select_halflife.py |
| 实盘状态 | 每天保存 `s_prev` | EMA 只依赖上一期状态，更新只需一行 |

回测（backtest.py）、实盘（update.py）、截断一致性测试（test_truncation.py）
三处共用本模块，保证同一份代码、同一份结果。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- 基础工具

def leverage(vol: pd.Series, target_vol: float = 0.10) -> pd.Series:
    """波动率目标杠杆 `min(1, target_vol / σ̂)`。

    σ̂ 无效（NaN / 非正）时取 1（不缩放），与主模型 `run_model` 的口径一致。
    """
    v = vol.astype(float) if isinstance(vol, pd.Series) else pd.Series(vol, dtype=float)
    vv = v.to_numpy(dtype=float)
    lev = np.ones(len(vv), dtype=float)
    ok = np.isfinite(vv) & (vv > 0)
    lev[ok] = np.minimum(1.0, target_vol / vv[ok])
    return pd.Series(lev, index=v.index, name="lev")


# ---------------------------------------------------------------- EMA 平滑

def ema_recursive(z: pd.Series, halflife: float) -> pd.Series:
    """因果递归 EMA：s_t 只用 z_≤t。缺失日沿用上一期，不更新。

    与 `pandas.Series.ewm(halflife=..., adjust=False)` 的差别只在**首个有效值**
    的处理上：本实现把第一个有效 z 直接作为 s 的初值（不做 λ 衰减启动），
    缺失日沿用上一期而不做时间衰减。这两点保证回测与实盘逐位一致。

    halflife=1 时 λ=0.5，每期权重一半来自上一期。
    """
    z = z.astype(float) if isinstance(z, pd.Series) else pd.Series(z, dtype=float)
    lam = 0.5 ** (1.0 / float(halflife))
    vals = z.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan, dtype=float)
    prev = np.nan
    for i, v in enumerate(vals):
        if np.isfinite(v):
            prev = v if np.isnan(prev) else lam * prev + (1.0 - lam) * v
        out[i] = prev          # z 尚未出现时 prev 为 NaN -> 不建仓
    return pd.Series(out, index=z.index, name="s_ema_raw")


def fit_scale(z: pd.Series, s: pd.Series, fit_end) -> float:
    """平滑后打分的尺度系数 k = std(z_训练段) / std(s_训练段)。

    **只用训练段**（fit_end = 2022-12-31），避免用到验证期/测试期信息。
    """
    zt = z.loc[:fit_end].dropna()
    st = s.loc[:fit_end].dropna()
    if len(zt) < 2 or len(st) < 2:
        return 1.0
    sd_s = float(st.std(ddof=0))
    sd_z = float(zt.std(ddof=0))
    if not np.isfinite(sd_s) or sd_s == 0 or not np.isfinite(sd_z):
        return 1.0
    return sd_z / sd_s


def build_positions(z: pd.Series, vol60: pd.Series, halflife: float, k: float,
                    target_vol: float = 0.10) -> tuple[pd.Series, pd.Series]:
    """构造平滑版仓位。

    返回 (pos, s)：
      s  = ema_recursive(z, halflife) * k     （已调回原量级，未截断）
      pos = clip(s, -1, 1) * min(1, target_vol / σ̂)

    `s` 一并返回，供截断一致性测试逐位比对、以及实盘每日保存 EMA 状态。
    """
    s = ema_recursive(z, halflife) * float(k)
    raw = s.clip(-1.0, 1.0)
    lev = leverage(vol60, target_vol)
    pos = raw * lev
    return pos, s


def main_positions(z: pd.Series, vol60: pd.Series,
                    target_vol: float = 0.10) -> pd.Series:
    """主模型仓位口径：clip(z, -1, 1) * min(1, target_vol / σ̂)。

    与 `models.run_model` 内部完全一致，抽出来是为了让各变体共用同一个基准。
    """
    raw = z.astype(float).clip(-1.0, 1.0)
    return raw * leverage(vol60, target_vol)


# ---------------------------------------------------------------- 仓位规则变体

def week_start_mask(index: pd.Index) -> pd.Series:
    """每周的**第一个交易日**为 True。

    不用「星期一是交易日」来判断：如果周一是法定假日，那一周就没有任何一次调仓，
    持仓会一直沿用到下周一，白白多持有一周。正确口径是
    「每个自然周里的第一个交易日调仓」——假日落在周一时，周二调仓。

    index 必须是交易日索引（升序、唯一）。
    """
    idx = pd.DatetimeIndex(index)
    iso = pd.Series(idx.isocalendar().year.values * 100
                    + idx.isocalendar().week.values, index=idx)
    return iso.ne(iso.shift(1)).fillna(True)


def hold_on(mask_update: pd.Series, target: pd.Series) -> pd.Series:
    """只在 `mask_update` 为 True 的日期更新仓位，其余日期沿用上一期。

    用于「每周一调仓」：mask_update = 该周的第一个交易日（见 `week_start_mask`）。
    目标仓位为 NaN 时也保持上一期（不因为缺信号而平仓）。
    """
    target = target.astype(float)
    out = pd.Series(np.nan, index=target.index, name="pos")
    prev = np.nan
    tv = target.to_numpy(dtype=float)
    mv = mask_update.fillna(False).to_numpy(dtype=bool)
    for i in range(len(tv)):
        if mv[i] and np.isfinite(tv[i]):
            prev = tv[i]
        out.iloc[i] = prev
    return out


def deadband(target: pd.Series, band: float = 0.4) -> pd.Series:
    """不交易带：目标仓位变动幅度 < band 时**不换仓**，沿用上一期仓位。

    目的同样是压换手；band=0 时退化为原序列。
    """
    target = target.astype(float)
    out = pd.Series(np.nan, index=target.index, name="pos")
    prev = np.nan
    tv = target.to_numpy(dtype=float)
    for i in range(len(tv)):
        if not np.isfinite(tv[i]):
            out.iloc[i] = prev
            continue
        if np.isnan(prev) or abs(tv[i] - prev) >= band:
            prev = tv[i]
        out.iloc[i] = prev
    return out


def long_only(s: pd.Series, lev: pd.Series) -> pd.Series:
    """只做多：把截断后的仓位下限抬到 0（上界仍为 1）。

    `s` 必须是**未乘杠杆**的打分；杠杆只在这里乘一次。
    """
    return s.clip(0.0, 1.0) * lev


def build_positions_long_only(z: pd.Series, vol60: pd.Series, halflife: float,
                               k: float, target_vol: float = 0.10
                               ) -> tuple[pd.Series, pd.Series]:
    """只做多 + EMA 平滑：把打分截断到 [0, 1]（而不是 [-1, 1]），再乘波动率目标。

    与 `build_positions` 的唯一区别是截断区间；杠杆只乘一次。
    """
    s = ema_recursive(z, halflife) * float(k)
    raw = s.clip(0.0, 1.0)
    return raw * leverage(vol60, target_vol), s


# ---------------------------------------------------------------- 网格与选参

def ema_positions(res_z: pd.Series, vol60: pd.Series, halflife_grid,
                  fit_end, target_vol: float = 0.10) -> list:
    """对网格里的每个半衰期算 (halflife, k, pos, s)。

    尺度系数 k 每次都用**训练段**重新拟合（只依赖训练段，不含验证/测试信息）。
    """
    out = []
    for hl in halflife_grid:
        s0 = ema_recursive(res_z, hl)
        k = fit_scale(res_z, s0, fit_end)
        pos, s = build_positions(res_z, vol60, hl, k, target_vol)
        out.append(dict(halflife=float(hl), scale_k=float(k), pos=pos, s=s))
    return out


def select_halflife(valid_sharpe: pd.Series, halflife_grid) -> float:
    """按**验证期**净 Sharpe 选出半衰期。

    这个函数**只接受验证期这一个入参**——测试期指标在函数外单独计算，
    因此不可能反向影响选择。这是「测试集只跑一次」在代码层面的保证。
    """
    s = valid_sharpe.dropna()
    if len(s) == 0:
        return float(list(halflife_grid)[0])
    return float(s.idxmax())
