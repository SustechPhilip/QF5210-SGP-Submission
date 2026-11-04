"""SGP 日频择时：因子模块（A–E 类）。

设计约束（见 QF5210_SGP回测详细计划_补充v2.md）：
  * 因子只能读 daily_dec.csv 的第 t 行，和 daily_full.csv 的 shift(1) 行。
  * B 类需要分钟级重算：从 minutes_tagged.pkl 取 session=='day' 且 hm<='16:59' 的 bar。
  * B11_CLOSESETTLE / B12_TAIL15 需要前一日的收盘附近分钟数据，用 full.shift(1) 口径。
  * 坏日（thin_day 稀疏日 或 data_gap 盘中数据中断）当日口径的 B 类因子置 NaN；
    用 t-1 数据的 B 类因子改为「前一日坏掉才置空」。预测时由调用方用最近一次
    可用值填充（as-of）。

约定：
  P(hh:mm) = 该时刻**及之前**最后一根在场 bar 的收盘价（缺失分钟视为无成交，不前值补价）。
  例外：P_0830 缺失时回退到当日开盘价 O_t（仅缺失时回退，不改变定义）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PREV_DAY_B_FACTORS  # noqa: E402

WINDOWS = (1, 5, 20, 60, 120)

# B 类用到的时刻锚点
B_TIMES = ("08:30", "09:00", "12:00", "13:00",
           "16:29", "16:39", "16:49", "16:50", "16:54", "16:59")


def _hm2min(hm: str) -> int:
    h, m = hm.split(":")
    return int(h) * 60 + int(m)


def _ma(s: pd.Series, k: int) -> pd.Series:
    return s.rolling(k, min_periods=k).mean()


def _sd(s: pd.Series, k: int) -> pd.Series:
    return s.rolling(k, min_periods=k).std()


def _ema(s: pd.Series, k: int) -> pd.Series:
    return s.ewm(span=k, adjust=False, min_periods=k).mean()


def _pct_rank(s: pd.Series, k: int) -> pd.Series:
    """s 在过去 k 个交易日（含当日）中的分位数，0~1。"""
    return s.rolling(k, min_periods=k).rank(pct=True)


def _true_range(h: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev = close.shift(1)
    return pd.concat([h - low, (h - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)


# ---------------------------------------------------------------- 分钟级聚合

class MinuteAgg:
    """分钟 bar -> B 类因子中间量。

    decision：只含 <=16:59 的日盘 bar，用于当日 B 类因子。
    full_tail：含 17:00 之后的日盘 bar，用于前一日收盘附近口径（B11/B12），只用 shift(1)。
    """

    def __init__(self, minutes: pd.DataFrame, anchor_open: bool = False):
        d = minutes[minutes["session"] == "day"].sort_values(["td", "t"])
        self.full = d
        self.dec = d[d["hm"] <= "16:59"]
        self.anchor_open = anchor_open

    # -- 时刻价格 --
    def _price_at(self, frame: pd.DataFrame, times=B_TIMES) -> pd.DataFrame:
        out = {}
        for hhmm in times:
            sel = frame[frame["hm"] <= hhmm].groupby("td")["c"].last()
            out[f"P_{hhmm.replace(':', '')}"] = sel
        return pd.DataFrame(out)

    def _anchor_open(self, out: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
        """开盘锚点口径。

        主口径：P_0830 缺失时**才**回退到当日开盘价 O_t（实测只有 2025-07-30 一天
        第一根 bar 是 08:31）。不改变 P_0830 的定义。

        稳健性口径（anchor_open=True）：所有交易日都用 O_t 当锚点。
        P_0830 与 O_t 全样本平均差 12.7bp、p95 35.9bp、仅 4.5% 的日子相等，
        会改变全部日子的 B3_SEG_AM / B3_SEG_OPEN30 —— 属于事后重新定义因子，
        只进稳健性，不进主结果。
        """
        o = d.groupby("td")["o"].first()
        if self.anchor_open:
            out["P_0830"] = o.reindex(out.index)
        else:
            out["P_0830"] = out["P_0830"].fillna(o.reindex(out.index))
        return out

    # -- VWAP 窗口 --
    def _vwap_window(self, frame: pd.DataFrame, lo: str, hi: str) -> pd.Series:
        w = frame[(frame["hm"] >= lo) & (frame["hm"] <= hi)]
        return (w["c"] * w["v"]).groupby(w["td"]).sum() / w["v"].groupby(w["td"]).sum()

    def build_decision(self) -> pd.DataFrame:
        d = self.dec
        idx = pd.Index(sorted(d["td"].unique()), name="td")
        out = self._price_at(d).reindex(idx)

        g = d.groupby("td")
        out["m_n"] = g.size()
        out["m_open"] = g["o"].first()
        out = self._anchor_open(out, d)
        out["m_high"] = g["h"].max()
        out["m_low"] = g["l"].min()
        out["m_vwap"] = (d["c"] * d["v"]).groupby(d["td"]).sum() / g["v"].sum()
        out["vwap_1650_1659"] = self._vwap_window(d, "16:50", "16:59")
        out["tail_vol_share"] = (
            d[(d["hm"] >= "16:29") & (d["hm"] <= "16:59")].groupby("td")["v"].sum()
            / g["v"].sum()
        )

        # ---- 分钟收益（在场 bar 之间） ----
        r = np.log(d["c"]).groupby(d["td"]).diff()
        dv = d["v"].groupby(d["td"]).diff()
        a = pd.DataFrame({"td": d["td"].values, "r": r.values, "v": d["v"].values, "dv": dv.values})
        gr = a.groupby("td")

        out["rv"] = gr["r"].apply(lambda s: np.sqrt(np.nansum(s.to_numpy() ** 2)))
        out["rv_up"] = gr["r"].apply(lambda s: np.sqrt(np.nansum(s.clip(lower=0).to_numpy() ** 2)))
        out["rv_dn"] = gr["r"].apply(lambda s: np.sqrt(np.nansum(s.clip(upper=0).to_numpy() ** 2)))

        # ---- 涨跌量比 ----
        # 定义（v2 §3 B7）：**上涨分钟成交量之和 ÷ 下跌分钟成交量之和**。
        # 注意是成交量的和，不是「收益×成交量」的和——后者量纲错误，且分母恒为负，
        # 会让该因子在全样本恒为负（历史 bug，已修）。
        upv = d["v"].where(r > 0, 0.0)
        dnv = d["v"].where(r < 0, 0.0)
        uv = pd.DataFrame({"td": d["td"].values, "up": upv.values, "dn": dnv.values}).groupby("td").sum()
        out["ud_vol"] = uv["up"] / uv["dn"].replace(0, np.nan)

        # ---- 分组均值/标准差 ----
        # 注意：groupby.transform 返回**原索引**（a 的 RangeIndex），而 groupby.agg 返回
        # 唯一 td 索引。两者混用会索引对不上 -> 静默全 NaN。故这里显式分成两套：
        #   *_row : 行级（与 a 同序 RangeIndex），只用于 a 内部逐行运算
        #   *_g   : 组级（唯一 td 索引），用于与 c3/c4/cov 等 td 索引对象做除法
        rm_row = gr["r"].transform("mean")
        rs_row = gr["r"].transform("std")
        dvm_row = gr["dv"].transform("mean")
        dvs_row = gr["dv"].transform("std")
        vm_row = gr["v"].transform("mean")
        vs_row = gr["v"].transform("std")

        rm_g = gr["r"].mean()
        rs_g = gr["r"].std()
        dvm_g = gr["dv"].mean()
        dvs_g = gr["dv"].std()
        vm_g = gr["v"].mean()
        vs_g = gr["v"].std()

        # ---- 量价相关 corr(r_i, dv_i) ----
        cov = ((a["r"] - rm_row) * (a["dv"] - dvm_row)).groupby(a["td"]).mean()
        out["pv_corr"] = cov / (rs_g * dvs_g).replace(0, np.nan)

        # ---- 分钟收益偏度 / 超额峰度 ----
        c3 = ((a["r"] - rm_row) ** 3).groupby(a["td"]).mean()
        c4 = ((a["r"] - rm_row) ** 4).groupby(a["td"]).mean()
        out["m_skew"] = c3 / rs_g ** 3
        out["m_kurt"] = c4 / rs_g ** 4 - 3.0

        # ---- 分钟成交量 1-4 阶矩 ----
        out["vm1"] = vm_g
        out["vm2"] = vs_g
        out["vm3"] = (((a["v"] - vm_row) ** 3).groupby(a["td"]).mean()) / vs_g ** 3
        out["vm4"] = (((a["v"] - vm_row) ** 4).groupby(a["td"]).mean()) / vs_g ** 4 - 3.0

        # ---- 高低点时刻（归一化到 [0,1]，基准 08:30 -> 16:59） ----
        span = _hm2min("16:59") - _hm2min("08:30")
        gd = d.groupby("td")
        hi_idx = gd["h"].idxmax()
        lo_idx = gd["l"].idxmin()
        out["high_t"] = hi_idx.map(lambda t: (_hm2min(d.at[t, "hm"]) - _hm2min("08:30")) / span)
        out["low_t"] = lo_idx.map(lambda t: (_hm2min(d.at[t, "hm"]) - _hm2min("08:30")) / span)

        return out

    def build_full_tail(self) -> pd.DataFrame:
        """完整日盘的收盘附近信息：P(17:00)、P(17:14)、17:05-17:14 VWAP。"""
        f = self.full
        idx = pd.Index(sorted(f["td"].unique()), name="td")
        out = pd.DataFrame(index=idx)
        for hhmm in ("17:00", "17:14"):
            out[f"P_{hhmm.replace(':', '')}"] = (
                f[f["hm"] <= hhmm].groupby("td")["c"].last().reindex(idx))
        out["vwap_1705_1714"] = self._vwap_window(f, "17:05", "17:14").reindex(idx)
        return out


# ---------------------------------------------------------------- A 类

def build_A(dec: pd.DataFrame) -> pd.DataFrame:
    """趋势与动量。基准序列 C* = dec.C_star。"""
    c = dec["C_star"]
    vwap = dec["d_vwap"]
    out = pd.DataFrame(index=dec.index)

    for k in WINDOWS:
        out[f"A1_MOM_{k}"] = np.log(c / c.shift(k))
    for k in (5, 20, 60):
        out[f"A2_MOMVWAP_{k}"] = np.log(vwap / vwap.shift(k))
    for k in (20, 60):
        out[f"A3_TS_{k}"] = (c - _ma(c, k)) / _sd(c, k)
        lo = c.rolling(k, min_periods=k).min()
        hi = c.rolling(k, min_periods=k).max()
        out[f"A4_DC_{k}"] = (c - lo) / (hi - lo).replace(0, np.nan)
    out["A5_MMS"] = (_ma(c, 5) - _ma(c, 20)) / c
    return out


# ---------------------------------------------------------------- B 类

def build_B(dec: pd.DataFrame, full_prev: pd.DataFrame,
             tail_prev: pd.DataFrame, ma: MinuteAgg) -> pd.DataFrame:
    """日内价量。

    full_prev : daily_full.shift(1)（完整日盘，含 C 与 C_vwap_1705_1714）
    tail_prev : 分钟级完整日盘尾部的 shift(1)（P_1700 / P_1714）
    """
    out = pd.DataFrame(index=dec.index)
    m = ma.build_decision()

    c_star = dec["C_star"]
    vwap = dec["d_vwap"]

    # B1 尾盘收益：P(16:59-n)
    for n, key in ((5, "P_1654"), (10, "P_1649"), (30, "P_1629")):
        out[f"B1_TAIL_{n}"] = np.log(c_star / m[key])

    # B2 VWAP 偏离
    out["B2_VDEV"] = (c_star - vwap) / vwap

    # B3 分时段收益
    out["B3_SEG_AM"] = np.log(m["P_1200"] / m["P_0830"])
    out["B3_SEG_NOON"] = np.log(m["P_1300"] / m["P_1200"])
    out["B3_SEG_PM"] = np.log(c_star / m["P_1300"])
    out["B3_SEG_OPEN30"] = np.log(m["P_0900"] / m["P_0830"])

    # B4 已实现波动
    out["B4_RV"] = m["rv"]
    out["B4_RVUP"] = m["rv_up"]
    out["B4_RVDN"] = m["rv_dn"]
    out["B4_UDRATIO"] = m["rv_up"] / m["rv_dn"].replace(0, np.nan)

    # B5 分钟收益分布
    out["B5_SKEW"] = m["m_skew"]
    out["B5_KURT"] = m["m_kurt"]

    # B6 分钟成交量矩
    out["B6_VMOM1"] = m["vm1"]
    out["B6_VMOM2"] = m["vm2"]
    out["B6_VMOM3"] = m["vm3"]
    out["B6_VMOM4"] = m["vm4"]

    # B7 涨跌量比
    out["B7_UDVOL"] = m["ud_vol"]

    # B8 量价相关
    out["B8_PVCORR"] = m["pv_corr"]

    # B9 高低点时刻
    out["B9_HIGHT"] = m["high_t"]
    out["B9_LOWT"] = m["low_t"]

    # B10 尾盘成交占比（分母为决策版全日量）
    out["B10_TAILVOL"] = m["tail_vol_share"]

    # B11 收盘 vs 17:05-17:14 VWAP 偏离（前一日，研报对照口径）
    out["B11_CLOSESETTLE"] = ((full_prev["C"] - full_prev["C_vwap_1705_1714"])
                              / full_prev["C_vwap_1705_1714"])

    # B11 当日 pre-close 版（16:50-16:59 VWAP 偏离）
    out["B11_PRECLOSE"] = (c_star - m["vwap_1650_1659"]) / m["vwap_1650_1659"]

    # B12 前一日 17:00 -> 17:14 收益
    out["B12_TAIL15"] = np.log(tail_prev["P_1714"] / tail_prev["P_1700"])
    return out


# ---------------------------------------------------------------- C 类

def build_C(dec: pd.DataFrame) -> pd.DataFrame:
    """技术指标（在 C* 日序列上）。参数为常规技术分析参数，属窗口规则的已声明例外。"""
    c = dec["C_star"]
    h = dec["d_high"]
    l = dec["d_low"]
    out = pd.DataFrame(index=dec.index)

    # MACD(12,26,9)
    dif = _ema(c, 12) - _ema(c, 26)
    dea = _ema(dif, 9)
    out["C1_DIF"] = dif
    out["C2_DEA"] = dea
    out["C3_HIST"] = dif - dea

    # RSI(14) Wilder
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    out["C4_RSI14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    # KDJ(9,3,3)
    lo9 = l.rolling(9, min_periods=9).min()
    hi9 = h.rolling(9, min_periods=9).max()
    rsv = (c - lo9) / (hi9 - lo9).replace(0, np.nan) * 100
    k = rsv.ewm(alpha=1 / 3, adjust=False, min_periods=3).mean()
    d = k.ewm(alpha=1 / 3, adjust=False, min_periods=3).mean()
    out["C5_K"] = k
    out["C6_D"] = d
    out["C7_J"] = 3 * k - 2 * d

    # 布林带位置
    out["C8_BOLL"] = (c - _ma(c, 20)) / (2 * _sd(c, 20))

    # ATR(14)/C*
    atr = _true_range(h, l, c).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    out["C9_ATR14"] = atr / c

    # 均线偏离
    for k in (5, 10, 20, 60, 120):
        out[f"C10_MADEV_{k}"] = (c - _ma(c, k)) / _ma(c, k)

    # CCI(20)：典型价
    tp = (h + l + c) / 3
    md = tp.rolling(20, min_periods=20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    out["C11_CCI20"] = (tp - _ma(tp, 20)) / (0.015 * md)

    # Williams %R(14)
    hi14 = h.rolling(14, min_periods=14).max()
    lo14 = l.rolling(14, min_periods=14).min()
    out["C12_WILLR14"] = (hi14 - c) / (hi14 - lo14).replace(0, np.nan) * -100

    # 120 日分位数位置
    out["C13_PCT120"] = _pct_rank(c, 120)

    # 连续涨/跌天数（截断 ±20）
    sign = np.sign(c.diff())
    grp = (sign != sign.shift()).cumsum()
    out["C14_STREAK"] = sign.groupby(grp).cumsum().clip(-20, 20)
    return out


# ---------------------------------------------------------------- D 类

def build_D(dec: pd.DataFrame) -> pd.DataFrame:
    """量能与拥挤。全部用决策版 V* = dec.d_vol。"""
    v = dec["d_vol"]
    c = dec["C_star"]
    out = pd.DataFrame(index=dec.index)
    out["D1_VOLQ_120"] = _pct_rank(v, 120)
    out["D2_VOLCROW"] = v / _ma(v, 20) - 1
    out["D3_TURNCHG"] = np.log(v * c * 100).diff()
    return out


# ---------------------------------------------------------------- E 类

def build_E(dec: pd.DataFrame) -> pd.DataFrame:
    """夜盘与日盘。主版本用共有窗口 nc_*（17:50-04:44）。"""
    out = pd.DataFrame(index=dec.index)
    no, nc = dec["nc_open"], dec["nc_close"]
    nv = dec["nc_vol"]
    vstar = dec["d_vol"]
    out["E1_NRET"] = np.log(nc / no)
    out["E2_NRV"] = dec["nc_rv"]
    out["E3_NSHARE"] = nv / (nv + vstar)
    out["E4_GAP"] = np.log(dec["d_open"] / nc)
    out["E5_DAYNIGHT"] = np.log(dec["C_star"] / dec["d_open"]) - np.log(nc / no)
    return out


def build_E_robust(dec: pd.DataFrame) -> pd.DataFrame:
    """稳健性版本：全窗口夜盘 n_*、每分钟均量。"""
    out = pd.DataFrame(index=dec.index)
    out["E1F_NRET_FULL"] = np.log(dec["n_close"] / dec["n_open"])
    out["E2F_NRV_FULL"] = dec["n_rv"]
    out["E3F_NSHARE_FULL"] = dec["n_vol"] / (dec["n_vol"] + dec["d_vol"])
    out["E6_NPM"] = dec["nc_vol"] / dec["nc_nmin"].replace(0, np.nan)
    out["E7_DPM"] = dec["d_vol"] / dec["d_nmin"].replace(0, np.nan)
    return out


# ---------------------------------------------------------------- 组装

CLASS_OF_PREFIX = {"A": "A", "B": "B", "C": "C", "D": "D", "E": "E"}


def factor_class(col: str) -> str:
    return col[0]


def b7_legacy(minutes: pd.DataFrame) -> pd.Series:
    """B7_UDVOL 的**历史（错误）公式**，仅用于「实现修正对照」。

    历史实现写成了「收益×成交量」之和的比值：
        upv = r.where(r>0,0) * v ;  dnv = r.where(r<0,0) * v
    分母恒为负 -> 因子在全样本恒为负，且量纲错误。
    正确定义见 build_decision()：上涨分钟成交量之和 ÷ 下跌分钟成交量之和。

    保留这个函数是为了能复现「只修 B7」这一版的结果，说明正负号为什么会翻转。
    """
    d = minutes[minutes["session"] == "day"].sort_values(["td", "t"])
    d = d[d["hm"] <= "16:59"]
    r = np.log(d["c"]).groupby(d["td"]).diff()
    upv = (r.where(r > 0, 0.0) * d["v"].values)
    dnv = (r.where(r < 0, 0.0) * d["v"].values)
    uv = pd.DataFrame({"td": d["td"].values, "up": upv, "dn": dnv}).groupby("td").sum()
    out = uv["up"] / uv["dn"].replace(0, np.nan)
    out.name = "B7_UDVOL"
    return out


def build_all(dec: pd.DataFrame, full: pd.DataFrame, minutes: pd.DataFrame,
              robust: bool = False, anchor_open: bool = False) -> pd.DataFrame:
    """组装全部因子。

    dec   : daily_dec.csv（t 日夜盘 + t 日日盘 <=16:59）
    full  : daily_full.csv（完整日盘），因子只用 shift(1)
    minutes: minutes_tagged.pkl
    anchor_open : True 时所有交易日用当日开盘价 O_t 当 08:30 锚点（稳健性口径）
    """
    ma = MinuteAgg(minutes, anchor_open=anchor_open)
    tail = ma.build_full_tail()

    full_prev = full.shift(1)
    tail_prev = tail.shift(1)

    parts = [build_A(dec), build_B(dec, full_prev, tail_prev, ma),
             build_C(dec), build_D(dec), build_E(dec)]
    X = pd.concat(parts, axis=1)
    X = X.reindex(dec.index)

    # ---- 坏日掩码：thin_day（稀疏日）OR data_gap（盘中数据中断）----
    # 这两天当日分钟数据不可靠，当日口径的 B 类因子置 NaN。
    thin = full["thin_day"].astype(bool).reindex(X.index).fillna(False)
    gap = full["data_gap"].astype(bool).reindex(X.index).fillna(False)
    bad = thin | gap

    bcols = [c for c in X.columns if c.startswith("B")]
    # 用 t-1 数据的 B 类因子：它们在**当日**坏掉时仍然有效（t-1 数据是好的），
    # 只有**前一日**坏掉时才置空。历史 bug：把当日坏掉的全部 B 类一律置空，
    # 白白删掉 B11_CLOSESETTLE / B12_TAIL15，而次日反而用到了坏日的薄尾盘数据。
    prev_day_b = [c for c in PREV_DAY_B_FACTORS if c in X.columns]
    same_day_b = [c for c in bcols if c not in prev_day_b]

    X.loc[bad, same_day_b] = np.nan
    prev_bad = bad.shift(1, fill_value=True)
    X.loc[prev_bad, prev_day_b] = np.nan

    if robust:
        X = pd.concat([X, build_E_robust(dec)], axis=1)
    return X
