"""日频聚合：决策表 dec（t 日只用 <=16:59 日盘 + t 日夜盘）与完整表 full。

因子只能读 dec.loc[:t] 与 full.shift(1)；full 当日行只用于标签、执行和复核。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (GAP_MIN_MINUTES, THIN_EXPECTED_MINUTES,  # noqa: E402
                    THIN_RATIO)
from sessions import COMMON_CLOSE_VWAP, COMMON_NIGHT, DECISION_CUTOFF


def _ohlcv(g: pd.DataFrame, p: str) -> pd.DataFrame:
    g = g.sort_values("t")
    agg = g.groupby("td").agg(
        **{
            f"{p}open": ("o", "first"),
            f"{p}high": ("h", "max"),
            f"{p}low": ("l", "min"),
            f"{p}close": ("c", "last"),
            f"{p}vol": ("v", "sum"),
            f"{p}nmin": ("v", "size"),
            f"{p}last_hm": ("hm", "last"),
        }
    )
    pv = (g["c"] * g["v"]).groupby(g["td"]).sum()
    agg[f"{p}vwap"] = pv / agg[f"{p}vol"]  # 价格用分钟收盘价近似
    return agg


def _rv(g: pd.DataFrame, p: str) -> pd.DataFrame:
    g = g.sort_values("t")
    r = np.log(g["c"]).groupby(g["td"]).diff()
    gg = g.assign(r=r)
    out = gg.groupby("td")["r"].agg(
        **{f"{p}rv": lambda x: np.sqrt(np.nansum(x**2)),
           f"{p}rv_up": lambda x: np.sqrt(np.nansum(np.clip(x, 0, None) ** 2)),
           f"{p}rv_dn": lambda x: np.sqrt(np.nansum(np.clip(x, None, 0) ** 2))}
    )
    return out


def build_daily(m: pd.DataFrame, cal: pd.DataFrame,
                gap_min_minutes: int = GAP_MIN_MINUTES,
                thin_ratio: float = THIN_RATIO,
                thin_expected: int = THIN_EXPECTED_MINUTES
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """日频聚合：决策表 dec（t 日只用 <=16:59 日盘 + t 日夜盘）与完整表 full。

    因子只能读 dec.loc[:t] 与 full.shift(1)；full 当日行只用于标签、执行和复核。

    **坏日判定（thin_day / data_gap）在本函数内一次完成**，与正式回测管线
    完全同源——截断一致性测试直接调用 build_daily，不再另外赋值 data_gap，
    保证测试跑的管线与出结果的管线是同一条。
    """
    m = m[m["session"].isin(["day", "night"])]  # 排除 pending 与 offbook
    day = m[m["session"] == "day"]
    night = m[m["session"] == "night"]
    day_dec = day[day["hm"] <= DECISION_CUTOFF]
    night_common = night[(night["hm"] >= COMMON_NIGHT[0]) | (night["hm"] <= COMMON_NIGHT[1])]

    idx = pd.Index(cal["td"], name="td")
    base = cal.set_index("td")[["regime", "decision_ts"]]

    # ---------- 决策表：t 日 17:00 前全部可知 ----------
    dec = base.join(_ohlcv(day_dec, "d_")).join(_rv(day_dec, "d_"))
    dec = dec.join(_ohlcv(night, "n_")).join(_rv(night, "n_"))
    dec = dec.join(_ohlcv(night_common, "nc_")).join(_rv(night_common, "nc_"))
    dec = dec.rename(columns={"d_close": "C_star"})
    dec["C_star_filled"] = (dec["d_last_hm"] != DECISION_CUTOFF).astype(int)
    dec = dec.reindex(idx)

    # ---------- 完整表：含 17:00 之后，t 日当天不可用于因子 ----------
    full = base.join(_ohlcv(day, "D_")).join(_rv(day, "D_"))
    full = full.rename(columns={"D_close": "C", "D_open": "O"})
    day_last = cal.set_index("td")["day_last"]
    full["close_filled"] = (full["D_last_hm"] != day_last).astype(int)
    c1714 = day[day["hm"] <= "17:14"].sort_values("t").groupby("td")["c"].last()
    full["C_1714"] = c1714
    w = day[(day["hm"] >= COMMON_CLOSE_VWAP[0]) & (day["hm"] <= COMMON_CLOSE_VWAP[1])]
    full["C_vwap_1705_1714"] = (w["c"] * w["v"]).groupby(w["td"]).sum() / w["v"].groupby(w["td"]).sum()
    full["short_day"] = (full["D_nmin"] < 300).astype(int)

    # thin_day：稀疏交易日。现货休市（耶稣受难日、圣诞节、节礼日、平安夜、除夕、
    # 农历新年）导致期货成交稀，分钟数少但仍正常交易到当期收盘，不是半日市。
    #
    # **只用决策窗内（<=16:59）的 bar 数判定**：判定「今天的数据够不够用」时
    # 不能看 17:00 之后的信息。决策窗 08:30-16:59 固定 510 根，四个交易规则时期
    # 都一样，因此与时期无关（不再按 regime 取 525/530）。
    # 实测：改用决策窗口径后命中的 15 天与原全日口径**完全相同**（0 处分歧）。
    dec_bars = day_dec.groupby("td").size().reindex(idx).fillna(0)
    full["thin_day"] = (dec_bars < thin_ratio * thin_expected).astype(int)

    # data_gap：盘中长时间无成交的数据中断/暂停（目前只有 2024-11-25 13:36-15:45）
    # 同样只用决策窗内的 bar 判定；判定阈值读 config，不在此硬编码。
    full["data_gap"] = (find_data_gaps(m, cal, gap_min_minutes)
                        .reindex(idx).fillna(0).astype(int))
    full = full.reindex(idx)
    return dec, full


def find_data_gaps(m: pd.DataFrame, cal: pd.DataFrame,
                   min_gap_minutes: int = GAP_MIN_MINUTES) -> pd.Series:
    """找日盘内连续无成交超过 min_gap_minutes 的段落，返回 td -> 1 的标记。

    只看当日日盘 session（不含夜盘），夜盘的正常稀疏不算数据中断；
    只看**决策窗内（<=16:59）**的 bar——17:00 决策时点之后的空档不影响决策。
    """
    day = m[(m["session"] == "day") & (m["hm"] <= DECISION_CUTOFF)].sort_values("t")

    def mins(hm: str) -> int:
        h, mm = hm.split(":")
        return int(h) * 60 + int(mm)

    gap_flag = {}
    for td, g in day.groupby("td"):
        hm = g["hm"].tolist()
        worst = 0
        for a, b in zip(hm[:-1], hm[1:]):
            worst = max(worst, mins(b) - mins(a) - 1)
        if worst >= min_gap_minutes:
            gap_flag[td] = 1
    return pd.Series(gap_flag, dtype="int64").reindex(cal["td"]).fillna(0).astype(int)


def build_labels(full: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:
    """标签与其实现时间。训练只可用 label_known_ts < decision_ts[t] 的行。"""
    c = cal.set_index("td")
    lab = pd.DataFrame(index=full.index)
    # 方案一：t 日 17:00 决策、t 日收盘成交，y = C_{t+1}/C_t - 1，于 t+1 日收盘实现
    lab["y_cc"] = full["C"].shift(-1) / full["C"] - 1
    lab["y_cc_known_ts"] = c["close_ts"].shift(-1)
    # 稳健性收盘口径
    for k in ["C_1714", "C_vwap_1705_1714"]:
        lab[f"y_{k}"] = full[k].shift(-1) / full[k] - 1
    # 方案二（研报口径）：t+1 日 08:25 决策，y = O_{t+2}/O_{t+1} - 1，于 t+2 日开盘实现
    lab["y_oo"] = full["O"].shift(-2) / full["O"].shift(-1) - 1
    lab["y_oo_known_ts"] = pd.Series(c.index, index=c.index).shift(-2) + pd.Timedelta("08:30:00")
    return lab
