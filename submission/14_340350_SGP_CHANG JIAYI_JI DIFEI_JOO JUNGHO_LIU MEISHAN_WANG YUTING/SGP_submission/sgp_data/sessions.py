"""SGP 分钟数据：读入、tick 化、按历史规则做会话切分与校验。

规则（point-in-time，不用现行规则倒推历史）：
  R0 2020-06-15  日盘末根 17:14  夜盘首 17:50  夜盘末根 04:44  （2020 交易日历；05:15 收盘在数据中自 2021-05-20 起）
  R1 2021-05-20  17:14  17:50  05:14  （数据实测）
  R2 2024-05-13  17:19  17:50  05:14  （数据实测：日盘延至 17:20，官方通告待核）
  R3 2025-04-07  17:19  17:35  05:14  （SGX 2025-04-07 T+1 提前开盘，见报告出处）

归属规则不写死时刻：每个交易日 D 的收盘边界 close_ts[D] = D + 日盘末根 + 1 分钟，
任一 bar 归属于 close_ts 严格大于其时间戳的第一个交易日（夜盘、周五夜盘、节前夜盘自动后移）。
规则表只用于断言校验。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TICK = 0.05
DAY_OPEN = "08:30"
DECISION_CUTOFF = "16:59"          # 决策截断：只用 <=16:59 的日盘 bar（各时期一致）
COMMON_NIGHT = ("17:50", "04:44")  # 各时期共有的夜盘窗口
COMMON_CLOSE_VWAP = ("17:05", "17:14")  # 各时期都存在、覆盖 17:06-17:09 对冲高峰

REGIMES = pd.DataFrame(
    [
        ("R0", "2020-06-15", "17:14", "17:50", "04:44"),
        ("R1", "2021-05-20", "17:14", "17:50", "05:14"),
        ("R2", "2024-05-13", "17:19", "17:50", "05:14"),
        ("R3", "2025-04-07", "17:19", "17:35", "05:14"),
    ],
    columns=["regime", "start", "day_last", "night_first", "night_last"],
)
REGIMES["start"] = pd.to_datetime(REGIMES["start"])


def regime_of(dates: pd.Series | pd.DatetimeIndex) -> pd.DataFrame:
    """返回每个日期对应的规则行（按 start 向前匹配）。"""
    d = pd.to_datetime(pd.Series(dates)).dt.normalize()
    i = np.searchsorted(REGIMES["start"].values, d.values, side="right") - 1
    if (i < 0).any():
        raise ValueError("日期早于第一个规则时期")
    return REGIMES.iloc[i].reset_index(drop=True)


def load_minutes(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = ["t", "o", "h", "l", "c", "v"]
    df["t"] = pd.to_datetime(df["t"])
    df = df.sort_values("t").drop_duplicates("t").reset_index(drop=True)
    # 整数 tick 存储，消除 505.20001 类浮点误差
    for k in ["o", "h", "l", "c"]:
        tk = np.rint(df[k] / TICK).astype(np.int64)
        off = (df[k] / TICK - tk).abs().max()
        assert off < 1e-3, f"{k} 存在不在 0.05 格点上的价格（最大偏离 {off:.4f} tick）"
        df[k + "_tk"] = tk
        df[k] = tk * TICK
    df["hm"] = df["t"].dt.strftime("%H:%M")
    df["date"] = df["t"].dt.normalize()
    return df


def build_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """交易日 = 在当期日盘窗口 [08:30, day_last] 内有成交 bar 的日期。"""
    reg = regime_of(df["date"])
    in_day = (df["hm"].values >= DAY_OPEN) & (df["hm"].values <= reg["day_last"].values)
    days = pd.DatetimeIndex(sorted(df.loc[in_day, "date"].unique()))
    cal = pd.DataFrame({"td": days})
    r = regime_of(cal["td"])
    cal["regime"] = r["regime"].values
    cal["day_last"] = r["day_last"].values
    cal["close_ts"] = cal["td"] + pd.to_timedelta(cal["day_last"] + ":00") + pd.Timedelta(minutes=1)
    cal["decision_ts"] = cal["td"] + pd.Timedelta(hours=17)  # 方案一：17:00 决策
    return cal


def assign_sessions(df: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:
    """给每根 bar 标注交易日 td 与 session（day/night）；样本尾部尚无归属日的夜盘标为 pending。"""
    close_ts = cal["close_ts"].values
    k = np.searchsorted(close_ts, df["t"].values, side="right")
    out = df.copy()
    out["td"] = pd.NaT
    ok = k < len(cal)
    out.loc[ok, "td"] = cal["td"].values[k[ok]]
    day_last = pd.Series(cal["day_last"].values, index=cal["td"])
    dl = out["td"].map(day_last)
    is_day = ok & (out["date"] == out["td"]) & (out["hm"] >= DAY_OPEN) & (out["hm"] <= dl.fillna(""))
    out["session"] = np.where(~ok, "pending", np.where(is_day, "day", "night"))
    # 当日 05:15–08:29 的 bar：不在任何时期的交易时段内（R3 出现 42 笔，整数手、多在集合竞价/撮合空档），
    # 按场外登记的大宗成交处理：不进价格与因子，单独记录
    offbook = ok & (out["date"] == out["td"]) & (out["hm"] > "05:14") & (out["hm"] < DAY_OPEN)
    out.loc[offbook, "session"] = "offbook"
    return out


def validate(m: pd.DataFrame) -> pd.DataFrame:
    """按 bar 自身日期所属规则检查是否落在允许时段；返回越界 bar 明细。"""
    reg = regime_of(m["date"])
    hm = m["hm"].values
    day_ok = (hm >= DAY_OPEN) & (hm <= reg["day_last"].values)
    eve_ok = hm >= reg["night_first"].values
    morn_ok = hm <= reg["night_last"].values
    bad = ~(day_ok | eve_ok | morn_ok)
    # 日盘 bar 必须在日盘窗口；夜盘 bar 不得落在日盘窗口
    bad = bad & (m["session"] != "offbook").values  # offbook 单独统计
    sess_bad = ((m["session"] == "day").values & ~day_ok) | ((m["session"] == "night").values & day_ok & (m["date"] == m["td"]).values)
    return m.loc[bad | sess_bad, ["t", "hm", "v", "session", "td"]].assign(regime=reg["regime"].values[bad | sess_bad])
