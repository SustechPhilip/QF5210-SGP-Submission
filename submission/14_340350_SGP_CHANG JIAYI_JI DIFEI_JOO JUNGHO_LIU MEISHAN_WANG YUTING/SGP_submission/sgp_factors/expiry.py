"""到期日（每月倒数第二个营业日）与其对应交易日的计算。

**唯一实现。** `backtest.py` 的换月检验（R11）与扩展因子的 K1_EXPIRY_WK
必须用同一套定义，否则两处说的「到期日」不是同一天。
本模块从 `backtest.py` 里原样搬出，行为逐位不变。

两个历史 bug（搬过来时保留注释，以免重犯）：

* 原函数名 `last_bd_2nd` 但实际返回的是每月**最后一个**营业日
  （循环里每次覆盖 prev），导致到期日错位。
* 遇到该月不是交易日时**直接跳过该月**，只找到 76 个月（样本 77 个月），
  换月检验因此少了最后一个月的到期日窗口。现在改为**顺延到上一个交易日**。
"""
from __future__ import annotations

import datetime as dt

import pandas as pd


def bd_2nd_last(yy: int, mm: int) -> dt.date | None:
    """每月**倒数第二个营业日**（周一至周五，不排除法定假日）。"""
    d = dt.date(yy, mm, 1)
    bds = []
    while d.month == mm:
        if d.weekday() < 5:
            bds.append(d)
        d += dt.timedelta(days=1)
    return bds[-2] if len(bds) >= 2 else None


def expiry_trading_day(yy: int, mm: int,
                        cal_index: pd.DatetimeIndex) -> pd.Timestamp | None:
    """每月倒数第二个营业日对应的**交易日**。

    若该日不是交易日（法定假日，或落在样本区间之外），**顺延到上一个交易日**。
    """
    e = bd_2nd_last(yy, mm)
    if e is None:
        return None
    ts = pd.Timestamp(e)
    idx = cal_index[cal_index <= ts]
    if len(idx) == 0:
        return None
    return idx[-1]


def expiry_positions(cal_index: pd.DatetimeIndex):
    """样本区间内每个月的到期日 -> 交易日历位置。

    返回 (expiry_pos, n_months, n_rolled, rolled_months)：
    * expiry_pos       : 到期日在 cal_index 中的位置（去重、升序）
    * n_months         : 样本覆盖到的月份数
    * n_rolled         : 到期日顺延到上一个交易日的月份数
    * rolled_months    : 顺延明细，形如 "2021-05->2021-05-28"
    """
    expiry_pos: list[int] = []
    n_months = 0
    n_rolled = 0
    rolled_months: list[str] = []
    # 只遍历**样本覆盖到的月份**（2020-06 ~ 2026-10 共 77 个）；
    # 样本之外的月份（如 2026-11、2026-12）不属于换月检验的范围。
    for p in pd.period_range(cal_index.min(), cal_index.max(), freq="M"):
        n_months += 1
        d = expiry_trading_day(p.year, p.month, cal_index)
        if d is None:
            continue
        if pd.Timestamp(bd_2nd_last(p.year, p.month)) != d:
            n_rolled += 1
            rolled_months.append(f"{p}->{d.date()}")
        expiry_pos.append(cal_index.get_loc(d))
    return sorted(set(expiry_pos)), n_months, n_rolled, rolled_months


def expiry_window_positions(expiry_pos: list[int],
                             window_days: int) -> list[int]:
    """到期窗口内的交易日历位置。

    每个到期日往前取 window_days-1 个交易日（含到期日共 window_days 个交易日）。
    """
    excl_idx: set[int] = set()
    for i in expiry_pos:
        for k in range(window_days):
            j = i - k
            if j >= 0:
                excl_idx.add(j)
    return sorted(excl_idx)
