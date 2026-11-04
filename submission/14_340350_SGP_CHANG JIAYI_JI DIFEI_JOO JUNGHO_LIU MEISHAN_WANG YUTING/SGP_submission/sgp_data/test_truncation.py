"""未来函数测试（数据层）：把分钟数据截到 t 日 16:59:59 重算，dec 的 t 行及以前、full 的 t-1 行及以前必须与全量一致。

python test_truncation.py <SGP_2.csv> [n]
（因子层 + 模型层的扩展版见 ../sgp_factors/test_truncation.py）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from daily import build_daily
from sessions import assign_sessions, build_calendar, load_minutes


def rebuild(raw: pd.DataFrame):
    cal = build_calendar(raw)
    m = assign_sessions(raw, cal)
    return build_daily(m, cal)


def main(src: str, n: int = 20, seed: int = 0) -> None:
    raw = load_minutes(src)
    dec_all, full_all = rebuild(raw)
    num = dec_all.select_dtypes("number").columns
    numf = full_all.select_dtypes("number").columns
    rng = np.random.default_rng(seed)
    days = dec_all.index[300:]
    picks = sorted(rng.choice(days, size=n, replace=False))
    for t in picks:
        cut = raw[raw["t"] <= pd.Timestamp(t) + pd.Timedelta("16:59:59")]
        dec_c, full_c = rebuild(cut)
        a, b = dec_all.loc[:t, num], dec_c.loc[:t, num]
        assert a.index.equals(b.index), f"{t}: 交易日历不一致"
        assert np.allclose(a.values, b.values, equal_nan=True), f"{t}: dec 行不一致"
        prev = full_all.index[full_all.index.get_loc(t) - 1]
        assert np.allclose(full_all.loc[:prev, numf].values, full_c.loc[:prev, numf].values, equal_nan=True), f"{t}: full(t-1) 不一致"
        print("ok", pd.Timestamp(t).date())
    print(f"截断测试通过：{n} 个日期")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "data/SGP_2.csv", int(a[1]) if len(a) > 1 else 20)
