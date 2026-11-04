"""验证 MinuteAgg 中原先全 NaN 的 7 个因子已修复（索引对齐问题）。

用法: python3 test_minute_agg_fix.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from features import MinuteAgg  # noqa: E402

FIXED = ["m_skew", "m_kurt", "vm1", "vm2", "vm3", "vm4", "pv_corr"]


def synth(n_days: int = 30, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        td = pd.Timestamp("2023-01-02") + pd.Timedelta(days=d)
        if td.weekday() >= 5:
            continue
        t0 = pd.Timestamp("2023-01-02 08:30")
        times = pd.date_range(t0, periods=500, freq="1min") + pd.Timedelta(days=d)
        n = len(times)
        c = 4000 * np.exp(np.cumsum(rng.normal(0, 2e-4, n)))
        v = rng.lognormal(10, 0.4, n)
        rows.append(pd.DataFrame({
            "td": td, "t": times, "hm": times.strftime("%H:%M"),
            "o": c, "h": c * 1.0005, "l": c * 0.9995, "c": c, "v": v,
            "session": "day",
        }))
    return pd.concat(rows, ignore_index=True)


def main() -> int:
    m = MinuteAgg(synth())
    out = m.build_decision()
    print("build_decision shape:", out.shape)
    bad = []
    for c in FIXED:
        nn = int(out[c].notna().sum())
        print(f"  {c:10s} notna={nn:4d}/{len(out)}")
        if nn < len(out) * 0.9:
            bad.append(c)
    # 数值合理性：skew/kurt 应有限；vm3/vm4 不应为 0（除退化情形）
    for c in ["m_skew", "m_kurt", "vm3", "vm4"]:
        v = out[c].replace([np.inf, -np.inf], np.nan).dropna()
        print(f"  {c:10s} min={v.min():.4g} max={v.max():.4g} std={v.std():.4g}")
    if bad:
        print("FAIL: 仍全 NaN 的因子:", bad)
        return 1
    print("PASS: 7 个因子全部修复")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
