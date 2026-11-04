"""因子模块主脚本：构建 A–E 类因子，输出因子表与登记表。

python build_factors.py [--robust]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "sgp_data" / "out"
OUT = HERE / "out"

sys.path.insert(0, str(HERE.parent))
from config import rel  # noqa: E402
from features import build_all, factor_class  # noqa: E402


def load():
    dec = pd.read_csv(DATA / "daily_dec.csv", parse_dates=["td"]).set_index("td")
    full = pd.read_csv(DATA / "daily_full.csv", parse_dates=["td"]).set_index("td")
    labels = pd.read_csv(DATA / "labels.csv", parse_dates=["td"]).set_index("td")
    cal = pd.read_csv(DATA / "calendar.csv", parse_dates=["td"]).set_index("td")
    minutes = pd.read_pickle(DATA / "minutes_tagged.pkl")
    minutes["td"] = pd.to_datetime(minutes["td"])
    return dec, full, labels, cal, minutes


def registry(X: pd.DataFrame) -> pd.DataFrame:
    """因子登记表：id、类别、窗口、数据来源、first_valid。"""
    rows = []
    for c in X.columns:
        cls = factor_class(c)
        if cls == "A":
            src = "dec.C_star / dec.d_vwap"
        elif cls == "B":
            src = "分钟重算(<=16:59) / full.shift(1)"
        elif cls == "C":
            src = "dec.C_star / dec.d_high / dec.d_low"
        elif cls == "D":
            src = "dec.d_vol / dec.C_star"
        else:
            src = "dec.nc_* / dec.d_*"
        fv = X[c].first_valid_index()
        rows.append(dict(
            id=c, cls=cls,
            window="常规技术参数" if cls == "C" else "1/5/20/60/120",
            window_rule_exception="Y" if cls == "C" else "N",
            source=src,
            first_valid=None if fv is None else fv.strftime("%Y-%m-%d"),
            n_valid=int(X[c].notna().sum()),
            nan_frac=round(float(X[c].isna().mean()), 4),
            # 下面三列由 backtest.py 在筛选+RFE 之后回填；
            # 这里先写 N，不能一律标 Y（历史 bug：65 个因子全标「进入主模型」，
            # 实际进入主模型的只有 3 个）。
            screened_in="N",
            rfe_kept="N",
            in_main_model="N",
        ))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robust", action="store_true", help="附加 E 类稳健性版本")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    dec, full, labels, cal, minutes = load()
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    X = build_all(dec, full, minutes, robust=args.robust)
    print(f"factors built in {time.time()-t0:.1f}s, shape={X.shape}", flush=True)

    print("类别因子数:", X.columns.map(factor_class).value_counts().to_dict())
    for cls in "ABCDE":
        cols = [c for c in X.columns if factor_class(c) == cls]
        if not cols:
            continue
        fv = X[cols].notna().idxmax().max()
        print(f"  {cls}: n={len(cols)} 类别 first_valid={fv.date() if pd.notna(fv) else None}")
    all_fv = X.notna().idxmax().max()
    print(f"全部因子同时可用的起点: {all_fv.date()}", flush=True)

    X.to_csv(OUT / "factors.csv")
    reg = registry(X)
    reg.to_csv(OUT / "factor_registry.csv", index=False)
    print(f"wrote {rel(OUT/'factors.csv')} and {rel(OUT/'factor_registry.csv')}",
          flush=True)

    tr = X.loc[:"2022-12-31"]
    print("\n训练段各因子 |Spearman| 中位数 vs 其他:")
    corr = tr.corr(method="spearman", min_periods=100)
    med = corr.abs().where(~np.eye(len(corr), dtype=bool)).median()
    for cls in "ABCDE":
        cols = [c for c in corr.columns if factor_class(c) == cls]
        if cols:
            print(f"  {cls}: {med[cols].median():.3f} (min {med[cols].min():.3f})")


if __name__ == "__main__":
    main()
