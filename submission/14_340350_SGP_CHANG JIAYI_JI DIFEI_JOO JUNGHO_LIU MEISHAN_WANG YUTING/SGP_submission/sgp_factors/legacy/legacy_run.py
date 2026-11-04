"""原版实现快照 + 「只修 B7」对照。

本目录下的 `features.py` / `models.py` / `screen.py` 是**实现修正之前**的原样快照
（只加了一个 `B7_LEGACY` 开关）。保留它们是为了让下面这个结论**可复现**：

    测试期 Sharpe 的**正负号会随实现细节翻转**——
    原版 −0.133 / 只修 B7 +0.104 / 全部修正 −0.021。

不要在本目录里加新功能。

用法：
  python3 legacy/legacy_run.py             # 原版（全部实现缺陷都在）
  python3 legacy/legacy_run.py --fix-b7    # 只修 B7_UDVOL 公式，其余不动
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT.parent / "sgp_data" / "out"
OUT = ROOT / "out"

# legacy 目录优先，确保 import 到的是快照版本
sys.path.insert(0, str(HERE))

import features as lf  # noqa: E402
import models as lm  # noqa: E402
import screen as ls  # noqa: E402

TRAIN_END = pd.Timestamp("2022-12-31")
VALID_START = pd.Timestamp("2023-01-01")
VALID_END = pd.Timestamp("2023-12-31")
TEST_START = pd.Timestamp("2024-01-01")
FEE = 2e-4


def ann_factor(cal: pd.DataFrame) -> float:
    idx = pd.DatetimeIndex(cal["td"] if "td" in cal.columns else cal.index)
    return len(idx) / ((idx.max() - idx.min()).days / 365.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix-b7", action="store_true", help="把 B7 换成修正后的公式")
    ap.add_argument("--out", default=None, help="把结果写成 JSON")
    args = ap.parse_args()

    lf.B7_LEGACY = not args.fix_b7
    variant = "b7_only" if args.fix_b7 else "original"

    dec = pd.read_csv(DATA / "daily_dec.csv", parse_dates=["td"]).set_index("td")
    full = pd.read_csv(DATA / "daily_full.csv", parse_dates=["td"]).set_index("td")
    labels = pd.read_csv(DATA / "labels.csv", parse_dates=["td"]).set_index("td")
    cal = pd.read_csv(DATA / "calendar.csv", parse_dates=["td"]).set_index("td")
    minutes = pd.read_pickle(DATA / "minutes_tagged.pkl")
    minutes["td"] = pd.to_datetime(minutes["td"])

    ann = ann_factor(cal)
    ls.set_ann(ann)
    X = lf.build_all(dec, full, minutes, robust=False)
    y = labels["y_cc"]
    price = full["C"]
    vol = lm.vol_target_60(full, ann)

    sigs, sh = ls.single_factor_signals(X, y, price, FEE, sharpe_end=TRAIN_END)
    sel = ls.screen_factors(X, sh, "loose", TRAIN_END)
    cols = ls.rfe_ols(X, y, sel, TRAIN_END, n_keep=10)
    res = lm.run_model(X, y, price, cols, model="ols", window="expanding", fee=FEE,
                       vol_series=vol, ann=ann, min_train=250)
    R = ls.net_returns(res["pos"], y.reindex(X.index), price.reindex(X.index), FEE)

    v = lm.metrics(R.loc[VALID_START:VALID_END], ann)
    t = lm.metrics(R.loc[TEST_START:], ann)
    out = dict(variant=variant, features=cols,
               valid_sharpe=v["sharpe"], test_sharpe=t["sharpe"],
               test_cagr=t["cagr"], test_max_dd=t["max_dd"],
               n_valid=int(v["n"]), n_test=int(t["n"]))
    print(f"[{variant}] 入选特征 {cols}")
    print(f"  验证 Sharpe={v['sharpe']:+.4f}  测试 Sharpe={t['sharpe']:+.4f}  "
          f"CAGR={t['cagr']:+.4f}  maxDD={t['max_dd']:+.4f}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
