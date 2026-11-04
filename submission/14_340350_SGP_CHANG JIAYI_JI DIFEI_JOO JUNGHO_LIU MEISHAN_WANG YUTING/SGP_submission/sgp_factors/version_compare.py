"""实现修正对照：把三版结果并排跑出来，说明「正负号会随实现细节翻转」。

三版：
  original  修复前的原版实现（12 处实现缺陷同时存在）。由 legacy/ 快照目录跑，
            那里保存的是**修复前的原样代码**，不是改出来的。
  b7_only   只修 B7_UDVOL 公式，其余 11 处缺陷保留。同样由 legacy/ 跑。
  fixed     全部 12 处实现缺陷都修（= 当前主结果）。

用法：
  python3 version_compare.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "sgp_data" / "out"
OUT = HERE / "out"

sys.path.insert(0, str(HERE.parent))
from config import (FEE, RFE_KEEP, SCREEN_PRESET, TEST_START, TRAIN_END,  # noqa: E402
                    VALID_END, VALID_START, WINDOW, rel)
from features import build_all  # noqa: E402
from models import MIN_TRAIN, metrics, run_model, vol_target_60  # noqa: E402
from screen import (net_returns, rfe_ols, screen_factors,  # noqa: E402
                    set_ann, single_factor_signals)


def ann_factor(cal: pd.DataFrame) -> float:
    idx = pd.DatetimeIndex(cal["td"] if "td" in cal.columns else cal.index)
    return len(idx) / ((idx.max() - idx.min()).days / 365.25)


def load():
    dec = pd.read_csv(DATA / "daily_dec.csv", parse_dates=["td"]).set_index("td")
    full = pd.read_csv(DATA / "daily_full.csv", parse_dates=["td"]).set_index("td")
    labels = pd.read_csv(DATA / "labels.csv", parse_dates=["td"]).set_index("td")
    cal = pd.read_csv(DATA / "calendar.csv", parse_dates=["td"]).set_index("td")
    minutes = pd.read_pickle(DATA / "minutes_tagged.pkl")
    minutes["td"] = pd.to_datetime(minutes["td"])
    return dec, full, labels, cal, minutes


def run_variant(name: str, X: pd.DataFrame, y, price, vol, ann) -> dict:
    sigs, sh = single_factor_signals(X, y, price, FEE, sharpe_end=TRAIN_END)
    sel = screen_factors(X, sh, SCREEN_PRESET, TRAIN_END)
    cols = rfe_ols(X, y, sel, TRAIN_END, n_keep=RFE_KEEP)
    res = run_model(X, y, price, cols, model="ols", window=WINDOW, fee=FEE,
                    vol_series=vol, ann=ann, min_train=MIN_TRAIN)
    R = net_returns(res["pos"], y.reindex(X.index), price.reindex(X.index), FEE)
    v = metrics(R.loc[VALID_START:VALID_END], ann)
    t = metrics(R.loc[TEST_START:], ann)
    out = dict(variant=name, features=cols,
               valid_sharpe=v["sharpe"], test_sharpe=t["sharpe"],
               test_cagr=t["cagr"], test_max_dd=t["max_dd"])
    print(f"{name:10s} 入选 {cols}")
    print(f"{'':10s} 验证 Sharpe={v['sharpe']:+.4f}  测试 Sharpe={t['sharpe']:+.4f}  "
          f"CAGR={t['cagr']:+.4f}")
    return out


def main():
    dec, full, labels, cal, minutes = load()
    ann = ann_factor(cal)
    set_ann(ann)
    y = labels["y_cc"]
    price = full["C"]
    vol = vol_target_60(full, ann)

    rows = []

    # --- 原版 / 只修 B7：由 legacy/ 快照跑（这两版是原样代码，不是改出来的）---
    import subprocess
    for variant, flag in (("original", []), ("b7_only", ["--fix-b7"])):
        tmp = OUT / f"_legacy_{variant}.json"
        subprocess.run([sys.executable, str(HERE / "legacy" / "legacy_run.py"),
                        *flag, "--out", str(tmp)], check=True,
                       cwd=str(HERE / "legacy"))
        rows.append(json.loads(tmp.read_text()))
        tmp.unlink()

    # --- 全部修正：当前主结果 ---
    X_fixed = build_all(dec, full, minutes, robust=False)
    rows.append(run_variant("fixed", X_fixed, y, price, vol, ann))

    order = {"original": 0, "b7_only": 1, "fixed": 2}
    rows.sort(key=lambda d: order[d["variant"]])
    df = pd.DataFrame(rows)[["variant", "features", "valid_sharpe", "test_sharpe",
                             "test_cagr", "test_max_dd"]]
    df.to_csv(OUT / "version_compare.csv", index=False)
    with open(OUT / "version_compare.json", "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print("\n=== 实现修正对照 ===")
    print(df.to_string(index=False))
    print(f"\nwrote {rel(OUT/'version_compare.csv')}")


if __name__ == "__main__":
    main()
