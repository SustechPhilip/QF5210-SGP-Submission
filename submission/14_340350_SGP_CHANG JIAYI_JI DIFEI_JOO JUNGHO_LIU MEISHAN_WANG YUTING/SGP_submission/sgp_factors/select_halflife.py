"""选定 EMA 半衰期与尺度系数：**只用验证期**，写回 config.yaml 后冻结。

流程（事先写定的规则，不看测试期）：
  1. 用与主回测**完全相同**的管线（loose 筛选 -> RFE 10 -> OLS + expanding）跑出打分 z；
  2. 对半衰期网格 {1,2,3,5,7,10} 逐个算递归 EMA、训练段尺度系数 k、平滑后仓位；
  3. **只按 2023 验证期的净 Sharpe 取最大**（`select_on = valid`）。
     选参函数 `position.select_halflife` **只接受验证期这一个入参**，
     测试期指标在选出之后才计算，结构上不可能反向影响选择；
  4. 选出后写回 `config.yaml` 的 `smoothing.chosen_halflife` / `smoothing.scale_k`，
     之后冻结——回测（backtest.py）与实盘（update.py）都只读这两个值，
     保证逐位一致；
  5. 整个网格（含测试期结果，**仅作披露**）存到 `out/ema_grid_valid.csv`。

用法：
  python3 select_halflife.py            # 选参并写回 config.yaml
  python3 select_halflife.py --dry-run  # 只打印，不写回
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "sgp_data" / "out"
OUT = HERE / "out"
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from config import (FEE, HALFLIFE_GRID, N_SHIFT, RFE_KEEP, SCREEN_PRESET,  # noqa: E402
                    SEED, SMOOTH_SCALE_FIT_END, TARGET_VOL, TEST_START,
                    TRAIN_END, VALID_END, VALID_START, WINDOW, rel,
                    write_back_smoothing)
from models import (MIN_TRAIN, circular_shift_null, run_model, turnover,  # noqa: E402
                    turnover_daily, vol_target_60)
from position import ema_positions, main_positions, select_halflife  # noqa: E402
from screen import (ann_sharpe, net_returns, rfe_ols, screen_factors,  # noqa: E402
                    set_ann, single_factor_signals)


def ann_factor(cal: pd.DataFrame) -> float:
    span_years = (cal.index.max() - cal.index.min()).days / 365.25
    return len(cal) / span_years


def cagr(r: pd.Series, ann: float) -> float:
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    years = len(r) / ann
    return float((1 + r).cumprod().iloc[-1] ** (1 / years) - 1) if years > 0 else np.nan


def main(dry_run: bool = False) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    X = pd.read_csv(OUT / "factors.csv", parse_dates=[0], index_col=0)
    X.index = pd.to_datetime(X.index)
    full = pd.read_csv(DATA / "daily_full.csv", parse_dates=["td"]).set_index("td")
    labels = pd.read_csv(DATA / "labels.csv", parse_dates=["td"]).set_index("td")
    cal = pd.read_csv(DATA / "calendar.csv", parse_dates=["td"]).set_index("td")

    ann = ann_factor(cal)
    set_ann(ann)
    y = labels["y_cc"]
    price = full["C"]
    vol = vol_target_60(full, ann)
    dates = X.index

    # 与主回测同一套特征集合（训练段定一次）
    _, sh = single_factor_signals(X, y, price, FEE, sharpe_end=TRAIN_END)
    sel = screen_factors(X, sh, SCREEN_PRESET, TRAIN_END)
    cols = rfe_ols(X, y, sel, TRAIN_END, n_keep=RFE_KEEP)
    print(f"入选特征: {cols}", flush=True)

    res = run_model(X, y, price, cols, model="ols", window=WINDOW, fee=FEE,
                    vol_series=vol, ann=ann, min_train=MIN_TRAIN)
    print(f"首个可预测日 {res['pos'].first_valid_index().date()}", flush=True)

    # ---------- 第一步：只算验证期 ----------
    rows = []
    for item in ema_positions(res["z"], vol, HALFLIFE_GRID,
                              SMOOTH_SCALE_FIT_END, TARGET_VOL):
        pos = item["pos"]
        R = net_returns(pos, y.reindex(dates), price.reindex(dates), FEE)
        dp = turnover_daily(pos)
        rows.append(dict(
            method="ema", halflife=item["halflife"], scale_k=item["scale_k"],
            valid_sharpe=ann_sharpe(R.loc[VALID_START:VALID_END], ann),
            valid_cagr=cagr(R.loc[VALID_START:VALID_END], ann),
            valid_turnover=turnover(pos.loc[VALID_START:VALID_END], ann, dp=dp),
        ))
    grid = pd.DataFrame(rows)

    # 不平滑基准（主结果口径）
    pos0 = main_positions(res["z"], vol, TARGET_VOL)
    R0 = net_returns(pos0, y.reindex(dates), price.reindex(dates), FEE)
    dp0 = turnover_daily(pos0)
    base = dict(method="none", halflife=0.0, scale_k=1.0,
                valid_sharpe=ann_sharpe(R0.loc[VALID_START:VALID_END], ann),
                valid_cagr=cagr(R0.loc[VALID_START:VALID_END], ann),
                valid_turnover=turnover(pos0.loc[VALID_START:VALID_END], ann, dp=dp0))

    # ---------- 选择：只用验证期净 Sharpe ----------
    # 索引设成半衰期本身，`select_halflife` 返回的才是半衰期而不是行号。
    grid = grid.set_index("halflife")
    best_hl = select_halflife(grid["valid_sharpe"], HALFLIFE_GRID)
    best = grid.loc[best_hl]
    print(f"\n按验证期净 Sharpe 选出半衰期 = {best_hl:.0f} 日"
          f"（k = {best['scale_k']:.6f}）", flush=True)

    # ---------- 第二步：测试期指标，纯披露，不参与选择 ----------
    test_rows = []
    for item in ema_positions(res["z"], vol, HALFLIFE_GRID,
                              SMOOTH_SCALE_FIT_END, TARGET_VOL):
        pos = item["pos"]
        R = net_returns(pos, y.reindex(dates), price.reindex(dates), FEE)
        dp = turnover_daily(pos)
        test_rows.append(dict(
            test_sharpe=ann_sharpe(R.loc[TEST_START:], ann),
            test_cagr=cagr(R.loc[TEST_START:], ann),
            test_turnover=turnover(pos.loc[TEST_START:], ann, dp=dp),
        ))
    base["test_sharpe"] = ann_sharpe(R0.loc[TEST_START:], ann)
    base["test_cagr"] = cagr(R0.loc[TEST_START:], ann)
    base["test_turnover"] = turnover(pos0.loc[TEST_START:], ann, dp=dp0)

    # 按位置对齐（两段都是 6 行 EMA，顺序一致），再把基准行放到最前
    te = pd.DataFrame(test_rows)
    assert len(te) == len(grid), "网格与测试期行数不一致"
    grid = pd.concat([grid.reset_index(), te.reset_index(drop=True)], axis=1)
    grid = pd.concat([pd.DataFrame([base]), grid], ignore_index=True)
    grid["selected"] = (grid["method"] == "ema") & (grid["halflife"] == best_hl)

    out_csv = OUT / "ema_grid_valid.csv"
    grid.to_csv(out_csv, index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print("\n半衰期网格（选择只用验证期；测试期仅作披露）：")
        print(grid.round(4).to_string(index=False))
    print(f"\nwrote {rel(out_csv)}", flush=True)

    # ---------- 披露：各半衰期的循环平移 p 值 ----------
    # 报告 §6.4 说「改进版单侧 p = 0.051，按半衰期 3 日算是 0.052」。
    # 这两个数必须能用包内代码复现，所以在这里一并算出来。
    # 用的是 `models.circular_shift_null`（与 backtest.py 的 R4 完全同一份实现），
    # **不是** pandas `ewm`——两者在首个有效值与缺失日的处理上不同，
    # 用 ewm 算出来的数与回测口径不一致（这是报告早期版本的一个错误）。
    print("\n各半衰期的循环平移随机对照（测试期，1000 次；仅作披露）:")
    p_rows = []
    for item in ema_positions(res["z"], vol, HALFLIFE_GRID,
                              SMOOTH_SCALE_FIT_END, TARGET_VOL):
        pos = item["pos"]
        R = net_returns(pos, y.reindex(dates), price.reindex(dates), FEE)
        real = ann_sharpe(R.loc[TEST_START:], ann)
        null = circular_shift_null(pos.loc[TEST_START:], y, price, FEE,
                                    n_boot=N_SHIFT, ann=ann, seed=SEED)
        pct = float((null < real).mean())
        p_rows.append(dict(halflife=item["halflife"], scale_k=item["scale_k"],
                            test_sharpe=real,
                            null_mean=float(null.mean()),
                            null_p95=float(null.quantile(0.95)),
                            percentile=pct, p_one_sided=float(1.0 - pct)))
        tag = "  <== 冻结值" if item["halflife"] == float(best_hl) else ""
        print(f"  半衰期 {item['halflife']:.0f} 日: Sharpe={real:.4f} "
              f"零假设均值={null.mean():.4f} 分位={pct:.4f} "
              f"单侧 p={1-pct:.4f}{tag}", flush=True)
    p_df = pd.DataFrame(p_rows)
    p_csv = OUT / "ema_pvalue_by_halflife.csv"
    p_df.to_csv(p_csv, index=False)
    print(f"wrote {rel(p_csv)}", flush=True)

    if dry_run:
        print("dry-run：未写回 config.yaml")
        return 0
    write_back_smoothing(float(best_hl), float(best["scale_k"]))
    print(f"已写回 config.yaml: chosen_halflife={float(best_hl)}, "
          f"scale_k={float(best['scale_k']):.10f}", flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    raise SystemExit(main(dry_run=ap.parse_args().dry_run))
