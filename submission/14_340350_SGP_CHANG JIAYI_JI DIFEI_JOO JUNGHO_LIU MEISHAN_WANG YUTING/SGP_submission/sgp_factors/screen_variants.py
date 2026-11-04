"""筛选方法变体对照：复现报告 §6.5「只修筛选方法没用」这个结论。

背景
----
主结果的因子筛选是「滚动外推单因子 Sharpe > 0 且与入选因子 |ρ| < 0.2」的
loose 档。§6.5 声称：**换掉筛选方法本身并不能救主结果，降换手才是有效的**。
这个结论必须能被读者核对，所以这里把它写成脚本。

两个筛选方法变体
----------------
1. **去均值信号**（`demean_signal=True`）
   默认的单因子信号 s_t = clip(ŷ_t / sd_fit)，用的是拟合值本身；变体先减掉
   训练窗内拟合值的均值 alpha + beta·fbar，再标准化：
       s_t = clip( (ŷ_t − mean_fit) / sd_fit, -1, 1 )
   含义：信号只保留「因子相对训练窗均值的偏离」，去掉水平漂移。

2. **统一评估窗口**（`eval_start=...`）
   默认每个因子的外推 Sharpe 在**自己的**信号起始日之后计算，不同因子的
   可用起点不同，横比不公平；变体把所有因子的 Sharpe 限制在同一个窗口里：
   左端点取在「最后一个因子开始出信号的那天」
   （`screen.common_signal_start`），右端点仍是训练段末日。

每个变体都跑**完整管线**：单因子外推 -> 筛选 -> RFE -> OLS 模型 -> 仓位。
并且对每个变体都算两个仓位口径：
  * **不平滑**（主结果口径）
  * **EMA 平滑**（半衰期取冻结值 config.yaml 的 `chosen_halflife`）

用法：
  python3 screen_variants.py
输出：
  out/screen_variants.csv   每个变体一行的汇总表
  logs/screen_variants.log  （由外层重定向产生）
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "sgp_data" / "out"
OUT = HERE / "out"
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from config import (CHOSEN_HALFLIFE, FEE, RFE_KEEP, SCALE_K, SCREEN_PRESET,  # noqa: E402
                    SMOOTHING_ENABLED, TARGET_VOL, TEST_START, TRAIN_END,
                    VALID_END, VALID_START, WINDOW, rel)
from models import MIN_TRAIN, run_model, turnover, turnover_daily, vol_target_60  # noqa: E402
from position import ema_positions  # noqa: E402
from screen import (ann_sharpe, common_signal_start, net_returns, rfe_ols,  # noqa: E402
                    screen_factors, set_ann, single_factor_signals)


def ann_factor(cal: pd.DataFrame) -> float:
    span_years = (cal.index.max() - cal.index.min()).days / 365.25
    return len(cal) / span_years


def main() -> int:
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
    print(f"ANN={ann:.1f}，因子 {X.shape[1]} 个，交易日 {len(dates)}", flush=True)

    # ---- 基线信号（默认口径），用来定「统一评估窗口」的左端点 ----
    sig_base, sh_base = single_factor_signals(X, y, price, FEE,
                                               sharpe_end=TRAIN_END)
    eval_start = common_signal_start(sig_base)
    print(f"统一评估窗口左端点（最后一个因子开始出信号的日子）: {eval_start.date()}",
          flush=True)

    variants = [
        dict(name="基线(主结果口径)", demean_signal=False, eval_start=None),
        dict(name="去均值信号", demean_signal=True, eval_start=None),
        dict(name="统一评估窗口", demean_signal=False, eval_start=eval_start),
    ]

    rows = []
    for v in variants:
        t0 = time.time()
        sig, sh = single_factor_signals(X, y, price, FEE, sharpe_end=TRAIN_END,
                                         demean_signal=v["demean_signal"],
                                         eval_start=v["eval_start"])
        sel = screen_factors(X, sh, SCREEN_PRESET, TRAIN_END)
        cols = rfe_ols(X, y, sel, TRAIN_END, n_keep=RFE_KEEP)
        res = run_model(X, y, price, cols, model="ols", window=WINDOW, fee=FEE,
                        vol_series=vol, ann=ann, min_train=MIN_TRAIN)
        pos = res["pos"]
        R = net_returns(pos, y.reindex(dates), price.reindex(dates), FEE)
        dp = turnover_daily(pos)
        row = dict(
            variant=v["name"],
            n_screened=len(sel),
            features=";".join(cols),
            valid_sharpe=ann_sharpe(R.loc[VALID_START:VALID_END], ann),
            test_sharpe=ann_sharpe(R.loc[TEST_START:], ann),
            valid_turnover=turnover(pos.loc[VALID_START:VALID_END], ann, dp=dp),
            test_turnover=turnover(pos.loc[TEST_START:], ann, dp=dp),
            n_features=len(cols),
        )
        print(f"\n--- {v['name']} ---")
        print(f"筛选入选 {len(sel)} 个，RFE 后 {cols}")
        print(f"不平滑: valid={row['valid_sharpe']:.3f} "
              f"test={row['test_sharpe']:.3f} "
              f"({time.time()-t0:.1f}s)", flush=True)

        # 同一变体下再算 EMA 平滑版（用冻结的半衰期与尺度系数）
        if SMOOTHING_ENABLED and CHOSEN_HALFLIFE is not None and SCALE_K is not None:
            hl, k = float(CHOSEN_HALFLIFE), float(SCALE_K)
            s0 = ema_positions(res["z"], vol, [hl], TRAIN_END, TARGET_VOL)[0]
            pos_e, _ = s0["pos"], s0["s"]
            Re = net_returns(pos_e, y.reindex(dates), price.reindex(dates), FEE)
            dpe = turnover_daily(pos_e)
            row["ema_valid_sharpe"] = ann_sharpe(Re.loc[VALID_START:VALID_END], ann)
            row["ema_test_sharpe"] = ann_sharpe(Re.loc[TEST_START:], ann)
            row["ema_valid_turnover"] = turnover(pos_e.loc[VALID_START:VALID_END],
                                                  ann, dp=dpe)
            row["ema_test_turnover"] = turnover(pos_e.loc[TEST_START:], ann, dp=dpe)
            print(f"EMA 平滑(半衰期{hl:.0f}日): valid={row['ema_valid_sharpe']:.3f} "
                  f"test={row['ema_test_sharpe']:.3f}", flush=True)
        rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = OUT / "screen_variants.csv"
    df.to_csv(out_csv, index=False)
    print("\n=== 汇总 ===")
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(df[["variant", "n_screened", "n_features",
                  "valid_sharpe", "test_sharpe", "valid_turnover", "test_turnover",
                  "ema_valid_sharpe", "ema_test_sharpe"]].round(4).to_string(index=False))
    print(f"\nwrote {rel(out_csv)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
