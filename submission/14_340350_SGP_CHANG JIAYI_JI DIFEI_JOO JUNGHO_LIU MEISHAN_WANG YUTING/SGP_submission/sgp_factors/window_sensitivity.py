"""筛选窗口敏感性扫描：把「换窗口能不能救主结果」这件事做成可复现的脚本。

背景
----
主结果的因子筛选依赖两个窗口设定：

1. **单因子滚动外推的拟合窗**（`fit_window`）：每个因子的外推信号由一段历史拟合
   出来，再算它的外推 Sharpe。默认 `expanding`（用全部可用历史）。
2. **单因子外推 Sharpe 的评估窗**（`eval_window`）：Sharpe 只在训练段末尾的
   N 个交易日里算（`all` = 每个因子从自己开始出信号算到训练段末日）。

这两个窗口都属于「筛选口径」的一部分，事先没有在验证期上单独调过。本脚本
把它们的组合全部跑一遍，回答两个问题：

* 有没有哪个窗口组合能显著改写结论（尤其是测试期）？
* 如果有，它是**事先登记**的还是**事后从几十个组合里挑出来的**？

**重要**：本脚本是**事后**的敏感性分析，不是选参工具。主结果的口径
（`fit_window=expanding`、`eval_window=all`、不去均值）是在测试集结果出来
之前就登记在计划里的（v2 §4.1），**不因本脚本的结果而改变**。

输出
----
* `out/window_sensitivity.csv`：每个组合一行
  （去均值开关 × 拟合窗 × 评估窗 × 入选因子 × 验证/测试期 Sharpe）

用法
----
    python3 sgp_factors/window_sensitivity.py            # 全量（140 组，约 10 分钟）
    python3 sgp_factors/window_sensitivity.py --quick    # 8 组冒烟（约 2 分钟）
"""
from __future__ import annotations

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

from config import (FEE, MIN_TRAIN, MIN_TRAIN_SCREEN, RFE_KEEP, SCREEN_PRESET,  # noqa: E402
                    TEST_START, TRAIN_END, VALID_END, VALID_START, WINDOW, rel)
from models import run_model, vol_target_60  # noqa: E402
from screen import (ann_sharpe, net_returns, rfe_ols, screen_factors,  # noqa: E402
                    set_ann, single_factor_signals)

# 拟合窗：expanding + 一组滚动长度
FIT_WINDOWS = [("expanding", None)] + [("rolling", n) for n in
                                       (100, 130, 160, 190, 210, 240, 280, 320, 360)]
# 评估窗（训练段末尾的交易日数）；None = 每个因子从自己开始出信号算到训练段末日
EVAL_WINDOWS = [None, 150, 200, 250, 300, 350, 450]


def ann_factor(cal: pd.DataFrame) -> float:
    span_years = (cal.index.max() - cal.index.min()).days / 365.25
    return len(cal) / span_years


def eval_window_start(cal: pd.DataFrame, n_days: int | None) -> pd.Timestamp | None:
    """评估窗左端点：训练段最后 n_days 个交易日的第一天。None = 不限制。"""
    if n_days is None:
        return None
    tr = cal.index[cal.index <= TRAIN_END]
    if len(tr) <= n_days:
        return tr[0]
    return tr[-n_days]


def sharpe_on_window(sig: pd.DataFrame, yv: pd.Series, price: pd.Series,
                     fee: float, start: pd.Timestamp | None,
                     sharpe_end: pd.Timestamp) -> pd.Series:
    """与 `screen.single_factor_signals` 内部完全一致的口径：

    ev = (日期 <= sharpe_end) & (日期 >= start)，逐因子算净 Sharpe。
    """
    ev = sig.index
    ev = ev[ev <= sharpe_end]
    if start is not None:
        ev = ev[ev >= start]
    out = {}
    for col in sig.columns:
        s = sig[col].reindex(ev)
        r = net_returns(s, yv.reindex(ev), price.reindex(ev), fee)
        out[col] = ann_sharpe(r)
    return pd.Series(out)


def main() -> int:
    quick = "--quick" in sys.argv
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

    fit_windows = FIT_WINDOWS
    eval_windows = EVAL_WINDOWS
    if quick:
        fit_windows = [("expanding", None), ("rolling", 240)]
        eval_windows = [None, 250]

    rows = []
    for demean in (False, True):
        for kind, win_len in fit_windows:
            t0 = time.time()
            # 1) 单因子滚动外推信号（与主结果同一个函数，只是换拟合窗/去均值）
            sig, _ = single_factor_signals(
                X, y, price, FEE, min_train=MIN_TRAIN_SCREEN,
                window=kind, win_len=(win_len or 0), sharpe_end=TRAIN_END,
                demean_signal=demean, eval_start=None)
            yv = y.reindex(dates)
            for n_ev in eval_windows:
                start = eval_window_start(cal, n_ev)
                sh = sharpe_on_window(sig, yv, price, FEE, start, TRAIN_END)
                sel = screen_factors(X, sh, SCREEN_PRESET, TRAIN_END)
                cols = rfe_ols(X, y, sel, TRAIN_END, n_keep=RFE_KEEP)
                res = run_model(X, y, price, cols, model="ols", window=WINDOW,
                                fee=FEE, vol_series=vol, ann=ann,
                                min_train=MIN_TRAIN)
                R = net_returns(res["pos"], yv, price.reindex(dates), FEE)
                row = dict(
                    demean=int(demean),
                    fit_window=kind if kind == "expanding" else f"rolling{win_len}",
                    eval_window="all" if n_ev is None else str(n_ev),
                    n_screened=len(sel),
                    n_features=len(cols),
                    features=";".join(cols),
                    valid_sharpe=ann_sharpe(R.loc[VALID_START:VALID_END], ann),
                    test_sharpe=ann_sharpe(R.loc[TEST_START:], ann),
                    # **退化行标记**：筛选一个因子都没选出来时，模型退化成
                    # 「零特征 OLS」——`models.run_model` 在 cols 为空时仍会拟合
                    # 截距项，z 变成天文数字、仓位近似常多，于是照样算出一个
                    # Sharpe（例如 +0.45）。**这不是策略表现，是没有策略**，
                    # 统计时必须剔除。
                    degenerate=int(len(cols) == 0),
                )
                rows.append(row)
                print(f"demean={int(demean)} fit={row['fit_window']} "
                      f"eval={row['eval_window']}: screen={len(sel)} "
                      f"feats={cols} valid={row['valid_sharpe']:.4f} "
                      f"test={row['test_sharpe']:.4f}", flush=True)
            print(f"  (fit={kind}{win_len or ''} 用时 {time.time()-t0:.1f}s)",
                  flush=True)

    df = pd.DataFrame(rows)
    out_csv = OUT / "window_sensitivity.csv"
    df.to_csv(out_csv, index=False)

    live = df[df.degenerate == 0]
    print("\n=== 汇总（有效行：至少选入 1 个因子；按测试期 Sharpe 排序，前 10） ===")
    top = live.sort_values("test_sharpe", ascending=False).head(10)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(top[["demean", "fit_window", "eval_window", "n_screened",
                   "valid_sharpe", "test_sharpe"]].round(4).to_string(index=False))
    print("\n=== 主结果登记口径（demean=0, expanding, all） ===")
    base = df[(df.demean == 0) & (df.fit_window == "expanding")
              & (df.eval_window == "all")]
    print(base[["n_screened", "features", "valid_sharpe", "test_sharpe"]]
          .round(4).to_string(index=False))
    n_deg = int(df.degenerate.sum())
    print(f"\n共 {len(df)} 组；其中退化行（筛选无因子入选）{n_deg} 组，不参与统计")
    print(f"有效行 {len(live)} 组：测试期 Sharpe 为正的有 "
          f"{int((live.test_sharpe > 0).sum())} 组，"
          f"验证期与测试期同时为正的有 "
          f"{int(((live.test_sharpe > 0) & (live.valid_sharpe > 0)).sum())} 组")
    print(f"有效行的测试期 Sharpe：最大 {live.test_sharpe.max():.4f} / "
          f"中位 {live.test_sharpe.median():.4f} / 最小 {live.test_sharpe.min():.4f}")
    print(f"wrote {rel(out_csv)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
