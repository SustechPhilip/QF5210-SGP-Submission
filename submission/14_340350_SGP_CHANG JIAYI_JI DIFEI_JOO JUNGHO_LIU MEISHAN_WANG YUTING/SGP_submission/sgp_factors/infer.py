"""P0 推断检验：最好单因子基准、Deflated Sharpe、Sharpe 差异检验、风险情景表。

一次性产出四组结果，全部写进 `out/`：

1. `out/baseline_single_factor.csv`
   「最好单因子」基准：组合 vs 单因子的增益到底有多少。
2. `out/dsr.csv`
   Deflated Sharpe Ratio：扣掉多重检验之后还剩多少显著性。
3. `out/sharpe_diff.csv`
   两个策略的 Sharpe 差是否显著（配对平稳块自助法）。
4. `out/risk_scenarios.csv`
   风险情景表：目标波动与杠杆上限同比例放大。

**「最好单因子」基准的口径（重要）**

* 选因子：在**训练段**（≤2022-12-31）按外推 Sharpe 选出最好的单因子，
  不用全样本 Sharpe（否则等于用测试期信息选因子）。
* 输入信号：单因子滚动外推的**未截断标准化预测值** ŷ/sd(拟合值)。
  截断到 ±1 之后再做仓位映射会丢掉「信号有多强」的信息，
  单因子基准必须用未截断的版本才能与组合版（模型 z 未截断）可比。
* 尺度系数 k：只用训练段重新拟合。
* 两套仓位映射各出一行：
  - 主结果映射（`clip(z,-1,1) * min(1, 目标波动/σ̂)`），与主结果可比；
  - 改进版映射（EMA 平滑 + 训练段尺度系数），与改进版可比。

用法：
  python3 sgp_factors/infer.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from config import (CHOSEN_HALFLIFE, FEE, MIN_TRAIN, MIN_TRAIN_SCREEN,  # noqa: E402
                    RFE_KEEP, SCALE_K, SCREEN_PRESET, TARGET_VOL, TRAIN_END,
                    VALID_END, VALID_START, TEST_START)
from inference import (daily_sharpe, deflated_sr, expected_max_sr,  # noqa: E402
                       sharpe_diff_bootstrap)
from models import run_model, vol_target_60  # noqa: E402
from position import (build_positions, ema_recursive, fit_scale,  # noqa: E402
                      main_positions)
from screen import (ann_sharpe, net_returns, rfe_ols, screen_factors,  # noqa: E402
                    set_ann, single_factor_signals)
from trial_registry import (add_new_trials, build_registry,  # noqa: E402
                             trial_sharpes)

OUT = HERE / "out"
DATA = ROOT / "sgp_data" / "out"


def ann_factor(cal: pd.DataFrame) -> float:
    idx = pd.DatetimeIndex(cal["td"] if "td" in cal.columns else cal.index)
    span_years = (idx.max() - idx.min()).days / 365.25
    return len(idx) / span_years


def load():
    cal = pd.read_csv(DATA / "calendar.csv", parse_dates=["td"]).set_index("td")
    labels = pd.read_csv(DATA / "labels.csv", parse_dates=["td"]).set_index("td")
    full = pd.read_csv(DATA / "daily_full.csv", parse_dates=["td"]).set_index("td")
    X = pd.read_csv(HERE / "out" / "factors.csv", parse_dates=[0], index_col=0)
    X.index = pd.to_datetime(X.index)
    ann = ann_factor(cal)
    set_ann(ann)
    return cal, labels, full, X, ann


# ---------------------------------------------------------------------------
# 1. 最好单因子基准
# ---------------------------------------------------------------------------

def raw_single_factor_signals(X: pd.DataFrame, y: pd.Series, col: str,
                               min_train: int, window: str = "expanding"
                               ) -> pd.Series:
    """单因子滚动外推的**未截断**标准化预测值 ŷ/sd(拟合值)。

    `screen.single_factor_signals` 会把信号截断到 ±1；单因子基准要用
    未截断版本，否则与组合版（模型 z 未截断）不可比。

    这里复用 `screen._ols_stats` 的同一套滚动拟合口径，保证与筛选阶段
    的信号生成方式完全一致（同样的 as-of 填充、同样的训练窗）。
    """
    from screen import _ols_stats

    dates = X.index
    yv = y.reindex(dates)
    base_valid = yv.notna()
    f_raw = X[col].reindex(dates)
    f = f_raw.ffill()
    valid = base_valid & f_raw.notna()
    beta, alpha, n, sd_fit, _fbar = _ols_stats(f_raw, yv, valid, window, 250)
    yhat = alpha + beta * f
    z = (yhat / sd_fit.replace(0.0, np.nan)).where(n >= min_train)
    z.name = col
    return z


def best_single_factor_baseline(cal, labels, full, X, ann) -> tuple[pd.DataFrame, dict]:
    """产出「最好单因子」基准的两行（主结果映射 / 改进版映射）。"""
    y = labels["y_cc"]
    price = full["C"]
    vol = vol_target_60(full, ann)
    dates = X.index

    # 训练段外推 Sharpe -> 选最好的单因子
    sigs, sh = single_factor_signals(X, y, price, FEE, min_train=MIN_TRAIN_SCREEN,
                                      window="expanding", sharpe_end=TRAIN_END)
    best = sh.idxmax()
    train_sharpe = float(sh[best])

    z = raw_single_factor_signals(X, y, best, MIN_TRAIN_SCREEN)

    rows = []

    # (a) 主结果映射
    pos_main = main_positions(z, vol, TARGET_VOL)
    R_main = net_returns(pos_main, y.reindex(dates), price.reindex(dates), FEE)

    # (b) 改进版映射：EMA 平滑 + 训练段尺度系数
    s0 = ema_recursive(z, float(CHOSEN_HALFLIFE))
    k = fit_scale(z, s0, TRAIN_END)
    pos_ema, s_ema = build_positions(z, vol, float(CHOSEN_HALFLIFE), k, TARGET_VOL)
    R_ema = net_returns(pos_ema, y.reindex(dates), price.reindex(dates), FEE)

    for name, pos, R in [("best_single_main_map", pos_main, R_main),
                         ("best_single_ema_map", pos_ema, R_ema)]:
        tr = R.loc[:TRAIN_END]
        va = R.loc[VALID_START:VALID_END]
        te = R.loc[TEST_START:]
        rows.append(dict(
            name=name,
            factor=best,
            train_sharpe_train=train_sharpe,
            valid_sharpe=ann_sharpe(va, ann),
            test_sharpe=ann_sharpe(te, ann),
            test_sharpe_daily=daily_sharpe(te),
            n_test=int(te.notna().sum()),
            scale_k=float(k) if name.endswith("ema_map") else np.nan,
        ))

    info = dict(best_factor=best, train_sharpe=train_sharpe,
                R_main=R_main, R_ema=R_ema, pos_main=pos_main, pos_ema=pos_ema)
    return pd.DataFrame(rows), info


# ---------------------------------------------------------------------------
# 2. 风险情景表
# ---------------------------------------------------------------------------

def risk_scenarios(pos: pd.Series, y: pd.Series, price: pd.Series,
                   ann: float) -> pd.DataFrame:
    """目标波动与杠杆上限**同比例放大**的三档情景。

    (目标波动, 杠杆上限) = (10%, 1.0) / (15%, 1.5) / (20%, 2.0)。

    实现：把最终仓位整体乘以 k（k = 1.0 / 1.5 / 2.0）。收益与**交易成本**
    同时放大 k 倍，所以净收益也是 k 倍——**Sharpe 不变**，CAGR 与回撤同比放大。
    这正是「杠杆超过 1 倍偏离了原计划（研报系列 4 设定 1 倍上限），
    属于风险偏好设定，不是策略改进」这句话的数学含义。

    注意只提高目标波动、不提高上限的组合**不放进表里**：那等于把
    波动率缩放关掉了，不是风险情景，是另一个策略。
    """
    rows = []
    p = pos.astype(float)
    yv = y.reindex(p.index)
    pv = price.reindex(p.index)
    for tv, cap in [(0.10, 1.0), (0.15, 1.5), (0.20, 2.0)]:
        k = cap
        pk = p * k
        R = net_returns(pk, yv, pv, FEE)
        te = R.loc[TEST_START:]
        # 收益序列里可能夹着 NaN（标签缺失日）。cumprod 遇 NaN 会一路传播成 NaN，
        # 所以做累计净值之前必须先去掉 NaN，否则末值恒为 NaN、CAGR 全空。
        te = te.dropna()
        cum = (1.0 + te).cumprod()
        rows.append(dict(
            target_vol=tv,
            leverage_cap=cap,
            scale=k,
            test_sharpe=ann_sharpe(te, ann),
            test_cagr=float(cum.iloc[-1] ** (ann / max(len(te), 1)) - 1.0),
            test_max_dd=float((cum / cum.cummax() - 1.0).min()),
            n_test=int(len(te)),
        ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    cal, labels, full, X, ann = load()
    y = labels["y_cc"]
    price = full["C"]
    vol = vol_target_60(full, ann)
    print(f"样本 {X.index[0].date()} ~ {X.index[-1].date()}，{len(X)} 个交易日，ANN={ann:.2f}")

    # ---- 1. 最好单因子基准 -------------------------------------------
    print("\n[1] 最好单因子基准")
    base_df, base_info = best_single_factor_baseline(cal, labels, full, X, ann)
    best = base_info["best_factor"]
    print(f"    训练段最优单因子: {best}（训练段外推 Sharpe {base_info['train_sharpe']:.3f}）")
    for _, r in base_df.iterrows():
        print(f"    {r['name']:<24} 验证 {r['valid_sharpe']:+.3f}  测试 {r['test_sharpe']:+.3f}")
    base_df.to_csv(OUT / "baseline_single_factor.csv", index=False)

    # ---- 主结果与改进版的日度收益 ------------------------------------
    # 主结果：直接读回测产出的仓位（避免重跑模型引入差异）
    daily = pd.read_csv(OUT / "daily_backtest.csv", parse_dates=[0], index_col=0)
    R_main = daily["R_net"]
    pos_main = daily["pos"]
    improved = json.loads((OUT / "robustness.json").read_text())["improved"]
    # 改进版的仓位：主结果仓位 -> EMA 平滑 -> 波动率目标（与 backtest 同一口径）
    res_z = daily["z"]
    s_ema = ema_recursive(res_z, float(CHOSEN_HALFLIFE))
    k_imp = float(SCALE_K) if SCALE_K is not None else fit_scale(res_z, s_ema, TRAIN_END)
    pos_ema, s_ema = build_positions(res_z, vol, float(CHOSEN_HALFLIFE), k_imp,
                                     TARGET_VOL)
    R_ema = net_returns(pos_ema, y.reindex(daily.index),
                         price.reindex(daily.index), FEE)
    # 与回测产出的改进版逐位核对
    if "R_net_ema" in daily.columns:
        d = (R_ema - daily["R_net_ema"]).abs().max()
        print(f"    改进版重算与回测最大偏差: {d:.2e}")
        assert d < 1e-9, "改进版重算与回测不一致"

    # ---- 2. 试验登记册 ------------------------------------------------
    print("\n[2] 试验登记册")
    reg = build_registry(OUT, ann)
    new_rows = [
        dict(name="best_single_main_map", caliber="strategy",
             category="新增基准", test_sharpe_ann=float(base_df.loc[0, "test_sharpe"]),
             test_sharpe_daily=float(base_df.loc[0, "test_sharpe_daily"]),
             n_test=int(base_df.loc[0, "n_test"]),
             note="训练段最优单因子 B4_RVDN + 主结果仓位映射"),
        dict(name="best_single_ema_map", caliber="strategy",
             category="新增基准", test_sharpe_ann=float(base_df.loc[1, "test_sharpe"]),
             test_sharpe_daily=float(base_df.loc[1, "test_sharpe_daily"]),
             n_test=int(base_df.loc[1, "n_test"]),
             note="训练段最优单因子 + 改进版仓位映射"),
    ]
    reg = add_new_trials(reg, new_rows)
    reg.to_csv(OUT / "trials.csv", index=False)
    for cal_name in ("full", "strategy"):
        s = trial_sharpes(reg, cal_name)
        print(f"    {cal_name:<9} 登记 {len(s):>4} 个方案，"
              f"日度 Sharpe 均值 {s.mean():+.4f} 标准差 {s.std(ddof=1):.4f}")

    # ---- 3. Deflated Sharpe ------------------------------------------
    print("\n[3] Deflated Sharpe Ratio")
    dsr_rows = []
    targets = [("主结果(ols_expanding)", R_main),
               ("EMA改进版", R_ema),
               ("最好单因子(主结果映射)", base_info["R_main"]),
               ("最好单因子(改进版映射)", base_info["R_ema"])]
    for cal_name in ("full", "strategy"):
        s_trials = trial_sharpes(reg, cal_name)
        for nm, R in targets:
            d = deflated_sr(R.loc[TEST_START:], s_trials, ann=ann)
            d["target"] = nm
            d["caliber"] = cal_name
            dsr_rows.append(d)
    dsr = pd.DataFrame(dsr_rows)
    cols = ["target", "caliber", "n", "sr_daily", "sr_annualized", "n_trials",
            "sr_star", "psr", "psr0", "dsr", "p_value", "skew", "kurt"]
    dsr = dsr[cols]
    dsr.to_csv(OUT / "dsr.csv", index=False)
    print(dsr.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    # ---- 4. Sharpe 差异检验 ------------------------------------------
    print("\n[4] Sharpe 差异检验（配对平稳块自助法，块长 10 天，5000 次）")
    pairs = [
        ("EMA改进版 - 主结果", R_ema, R_main),
        ("组合版 - 最好单因子(主结果映射)", R_main, base_info["R_main"]),
        ("组合版(改进版映射) - 最好单因子(改进版映射)", R_ema, base_info["R_ema"]),
    ]
    sd_rows = []
    for nm, a, b in pairs:
        d = sharpe_diff_bootstrap(a.loc[TEST_START:], b.loc[TEST_START:],
                                  n=5000, mean_block=10, seed=20261030,
                                  ann=ann)
        d["comparison"] = nm
        sd_rows.append(d)
    sd = pd.DataFrame(sd_rows)
    sd = sd[["comparison", "n_obs", "obs_diff", "p_value", "boot_diff_mean",
             "boot_diff_ci", "n_boot", "mean_block"]]
    sd.to_csv(OUT / "sharpe_diff.csv", index=False)
    for _, r in sd.iterrows():
        ci = r["boot_diff_ci"]
        print(f"    {r['comparison']:<44} ΔSharpe {r['obs_diff']:+.3f}  "
              f"p={r['p_value']:.4f}  95%CI [{ci[0]:+.3f}, {ci[1]:+.3f}]")

    # ---- 5. 风险情景表 ------------------------------------------------
    print("\n[5] 风险情景表（目标波动与杠杆上限同比例放大）")
    scen = risk_scenarios(pos_ema, y.reindex(daily.index),
                          price.reindex(daily.index), ann)
    scen_main = risk_scenarios(pos_main, y.reindex(daily.index),
                               price.reindex(daily.index), ann)
    scen["strategy"] = "EMA改进版"
    scen_main["strategy"] = "主结果"
    scen_all = pd.concat([scen, scen_main], ignore_index=True)
    scen_all.to_csv(OUT / "risk_scenarios.csv", index=False)
    print(scen_all.to_string(index=False,
                              float_format=lambda v: f"{v:+.4f}"))

    print("\n完成。产物：")
    for f in ("baseline_single_factor.csv", "trials.csv", "dsr.csv",
              "sharpe_diff.csv", "risk_scenarios.csv"):
        print(f"  out/{f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
