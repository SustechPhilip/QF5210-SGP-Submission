"""回测主脚本：筛选 -> RFE -> 模型 -> 指标 -> 稳健性。

对应 v2 §4 / §5 / §6 / §7。主结果预先登记为
OLS + RFE 10 + loose + expanding。参数一律从 config.py 取。

python backtest.py [--quick]
"""
from __future__ import annotations

import argparse
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
from config import (CHOSEN_HALFLIFE, EXPIRY_WINDOW_DAYS, FEE, FEE_GRID,  # noqa: E402
                    HALFLIFE_GRID, N_SHIFT, RIDGE_ALPHAS, RFE_KEEP, SCALE_K,
                    SCREEN_PRESET, SEED, SMOOTHING_ENABLED, TARGET_VOL,
                    TEST_START, TRAIN_END, VALID_END, VALID_START, WINDOW,
                    rel)
from features import factor_class  # noqa: E402
from expiry import expiry_positions, expiry_window_positions  # noqa: E402
from models import (MIN_TRAIN, add_turnover, circular_shift_null, metrics,  # noqa: E402
                    position_stats, run_buyhold, run_long_vt, run_model,
                    run_mom20, smoothed_positions, turnover, turnover_daily,
                    vol_target_60)
from position import (build_positions, build_positions_long_only,  # noqa: E402
                      deadband, ema_recursive, fit_scale, hold_on,
                      long_only, main_positions, week_start_mask)
from screen import (ann_sharpe, net_returns, rfe_ols, screen_factors,  # noqa: E402
                    set_ann, single_factor_signals)


def ann_factor(cal: pd.DataFrame) -> float:
    """ANN = 交易日数 / 样本年数。"""
    span_years = (cal.index.max() - cal.index.min()).days / 365.25
    return len(cal) / span_years


def load():
    X = pd.read_csv(OUT / "factors.csv", parse_dates=[0], index_col=0)
    X.index = pd.to_datetime(X.index)
    full = pd.read_csv(DATA / "daily_full.csv", parse_dates=["td"]).set_index("td")
    labels = pd.read_csv(DATA / "labels.csv", parse_dates=["td"]).set_index("td")
    cal = pd.read_csv(DATA / "calendar.csv", parse_dates=["td"]).set_index("td")
    return X, full, labels, cal


def seg_metrics(R: pd.Series, ann: float) -> dict:
    """按训练/验证/测试三段算指标。"""
    out = {}
    for name, sl in [("train", R.loc[:TRAIN_END]),
                     ("valid", R.loc[VALID_START:TRAIN_END + pd.Timedelta(days=365)]),
                     ("test", R.loc[TEST_START:])]:
        m = metrics(sl, ann)
        out[name] = m
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="跳过稳健性检验")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    X, full, labels, cal = load()
    ann = ann_factor(cal)
    set_ann(ann)
    print(f"ANN = {ann:.1f}（交易日 {len(cal)}，"
          f"{cal.index.min().date()} ~ {cal.index.max().date()}）", flush=True)

    y = labels["y_cc"]
    price = full["C"]
    vol = vol_target_60(full, ann)
    dates = X.index

    # ---------------- 单因子外推 Sharpe ----------------
    t0 = time.time()
    # 筛选只用训练段（≤TRAIN_END）的外推 Sharpe：用全样本 Sharpe 挑因子等于把
    # 测试期信息泄漏进特征选择（v2 §4.2/§4.3）。全样本 Sharpe 只作描述性披露。
    sigs, sharpes_train = single_factor_signals(X, y, price, FEE, window="expanding",
                                                 sharpe_end=TRAIN_END)
    sigs_full, sharpes_full = single_factor_signals(X, y, price, FEE, window="expanding")
    print(f"\n单因子外推完成 {time.time()-t0:.1f}s", flush=True)
    sigs.to_csv(OUT / "single_factor_signals.csv")
    sharpes_full.sort_values(ascending=False).to_csv(OUT / "factor_sharpe_extrap.csv",
                                                      header=["sharpe_full_sample"])
    sharpes_train.reindex(sharpes_full.sort_values(ascending=False).index).to_csv(
        OUT / "factor_sharpe_extrap_train.csv", header=["sharpe_train_period"])
    print("训练段外推 Sharpe 前 12:")
    print(sharpes_train.sort_values(ascending=False).head(12).round(3).to_string())
    print("训练段外推 Sharpe > 0 的因子数:", int((sharpes_train > 0).sum()),
          "/", sharpes_train.notna().sum())
    print("（披露用）全样本外推 Sharpe > 0 的因子数:", int((sharpes_full > 0).sum()),
          "/", sharpes_full.notna().sum())

    # ---------------- 筛选 + RFE ----------------
    selected = screen_factors(X, sharpes_train, SCREEN_PRESET, TRAIN_END)
    print(f"\n{SCREEN_PRESET} 筛选入选 {len(selected)} 个因子", flush=True)
    final_cols = rfe_ols(X, y, selected, TRAIN_END, n_keep=RFE_KEEP)
    print("RFE 保留:", final_cols, flush=True)
    pd.Series(final_cols).to_csv(OUT / "selected_features.csv", index=False,
                                 header=["factor"])
    pd.Series(selected).to_csv(OUT / "selected_features_pre_rfe.csv", index=False,
                               header=["factor"])

    # 因子登记表回填：哪些进了筛选、哪些过了 RFE、哪些真正进入主模型
    reg_path = OUT / "factor_registry.csv"
    if reg_path.exists():
        reg = pd.read_csv(reg_path)
        reg["screened_in"] = reg["id"].isin(selected).map({True: "Y", False: "N"})
        reg["rfe_kept"] = reg["id"].isin(final_cols).map({True: "Y", False: "N"})
        reg["in_main_model"] = reg["id"].isin(final_cols).map({True: "Y", False: "N"})
        reg["sharpe_train"] = reg["id"].map(sharpes_train).round(4)
        reg.to_csv(reg_path, index=False)
        print(f"因子登记表已回填：进入主模型 {int((reg['in_main_model']=='Y').sum())} 个 "
              f"/ {len(reg)} 个", flush=True)

    # ---------------- 模型 ----------------
    def run(name, model="ols", window=WINDOW, alpha=1.0, fee=FEE,
            cols=None, start=None, y_series=None, price_series=None):
        cols = final_cols if cols is None else cols
        ys = y if y_series is None else y_series
        ps = price if price_series is None else price_series
        res = run_model(X, ys, ps, cols, model=model, alpha=alpha,
                        window=window, fee=fee, vol_series=vol, ann=ann,
                        min_train=MIN_TRAIN, start=start)
        R = net_returns(res["pos"], ys.reindex(dates), ps.reindex(dates), fee)
        return res, R

    res, R = run("ols_exp")
    print(f"\n首个可预测日: {res['pos'].first_valid_index().date()}", flush=True)
    print(f"训练段内预测天数: {res['pos'].loc[:TRAIN_END].notna().sum()}", flush=True)

    # 基准
    bench = {}
    bench["buyhold"] = net_returns(run_buyhold(dates, price, y, FEE), y, price, FEE)
    bench["long_vt"] = net_returns(run_long_vt(dates, vol, y, price, FEE),
                                   y, price, FEE)
    bench["mom20"] = net_returns(run_mom20(X, dates, vol), y, price, FEE)
    # buyhold 在测试期重新建仓
    bh_test = run_buyhold(dates, price, y, FEE, start=TEST_START)
    bench["buyhold_test_rebased"] = net_returns(bh_test, y, price, FEE)

    # ---------------- 指标 ----------------
    rows = []
    def report(name, R, pos=None):
        # 分段换手用**全样本**的逐日换手序列切片，避免每段首日被当成从 0 建仓，
        # 与 R_net 实际扣的成本一致。
        # `pos` 传仓位序列（策略或基准都传），基准因此也有年换手：
        # 买入持有 -> 0.0（全程 pos≡1，无换手）；常多+波动率目标 -> 约 2.5 倍/年。
        dp_full = turnover_daily(pos) if pos is not None else None
        for seg, sl in [("train", R.loc[:TRAIN_END]),
                        ("valid", R.loc[VALID_START:VALID_END]),
                        ("test", R.loc[TEST_START:])]:
            m = metrics(sl, ann)
            if pos is not None:
                p = pos.reindex(sl.index)
                m = add_turnover(m, p, ann, dp=dp_full)
            rows.append(dict(strategy=name, segment=seg, **m))
        print(f"\n--- {name} ---")
        print(pd.DataFrame(rows).query("strategy == @name")[
            ["segment", "n", "cagr", "ann_vol", "sharpe", "max_dd", "calmar",
             "win_rate", "ann_turnover"]].round(4).to_string(index=False))

    report("ols_exp", R, res["pos"])
    bench_pos = {
        "buyhold": run_buyhold(dates, price, y, FEE),
        "long_vt": run_long_vt(dates, vol, y, price, FEE),
        "mom20": run_mom20(X, dates, vol),
    }
    bench = {k: net_returns(v, y, price, FEE) for k, v in bench_pos.items()}
    # buyhold 在测试期重新建仓
    bh_test = run_buyhold(dates, price, y, FEE, start=TEST_START)
    bench["buyhold_test_rebased"] = net_returns(bh_test, y, price, FEE)
    bench_pos["buyhold_test_rebased"] = bh_test
    for k, v in bench.items():
        report(k, v, bench_pos.get(k))

    # ---------------- 事后改进版（不替换主结果）----------------
    # 以下全部是**看过测试集结果之后**才做的调整，单列报告。
    # 主结果仍是上面的 `ols_exp`（不平滑），改进版只作披露。
    print("\n=== 事后改进版：降换手（不替换主结果）===", flush=True)
    improved = {}
    pos_main = res["pos"]
    dp_main = turnover_daily(pos_main)

    def _variant_row(name, pos):
        Rv = net_returns(pos, y.reindex(dates), price.reindex(dates), FEE)
        dpv = turnover_daily(pos)
        row = dict(
            name=name,
            valid_sharpe=ann_sharpe(Rv.loc[VALID_START:VALID_END], ann),
            test_sharpe=ann_sharpe(Rv.loc[TEST_START:], ann),
            valid_turnover=turnover(pos.loc[VALID_START:VALID_END], ann, dp=dpv),
            test_turnover=turnover(pos.loc[TEST_START:], ann, dp=dpv),
        )
        return row, Rv

    # (1) EMA 平滑改进版：半衰期与尺度系数已由 select_halflife.py 按验证期规则冻结
    pos_ema = s_ema = None
    R_ema = None
    if SMOOTHING_ENABLED:
        if CHOSEN_HALFLIFE is None or SCALE_K is None:
            print("config.smoothing.chosen_halflife / scale_k 尚未冻结 —— "
                  "请先运行 `python3 sgp_factors/select_halflife.py`", flush=True)
        else:
            hl_f, k_f = float(CHOSEN_HALFLIFE), float(SCALE_K)
            pos_ema, s_ema = smoothed_positions(res, vol, hl_f, k_f, TARGET_VOL)
            R_ema = net_returns(pos_ema, y.reindex(dates), price.reindex(dates), FEE)
            report("ema_smoothed", R_ema, pos_ema)
            improved["ema"] = dict(halflife=hl_f, scale_k=k_f,
                                   select_on="2023 验证期净 Sharpe（测试集只跑一次）")
            # 网格（披露）：整个网格都在 out/ema_grid_valid.csv 里
            grid_path = OUT / "ema_grid_valid.csv"
            if grid_path.exists():
                grid = pd.read_csv(grid_path)
                improved["ema"]["grid"] = grid.to_dict("records")
                print("\n半衰期网格（披露；选择只用验证期）：")
                print(grid.round(4).to_string(index=False))

    # (2) 三个仓位规则变体：都作用在**同一个主模型信号**上，只改仓位规则
    variant_rows = []
    r_main, _ = _variant_row("main(不平滑)", pos_main)
    variant_rows.append(r_main)
    if pos_ema is not None:
        r_ema, _ = _variant_row(f"ema(半衰期{improved['ema']['halflife']:.0f}日)", pos_ema)
        variant_rows.append(r_ema)
    # 每周调仓：取**每周第一个交易日**调仓，而不是「星期一」。
    # 若周一是法定假日，「星期一调仓」会导致那一周完全不调仓（持仓沿用一周）；
    # 正确口径是假日落在周一时由周二调仓。见 position.week_start_mask。
    is_week_start = week_start_mask(pos_main.index)
    p_week, _ = _variant_row("每周调仓（周内首个交易日）", hold_on(is_week_start, pos_main))
    variant_rows.append(p_week)
    # 不交易带 0.4
    p_band, _ = _variant_row("不交易带0.4", deadband(pos_main, 0.4))
    variant_rows.append(p_band)
    # 只做多 + 半衰期 3 日
    # 注意：这里的 3 日是**固定值**，不是按验证期选出来的（验证期选出的是 2 日）。
    # 保留这一行是因为它是「只做多」这条规则的对照：同一个 3 日半衰期下，
    # 平滑版的仓位可以是负，只做多版被截到 [0,1]。两行放在一起才好读。
    if pos_ema is not None:
        hl3 = 3.0
        s3 = ema_recursive(res["z"], hl3)
        k3 = fit_scale(res["z"], s3, TRAIN_END)
        p_lo, _ = build_positions_long_only(res["z"], vol, hl3, k3, TARGET_VOL)
        r_lo, _ = _variant_row("只做多+半衰期3日(固定3日,非验证期选出)", p_lo)
        variant_rows.append(r_lo)
    var_df = pd.DataFrame(variant_rows)
    improved["variants"] = var_df.to_dict("records")
    print("\n仓位规则变体（同一信号，只改仓位规则）：")
    print(var_df.round(4).to_string(index=False))

    # 仓位行为统计（**测试期**口径；主结果 + 平滑版）
    improved["position_stats_main"] = position_stats(
        pos_main.loc[TEST_START:], res["z"].loc[TEST_START:], ann)
    if pos_ema is not None:
        improved["position_stats_ema"] = position_stats(
            pos_ema.loc[TEST_START:], res["z"].loc[TEST_START:], ann)
    print("\n仓位统计（主结果，测试期）:")
    print(json.dumps(improved["position_stats_main"], indent=2, default=str))

    # 稳健性 ----------------
    robust = {}
    if not args.quick:
        print("\n=== 稳健性 ===", flush=True)
        # R1 成本三档
        for f in FEE_GRID:
            _, Rf = run("ols", fee=f)
            m = metrics(Rf.loc[TEST_START:], ann)
            robust[f"cost_{f:g}"] = m
            print(f"cost fee={f:g}: test Sharpe={m['sharpe']:.3f} CAGR={m['cagr']:.3f}")

        # R2 分年
        by_year = {}
        for yr in range(2020, 2027):
            sl = R.loc[f"{yr}-01-01":f"{yr}-12-31"]
            by_year[yr] = metrics(sl, ann)
        robust["by_year"] = by_year

        # R3 类别消融（连同筛选与 RFE 一起重跑）
        ablation = {}
        for cls in "ABCDE":
            Xc = X[[c for c in X.columns if factor_class(c) != cls]]
            sigs_c, sh_c = single_factor_signals(Xc, y, price, FEE,
                                                 sharpe_end=TRAIN_END)
            sel_c = screen_factors(Xc, sh_c, SCREEN_PRESET, TRAIN_END)
            cols_c = rfe_ols(Xc, y, sel_c, TRAIN_END, n_keep=RFE_KEEP)
            res_c = run_model(Xc, y, price, cols_c, model="ols", window=WINDOW,
                              fee=FEE, vol_series=vol, ann=ann, min_train=MIN_TRAIN)
            Rc = net_returns(res_c["pos"], y.reindex(dates), price.reindex(dates), FEE)
            ablation[f"drop_{cls}"] = dict(
                n_features=len(cols_c),
                features=cols_c,
                test=metrics(Rc.loc[TEST_START:], ann))
            print(f"ablation drop {cls}: features={cols_c} "
                  f"test Sharpe={ablation[f'drop_{cls}']['test']['sharpe']:.3f}")
        robust["ablation"] = ablation

        # R4 随机信号：循环平移
        null = circular_shift_null(res["pos"], y, price, FEE, n_boot=N_SHIFT, ann=ann,
                                    seed=SEED)
        test_null = circular_shift_null(res["pos"].loc[TEST_START:], y, price,
                                         FEE, n_boot=N_SHIFT, ann=ann, seed=SEED)
        real_test = metrics(R.loc[TEST_START:], ann)["sharpe"]
        pct = float((test_null < real_test).mean())
        robust["circular_shift"] = dict(
            full_null_mean=float(null.mean()), full_null_p95=float(null.quantile(0.95)),
            test_null_mean=float(test_null.mean()), test_null_p95=float(test_null.quantile(0.95)),
            real_test_sharpe=real_test,
            test_percentile=pct,
            # 分位 p 的含义：策略 Sharpe 高于 p 比例的平移序列 -> 单侧 p = 1 - p
            test_p_one_sided=float(1.0 - pct),
            # 文字由实际数值生成，不写死（历史 bug：写死成旧值 0.723 / p≈0.28）
            percentile_meaning=(f"分位 {pct:.3f} 表示策略 Sharpe 高于 {pct*100:.1f}% 的平移序列，"
                                f"单侧 p≈{1.0-pct:.3f}"))
        print(f"循环平移零假设: test null mean={test_null.mean():.3f} "
              f"p95={test_null.quantile(0.95):.3f}, 策略={real_test:.3f}, "
              f"分位={pct:.3f} (单侧 p={1-pct:.3f})")

        # R5 替代收盘口径
        alt = {}
        for k in ("y_C_1714", "y_C_vwap_1705_1714"):
            ys = labels[k]
            _, Ra = run("ols", y_series=ys)
            alt[k] = metrics(Ra.loc[TEST_START:], ann)
            print(f"alt_close {k}: test Sharpe={alt[k]['sharpe']:.3f}")
        robust["alt_close"] = alt

        # R6 训练窗对比
        win = {}
        for wname in ("expanding", "250"):
            _, Rw = run("ols", window=wname)
            win[wname] = metrics(Rw.loc[TEST_START:], ann)
            print(f"window {wname}: test Sharpe={win[wname]['sharpe']:.3f}")
        robust["window"] = win

        # R6b Ridge
        ridge = {}
        for a in RIDGE_ALPHAS:
            _, Rr = run("ols", model="ridge", alpha=a)
            v = metrics(Rr.loc[VALID_START:VALID_END], ann)["sharpe"]
            t = metrics(Rr.loc[TEST_START:], ann)
            ridge[a] = dict(valid_sharpe=v, test=t)
            print(f"ridge alpha={a}: valid Sharpe={v:.3f} test Sharpe={t['sharpe']:.3f}")
        robust["ridge"] = ridge

        # R13 开盘锚点口径（稳健性，不进主结果）
        # 主口径只在 P_0830 缺失时回退到 O_t；这里改成所有交易日都用 O_t 当锚点，
        # 看主结果对这一口径有多敏感。
        try:
            from features import build_all
            dec_df = pd.read_csv(DATA / "daily_dec.csv", parse_dates=["td"]).set_index("td")
            minutes_pkl = pd.read_pickle(DATA / "minutes_tagged.pkl")
            minutes_pkl["td"] = pd.to_datetime(minutes_pkl["td"])
            X_o = build_all(dec_df, full, minutes_pkl, anchor_open=True)
            sig_o, sh_o = single_factor_signals(X_o, y, price, FEE, sharpe_end=TRAIN_END)
            sel_o = screen_factors(X_o, sh_o, SCREEN_PRESET, TRAIN_END)
            cols_o = rfe_ols(X_o, y, sel_o, TRAIN_END, n_keep=RFE_KEEP)
            res_o = run_model(X_o, y, price, cols_o, model="ols", window=WINDOW,
                              fee=FEE, vol_series=vol, ann=ann, min_train=MIN_TRAIN)
            R_o = net_returns(res_o["pos"], y.reindex(dates), price.reindex(dates), FEE)
            robust["anchor_open"] = dict(
                features=cols_o,
                test=metrics(R_o.loc[TEST_START:], ann))
            print(f"anchor O_t 全量口径: features={cols_o} "
                  f"test Sharpe={robust['anchor_open']['test']['sharpe']:.3f}")
        except Exception as e:            # pragma: no cover
            robust["anchor_open"] = dict(error=str(e))
            print("anchor O_t 口径跳过:", e)

        # R11 换月检验
        # (a) 到期日虚拟变量回归：对日收益回归，虚拟变量 = 每月倒数第二个营业日
        #     **及其之前 3 个交易日**（共 4 个交易日）。
        #     历史 bug：窗口用自然日（pd.Timedelta(days=k)）而不是交易日；
        #     且 t 值用残差标准差当分母（那是 sigma，不是系数的标准误）。
        # (b) 标签置 NaN 对照：把到期日窗口内的标签置 NaN，重跑整条管线（筛选+RFE+模型）
        # 到期日 -> 交易日历位置。
        # 实现放在 `expiry.py`（唯一实现），扩展因子的 K1_EXPIRY_WK 用同一套定义，
        # 否则两处说的「到期日」可能不是同一天。
        expiry_pos, n_months, n_rolled, rolled_months = expiry_positions(cal.index)

        # 每个到期日往前取 3 个交易日（含到期日共 4 个交易日）
        excl_idx = set(expiry_window_positions(expiry_pos, EXPIRY_WINDOW_DAYS))
        excl_days = [cal.index[i] for i in sorted(excl_idx)]

        dummy = pd.Series(0.0, index=dates)
        dummy.loc[dummy.index.isin(excl_days)] = 1.0
        r_full = y.reindex(dates)
        m_ = dummy.notna() & r_full.notna()
        Xd = np.column_stack([np.ones(int(m_.sum())), dummy[m_].to_numpy()])
        yd = r_full[m_].to_numpy()
        beta, *_ = np.linalg.lstsq(Xd, yd, rcond=None)
        resid = yd - Xd @ beta
        dof = max(len(yd) - 2, 1)
        sigma2 = float(resid @ resid) / dof
        # 系数的标准误：se(b) = sigma * sqrt(diag((X'X)^-1))，不是 sigma 本身
        XtX_inv = np.linalg.pinv(Xd.T @ Xd)
        se_beta = np.sqrt(np.maximum(sigma2 * np.diag(XtX_inv), 1e-300))
        tstat = float(beta[1] / se_beta[1]) if se_beta[1] > 0 else np.nan

        # (b) 标签置 NaN 重跑：先在训练段做筛选与 RFE（标签变了，筛选也变），再跑模型
        y_ex = y.copy()
        excl_pos = [i for i in sorted(excl_idx) if i < len(y_ex)]
        y_ex.iloc[excl_pos] = np.nan
        sigs_ex, sh_ex = single_factor_signals(X, y_ex, price, FEE, sharpe_end=TRAIN_END)
        sel_ex = screen_factors(X, sh_ex, SCREEN_PRESET, TRAIN_END)
        cols_ex = rfe_ols(X, y_ex, sel_ex, TRAIN_END, n_keep=RFE_KEEP)
        res_ex = run_model(X, y_ex, price, cols_ex, model="ols", window=WINDOW,
                           fee=FEE, vol_series=vol, ann=ann, min_train=MIN_TRAIN)
        R_ex = net_returns(res_ex["pos"], y_ex.reindex(dates), price.reindex(dates), FEE)
        robust["expiry"] = dict(
            n_expiry_days=len(expiry_pos),
            n_months_in_sample=n_months,
            n_months_rolled_back=n_rolled,
            rolled_back_months=rolled_months,
            n_excluded_labels=int(len(excl_pos)),
            expiry_window_trading_days=EXPIRY_WINDOW_DAYS,
            dummy_coef_bp=float(beta[1] * 1e4),
            dummy_se_bp=float(se_beta[1] * 1e4),
            dummy_tstat=float(tstat),
            test_sharpe_original=metrics(R.loc[TEST_START:], ann)["sharpe"],
            test_sharpe_with_expiry_labels=metrics(R_ex.loc[TEST_START:], ann)["sharpe"],
            features_original=final_cols,
            features_expiry=cols_ex,
        )
        print(f"换月检验: 到期日 {len(expiry_pos)} 天, 窗口内标签 {len(excl_pos)} 条, "
              f"虚拟变量系数 {beta[1]*1e4:.3f} bp (se={se_beta[1]*1e4:.3f} bp, "
              f"t={tstat:.2f}); 标签置 NaN 后 test Sharpe="
              f"{robust['expiry']['test_sharpe_with_expiry_labels']:.3f}")

        # R12 按规则时期分段：**只统计测试期**（历史 bug：R1 时期大部分在训练段，
        # 混入会把样本内表现当成样本外证据）
        by_reg = {}
        for rg in ("R0", "R1", "R2", "R3"):
            days = [d for d in full.index[full["regime"] == rg] if d >= TEST_START]
            sel = R.loc[days] if len(days) else R.iloc[0:0]
            by_reg[rg] = metrics(sel, ann)
            # 该时期**不在测试期**的天数（= 训练段 + 验证段）。
            # 历史 bug：键名写作 n_in_sample_days，读者会误以为都是训练段样本内，
            # 实际里面还混着 259 天验证期（样本外）。键名改为 n_not_in_test_days。
            by_reg[rg]["n_not_in_test_days"] = int(
                (full["regime"] == rg).sum() - len(days))
        robust["by_regime"] = by_reg

        # ---------------- 改进版（EMA 平滑）的稳健性 ----------------
        if pos_ema is not None:
            print("\n=== 改进版（EMA 平滑）稳健性 ===", flush=True)
            ema_rob = dict(halflife=improved["ema"]["halflife"],
                           scale_k=improved["ema"]["scale_k"])
            # 成本三档
            ema_cost = {}
            for f in FEE_GRID:
                Re = net_returns(pos_ema, y.reindex(dates), price.reindex(dates), f)
                me = metrics(Re.loc[TEST_START:], ann)
                ema_cost[f"{f:g}"] = me
                print(f"ema cost fee={f:g}: test Sharpe={me['sharpe']:.3f} "
                      f"CAGR={me['cagr']:.3f}")
            ema_rob["cost"] = ema_cost
            # 分年
            ema_year = {}
            for yr in range(2020, 2027):
                sl = R_ema.loc[f"{yr}-01-01":f"{yr}-12-31"]
                ema_year[yr] = metrics(sl, ann)
            ema_rob["by_year"] = ema_year
            print("ema 分年 Sharpe:", {y: round(v["sharpe"], 3)
                                       for y, v in ema_year.items() if np.isfinite(v["sharpe"])})
            # 循环平移
            enull = circular_shift_null(pos_ema.loc[TEST_START:], y, price, FEE,
                                         n_boot=N_SHIFT, ann=ann, seed=SEED)
            real_e = metrics(R_ema.loc[TEST_START:], ann)["sharpe"]
            epct = float((enull < real_e).mean())
            ema_rob["circular_shift"] = dict(
                test_null_mean=float(enull.mean()),
                test_null_p95=float(enull.quantile(0.95)),
                real_test_sharpe=real_e,
                test_percentile=epct,
                test_p_one_sided=float(1.0 - epct),
                percentile_meaning=(f"分位 {epct:.3f} 表示策略 Sharpe 高于 "
                                    f"{epct*100:.1f}% 的平移序列，单侧 p≈{1.0-epct:.3f}"))
            print(f"ema 循环平移: null mean={enull.mean():.3f} "
                  f"p95={enull.quantile(0.95):.3f}, 策略={real_e:.3f}, "
                  f"分位={epct:.3f} (单侧 p={1-epct:.3f})")
            improved["ema"]["robustness"] = ema_rob

    # ---------------- 输出 ----------------
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(OUT / "metrics.csv", index=False)
    daily = pd.DataFrame({"pos": res["pos"], "yhat": res["yhat"], "z": res["z"],
                          "n_train": res["n_train"], "y": y.reindex(dates),
                          "R_net": R, "vol60": vol.reindex(dates),
                          "C": price.reindex(dates)})
    daily.to_csv(OUT / "daily_backtest.csv")
    # 改进版单独存一份，主结果 daily_backtest.csv 不含它
    if pos_ema is not None:
        pd.DataFrame({"pos_ema": pos_ema, "s_ema": s_ema,
                      "z": res["z"], "y": y.reindex(dates), "R_net_ema": R_ema,
                      "vol60": vol.reindex(dates),
                      "C": price.reindex(dates)}).to_csv(OUT / "daily_backtest_ema.csv")
    robust["improved"] = improved
    with open(OUT / "robustness.json", "w") as f:
        json.dump(robust, f, indent=2, default=str)
    print(f"\nwrote metrics.csv / daily_backtest.csv / robustness.json to {rel(OUT)}")


if __name__ == "__main__":
    main()
