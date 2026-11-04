"""每日更新：把策略推进到最新数据日。

回测与每日更新用**同一份代码**，只换数据截止日（v2 §2.4）。

用法：
  python3 update.py --asof 2026-10-16          # 用 sgp_data/out 的现有产物重跑到 asof
  python3 update.py --asof 2026-10-16 --csv "..." # 指定新的分钟 CSV，先重建数据层

输出：
  out/positions_asof_<date>.csv   截止 asof 的每日仓位与预测
  out/perf_asof_<date>.csv        截止 asof 的绩效表
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "sgp_data"
OUT = HERE / "out"
sys.path.insert(0, str(HERE.parent))

from config import (CHOSEN_HALFLIFE, FEE, RFE_KEEP, SCALE_K, SCREEN_PRESET,  # noqa: E402
                    SMOOTHING_ENABLED, STATE_HISTORY_DAYS, TARGET_VOL,
                    TEST_START, TRAIN_END, WINDOW, rel)
from features import build_all  # noqa: E402
from models import (MIN_TRAIN, add_turnover, metrics, run_model,  # noqa: E402
                    smoothed_positions, turnover_daily, vol_target_60)
from screen import (ann_sharpe, net_returns, rfe_ols, screen_factors,  # noqa: E402
                    set_ann, single_factor_signals)


def ann_factor(cal: pd.DataFrame) -> float:
    idx = pd.DatetimeIndex(cal["td"] if "td" in cal.columns else cal.index)
    return len(idx) / ((idx.max() - idx.min()).days / 365.25)


def load_data(asof: pd.Timestamp):
    dec = pd.read_csv(DATA / "out" / "daily_dec.csv", parse_dates=["td"]).set_index("td")
    full = pd.read_csv(DATA / "out" / "daily_full.csv", parse_dates=["td"]).set_index("td")
    labels = pd.read_csv(DATA / "out" / "labels.csv", parse_dates=["td"]).set_index("td")
    cal = pd.read_csv(DATA / "out" / "calendar.csv", parse_dates=["td"]).set_index("td")
    minutes = pd.read_pickle(DATA / "out" / "minutes_tagged.pkl")
    minutes["td"] = pd.to_datetime(minutes["td"])
    dec, full, labels, cal = dec.loc[:asof], full.loc[:asof], labels.loc[:asof], cal.loc[:asof]
    minutes = minutes[minutes["td"] <= asof]
    return dec, full, labels, cal, minutes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", required=True, help="数据截止日 YYYY-MM-DD")
    ap.add_argument("--csv", default=None, help="新的分钟 CSV（可选，先重建数据层）")
    args = ap.parse_args()

    asof = pd.Timestamp(args.asof)
    OUT.mkdir(parents=True, exist_ok=True)

    if args.csv:
        sys.path.insert(0, str(DATA))
        from run_data import main as run_data_main  # noqa: E402
        print(f"重建数据层 <- {args.csv}", flush=True)
        run_data_main(args.csv, str(DATA / "out"))

    dec, full, labels, cal, minutes = load_data(asof)
    ann = ann_factor(cal)
    set_ann(ann)
    print(f"asof={asof.date()}，交易日 {len(cal)}，ANN={ann:.1f}", flush=True)

    X = build_all(dec, full, minutes, robust=False)
    y = labels["y_cc"]
    price = full["C"]
    vol = vol_target_60(full, ann)

    # 筛选 + RFE 只在训练段做，特征集合固定
    sigs, sh = single_factor_signals(X, y, price, FEE, sharpe_end=TRAIN_END)
    sel = screen_factors(X, sh, SCREEN_PRESET, TRAIN_END)
    cols = rfe_ols(X, y, sel, TRAIN_END, n_keep=RFE_KEEP)
    print(f"入选特征（训练段重跑，固定）: {cols}", flush=True)

    res = run_model(X, y, price, cols, model="ols", window=WINDOW, fee=FEE,
                    vol_series=vol, ann=ann, min_train=MIN_TRAIN)
    R = net_returns(res["pos"], y.reindex(X.index), price.reindex(X.index), FEE)

    pos_file = OUT / f"positions_asof_{asof.date()}.csv"
    out_df = pd.DataFrame({
        "pos": res["pos"], "yhat": res["yhat"], "z": res["z"],
        "n_train": res["n_train"], "C": price.reindex(X.index),
        "y_cc": y.reindex(X.index), "R_net": R,
    })

    # ---------------- EMA 平滑改进版：状态与实盘一致 ----------------
    # EMA 只依赖上一期的状态 s_prev，因此实盘每天只需要保存一行状态。
    state_path = OUT / "state_ema.json"
    pos_ema = R_ema = None
    if SMOOTHING_ENABLED and CHOSEN_HALFLIFE is not None and SCALE_K is not None:
        hl, k = float(CHOSEN_HALFLIFE), float(SCALE_K)
        pos_ema, s_ema = smoothed_positions(res, vol, hl, k, TARGET_VOL)
        R_ema = net_returns(pos_ema, y.reindex(X.index), price.reindex(X.index), FEE)
        out_df["pos_ema"] = pos_ema
        out_df["s_ema"] = s_ema
        out_df["R_net_ema"] = R_ema

        # 状态一致性校验：把状态文件里**历史保存过的每一天**都重新算一遍并比对。
        #
        # 为什么不能只比最后一天：状态文件里的最后一天就是本次运行写进去的，
        # 同一天重跑时拿自己和自己比，永远通过，校验等于没有。真正能抓出问题的是
        # 更早的那些天——它们是**上一次（或上几次）运行**写下的，本次重算若与它们
        # 不一致，说明中途改过冻结参数、改过代码，或数据层被重建过。
        # 所以状态文件保存最近 N 天的历史，每天追加，校验时逐条比对。
        entries: list = []
        if state_path.exists():
            with open(state_path, "r", encoding="utf-8") as f:
                st = json.load(f)
            entries = list(st.get("history", []))
            checked = 0
            for e in entries:
                d = pd.Timestamp(e["date"])
                if d not in s_ema.index:
                    continue
                a = float(s_ema.loc[d])
                b = float(e["s"])
                if not np.isclose(a, b, rtol=1e-9, atol=1e-12):
                    raise SystemExit(
                        f"EMA 状态不一致：{d.date()} 重算 s={a!r} != "
                        f"历史保存 s={b!r}。说明重算路径或冻结参数变了。")
                checked += 1
            if checked:
                print(f"EMA 状态校验通过：比对 {checked} 个历史交易日"
                      f"（最新 {pd.Timestamp(entries[-1]['date']).date()}）", flush=True)
            else:
                print("EMA 状态校验：状态文件里没有可比对的历史交易日（首次运行）",
                      flush=True)

        # 追加本次的日期到历史（保留最近 STATE_HISTORY_DAYS 天）
        entries = [e for e in entries if e["date"] != str(asof.date())]
        entries.append(dict(date=str(asof.date()),
                            s=float(s_ema.iloc[-1]) if len(s_ema) else None,
                            z=float(res["z"].iloc[-1]) if len(res["z"]) else None,
                            pos=float(pos_ema.iloc[-1]) if len(pos_ema) else None,
                            pos_main=float(res["pos"].iloc[-1]) if len(res["pos"]) else None))
        entries = entries[-STATE_HISTORY_DAYS:]
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(dict(date=str(asof.date()),
                           s_prev=float(s_ema.iloc[-1]) if len(s_ema) else None,
                           z_prev=float(res["z"].iloc[-1]) if len(res["z"]) else None,
                           pos=float(pos_ema.iloc[-1]) if len(pos_ema) else None,
                           pos_main=float(res["pos"].iloc[-1]) if len(res["pos"]) else None,
                           halflife=hl, scale_k=k,
                           history=entries), f, indent=2)
        print(f"wrote {rel(state_path)}", flush=True)

    out_df.to_csv(pos_file)
    print(f"wrote {rel(pos_file)}", flush=True)

    # 绩效表：**主结果与 EMA 改进版各一行**。
    # 课程提交与往后跟踪两个版本都交（报告 §10 说明口径），所以这里并列输出。
    seg = R.loc[TEST_START:]
    rows = []
    m = metrics(seg, ann)
    # 分段换手用全样本逐日换手切片，口径与回测一致
    m = add_turnover(m, res["pos"].reindex(seg.index), ann,
                     dp=turnover_daily(res["pos"]))
    m["version"] = "main_ols_exp"
    rows.append(m)
    if R_ema is not None:
        seg_e = R_ema.loc[TEST_START:]
        me = metrics(seg_e, ann)
        me = add_turnover(me, pos_ema.reindex(seg_e.index), ann,
                          dp=turnover_daily(pos_ema))
        me["version"] = f"ema_halflife_{CHOSEN_HALFLIFE:.0f}"
        rows.append(me)
    pd.DataFrame(rows).to_csv(OUT / f"perf_asof_{asof.date()}.csv", index=False)
    print(f"wrote {rel(OUT / f'perf_asof_{asof.date()}.csv')}", flush=True)
    for r in rows:
        print(f"测试期 [{r['version']}] Sharpe={r['sharpe']:.3f} "
              f"CAGR={r['cagr']:.3f}", flush=True)


if __name__ == "__main__":
    main()
