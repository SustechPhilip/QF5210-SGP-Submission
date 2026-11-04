"""绩效表：perf(start, end) 从同一份 R[t] 序列生成课程要求的绩效表。

口径与回测完全一致（v2 §2.4「净值与指标由同一份 R[t] 序列生成」）：
  R[t] = pos[t] × y_cc[t] − |Δpos[t]| × (fee + 0.05/C[t])

**两个版本并列输出**（课程提交与往后跟踪都交两个版本，见报告 §10）：
  * `main_ols_exp`      主结果，不平滑，预先登记的科学口径
  * `ema_halflife_N`    EMA 平滑改进版，半衰期由 2023 验证期选出后冻结

短窗口口径（n < 60 个交易日）：年化指标没有统计意义（Sharpe 的标准误约
√(260/n)，n=60 时约 2.1），此时年化类指标输出 N/A 并保留行位，同时补报
不年化的区间值。

用法：
  python3 perf.py                       # 默认 2026-10-01 ~ 2026-10-16
  python3 perf.py 2024-01-01 2026-10-16
  python3 perf.py 2026-10-01 2026-10-16 --file out/daily_backtest.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

sys.path.insert(0, str(HERE.parent))
from config import (CHOSEN_HALFLIFE, FEE, FREEZE_DATE, MIN_N_ANNUALIZED,  # noqa: E402
                    TICK, rel)


def ann_from_calendar(path: Path) -> float:
    cal = pd.read_csv(path, parse_dates=["td"]).set_index("td")
    return len(cal) / ((cal.index.max() - cal.index.min()).days / 365.25)


def perf(daily: pd.DataFrame, start: str, end: str, ann: float,
         fee: float = FEE, version: str = "main_ols_exp",
         pos_col: str = "pos", r_col: str = "R_net") -> dict:
    """区间绩效。daily 含 pos / R_net / C / y_cc。

    `pos_col` / `r_col` 用来切换版本：
      * 主结果      -> ("pos", "R_net")
      * EMA 改进版  -> ("pos_ema", "R_net_ema")

    换手口径：逐日 |Δpos| 取自**全样本**的 pos 序列，因此窗口第一天不会
    被当成「从 0 建仓」，与 R_net 里实际扣的成本一致。
    """
    d = daily.loc[start:end]
    d = d[d[r_col].notna()]
    if len(d) == 0:
        return dict(version=version, n=0, note="区间内无已实现收益")

    R = d[r_col]
    pos_all = daily[pos_col]
    ycol = "y_cc" if "y_cc" in d.columns else "y"

    # 逐日换手：全样本口径，首日不算换手（持仓延续）
    p_all = pos_all.fillna(0.0)
    dp_all = (p_all - p_all.shift(1).fillna(0.0)).abs()
    turnover_period = float(dp_all.reindex(d.index).sum())

    cum = (1 + R).cumprod()
    n = len(R)
    years = n / ann
    std = float(R.std(ddof=0))
    peak = cum.cummax()
    dd = cum / peak - 1
    mdd = float(dd.min())
    downside = np.minimum(R.to_numpy(), 0)
    dstd = float(np.sqrt(np.mean(downside ** 2)))
    gains = float(R[R > 0].sum())
    losses = float(-R[R < 0].sum())
    q05 = float(R.quantile(0.05))

    # ---- 区间值（不年化，任何窗口都报）----
    out = dict(
        version=version,
        # 样本状态：冻结日之后的日期才是「真样本外」（两个版本的参数都已定死）；
        # 冻结日及之前的数据早已在样本里，测试段更是已经被看过。
        sample_status=("冻结后新样本（真样本外）" if d.index.max() > FREEZE_DATE
                       else "已见样本（非新样本）"),
        n=int(n),
        start=str(d.index.min().date()),
        end=str(d.index.max().date()),
        total_return=float(cum.iloc[-1] - 1),
        mean_daily=float(R.mean()),
        std_daily=std,
        max_dd=mdd,
        win_rate=float((R > 0).mean()),
        profit_factor=float(gains / losses) if losses > 0 else np.nan,
        period_turnover=turnover_period,
        turnover_per_day=float(turnover_period / n),
        gross_sum=float((pos_all.reindex(d.index).fillna(0.0) * d[ycol]).sum()),
        cost_sum=float(R.sum() - (pos_all.reindex(d.index).fillna(0.0) * d[ycol]).sum()),
        avg_pos=float(pos_all.reindex(d.index).mean()),
        long_days=int((pos_all.reindex(d.index) > 0).sum()),
        short_days=int((pos_all.reindex(d.index) < 0).sum()),
        flat_days=int((pos_all.reindex(d.index) == 0).sum()),
    )

    # ---- 年化指标：n >= 60 才有统计意义 ----
    if n >= MIN_N_ANNUALIZED:
        cagr = float(cum.iloc[-1] ** (1 / years) - 1) if years > 0 else np.nan
        sharpe = float(R.mean() / std * np.sqrt(ann)) if std > 0 else np.nan
        sortino = float(R.mean() / dstd * np.sqrt(ann)) if dstd > 0 else np.nan
        out.update(
            cagr=cagr,
            ann_vol=float(std * np.sqrt(ann)),
            sharpe=sharpe,
            calmar=float(cagr / abs(mdd)) if mdd < 0 else np.nan,
            sortino=sortino,
            var95=q05,
            cvar95=float(R[R <= q05].mean()),
            ann_turnover=float(turnover_period / years),
            annualized=True,
        )
    else:
        note = f"N/A（n={n} < {MIN_N_ANNUALIZED}，年化无统计意义）"
        out.update(
            cagr=note, ann_vol=note, sharpe=note, calmar=note, sortino=note,
            var95=q05, cvar95=float(R[R <= q05].mean()),
            ann_turnover=note,
            annualized=False,
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("start", nargs="?", default="2026-10-01")
    ap.add_argument("end", nargs="?", default="2026-10-16")
    ap.add_argument("--file", default=str(OUT / "daily_backtest.csv"))
    ap.add_argument("--ema-file", default=str(OUT / "daily_backtest_ema.csv"))
    ap.add_argument("--ann", type=float, default=None)
    args = ap.parse_args()

    daily = pd.read_csv(args.file, parse_dates=[0], index_col=0)
    daily.index = pd.to_datetime(daily.index)
    ann = args.ann if args.ann else ann_from_calendar(
        HERE.parent / "sgp_data" / "out" / "calendar.csv")

    rows = [perf(daily, args.start, args.end, ann, version="main_ols_exp",
                 pos_col="pos", r_col="R_net")]

    # EMA 改进版：仓位与收益在 daily_backtest_ema.csv 里，两份按日期合并后算
    ema_path = Path(args.ema_file)
    if ema_path.exists():
        ema = pd.read_csv(ema_path, parse_dates=[0], index_col=0)
        ema.index = pd.to_datetime(ema.index)
        merged = daily.join(ema[["pos_ema", "R_net_ema"]], how="outer")
        rows.append(perf(merged, args.start, args.end, ann,
                         version=f"ema_halflife_{CHOSEN_HALFLIFE:.0f}",
                         pos_col="pos_ema", r_col="R_net_ema"))

    out = pd.DataFrame(rows)
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(out.T.to_string())
    out.to_csv(OUT / f"perf_{args.start}_{args.end}.csv", index=False)
    print(f"\nwrote {rel(OUT / f'perf_{args.start}_{args.end}.csv')}")


if __name__ == "__main__":
    main()
