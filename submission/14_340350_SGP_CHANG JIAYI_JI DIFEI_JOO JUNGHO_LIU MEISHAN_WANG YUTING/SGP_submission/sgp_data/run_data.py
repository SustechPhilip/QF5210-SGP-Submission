"""数据层主脚本：python run_data.py data/SGP_2.csv out/"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from daily import build_daily, build_labels
from sessions import REGIMES, assign_sessions, build_calendar, load_minutes, validate


def _drop_arrow_backing(df: pd.DataFrame) -> pd.DataFrame:
    """把 pandas 3 的 pyarrow 字符串列转成 object 列。

    pandas 3 默认把字符串列存成 pyarrow-backed 的 `str` dtype，pickle 里就带了
    pyarrow 的依赖：**换一台没装 pyarrow 的机器直接读不了**（实测 reviewer 环境
    卡在第一步）。转成 object 后，任何 pandas 2/3 都能读，不需要 pyarrow。

    **列名 Index 也要转**：pandas 3 的 `Index` 默认 dtype 就是 `str`
    （pyarrow-backed），只转数据列不转列名，pickle 里仍会残留
    `ArrowStringArray` / `pyarrow.lib` 的引用。
    """
    out = df.copy()
    for c in out.columns:
        if str(out[c].dtype) == "str":
            out[c] = out[c].astype(object)
    if str(out.columns.dtype) == "str":
        out.columns = out.columns.astype(object)
    if str(out.index.dtype) == "str":
        out.index = out.index.astype(object)
    return out



def main(src: str, out: str) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    raw = load_minutes(src)
    cal = build_calendar(raw)
    m = assign_sessions(raw, cal)
    bad = validate(m)
    dec, full = build_daily(m, cal)
    lab = build_labels(full, cal)

    # ---- 断言 ----
    assert m["t"].is_unique
    assert (m.loc[m.session == "day", "date"] == m.loc[m.session == "day", "td"]).all()
    fri_night = m[(m.session == "night") & (m.t.dt.weekday == 4) & (m.hm >= "17:35")]
    assert (fri_night["td"].dt.weekday == 0).mean() > 0.95, "周五夜盘应归属下周一（节假日除外）"
    assert dec["C_star"].notna().all() and full["C"].notna().all()

    # ---- 输出 ----
    dec.to_csv(out / "daily_dec.csv")
    full.to_csv(out / "daily_full.csv")
    lab.to_csv(out / "labels.csv")
    cal.to_csv(out / "calendar.csv", index=False)
    bad.to_csv(out / "out_of_session_bars.csv", index=False)
    # 存成不依赖 pyarrow 的格式（见 _drop_arrow_backing 的注释）
    m_out = _drop_arrow_backing(m)
    m_out.to_pickle(out / "minutes_tagged.pkl")
    # 自检：存出来的 pickle 必须能在没有 pyarrow 的环境里读
    chk = pd.read_pickle(out / "minutes_tagged.pkl")
    assert len(chk) == len(m)
    assert not any(str(chk[c].dtype) == "str" for c in chk.columns), \
        "pickle 里仍有 pyarrow-backed 字符串列，换机打不开"
    assert str(chk.columns.dtype) != "str", \
        "pickle 的列名 Index 仍是 pyarrow-backed，换机打不开"
    pd.DataFrame({"date": full.index.strftime("%Y-%m-%d"), "SGP": full["C"].round(2).values}).to_csv(
        out / "SGP_close.csv", index=False
    )

    # ---- 按时期汇总报告 ----
    rows = []
    for r in REGIMES.itertuples():
        mm = m[(m.td.isin(cal.loc[cal.regime == r.regime, "td"]))]
        d = mm[mm.session == "day"]
        n = mm[mm.session == "night"]
        nd = d.td.nunique()
        eve = n[n.hm >= "12:00"].groupby("td").hm.min()
        morn = n[n.hm < "12:00"].groupby("td").hm.max()
        f = full[full.regime == r.regime]
        de = dec[dec.regime == r.regime]
        gap = np.log(f["C"] / de["C_star"]) * 1e4
        rows.append(
            dict(
                regime=r.regime, start=r.start.date(), rule_day_last=r.day_last, rule_night=f"{r.night_first}-{r.night_last}",
                trading_days=nd, first_td=f.index.min().date(), last_td=f.index.max().date(),
                day_last_bar_mode=d.groupby("td").hm.max().mode()[0],
                night_first_mode=eve.mode()[0] if len(eve) else None,
                night_last_mode=morn.mode()[0] if len(morn) else None,
                day_bars_per_day=round(len(d) / nd, 1),
                close_filled_days=int(f.close_filled.sum()),
                short_days=int(f.short_day.sum()),
                thin_days=int(f.thin_day.sum()),
                data_gap_days=int(f.data_gap.sum()),
                night_vol_share=round(n.v.sum() / mm.v.sum(), 3),
                post1659_vol_share_of_day=round(d[d.hm > "16:59"].v.sum() / d.v.sum(), 3),
                gap_1659_to_close_absmean_bps=round(gap.abs().mean(), 2),
                gap_p95_bps=round(gap.abs().quantile(0.95), 2),
                out_of_session_bars=int((bad.regime == r.regime).sum()),
            )
        )
    rep = pd.DataFrame(rows)
    rep.to_csv(out / "regime_report.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(rep.T.to_string())
        print("交易日", len(cal), "越界 bar", len(bad), "pending bar", int((m.session == "pending").sum()),
              "offbook bar", int((m.session == "offbook").sum()))
        m[m.session == "offbook"][["t", "c", "v", "td"]].to_csv(out / "offbook_bars.csv", index=False)
        print("close_filled 日期:", list(full.index[full.close_filled == 1].strftime("%Y-%m-%d")))
        print("short_day 日期:", list(full.index[full.short_day == 1].strftime("%Y-%m-%d")))
        print("thin_day 日期:", list(full.index[full.thin_day == 1].strftime("%Y-%m-%d")))
        print("data_gap 日期:", list(full.index[full.data_gap == 1].strftime("%Y-%m-%d")))
        if len(bad):
            print(bad.groupby(["regime", "hm"]).size().sort_values(ascending=False).head(20))


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["data/SGP_2.csv", "out"]))
