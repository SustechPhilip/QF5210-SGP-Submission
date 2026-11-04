"""截断一致性测试（v2 §2.3，扩展到因子层 + 模型层 + 特征选择层）。

核心断言（对随机抽取的 N 个日期 t）：
  1. 把分钟数据截到 t 日 16:59:59，重跑**整条管线**（日历 -> 会话 -> 日频 -> 因子
     -> 筛选 -> RFE -> 模型）；
  2. 与全量数据跑出的结果比较：
       - 因子表 X 的所有 <= t 行必须逐位一致（np.allclose, equal_nan=True）
       - 标签 y 的所有 <= t-2 行必须一致
         （y[s] = C[s+1]/C[s]-1，s = t-1 需要 t 日收盘价，截断后必然不可得）
       - 模型输出 pos / yhat / z / n_train 的所有 <= t 行必须一致
  3. 任何一行不一致都报出具体日期与列名。

**特征选择层（已修）**：历史版本直接沿用全量数据选出的特征（`cols=cols_a`），
筛选与 RFE 根本没重跑，却宣称覆盖了这两个环节。

现在按日期分两种情形：
  * **t > 训练段末日（样本外）**：特征集合是用 <= 训练段末日 的数据定的，
    而训练段末日 < t，所以整套特征选择在 t 日 17:00 已经可得。
    -> 截断后**完整重跑筛选 + RFE**，并**断言入选特征与全量一致**。
  * **t <= 训练段末日（样本内）**：特征集合本身就是用 t 之后的数据选的
    （这是"特征集合在训练段定一次"的设计固有属性），截断后必然选不出一样的特征。
    -> 固定特征集合，只校验因子层与模型层。

只有因子模块 + 模型模块都通过，才能说明"t 日 17:00 决策时只用 <=16:59 的信息"。

数据源用 out/minutes_tagged.pkl（与 CSV 同源的已标注 pickle，加载 0.2s；
直接读 CSV 要 17 分钟，逐日期重跑不可行）。

用法：
  python3 test_truncation.py 20
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "sgp_data"
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(DATA))

from config import (CHOSEN_HALFLIFE, FEE, MIN_TRAIN_SCREEN, RFE_KEEP,  # noqa: E402
                    SCALE_K, SCREEN_PRESET, SMOOTHING_ENABLED, TARGET_VOL,
                    TRAIN_END)
from daily import build_daily, build_labels  # noqa: E402
from features import build_all  # noqa: E402
from models import MIN_TRAIN, run_model, vol_target_60  # noqa: E402
from position import build_positions  # noqa: E402
from screen import (rfe_ols, screen_factors, set_ann,  # noqa: E402
                    single_factor_signals)
from sessions import assign_sessions, build_calendar  # noqa: E402


def ann_factor(cal: pd.DataFrame) -> float:
    """ANN = 交易日数 / 样本年数。cal 可能是 DataFrame(td 列) 或以 td 为索引的表。"""
    idx = cal["td"] if "td" in cal.columns else cal.index
    idx = pd.DatetimeIndex(idx)
    span_years = (idx.max() - idx.min()).days / 365.25
    return len(idx) / span_years


def load_raw(pkl: Path) -> pd.DataFrame:
    """从已标注 pickle 还原 raw 结构（保留 t/o/h/l/c/v/hm/date）。"""
    m = pd.read_pickle(pkl)
    m["td"] = pd.to_datetime(m["td"])
    return m[["t", "o", "h", "l", "c", "v", "hm", "date"]].copy()


def pipeline(raw: pd.DataFrame, cols: list | None = None, ann: float | None = None):
    """从分钟数据跑完整管线，返回 (cal, dec, full, labels, X, res, cols, ann, pos_ema, s_ema)。

    EMA 平滑改进版用**冻结的** (半衰期, 尺度系数)：截断前后必须用同一组值，
    否则比的不是同一件事。冻结值来自 config.yaml（由 select_halflife.py 选出）。
    """
    cal = build_calendar(raw)
    m = assign_sessions(raw, cal)
    dec, full = build_daily(m, cal)
    labels = build_labels(full, cal)
    X = build_all(dec, full, m, robust=False)

    if ann is None:
        ann = ann_factor(cal)
    set_ann(ann)
    y = labels["y_cc"]
    price = full["C"]
    vol = vol_target_60(full, ann)

    if cols is None:
        sigs, sh = single_factor_signals(X, y, price, FEE, min_train=MIN_TRAIN_SCREEN,
                                          sharpe_end=TRAIN_END)
        sel = screen_factors(X, sh, SCREEN_PRESET, TRAIN_END)
        cols = rfe_ols(X, y, sel, TRAIN_END, n_keep=RFE_KEEP)

    res = run_model(X, y, price, cols, model="ols", window="expanding", fee=FEE,
                    vol_series=vol, ann=ann, min_train=MIN_TRAIN)
    pos_ema = s_ema = None
    if SMOOTHING_ENABLED and CHOSEN_HALFLIFE is not None and SCALE_K is not None:
        pos_ema, s_ema = build_positions(res["z"], vol, float(CHOSEN_HALFLIFE),
                                         float(SCALE_K), TARGET_VOL)
    return cal, dec, full, labels, X, res, cols, ann, pos_ema, s_ema


def diff_rows(a: pd.Series, b: pd.Series, idx: pd.Index, tol: float = 1e-12):
    """返回在 idx 上不一致的 (日期, 值a, 值b)。"""
    x = a.reindex(idx).to_numpy(dtype=float)
    z = b.reindex(idx).to_numpy(dtype=float)
    bad = []
    for i, d in enumerate(idx):
        if not np.allclose(x[i], z[i], equal_nan=True, rtol=tol, atol=tol):
            bad.append((d, x[i], z[i]))
    return bad


def main(n: int = 20, seed: int = 20261028) -> int:
    t0 = time.time()
    raw = load_raw(DATA / "out" / "minutes_tagged.pkl")
    print(f"加载分钟数据 {time.time()-t0:.1f}s，bar 数 {len(raw)}", flush=True)

    t0 = time.time()
    (cal_a, dec_a, full_a, lab_a, X_a, res_a, cols_a, ann_a,
     pos_ema_a, s_ema_a) = pipeline(raw)
    print(f"全量管线完成 {time.time()-t0:.1f}s：因子 {X_a.shape}，入选特征 {cols_a}",
          flush=True)

    rng = np.random.default_rng(seed)
    # 只取模型已能出预测的日期，否则截断后本来就没有仓位可比
    days = X_a.index[X_a.index > pd.Timestamp("2021-12-01")]
    picks = sorted(rng.choice(days, size=n, replace=False))

    fails = 0
    n_oos = 0
    for t in picks:
        tt = time.time()
        cut = raw[raw["t"] <= pd.Timestamp(t) + pd.Timedelta("16:59:59")]
        # 样本外日期：特征选择完全由 <= 训练段末日 的数据决定，t 日时已可得，
        # -> 完整重跑筛选 + RFE，并断言入选特征不变。
        # 样本内日期：特征集合本身是用 t 之后的数据选的，固定特征集合，
        # 只校验因子层与模型层。
        oos = t > TRAIN_END
        cols_b_arg = None if oos else cols_a
        (cal_b, dec_b, full_b, lab_b, X_b, res_b, cols_b, ann_b,
         pos_ema_b, s_ema_b) = pipeline(cut, cols=cols_b_arg, ann=ann_a)
        problems = []

        # --- 特征选择层：入选特征必须与全量一致（仅样本外日期）---
        if oos:
            n_oos += 1
            if list(cols_b) != list(cols_a):
                problems.append(("selected_features", str(cols_a), str(cols_b)))

        # --- 因子层：所有 <= t 的行、全部 65 列 ---
        idx_f = X_a.index[X_a.index <= t]
        for c in X_a.columns:
            problems += diff_rows(X_a[c], X_b[c], idx_f)

        # --- 标签层：y[s] 需要 C[s+1]，s <= t-2 才在截断数据里可得 ---
        pos_y = X_a.index.get_loc(t)
        idx_y = X_a.index[:pos_y - 1]      # <= t-2（位置口径，非日历日）
        problems += diff_rows(lab_a["y_cc"], lab_b["y_cc"], idx_y)

        # --- 模型层：pos / yhat / z / n_train 所有 <= t 的行 ---
        idx_m = X_a.index[X_a.index <= t]
        for c in ["pos", "yhat", "z", "n_train"]:
            problems += diff_rows(res_a[c], res_b[c], idx_m, tol=1e-10)

        # --- EMA 平滑改进版：s_ema / pos_ema（用冻结的半衰期与尺度系数）---
        if pos_ema_a is not None:
            problems += diff_rows(s_ema_a, s_ema_b, idx_m, tol=1e-10)
            problems += diff_rows(pos_ema_a, pos_ema_b, idx_m, tol=1e-10)

        if problems:
            fails += 1
            print(f"FAIL {pd.Timestamp(t).date()}: {len(problems)} 处不一致", flush=True)
            for p in problems[:10]:
                print(f"     {p[0]} : 全量={p[1]!r} 截断={p[2]!r}", flush=True)
        else:
            tag = "样本外(重跑筛选+RFE)" if oos else "样本内(固定特征集)"
            print(f"ok   {pd.Timestamp(t).date()}  {tag}  "
                  f"(因子 {len(idx_f)} 行, "
                  f"仓位 {int(res_b['pos'].reindex(idx_m).notna().sum())} 天, "
                  f"{time.time()-tt:.0f}s)", flush=True)

    print(f"\n{'='*62}")
    if fails == 0:
        extra = "、EMA 平滑改进版（s_ema / pos_ema）" if pos_ema_a is not None else ""
        print(f"截断测试通过：{n} 个日期（其中 {n_oos} 个样本外日期连同筛选与 RFE "
              f"一起重跑并断言入选特征不变），因子 / 标签 / 仓位三层{extra}全部逐位一致")
        return 0
    print(f"截断测试失败：{fails}/{n} 个日期不一致")
    return 1


if __name__ == "__main__":
    a = sys.argv[1:]
    raise SystemExit(main(int(a[0]) if a else 20))
