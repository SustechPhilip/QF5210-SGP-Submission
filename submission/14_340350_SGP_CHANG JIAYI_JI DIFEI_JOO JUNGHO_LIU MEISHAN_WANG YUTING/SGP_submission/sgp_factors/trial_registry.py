"""试验登记册 `out/trials.csv`。

**为什么需要这个文件。** 这个项目里试过几十种方案（主结果、EMA 网格、
筛选变体、仓位变体、Ridge、窗口扫描……）。单看一个策略的 p 值会**高估**显著性：
试得越多，偶然出现一个好看的 Sharpe 是迟早的事。Deflated Sharpe Ratio
（见 `inference.py`）要回答的就是「扣掉多重检验之后还剩多少显著性」，
而它需要知道**试验次数**和**各次试验的 Sharpe 分布**。

登记册的规则：

* **每一个看过测试期结果的方案都要登记**，包括无效的那些。
* 每行记录：方案名、类别、**测试期日度 Sharpe**（不年化）、测试期样本数。
* 两套口径：
  * `caliber=full`       —— 全部登记的方案（含 140 组窗口扫描）。
  * `caliber=strategy`   —— 只登记「策略级」方案（约 20 个），窗口扫描
    140 组不计入（它们是同一批因子的参数扫描，不是 140 个独立想法）。

`infer.py` 会用这两套口径分别算 DSR，并列报告。

**已知局限**：登记册只能收录**在 `out/` 里留下了结果文件**的方案。
中途失败或未落盘的尝试无法自动收录，只能手工补记——报告里要写明这一点。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

# 类别标签
CAT_MAIN = "主结果"
CAT_EMA = "EMA网格"
CAT_SCREEN = "筛选变体"
CAT_POS = "仓位变体"
CAT_MODEL = "模型变体"
CAT_WINDOW = "窗口扫描"
CAT_COST = "成本假设"
CAT_DATA = "数据口径"
CAT_NEW = "新增基准"


def daily_sharpe_from_ann(ann_sr: float, ann: float) -> float:
    """年化 Sharpe -> 日度 Sharpe。

    ann_sharpe = mean/std * sqrt(ann)，故日度 = 年化 / sqrt(ann)。
    """
    if ann is None or not np.isfinite(ann_sr) or ann <= 0:
        return np.nan
    return float(ann_sr / np.sqrt(ann))


def _row(name: str, caliber: str, category: str, ann_sr: float,
         ann: float, n_test: int, note: str = "") -> dict:
    return dict(
        name=name,
        caliber=caliber,
        category=category,
        test_sharpe_ann=ann_sr,
        test_sharpe_daily=daily_sharpe_from_ann(ann_sr, ann),
        n_test=n_test,
        note=note,
    )


def _from_metrics_block(block: dict, ann: float) -> tuple[float, int]:
    """robustness.json 里的 metrics 块 -> (年化 Sharpe, n)。"""
    if not isinstance(block, dict):
        return np.nan, 0
    test = block.get("test", block)
    return test.get("sharpe", np.nan), int(test.get("n", 0) or 0)


def build_registry(out_dir, ann: float) -> pd.DataFrame:
    """扫描 `out/` 下的既有结果，生成完整登记册。"""
    rows: list[dict] = []
    out = out_dir

    # ---- 1. 主结果（metrics.csv：strategy x segment）--------------------
    main_path = out / "metrics.csv"
    if main_path.exists():
        m = pd.read_csv(main_path)
        for _, r in m.iterrows():
            if str(r.get("segment")) != "test":
                continue
            rows.append(_row(str(r.get("strategy")), "strategy", CAT_MAIN,
                              r.get("sharpe"), ann, int(r.get("n", 0) or 0)))

    # ---- 2. EMA 网格 ----------------------------------------------------
    ema_grid = out / "ema_grid_valid.csv"
    if ema_grid.exists():
        g = pd.read_csv(ema_grid)
        for _, r in g.iterrows():
            hl = r.get("halflife")
            nm = "ema_hl=%s" % hl if hl is not None else "ema_none"
            rows.append(_row(nm, "strategy", CAT_EMA, r.get("test_sharpe"),
                              ann, int(r.get("n_test", 0) or 0),
                              note="半衰期按验证期选出"))

    # ---- 3. 筛选变体 ----------------------------------------------------
    sv = out / "screen_variants.csv"
    if sv.exists():
        s = pd.read_csv(sv)
        for _, r in s.iterrows():
            rows.append(_row(str(r.get("variant", "")), "strategy", CAT_SCREEN,
                              r.get("test_sharpe"), ann,
                              int(r.get("n_test", 0) or 0)))

    # ---- 4. 窗口扫描（140 组，只有 full 口径计入）-----------------------
    ws = out / "window_sensitivity.csv"
    if ws.exists():
        w = pd.read_csv(ws)
        for _, r in w.iterrows():
            nm = "ws_demean=%s_window=%s_all=%s" % (
                r.get("demean"), r.get("window"), r.get("all_factors"))
            rows.append(_row(nm, "full", CAT_WINDOW, r.get("test_sharpe"), ann,
                              int(r.get("n_test", 0) or 0),
                              note="degenerate" if r.get("degenerate") else ""))

    # ---- 5. 稳健性 json 里的方案 ---------------------------------------
    rj = out / "robustness.json"
    if rj.exists():
        rob = json.loads(rj.read_text())
        for key in ("cost_0.0001", "cost_0.0002", "cost_0.0005"):
            if key in rob:
                sr, n = _from_metrics_block(rob[key], ann)
                rows.append(_row("fee=%s" % key.split("_")[1], "strategy",
                                  CAT_COST, sr, ann, n))
        for a, d in (rob.get("ridge") or {}).items():
            sr, n = _from_metrics_block(d, ann)
            rows.append(_row("ridge_alpha=%s" % a, "strategy", CAT_MODEL, sr, ann, n))
        for a, d in (rob.get("alt_close") or {}).items():
            sr, n = _from_metrics_block(d, ann)
            rows.append(_row("alt_close=%s" % a, "strategy", CAT_DATA, sr, ann, n))
        for a, d in (rob.get("window") or {}).items():
            sr, n = _from_metrics_block(d, ann)
            rows.append(_row("window=%s" % a, "strategy", CAT_MODEL, sr, ann, n))
        for a, d in (rob.get("ablation") or {}).items():
            sr, n = _from_metrics_block(d, ann)
            rows.append(_row("ablation=%s" % a, "strategy", CAT_MODEL, sr, ann, n))
        for a, d in (rob.get("anchor_open") or {}).items():
            sr, n = _from_metrics_block(d, ann)
            rows.append(_row("anchor_open=%s" % a, "strategy", CAT_MODEL, sr, ann, n))
        imp = rob.get("improved") or {}
        for v in (imp.get("variants") or []):
            rows.append(_row("improved:%s" % v.get("name"), "strategy", CAT_POS,
                              v.get("test_sharpe"), ann,
                              int(v.get("n_test", 0) or 0)))

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["caliber", "category", "name"]).reset_index(drop=True)


def add_new_trials(df: pd.DataFrame, new_rows: list[dict]) -> pd.DataFrame:
    """把本次新增的方案（最好单因子基准、扩展因子版等）并入登记册。"""
    if not new_rows:
        return df
    add = pd.DataFrame(new_rows)
    if df.empty:
        return add
    return pd.concat([df, add], ignore_index=True).sort_values(
        ["caliber", "category", "name"]).reset_index(drop=True)


def trial_sharpes(df: pd.DataFrame, caliber: str) -> np.ndarray:
    """取某一口径下全部试验的**测试期日度 Sharpe**。"""
    s = df.loc[df["caliber"] == caliber, "test_sharpe_daily"].to_numpy(dtype=float)
    return s[np.isfinite(s)]
