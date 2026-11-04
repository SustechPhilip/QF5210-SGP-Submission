"""配置加载：config.yaml 是**唯一参数来源**。

config.yaml 用 **JSON 语法**书写。JSON 是 YAML 1.2 的子集，PyYAML 的
`yaml.safe_load` 与标准库的 `json.loads` 都能解析；但本模块用 `json.loads`，
因为它是标准库、零第三方依赖，保证任何环境都能直接运行（不需要装 PyYAML）。

本文件位于**包根目录**，与 `sgp_data/`、`sgp_factors/` 平级。
数据层（`sgp_data`）与因子层（`sgp_factors`）都从这里取参数，
两层之间不再各自硬编码同一常量（薄荷阈值 0.85、中断阈值 60 分钟等）。
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.yaml"


def rel(path) -> str:
    """把路径显示成**相对包根目录**的形式。

    包可以被任意移动或改名，所以日志里不应该出现 `~/Downloads/...`
    这种绝对路径——换个位置再跑，日志就对不上了，读者也会以为日志是别处跑的。
    统一显示成 `sgp_factors/out/xxx` 这样的相对路径，日志就跟包内布局对得上，
    与包放在哪里无关。
    """
    try:
        return Path(path).resolve().relative_to(HERE).as_posix()
    except ValueError:
        return Path(path).as_posix()


with open(CONFIG_PATH, "r", encoding="utf-8") as _f:
    CFG = json.load(_f)


# ---------------------------------------------------------------- 切分
TRAIN_END = pd.Timestamp(CFG["split"]["train_end"])
VALID_START = pd.Timestamp(CFG["split"]["valid_start"])
VALID_END = pd.Timestamp(CFG["split"]["valid_end"])
TEST_START = pd.Timestamp(CFG["split"]["test_start"])

# ---------------------------------------------------------------- 成本
FEE = float(CFG["cost"]["fee"])
TICK = float(CFG["cost"]["tick"])
FEE_GRID = [float(x) for x in CFG["cost"]["fee_grid"]]

# ---------------------------------------------------------------- 仓位
TARGET_VOL = float(CFG["target_vol"])
MIN_TRAIN = int(CFG["model"]["min_train"])
MIN_TRAIN_SCREEN = int(CFG["model"]["min_train_screen"])
RFE_KEEP = int(CFG["model"]["rfe_keep"])
WINDOW = str(CFG["model"]["window"])
RIDGE_ALPHAS = [float(a) for a in CFG["model"]["ridge_alphas"]]

# ---------------------------------------------------------------- 筛选
SCREEN_PRESET = str(CFG["screen"]["preset"])
SCREEN_PRESETS = {
    k: dict(sharpe=float(v["sharpe"]), corr=float(v["corr"]))
    for k, v in CFG["screen"]["presets"].items()
}

# ---------------------------------------------------------------- 数据质量
# 稀疏日判定：只用**决策窗内**（<=16:59）的 bar 数，避免用 17:00 之后的信息。
# 决策窗固定为 08:30-16:59，四个时期都是 510 根，因此与时期无关。
THIN_RATIO = float(CFG["data_quality"]["thin_day_ratio"])
THIN_EXPECTED_MINUTES = int(CFG["data_quality"]["thin_day_expected_minutes"])
# 盘中无成交中断阈值（分钟）；数据层与因子层同源
GAP_MIN_MINUTES = int(CFG["data_quality"]["gap_min_minutes"])
# 用 t-1 数据的 B 类因子：前一日坏掉时置空
PREV_DAY_B_FACTORS = list(CFG["data_quality"]["factor_mask_rule"]["prev_day_B"])

# ---------------------------------------------------------------- 稳健性
N_SHIFT = int(CFG["robust"]["n_shift"])
SEED = int(CFG["robust"]["seed"])
EXPIRY_WINDOW_DAYS = int(CFG["robust"]["expiry_window_trading_days"])

# ---------------------------------------------------------------- 绩效表
MIN_N_ANNUALIZED = int(CFG["perf"]["min_n_for_annualized"])

# ---------------------------------------------------------------- 冻结日
# 冻结日 = 改进版的参数（半衰期、尺度系数）确定时**能看到的最后一个交易日**。
# 该日之后追加的数据，对主结果与改进版都是**真正没见过的样本**（真样本外），
# 因为两个版本的参数此后都不再改动。
# 该日及之前的数据都已经在样本里（测试段尤其如此），标为「已见样本」。
FREEZE_DATE = pd.Timestamp(CFG.get("freeze", {}).get("date", "2026-10-16"))
FREEZE_NOTE = str(CFG.get("freeze", {}).get("note", ""))

# ---------------------------------------------------------------- 实盘状态
# update.py 保存最近多少天的 EMA 状态历史。同一天重跑时，最后一天是本次运行自己
# 写的、拿自己和自己比没有意义；真正能抓出「中途改了冻结参数/代码/数据层」的是
# 更早的那几天，所以状态文件保留一段历史供逐条比对。
STATE_HISTORY_DAYS = int(CFG.get("state_history_days", 30))

# ---------------------------------------------------------------- 平滑（事后改进版）
# 说明：这一段是**事后改进**（看过测试集结果之后才加的），单列为「改进版」，
# **不替换主结果**。主结果仍是不平滑的 OLS 仓位。
SMOOTHING = CFG.get("smoothing", {})
SMOOTHING_ENABLED = bool(SMOOTHING.get("enabled", False))
SMOOTHING_METHOD = str(SMOOTHING.get("method", "ema_recursive"))
HALFLIFE_GRID = [float(h) for h in SMOOTHING.get("halflife_grid", [])]
SMOOTH_SELECT_ON = str(SMOOTHING.get("select_on", "valid"))
SMOOTH_SELECT_METRIC = str(SMOOTHING.get("select_metric", "net_sharpe"))
SMOOTH_SCALE_FIT_END = pd.Timestamp(SMOOTHING.get("scale_fit_end", CFG["split"]["train_end"]))
# 由 select_halflife.py 写回，之后冻结；回测与实盘只读不改
CHOSEN_HALFLIFE = SMOOTHING.get("chosen_halflife", None)
SCALE_K = SMOOTHING.get("scale_k", None)


def write_back_smoothing(chosen_halflife, scale_k) -> None:
    """把选出的半衰期与尺度系数写回 config.yaml（冻结）。

    写回时保留文件原有缩进与键序，只替换 smoothing 段内的两个字段。
    """
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        raw = f.read()
    cfg = json.loads(raw)
    cfg.setdefault("smoothing", {})["chosen_halflife"] = chosen_halflife
    cfg["smoothing"]["scale_k"] = scale_k
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
        f.write("\n")
