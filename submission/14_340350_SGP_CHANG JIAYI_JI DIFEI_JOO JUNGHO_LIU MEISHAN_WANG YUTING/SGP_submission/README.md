# SGP（MSCI Singapore 股指期货）日频择时回测包

**课程**：QF5210 · **品种**：SGX MSCI Singapore 股指期货（SGP）主连 · **频率**：日频择时

一句话结论：**样本外 Sharpe 在 0 附近，没有可靠的择时能力；降换手这个方向有效，
但改进版仍跑输买入持有。** 完整论证见 `QF5210_SGP回测报告.md`。

---

## 1. 这个包是什么

从 1 分钟线出发，逐层往上做的一个**可复现**的日频择时回测：

```
SGP 2.csv (160 万根 1 分钟线)
  └─ sgp_data/    日历 → 会话划分 → 日频聚合 → 标签      （数据层）
       └─ sgp_factors/  65 个因子 → 滚动外推筛选 → RFE → OLS → 仓位 → 绩效
            └─ 稳健性（成本/分年/消融/随机对照/换月/规则时期…）
```

三个层次刻意分开：

| 层 | 目录 | 职责 |
|---|---|---|
| 数据层 | `sgp_data/` | 分钟线 → 交易日历、日频 bar、标签 `y_cc` |
| 因子层 | `sgp_factors/` | 因子计算、筛选、模型、仓位、回测、稳健性 |
| 参数层 | `config.yaml` + `config.py` | **唯一参数出口**，两层共用 |

`config.yaml` 是**唯一参数来源**，用 JSON 语法书写（JSON 是 YAML 1.2 的子集，
标准库 `json` 就能读，**不需要装 PyYAML**）。所有模块一律从 `config.py` 取参数，
不在代码里各自硬编码同一常量。

---

## 2. 快速开始

```bash
# 依赖：pandas>=2.0, numpy>=1.24, scipy>=1.10（scipy 仅推断检验用）
# 实测 Python 3.13 / pandas 3.0.5 / numpy 2.5.2 / scipy 1.18.0
pip install -r requirements.txt

# 包内已带全部产物，通常不用重跑。想从零复现见 §4。
```

先看结论，直接读 `QF5210_SGP回测报告.md`。

---

## 3. 目录结构

```
SGP_submission/
├── QF5210_SGP回测报告.md              ★ 主报告（10 章）
├── QF5210_SGP回测详细计划_补充v2.md    详细计划与修正记录
├── README.md                           本文件
├── requirements.txt                    依赖
├── config.yaml                        ★ 唯一参数来源
├── config.py                          配置加载出口
├── MANIFEST.sha256                     每个文件的 SHA256 校验清单
├── sgp_data/
│   ├── sessions.py                     会话划分（point-in-time 交易规则）
│   ├── daily.py                        日频聚合 + 标签
│   ├── run_data.py                     数据层入口
│   ├── test_truncation.py              数据层截断一致性测试
│   └── out/                            ★ 数据层产物（已含在包里）
│         minutes_tagged.pkl (177MB) daily_dec.csv daily_full.csv
│         labels.csv calendar.csv regime_report.csv SGP_close.csv
│         offbook_bars.csv out_of_session_bars.csv
├── sgp_factors/
│   ├── features.py                     65 个因子的计算
│   ├── build_factors.py                因子层入口
│   ├── screen.py                       滚动外推单因子 Sharpe → 贪心筛选 → RFE
│   ├── models.py                       滚动重估 OLS/Ridge、基准、指标、随机对照
│   ├── position.py                     ★ EMA 平滑仓位 + 仓位规则变体
│   ├── select_halflife.py              ★ 验证期选参 → 写回 config.yaml
│   │                                   （同时算各半衰期的循环平移 p 值）
│   ├── backtest.py                     ★ 主回测 + 全部稳健性
│   ├── screen_variants.py              ★ 筛选方法变体对照（报告 §6.5）
│   ├── window_sensitivity.py           ★ 筛选窗口敏感性扫描 140 组（报告 §5.4）
│   ├── expiry.py                       ★ 到期日 / 换月窗口的唯一实现
│   ├── inference.py                    ★ PSR / Deflated Sharpe / 配对平稳块自助法
│   ├── trial_registry.py               ★ 试验登记册（out/trials.csv）
│   ├── infer.py                        ★ 推断检验入口（报告 §5.5）
│   ├── update.py                       ★ 每日更新 + EMA 状态一致性校验
│   ├── perf.py                         ★ 课程要求的绩效表（两个版本并列）
│   ├── version_compare.py              ★ 三版实现对照（原版/只修B7/全修）
│   ├── legacy/                         ★ 修复前的代码快照
│   ├── test_truncation.py              ★ 截断一致性测试（因子+标签+模型+EMA 层）
│   ├── test_minute_agg_fix.py          索引错位修复的回归测试
│   └── out/                            ★ 因子层产物（已含在包里）
└── logs/                               每次运行的日志
```

所有路径都基于 `__file__` 相对解析，**整个包可以任意移动或改名**。

---

## 4. 复现步骤

```bash
# 1. 数据层（约 20 分钟，瓶颈是 CSV 解析；包内已有产物可跳过）
python3 sgp_data/run_data.py "~/Downloads/SGP 2.csv" sgp_data/out

# 2. 因子（约 10 秒）
python3 sgp_factors/build_factors.py

# 3. 主回测 + 全部稳健性（约 8 分钟）
python3 sgp_factors/backtest.py            # 加 --quick 跳过稳健性

# 3b. 事后改进版的选参（验证期选半衰期，写回 config.yaml 后冻结）
#     —— config.yaml 里已冻结好（chosen_halflife=2），通常不用重跑；
#        想核对就再跑一次（加 --dry-run 只看不写回），结果应完全一致。
#     同时会算出各半衰期的循环平移 p 值（报告 §6.4）
python3 sgp_factors/select_halflife.py --dry-run

# 4. 筛选方法变体对照（报告 §6.5）
python3 sgp_factors/screen_variants.py

# 4b. 筛选窗口敏感性扫描，140 组（报告 §5.4，约 30 分钟；--quick 为 8 组冒烟）
python3 sgp_factors/window_sensitivity.py

# 5. 实现修正对照：原版 / 只修 B7 / 全部修正
python3 sgp_factors/version_compare.py

# 6. 截断一致性测试（约 8 分钟，20 个随机日期）
python3 sgp_factors/test_truncation.py 20

# 7. 回归测试（索引错位修复）
python3 sgp_factors/test_minute_agg_fix.py

# 8. 绩效表与每日更新
python3 sgp_factors/perf.py 2026-10-01 2026-10-16
python3 sgp_factors/update.py --asof 2026-10-16

# 9. 推断检验（报告 §5.5，约 40 秒）
#    最好单因子基准 + Deflated Sharpe + Sharpe 差异检验 + 风险情景表
python3 sgp_factors/infer.py
```

**校验包完整性**：

```bash
cd SGP_submission && shasum -a 256 -c MANIFEST.sha256
```

**日志里的路径是相对包根目录的**（如 `sgp_factors/out/metrics.csv`），不含任何机器的绝对路径，
所以包换个位置重跑，日志依然和包内布局对得上。这一点已实测：把整个包复制到别处重跑，
产物与原目录**逐字节一致**。

---

## 5. 口径速查

| 项 | 口径 |
|---|---|
| 决策时点 | t 日 **17:00**；因子只用 **≤16:59** 的数据 |
| 执行价 | 当期日盘最后一笔成交价 `C_t` |
| 标签 | `y_cc[t] = C_{t+1}/C_t − 1` |
| 训练行 | 必须 **s ≤ t−2**（t−1 的标签要到 t 日收盘才实现） |
| 复权 | **不复权** |
| 成本 | `|Δpos| × (fee + 0.05/C_t)`，fee 默认 **2e-4** |
| 波动率目标 | **10%**，`min(1, 10%/σ̂₆₀)` |
| 样本切分 | 训练 ≤2022-12-31 / 验证 2023 / 测试 ≥2024-01-01，**ANN≈260** |
| 交易规则 | point-in-time：R0 日盘末 17:14 / 夜 17:50–04:44；R1 17:14 / 17:50–05:14；R2 17:19 / 17:50–05:14；R3 17:19 / 17:35–05:14 |

**主结果**（预先登记）：loose 筛选 × expanding 训练窗 × **OLS + RFE 10**；
Ridge 只作稳健性，不进主结果。

---

## 6. 两个策略版本

包里同时有**两个版本**，报告里分开写、不混用：

| 版本 | 口径 | 定位 |
|---|---|---|
| `main_ols_exp` | 不平滑 | **预先登记的科学口径**。所有「不看测试集」的约束都建立在这个版本上。 |
| `ema_halflife_2` | 模型打分做递归 EMA 平滑，半衰期按 **2023 验证期**净 Sharpe 选出（2 日），尺度系数用训练段拟合 | **事后改进版**，看过测试集结果之后才加的。降换手方向有效，但仍是跑输买入持有。 |

`perf.py` 和 `update.py` 的输出里**两个版本并列**，性能表可直接对比。

改进版的参数（`config.yaml` 的 `smoothing` 段）：

```yaml
"chosen_halflife": 2.0,
"scale_k": 2.149587570637219,
```

这两个值由 `select_halflife.py` 按**验证期**规则选出后写回并冻结，回测（`backtest.py`）
与实盘（`update.py`）都只读这两个值，保证逐位一致。

**冻结日 = 2026-10-16**（`config.yaml` 的 `freeze` 段）。该日之后追加的数据，
对两个版本都是**真正没见过的样本**；该日及之前的数据都已在样本里
（测试段尤其如此，10-01 至 10-15 就是「已见样本」）。
绩效表的 `sample_status` 列会标出这一状态。

`update.py` 每次运行会把最近 30 天的 EMA 状态存进 `out/state_ema.json`，
并**逐条比对历史日期**——同一天重跑不再是「拿自己和自己比」，
中途改了冻结参数、代码或数据层会立刻报错。

---

## 7. 未来函数防线

`sgp_factors/test_truncation.py` 是这个包最重要的一个测试。它对随机日期 t
**把分钟数据截到 t 日 16:59:59**，重跑整条管线（日历 → 会话 → 日频 → 因子 → 筛选 →
RFE → 模型），然后逐位比对：

- 因子表 X 的所有 ≤ t 行（65 列）
- 标签 y 的所有 ≤ t−2 行
- 模型输出 `pos` / `yhat` / `z` / `n_train` 的所有 ≤ t 行
- EMA 版 `s_ema` / `pos_ema` 的所有 ≤ t 行

样本外日期（t > 训练段末日）**连同筛选与 RFE 一起重跑**，并断言入选特征不变。

```bash
python3 sgp_factors/test_truncation.py 20    # 20 个随机日期
```

---

## 8. 已知边界

- **单品种**：只做了 SGP，没做多品种组合。
- **合规缺口**：课程要求的若干项未覆盖（`config.yaml` 的 `deferred` 段列了
  `xgb_reg` / `xgb_clf` / `gp_F` / `window750` / `scheme2_run` / `cat_equal`）。
- **改进版的地位**：半衰期只用验证期选，但「用平滑」这个想法是看过测试集之后才有的，
  所以改进版**不是**一个干净的样本外结果，只作披露，不替换主结果。
- **多重检验**：项目试过 140 种方案，单看一个 p 值会高估显著性。
  报告 §5.5 用 Deflated Sharpe 做了校正——主结果 p = 0.514、改进版 p = 0.206，
  组合相对最好单因子的增量 p = 0.519，**均不显著**。
  登记册只能收录在 `out/` 里留了结果文件的方案；真实试验数可能更多，
  那只会让 p 值更大（「不显著」的结论更强）。
- 详见报告 §9「局限性」。

---

## 9. 文件清单核对

`MANIFEST.sha256` 列出包内每个文件的 SHA256。打包脚本 `package_submission.py`
在生成它之后会立刻自校验一次，zip 内中文文件名已正确设置 UTF-8 标志位。
