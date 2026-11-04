# QF5210 · SGP Daily Timing — Complete English Speech Script

| Item | Detail |
| --- | --- |
| Deck | `QF5210_SGP_Presentation.pptx` (9 slides) |
| Length | **7.5–8 minutes** speaking + Q&A |
| Report | `SGP_Strategy_Report_.docx` / `QF5210_SGP回测报告.md` |
| Caliber | Historical test cut at **2026-09-30** (n = 715) |

## Timing overview

| # | Slide | Time |
| ---: | --- | ---: |
| 1 | Cover | ~20s |
| 2 | Research question | ~45s |
| 3 | Data & decision clock | ~50s |
| 4 | Factor funnel | ~50s |
| 5 | Main results | ~55s |
| 6 | Turnover | ~50s |
| 7 | Improvements (EMA / κ) | ~55s |
| 8 | Rigor (α/β · DSR) | ~50s |
| 9 | Close | ~35s |
| | **Total** | **~7.5–8 min** |

## Key numbers (must cite)

| Metric | Value |
| --- | --- |
| Main · hist-test Sharpe | **+0.045** |
| Buy-and-hold · hist-test Sharpe | **+1.389** |
| EMA half-life 2 · hist-test Sharpe | **+0.554** |
| Partial rebalance κ = 0.4 · hist-test | **+0.751** |
| Gross / net Sharpe (hist) | **+0.474 / +0.045** |
| Ann. turnover (main → EMA → κ) | **106.3 → 41.3 → 19.3** |
| DSR p = 1 − DSR (main / EMA, full 165) | **0.918 / 0.703** |
| October actual test (main) | −2.881 (n = 21; not meaningful) |

> Obsolete: old “test Sharpe −0.021” (cut at 10-16).  
> Cite **1 − DSR**. Do **not** cite uncorrected 1 − PSR.

---

## Full script (ready to speak)

### Slide 1 — Cover (~20s)

Good afternoon. We are presenting our QF5210 project on **daily timing of SGP** — that is, SGX MSCI Singapore index futures.

I will start with the conclusion. **We do not find reliable out-of-sample timing skill.**

Please keep three numbers in mind for the **historical test** window through **September 30, 2026**. Our pre-registered main result has Sharpe **plus 0.045**. Simple buy-and-hold on the same window is **plus 1.389**. A post-hoc EMA variant reaches **plus 0.554**. So smoothing helps relative to the main result, but it still does not beat the market, and it is not a claim that the strategy is profitable.

---

### Slide 2 — Research question (~45s)

The research question is narrow: using **only** the futures contract’s own minute price and volume, can we time SGP out of sample?

Our sample runs from **June 15, 2020 to October 30, 2026** — about **1.61 million** one-minute bars and **1,660** trading days. We split the sample into train through 2022, validation in 2023, a **historical test from 2024 through September 2026**, and then **October 2026 as an actual tracking period** with only **21** return days. That October slice is a short record, not a statistical verdict.

Before we ran the historical test, we locked the main specification: a **loose** factor screen, an **expanding-window OLS**, and **RFE with a cap of ten**. The test set for that scientific caliber is run once. After parameter freeze, data from mid-October onward is true out-of-sample for both the main result and the EMA tracking version.

---

### Slide 3 — Data and decision clock (~50s)

Trading hours changed three times in sample. We therefore use **point-in-time** session regimes, labeled R0 through R3. The R2 day-session extension is marked as **data-inferred**, not as a verified exchange notice.

Each day we decide at **17:00**. Factor information stops at **16:59**. We assume fill at the day-session close. The label is the next close-to-close return. Training rows may only use history through **t minus 2**, because the t-minus-1 label is not yet known at decision time.

We also ran a hard look-ahead check: truncate the raw minute data to 16:59:59 and rerun the pipeline. That truncation test passes **20 out of 20**. Annualization uses a fixed **260** day count. Data quality: zero out-of-window bars, forty-four off-book prints excluded in R3, fifteen thin days, and one data gap.

---

### Slide 4 — Factor funnel (~50s)

We built **sixty-five** factors in five families — momentum, intraday structure, technicals, activity, and day-versus-night — all from own price-volume.

On the training segment only, we keep factors with positive walk-forward Sharpe, then greedily remove highly correlated names with absolute Spearman rho below **0.2**. That funnel goes **65 to 15 to 3**. Recursive feature elimination is allowed to keep up to ten variables, but with only three candidates it deletes nothing.

The three kept factors are **B4_RVDN**, downside realized volatility; **B3_SEG_NOON**, the midday segment return; and **E5_DAYNIGHT**, the day-minus-night return gap. If we raise the Sharpe hurdle to 0.8, **zero** factors survive. So the bottleneck is weak single-factor strength, not an undersized library.

---

### Slide 5 — Main results (~55s)

Here is the pre-registered main result.

On **train**, Sharpe is about **plus 0.92**. On **validation**, it turns negative at about **minus 0.86**. On the **historical test**, Sharpe is **plus 0.045**, CAGR about **zero**, against buy-and-hold at **plus 1.389** and CAGR **plus 25 percent**.

October’s actual test prints a large negative annualized Sharpe around **minus 2.9**, but that is only **21** days. The standard error is huge, so we treat it as a diary entry, not evidence of skill or failure.

One more warning: tiny implementation fixes flip the historical-test sign — the original defective pipeline is about **minus 0.13**, fixing only B7 gives about **plus 0.10**, and the fully fixed main result is **plus 0.045**. So we report a near-zero finding honestly. The older number **minus 0.021** used a different test cut and is **obsolete**.

---

### Slide 6 — Turnover (~50s)

Why is the net result so weak if there seems to be a gross edge?

On the historical test, **gross** Sharpe is about **plus 0.47**. Annualized turnover is **106 times**. After costs, **net** Sharpe is **plus 0.045**. Annualized cost drag is about **3.6 percent**.

Positions behave like noise trading: absolute standardized signal exceeds one on about **23 percent** of days; position autocorrelation is only **0.28**; we see about **104** sign flips per year.

So the failure mode is **turnover**, not “we picked the wrong three names.” The natural next step is to lower turnover, not to rebuild the factor list.

---

### Slide 7 — Improvements (~55s)

After seeing the test outcome, we disclose a post-hoc lower-turnover fix: a recursive **EMA** on the forecast score. Half-life is chosen on **2023 validation only**; the historical test is run once. The selected half-life is **two days**.

That EMA version has validation Sharpe about **plus 0.79**, historical-test Sharpe **plus 0.554**, CAGR about **plus 4.6 percent**, and turnover down to **41**. A circular-shift one-sided p-value is **0.033**, but that is an unadjusted single comparison. Year **2025** is still negative, and buy-and-hold is still far ahead.

On top of EMA we also tried **partial rebalancing** with kappa equal to **0.4**: each day we move only part of the way to the target position. Historical-test Sharpe rises to about **plus 0.75**, turnover to about **19**. That is a **second** post-hoc layer, and it was **not** frozen as the operational tracking version.

Neither improvement replaces the scientific main result of **plus 0.045**.

---

### Slide 8 — Rigor (~50s)

We stress-test the pretty numbers.

Regressing strategy returns on the market, EMA alpha has a t-statistic around **0.5** — insignificant. A “core long” blend that mixes in a constant long sleeve can look better on Sharpe, but about **80 percent** of that return is **beta**, not timing alpha. We do not adopt it.

For multiple testing we use Deflated Sharpe. With the full trial registry, adjusted p-values — that is **one minus DSR** — are **0.918** for the main result and **0.703** for EMA. Neither is significant. Please cite these, not the old uncorrected one-minus-PSR figures.

A roll-calendar shock that blanks labels only for estimation, then evaluates both strategies on the **same 715** historical-test days, changes the selected factors and jumps Sharpe to about **plus 0.61**. That shows instability at noise scale, not a free lunch. Across screening-window grids, even picking the best validation configuration still leaves a **negative** historical test.

---

### Slide 9 — Close (~35s)

Three takeaways.

**First**, on the historical test the main strategy’s Sharpe is near zero — there is no reliable timing skill from own price-volume alone.  
**Second**, the economic failure mode is turnover eating a weak gross edge.  
**Third**, lower turnover helps in post-hoc experiments, but those versions are not live-ready and still lose to buy-and-hold.

We submit and track **both** the pre-registered main result and the frozen EMA variant. The package is reproducible from a single `config.yaml` and checksums.

Thank you. We are happy to take questions.

---

## Optional Q&A

| Question | Suggested answer |
| --- | --- |
| Why did −0.021 become +0.045? | Historical test now ends **2026-09-30** (715 days). October is a separate actual test. The old cut is not comparable. |
| Which DSR numbers do we cite? | **0.918 / 0.703** (= 1 − DSR, full 165 trials). |
| Why not make EMA or κ the main result? | Both are post-hoc. Scientific caliber stays with the pre-registered OLS path at +0.045. |
| Is October a failure of the strategy? | Only 21 days; SE ≈ 3.55. It is a short tracking record, not a statistical claim. |
| Why no open interest or basis? | The minute file has no contract ID, OI, or settlement fields — left as future work. |
| Is buy-and-hold a fair benchmark? | Same instrument, same historical-test window — it is the opportunity-cost benchmark. |
