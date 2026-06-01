# Airport Risk Intelligence

**Reply × LUISS 2026 — Project 2 (Classical vs Multi-Agent)**
Team: Daniele Giovanardi · Filippo Nannucci · Edoardo Riva

---

## Table of contents

1. [Executive summary](#1-executive-summary)
2. [Project context](#2-project-context)
3. [Dataset](#3-dataset)
4. [Methodology](#4-methodology)
5. [Multi-agent architecture](#5-multi-agent-architecture)
6. [Results](#6-results)
7. [Design rationale](#7-design-rationale)
8. [Limitations](#8-limitations)
9. [Future work](#9-future-work)
10. [Repository structure](#10-repository-structure)
11. [How to reproduce](#11-how-to-reproduce)
12. [Reproducibility note on figures and tables](#12-reproducibility-note-on-figures-and-tables)

---

## 1 — Executive summary

This project answers a single question that the Reply brief poses to the team: **does a multi-agent orchestration based on LangGraph deliver enough operational value to justify its complexity, compared to a classical sequential anomaly-detection pipeline?** We answer the question empirically: we implement the *same* anomaly-detection logic twice — once as a sequential script and once as a five-agent LangGraph DAG — and measure how much they actually disagree on a real Italian border-control dataset of 567 airport routes.

The headline numbers, all reproduced by `notebooks/08_report_assets.ipynb` from the canonical processed CSVs:

| Metric | Value |
|---|---|
| Routes analysed | 567 |
| Anomaly-label distribution (both pipelines) | **17 HIGH / 40 MEDIUM / 510 NORMAL** |
| Final-risk distribution (both pipelines) | **9 CRITICAL / 29 HIGH / 19 MEDIUM / 510 LOW** |
| Per-route label agreement (`anomaly_label`) | **100.00 %** (567 / 567) |
| Per-route label agreement (`final_risk`)    | **100.00 %** (567 / 567) |
| Pearson r on the final scalar score | **0.999987** |
| Spearman ρ on the final scalar score | **0.999974** |
| Business-rule hits (all 5 rules) | **Identical, delta = 0** (r = 1.000 by construction) |
| Top-50 anomalous routes overlap | **49 / 50** |
| Top-10 anomalous routes overlap | **10 / 10** |
| Ensemble weight choice | **Data-driven** — winner of a 4-simplex grid search (Section 4.4.1) |
| Threshold-sensitivity max swing on CRITICAL + HIGH (±10 %) | **2.6 %** (1 route) |

The two architectures converge on the **exact same answer**: bit-for-bit identical anomaly and final-risk labels on all 567 routes, residual Pearson r = 0.999987 only because of float-precision differences in reduction order between the two pipelines. The *real* difference between them is operational, not statistical: the multi-agent system buys an LLM-narrated route-by-route explanation, dynamic perimeter filtering, per-agent failure isolation, and a non-trivial feedback cycle to widen the search when a verifier disagrees with the first pass — none of which fall out for free from the sequential script.

The 100 % convergence is a code-level guarantee, not a coincidence. The Autoencoder — historically the only stochastic component — is trained by a single shared module (`shared/autoencoder.py`) that both pipelines call with a deterministically sorted input; early stopping is off so convergence depends only on `(data, random_state)`. Combined with the already-deterministic IF / LOF / Z components and the **canonical, threshold-aligned business-rule layer**, the verdict is identical end-to-end.

---

## 2 — Project context

### 2.1 The brief

Reply assigned the **NoiPA validation use case** as the framing for Project 2 of the *Backpropagation Bandits* LUISS course. NoiPA is the digital platform of the Italian Ministry of Economy and Finance (MEF) that ingests heterogeneous datasets from third-party authorities and validates them automatically. The deliverable asks for an anomaly-detection system that *could* be plugged into NoiPA — accepting heterogeneous tabular records, producing a route-level risk classification and a human-readable explanation per HIGH/MEDIUM route, and surfacing the results to an analyst.

The dataset Reply provided is **not** NoiPA itself: it is a sample of border-control passenger transits at Italian airports, used as the canonical example of the kind of heterogeneous third-party dataset that NoiPA would receive. The 567 origin/destination route pairs sit at the operational frontier of border control — they are exactly the routes a customs officer would care about.

### 2.2 What we built

Two implementations of the same detection logic, sharing the same preprocessing module, the same `FeatureBuilder` (54 numerical features per route), the same MAD-based baseline, the same four-model ensemble, the same five business rules and the same ensemble weights:

1. **Classical pipeline.** Seven sequential steps — EDA, preprocessing, feature engineering, baseline, ensemble, post-processing, evaluation. Inlined in `main.ipynb` so a reviewer can read it top-to-bottom without leaving the notebook. A standalone `classical_pipeline/main.py` orchestrator is also available for batch runs.
2. **Multi-agent pipeline.** A LangGraph DAG with five specialised agents — `DataAgent`, `BaselineAgent`, `OutlierAgent`, `RiskProfilingAgent`, `ReportAgent` — plus a `SupervisorAgent` verifier wired into the graph as a conditional branch with a bounded feedback cycle. Lives in `multiagent_pipeline/` and is imported by Sections 8–13 of `main.ipynb`.

### 2.3 Alignment guarantee

Both pipelines apply the **same five canonical business rules** with the **same thresholds**, and the labels they produce share the same **English vocabulary** (`HIGH / MEDIUM / NORMAL` for the ensemble layer, `CRITICAL / HIGH / MEDIUM / LOW` for the post-rule final classification). The previous Italian/English label drift between the two implementations was eliminated; the BR hit counts now match exactly on every route (zero delta on all five rules), and the residual disagreement on `anomaly_label` is concentrated entirely on the boundary between MEDIUM and NORMAL — i.e. on routes where the stochastic Autoencoder draws a slightly different decision surface from one fit to the next.

---

## 3 — Dataset

### 3.1 Raw layer

Reply provides two CSV files at the bottom of `data/raw/` (NDA-protected, **not** redistributed in this repository):

| File | Granularity | Rows | Key columns |
|---|---|---|---|
| `ALLARMI.csv` | one row per alarm (Interpol, SDI, NSIS) generated by a border control | 5 080 | route, year-month, motive, outcome (`chiuso` / `respinto` / `fermato` / `segnalato`) |
| `TIPOLOGIA_VIAGGIATORE.csv` | one row per traveller-profile transit on a route-month | 5 095 | nationality, route, year-month, total entered, alarmed, investigated |

### 3.2 Temporal coverage

The panel spans **three months only**: December 2023, January 2024, and February 2024. This is the single most consequential constraint on the methodology — it rules out STL-style decomposition (which needs ≥ 12 observations per series) and forces the temporal model to be a per-route linear-trend slope on the routes that appear in at least two months.

![Monthly volume](images/dataset_monthly_volume.png)

*Figure 1 — Monthly volume of raw records. The dataset spans Dec-2023 to Feb-2024 with one off-cycle record carrying a `MESE_PARTENZA = 12` value (treated as Dec 2024 by the cleaner; ignored by the temporal-trend module because it has no companion observations).*

### 3.3 Geographic coverage

Routes terminate or originate at Italian airports (FCO, MXP, LIN, BLQ, NAP, …). The 15 most active departure countries by passenger volume:

![Top countries](images/dataset_top_countries.png)

*Figure 2 — Top 15 departure countries by passenger volume entered into Italian territory.*

### 3.4 Unit of analysis

After cleaning and aggregating monthly, the unit of analysis is a **route**, i.e. an `airport_departure -> airport_arrival` pair (e.g. `CMN-FCO` for Casablanca → Rome Fiumicino). The cleaned panel contains **567 unique routes**, each described by **54 numerical features**. The full feature roster is documented in `data/processed/feature_cols.json`; the thirteen features used as inputs to the baseline are the ones reported in section 4.3 below.

---

## 4 — Methodology

The methodology splits naturally into six layers, identical on both pipelines and implemented through the same shared modules.

### 4.1 Preprocessing

`shared/preprocessing.py` performs cleaning and merging in one pass:

* **Date parsing** — `DATA_PARTENZA` is normalised to UTC; rows with unparseable timestamps are dropped.
* **Country-code normalisation** — ISO-2 codes are mapped to ISO-3 via an embedded lookup table; departure-country labels are stripped of stray casing and whitespace.
* **Gender normalisation** — the raw `GENERE` field carries inconsistent encodings (`M`, `m`, `M.`, `Maschio`); all are collapsed to `M` / `F` / `OTHER`.
* **Sparse-column drop** — columns with > 95 % nulls are dropped before the route-level merge.
* **Route-level merge** — alarms and traveller records are aggregated on `(AREOPORTO_PARTENZA, AREOPORTO_ARRIVO)`; the resulting `dataset_merged.csv` is the input of every downstream layer.

### 4.2 Feature engineering

`multiagent_pipeline/src/features.py` builds **54 numerical features per route**. They fall into four families:

| Family | Examples | Purpose |
|---|---|---|
| Volume | `tot_allarmi_sum`, `tot_allarmi_log`, `tot_entrati`, `tot_investigati` | Magnitude of the operational signal |
| Composition | `pct_interpol`, `pct_sdi`, `pct_nsis` | Which intelligence database the alarms come from |
| Rates | `tasso_chiusura`, `tasso_respinti`, `tasso_fermati`, `tasso_allarme_medio` | Outcome-side density |
| Stability | `false_positive_rate`, `alarm_per_invest` | Quality of the investigation pipeline |

### 4.3 Robust baseline

`BaselineAgent` (and the equivalent classical step) computes a robust z-score per feature using the **Median Absolute Deviation (MAD)**:

```
z_i = (x_i − median(X)) / (1.4826 · MAD(X))
```

with a fall-back to the standard deviation when MAD = 0 (sparse features where more than 50 % of the values equal the median would otherwise yield silently-zero z-scores). The robust scaling factor 1.4826 makes the MAD a consistent estimator of σ under a normal model — a standard choice that survives the heavy-tailed nature of border-alarm rates much better than the empirical standard deviation does.

The 13 features that participate in the baseline are:

`tot_allarmi_log`, `pct_interpol`, `pct_sdi`, `pct_nsis`, `tasso_chiusura`, `tasso_rilevanza`, `tasso_allarme_medio`, `tasso_inv_medio`, `score_rischio_esiti`, `tasso_respinti`, `tasso_fermati`, `false_positive_rate`, `alarm_per_invest`.

The composite `baseline_score` is the **mean of the absolute z-scores** across the 13 features. The visual evidence that MAD is a sounder choice than the standard deviation on this panel:

![MAD vs std](images/baseline_mad_vs_std.png)

*Figure 3 — Per-feature scale comparison: 1.4826·MAD vs standard deviation. On `tot_allarmi_log` and the rate features the standard deviation is inflated by the heavy tail, so the MAD-based scaling produces a tighter and more useful baseline.*

### 4.4 Ensemble anomaly detection

`OutlierAgent` (and the equivalent classical step) trains four independent detectors on the 13-feature matrix, normalises each output to [0, 1], and blends them into a single `ensemble_score`. The four weights below are **data-driven** — they are the winner of a grid search over the 4-simplex (Section 4.4.1) and supersede the original principled defaults that we initially borrowed from the literature.

| Detector | Weight | Hyper-parameters | Implementation |
|---|---|---|---|
| Isolation Forest | **0.40** | `contamination = 0.10`, `n_estimators = 200`, `random_state = 42` | `sklearn.ensemble.IsolationForest` |
| Local Outlier Factor | **0.15** | `n_neighbors = 20`, `contamination = 0.10` | `sklearn.neighbors.LocalOutlierFactor` |
| Z-score (MAD) | **0.30** | consumes `baseline_score` directly | shared module |
| Autoencoder (MLP) | **0.15** | architecture `13 → 8 → 4 → 8 → 13`, trained on the 510 normal routes via semi-supervision; deterministic single-module implementation (`shared/autoencoder.py`) — `early_stopping = False`, sort by route id, no per-run variability | `sklearn.neural_network.MLPRegressor` |

The ensemble degrades gracefully: with fewer than 30 normal routes available, the Autoencoder is excluded from the blend and its 0.15 weight is redistributed proportionally over IsolationForest, LOF and the Z-score.

#### 4.4.1 Data-driven weight choice — ablation and grid search

A reviewer can rightly ask whether the four weights are themselves principled or just plausible-looking defaults. We answer that question with two complementary analyses, both reproduced by `notebooks/08_report_assets.ipynb`.

**Ablation study** (`multiagent_pipeline/src/ensemble_ablation.py`). We drop one detector at a time, renormalise the remaining weights to sum to one, and compare the resulting top-17 HIGH set against the full ensemble. The results on the post-AE-alignment population:

| Subset | Top-17 overlap vs full | Business-rule rank correlation |
|---|---|---|
| IF only         | 0.765 | 0.580 |
| LOF only        | 0.000 | 0.200 |
| Z only          | 0.471 | 0.587 |
| AE only         | 0.471 | 0.356 |
| IF + LOF + Z    | 0.824 | 0.579 |
| IF + LOF + AE   | 0.824 | 0.520 |
| IF + Z + AE     | **1.000** | **0.558** |
| LOF + Z + AE    | 0.706 | 0.500 |
| IF + LOF + Z + AE (full) | 1.000 | 0.550 |

The single most informative reading: **dropping LOF leaves the top-17 unchanged** and slightly *improves* the operational alignment (rank correlation with `br_score`: 0.558 vs 0.550 for the full ensemble). LOF contributes mostly redundancy with the IF density signal — this is exactly the empirical justification for cutting its weight from 0.30 to 0.15 in the production blend. Dropping AE drops the top-17 overlap to 0.824 and the BR rank correlation to 0.520: AE earns its (lower) place by picking up non-linear feature combinations that the other three miss.

![Ensemble ablation](images/ensemble_ablation.png)

*Figure 5b — Ablation result. Blue bars show top-17 overlap with the full ensemble; orange bars show the Spearman correlation between the ensemble score and the business-rule score (operational alignment proxy).*

**Grid search** (`multiagent_pipeline/src/ensemble_grid_search.py`). We enumerate the 4-simplex of weight vectors at step 0.05 (all four weights strictly positive, summing to one — about 969 vectors) and score every vector by

```
objective = 0.5 · bootstrap_stability(top-17, 80% subsample) + 0.5 · ((BR_rank_corr + 1) / 2)
```

The objective rewards weight vectors whose top-17 HIGH set survives bootstrap resampling **and** rank-correlates with the business-rule score. Each half acts as a sanity check on the other.

| Weight vector | IF | LOF | Z | AE | Stability | BR rank corr | Objective |
|---|---|---|---|---|---|---|---|
| Original principled defaults | 0.35 | 0.30 | 0.15 | 0.20 | 0.797 | 0.499 | 0.773 |
| **Grid-search winner (current production)** | **0.40** | **0.15** | **0.30** | **0.15** | **0.833** | **0.550** | **0.804** |
| Gap | +0.05 | −0.15 | +0.15 | −0.05 | +0.036 | +0.051 | **+0.031 (+4.0 %)** |

The story is consistent with the ablation: IF stays the heaviest weight (+0.05); LOF is halved (−0.15) for the redundancy reason above; Z is doubled (+0.15) because it carries the highest individual BR rank correlation among the four; AE is trimmed (−0.05) but retained for non-linear coverage.

![Grid search heatmap](images/ensemble_grid_search_heatmap.png)

*Figure 5c — Marginal heat-map of the grid-search objective over (w_IF, w_Z), taking the max across w_LOF and w_AE. The black star marks the current production weights (= grid-search winner); the red ✕ would mark a different winner if one existed, but lies on top of the star. The objective surface is smooth around the winner, which is itself a robustness signal: small perturbations of the weights do not break the verdict.*

**Honest caveat we want to flag.** The Z component is built on MAD z-scores of the same 13 features the business rules look at, so a high `Z ↔ br_score` correlation is *partly* mechanical. We chose this objective deliberately because the operational alignment matters for the brief, but we do not claim the grid optimum is the One True Weighting — it is *the most defensible weighting given a stated, transparent objective*, which is the strongest claim available in an unsupervised setting.

The 567 routes split into three buckets at **data-driven thresholds** — the p97 of the ensemble score is the boundary between HIGH and MEDIUM, the p90 the boundary between MEDIUM and NORMAL:

![Ensemble distribution](images/ensemble_score_distribution.png)

*Figure 4 — Distribution of the ensemble anomaly score across 567 routes, with the data-driven p97 (HIGH) and p90 (MEDIUM) thresholds.*

The four detectors are not perfectly correlated — that is the whole point of blending them. The pairwise correlation between the individual model scores and the final ensemble illustrates that each detector contributes a distinct slice of the variance:

![Per-model correlation](images/ensemble_per_model_corr.png)

*Figure 5 — Pairwise correlation among the four detector scores and the final ensemble. The Autoencoder picks up non-linear feature combinations that IF, LOF, and Z miss; the residual correlation among IF/LOF/Z is high because they all rely on density in the same scaled feature space.*

### 4.5 Business rules

The post-processing layer applies **five canonical business rules** equivalent to the operational checks a customs officer would mentally run on a flagged route. Both pipelines apply them with identical thresholds; we audited the previous notebook (which carried a different rule set inline) and aligned every threshold to the canonical multi-agent definition.

| # | Rule id | Condition | Operational interpretation |
|---|---|---|---|
| 1 | `br_high_interpol` | `pct_interpol ≥ 0.30` | INTERPOL alarms dominate the route |
| 2 | `br_high_rejection` | `tasso_respinti ≥ 0.25` | Above-average traveller rejection at the border |
| 3 | `br_low_closure` | `tot_allarmi_log > 3` **and** `tasso_chiusura < 0.10` | Operational backlog: high alarm volume with low closure rate |
| 4 | `br_multi_source` | `pct_interpol ≥ 0.10` **and** `pct_sdi ≥ 0.10` | Multi-database corroboration (route shows up significantly in two distinct intelligence sources) |
| 5 | `br_high_alarm_rate` | `tasso_allarme_medio ≥ 0.50` | One traveller in two on this route triggers an alarm |

Each rule is binary. `br_score = mean(br_*) ∈ [0, 1]` is the aggregate.

**Note on `br_multi_source`.** The earlier implementation used a `pct_interpol > 0 AND pct_sdi > 0` rule. We reviewed the firing rate on the population and tightened it to `≥ 0.10` on both channels: under the old rule the BR fired on essentially any route where the two databases had a trace presence, which is not the operational signal we want to surface. The tighter rule fires on 152 routes (26.8 % of the population) — still material but now corresponding to real multi-source corroboration.

![Business rule hits](images/business_rule_hits.png)

*Figure 6 — Business-rule hit frequency on the 567-route population. Both pipelines produce identical counts on every rule (zero delta), so `br_score` Pearson r between the two pipelines is exactly 1.000 by construction.*

### 4.6 Final risk classification

`RiskProfilingAgent` (and the equivalent classical step) collapses the ML signal and the rule signal into a single ordinal label `final_risk ∈ {CRITICAL, HIGH, MEDIUM, LOW}`:

```
CRITICAL : anomaly_label == HIGH   AND br_score ≥ 0.4
HIGH     : anomaly_label == HIGH   OR  (anomaly_label == MEDIUM AND br_score ≥ 0.4)
MEDIUM   : anomaly_label == MEDIUM
LOW      : otherwise
```

The 0.4 boundary on `br_score` corresponds to "**at least two of the five rules fired**". A blended `confidence` score is also produced for ranking inside each bucket:

```
confidence = 0.60 · ensemble_score + 0.40 · br_score
```

The 60 / 40 ML / rules split reflects the brief: the ML signal carries more weight because it is statistically validated; the rules act as an interpretable amplifier.

---

## 5 — Multi-agent architecture

### 5.1 The five agents

The graph respects the Reply specification of five visible agents. The `SupervisorAgent` is a verifier wired in as a conditional branch — it does not count toward the spec headcount.

| # | Agent | Responsibility |
|---|---|---|
| 1 | `DataAgent` | Loads `ALLARMI.csv` + `TIPOLOGIA_VIAGGIATORE.csv`, applies the user-defined perimeter, and engineers the 54 numerical features per route via `FeatureBuilder`. |
| 2 | `BaselineAgent` | Computes robust MAD z-scores per baseline feature and aggregates them into a single `baseline_score` consumed downstream as the Z-component of the ensemble. |
| 3 | `OutlierAgent` | Trains the four-model weighted ensemble (IF + LOF + Z + AE) and produces `ensemble_score` and `anomaly_label` (HIGH / MEDIUM / NORMAL). |
| 4 | `RiskProfilingAgent` | Applies the five canonical business rules, computes `br_score`, blends ML and rules into `confidence`, and assigns `final_risk` (CRITICAL / HIGH / MEDIUM / LOW). Produces a per-route `risk_drivers` list of textual reason codes consumed by the LLM downstream. |
| 5 | `ReportAgent` (LLM) | Generates a Claude-authored natural-language explanation for each HIGH/MEDIUM route, weaving together the top z-score drivers from the BaselineAgent and the business rules that actually fired. |
| ★ | `SupervisorAgent` *(verifier, optional)* | Re-fits Isolation Forest at `contamination = 0.03` on the full population and tags first-pass HIGH routes as `robust_high = True` only if they survive the stricter rule. |

### 5.2 The DAG topology

The graph carries **four data-driven conditional edges** on top of the standard error-stop logic:

1. **after_baseline** — terminate early when the baseline signal is degenerate (fewer than five features available or `baseline_score` standard deviation below 0.01); the pipeline returns with a clear empty-output diagnostic.
2. **after_outlier** — route through `SupervisorAgent` only when the first pass produces ≥ 5 HIGH labels; otherwise short-circuit to the rule layer (refitting Isolation Forest on a tiny subset would be statistically meaningless).
3. **after_supervisor** — **cycle back to `OutlierAgent`** when the verifier downgrades more than 50 % of the first-pass HIGH labels, capped at two iterations to guarantee termination. This is the one place where the topology is genuinely non-linear.
4. **after_risk** — skip the LLM `ReportAgent` when there are no HIGH/MEDIUM routes worth narrating, saving API cost on quiet perimeters.

### 5.3 Where the multi-agent topology earns its weight

Three things drop out of the LangGraph design that would not survive a flat sequential script:

* **Per-agent failure isolation.** If `BaselineAgent` fails on a degenerate perimeter, the orchestrator still returns a meaningful partial state with a `baseline_meta.error` field and a human-readable `user_message`; the Streamlit dashboard renders the partial output and tells the analyst what to fix.
* **The supervisor → outlier cycle.** The verifier disagreeing with more than 50 % of the first-pass HIGH labels is a genuine signal that the contamination heuristic mis-calibrated the threshold for the current perimeter. A flat script cannot widen the search reactively; the multi-agent graph can.
* **Dynamic perimeter filtering.** `DataAgent` accepts a runtime perimeter dict (year, country, airport, zone) and the rest of the graph adapts. The Streamlit interactive dashboard at `streamlit_app/app.py` makes use of exactly this affordance to let an analyst restrict the analysis on the fly.

---

## 6 — Results

This section reproduces every claim in the executive summary and adds the residual diagnostics that justify it. Every figure and every table is generated deterministically from the canonical processed CSVs by `notebooks/08_report_assets.ipynb` (see §12 for the asset-reproducibility note).

### 6.1 Distribution convergence

Both pipelines produce **identical anomaly-label distributions** on the 567 routes — and after the AE alignment fix described in Section 4.4, also identical post-rule final-risk distributions:

| Label | Classical | Multi-agent |
|---|---|---|
| HIGH | 17 | 17 |
| MEDIUM | 40 | 40 |
| NORMAL | 510 | 510 |

| Final risk | Classical | Multi-agent |
|---|---|---|
| CRITICAL | 9 | 9 |
| HIGH | 29 | 29 |
| MEDIUM | 19 | 19 |
| LOW | 510 | 510 |

![Risk-label distribution](images/risk_label_distribution.png)

*Figure 7 — Side-by-side comparison of label distributions. The two pipelines produce identical splits at both the ensemble layer (HIGH/MEDIUM/NORMAL) and the post-rule layer (CRITICAL/HIGH/MEDIUM/LOW) — this is a code-level guarantee enforced by the shared AE module and the canonical business-rule layer.*

### 6.2 Top anomalous routes

The 15 routes with the highest ensemble score (multi-agent pipeline). All 15 sit above the p97 threshold and are labelled HIGH. Casablanca → Bologna (`CMN-BLQ`) leads by a clear margin at score 0.813 — the maximum observable on this dataset.

![Top routes](images/top_routes_anomaly_score.png)

*Figure 8 — Top 15 routes by ensemble anomaly score on the multi-agent pipeline. Colour encodes the anomaly_label.*

### 6.3 Per-route agreement

The two pipelines agree on **567 of 567 anomaly labels (100.00 %)** and on **567 of 567 final-risk labels (100.00 %)**. The confusion matrix is therefore strictly diagonal.

![Confusion matrix](images/anomaly_label_confusion_matrix.png)

*Figure 9 — Confusion matrix on anomaly_label (rows = classical, columns = multi-agent). All 567 routes sit on the diagonal — the off-diagonal entries are all zero, the formal expression of the 100 % agreement.*

### 6.4 Score correlation

The Pearson correlation between the two pipelines' final scalar scores is **0.999987**; the Spearman rank correlation is **0.999974**. The residual gap from 1.000000 is purely float-precision noise from differences in reduction order between the two pipelines (each computes the same weighted sum but iterates the arrays in marginally different memory layouts). No route's label depends on it.

![Score correlation](images/score_correlation_classical_vs_multiagent.png)

*Figure 10 — Per-route score correlation. Each point is a route, coloured by the multi-agent anomaly label. Dashed line: y = x. The vast majority of routes sit on the diagonal; the only meaningful spread lives in the NORMAL band, where small score perturbations matter little because the operational decision is unchanged.*

### 6.5 Business-rule alignment

The five canonical rules produce **identical hit counts on every rule** (delta = 0, see §4.5). This is the alignment guarantee that justifies the comparative analysis: any disagreement between the two pipelines is mechanically isolated to the ML ensemble, not introduced by drifting business rules.

### 6.6 Bootstrap CI on the agreement metric

To size the agreement metric properly we resample the merged 567-route DataFrame 1 000 times at 80 % subsample, recompute the row-level agreement on every resample, and report the 95 % percentile interval.

![Bootstrap CI](images/bootstrap_ci_agreement.png)

*Figure 11 — Bootstrap distribution of the per-route agreement. After the AE alignment fix the agreement is 100 % on every resample, so the distribution collapses to a single point and the 95 % CI degenerates to `[100 %, 100 %]`. This is what convergence-by-construction looks like; the chart is kept in the report as a visual marker of the difference between the pre-fix (98.24 % with CI [97.79 %, 98.90 %]) and post-fix regimes.*

### 6.7 Threshold sensitivity

We perturb each of the five BR thresholds independently by ±5 %, ±10 % and recompute the final-risk count. The dataset is structurally robust: only three thresholds (`high_alarm_rate`, `high_rejection_rate`, `multi_source_pct`) move the count of CRITICAL + HIGH routes at all, and at most by a single route (2.6 % swing relative to the 38-route baseline).

![Threshold sensitivity](images/threshold_sensitivity.png)

*Figure 12 — Sensitivity of the HIGH and CRITICAL counts to ±10 % / ±5 % perturbations of the five BR thresholds. Cells report the number of routes flagged at that level under each perturbation.*

### 6.8 Feature importance

A surrogate Gradient Boosting classifier trained to predict the ensemble flag from the 13 baseline features surfaces the drivers a customs operator would expect to see at the top: total alarm volume, average alarm rate, and the score on the outcome side (`score_rischio_esiti`).

![Feature importance and SHAP](images/feature_importance_shap.png)

*Figure 13 — Surrogate feature importance (left) and mean SHAP value (right) — top 10 features. The SHAP values are computed against the surrogate model and serve as an interpretability hint, not as a faithful explanation of the ensemble itself.*

### 6.9 Temporal trend

With only three months of data we cannot fit STL. As the closest proxy supported by the panel, we compute the linear-trend slope of `n_alarmed` per route on the subset of routes that appear in at least two months.

![Temporal trend](images/temporal_trend_distribution.png)

*Figure 14 — Distribution of per-route linear-trend slopes on `n_alarmed`. The slope is signed: positive means the route is producing more alarms over the panel, negative means it is producing fewer. The distribution is symmetric around zero, with most routes stable and a small set of rising and declining outliers.*

---

## 7 — Design rationale

Three choices look like deviations from the brief but were the right call given the data we had.

**On STL vs robust z-scores.** The brief mentions *"historical baseline using rolling averages and seasonal decomposition"*. We tried it and stepped back: the dataset spans only three months in total, so STL — which needs at least 12 observations per series — cannot run, and a rolling 3-month mean degenerates to the cross-sectional mean we already compute. Robust z-scores against the population are mathematically sounder for this sample size; we still added a per-route linear-trend slope (Section 14 of the report-assets notebook) to capture the temporal signal the dataset can actually support, on the subset of routes that appear in at least two months.

**On the four-model ensemble.** The brief lists *"IsolationForest, LOF or Z-score"*. We blend all four with a 20 % Autoencoder weight because the AE catches non-linear feature combinations that the three density-based detectors miss. The ensemble also degrades gracefully on small perimeters: below 30 normal samples the AE is excluded and the remaining weights are renormalised, so even narrow filters produce a coherent score.

**On the agent count.** The Reply slide specifies five agents — `DataAgent`, `BaselineAgent`, `OutlierAgent`, `RiskProfilingAgent`, `ReportAgent`. We respect the count exactly. We initially built feature engineering as its own agent and ended up with six boxes; merging `FeatureBuilder` into `DataAgent` removed orchestration overhead without changing the topology a reviewer sees. The `SupervisorAgent` is presented as a verifier rather than a sixth mandatory agent — its existence is a design strength but its absence would not invalidate the architecture.

---

## 8 — Limitations

1. **Single dataset.** The entire evaluation runs on a single Reply-provided dataset. We have not stress-tested either pipeline on a different schema, although `DataAgent` carries an LLM schema-normalisation layer that has not had to fire on this dataset because the canonical columns are all present.
2. **Three-month panel.** As discussed in §7, the temporal model we added is the most the panel can support; a longer panel would unlock STL and rolling means without changing the rest of the pipeline.
3. **LLM narratives are not programmatically validated.** The `ReportAgent` prompt forbids hallucination and is reviewed in spot checks, but we do not prove zero hallucination automatically. The prompt instruments enough structured context (top-3 z-score drivers, fired rules, ensemble score) that gross hallucinations are easy to spot in review, but not impossible to miss.
4. **Autoencoder determinism — historical note.** An earlier iteration of the project relied on `MLPRegressor(..., early_stopping=True)`; the validation split was data-order-dependent and produced run-to-run variability on a handful of MEDIUM ↔ NORMAL boundary routes (about 1.8 % of the population). The current code routes both pipelines through `shared/autoencoder.py`, which sorts the input by route id, disables early stopping, and trains for a fixed `max_iter`. The AE is now fully deterministic — the residual ~10⁻⁵ Pearson gap is float-precision noise, not algorithmic instability.
5. **No live data.** The Streamlit dashboard runs on the same processed CSVs as the analysis; there is no production ingestion pipeline.

---

## 9 — Future work

Three concrete extensions stand out.

A **`TrendAgent`** as a sixth optional node would extend the linear slope to STL once panels become long enough; the rest of the graph already supports an optional-agent wiring pattern (see how `ReportAgent` is conditionally added).

The **supervisor → outlier feedback cycle** currently widens the search only when the verifier disagrees with first-pass HIGH labels. A richer version could re-run on borderline MEDIUM routes too, which is where most of the residual disagreement actually lives.

A **multi-locale `ReportAgent`** would expose the narrative language as a runtime parameter so an Italian-speaking operator gets Italian narratives without prompt-hacking. The current prompt is hard-coded to English; localising it is a one-line change but the test surface grows.

---

## 10 — Repository structure

```
.
├── README.md                       This file
├── main.ipynb                      Single-notebook tour of the project
├── Oral_presentation_replay.pptx   Oral defence slides
├── requirements.txt
├── .env.example                    ANTHROPIC_API_KEY template
├── images/                         All PNG figures + tables/ CSV summaries
│   ├── *.png                       (generated by notebooks/08_report_assets.ipynb)
│   └── tables/*.csv
├── notebooks/
│   └── 08_report_assets.ipynb      Reproduces every figure and table in this README
├── shared/
│   ├── preprocessing.py            Cleaning + merge layer used by both pipelines
│   └── autoencoder.py              Deterministic AE — single source of truth
├── multiagent_pipeline/            LangGraph library
│   ├── main.py                     run_pipeline — graph orchestrator
│   ├── state.py                    AgentState schema + shared constants
│   ├── config.py                   API key + model config
│   ├── agents/
│   │   ├── data_agent.py
│   │   ├── baseline_agent.py
│   │   ├── outlier_agent.py
│   │   ├── supervisor_agent.py
│   │   ├── risk_profiling_agent.py
│   │   └── report_agent.py
│   ├── src/
│   │   ├── features.py
│   │   ├── bootstrap_ci.py
│   │   ├── threshold_sensitivity.py
│   │   ├── trend_analysis.py
│   │   ├── ensemble_ablation.py    Drop-one-detector study
│   │   └── ensemble_grid_search.py Data-driven ensemble weight selection
│   ├── tests/
│   │   ├── test_risk_profiling_agent.py   # 13 unit tests
│   │   └── e2e_validation.py
│   └── tools/
│       └── data_tools.py
└── streamlit_app/                  Interactive dashboard (optional)
    └── app.py
```

The classical pipeline is **inlined inside `main.ipynb`** so a reviewer can read the full implementation top-to-bottom without leaving the notebook. The multi-agent pipeline lives as a Python library because re-implementing the LangGraph DAG inline would erase the agent modularity that makes the orchestration meaningful.

---

## 11 — How to reproduce

### 11.1 Requirements

* Python ≥ 3.10
* The two raw CSVs provided by Reply under NDA — `data/raw/ALLARMI.csv` and `data/raw/TIPOLOGIA_VIAGGIATORE.csv`. They are **not** redistributed in this repository.
* Optionally, an Anthropic API key (only if you want the LLM narratives in §8 of the notebook).

### 11.2 Setup

```bash
git clone https://github.com/DanieleGiovanardi2408/BackPropBandits-815601-.git
cd BackPropBandits-815601-

python -m venv venv
source venv/bin/activate          # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Then drop the two NDA-protected CSVs into `data/raw/`:

```
data/raw/
├── ALLARMI.csv
└── TIPOLOGIA_VIAGGIATORE.csv
```

### 11.3 Optional LLM narratives

```bash
cp .env.example .env
# Edit .env and add: ANTHROPIC_API_KEY=sk-ant-...
```

Without a key, the report agent automatically falls back to a deterministic dry-run mode that emits template narratives and skips the API calls. All numerical results are unaffected.

### 11.4 End-to-end run

```bash
PYTHONPATH=. jupyter lab main.ipynb
```

then `Run All`. The notebook is structured in **thirteen sections** that follow the actual workflow:

| Section | Topic |
|---|---|
| 1 | EDA |
| 2 | Preprocessing |
| 3 | Feature engineering |
| 4 | Baseline construction |
| 5 | Anomaly detection (ensemble) |
| 6 | Post-processing (the five canonical BR + final_risk) |
| 7 | Evaluation (silhouette, stability, SHAP) |
| 8 | Multi-agent pipeline (LangGraph run) |
| 9 | Comparative analysis (classical vs multi-agent) |
| 10 | Bootstrap CI |
| 11 | Threshold sensitivity |
| 12 | Temporal coverage |
| 13 | Conclusions |

End-to-end runtime on the 2024 perimeter (567 routes):

* without the LLM → ~ 2 minutes
* with the LLM    → ~ 7 minutes (Claude generates one narrative per HIGH/MEDIUM route, ~ 57 calls)

### 11.5 Unit tests

```bash
PYTHONPATH=. python -m pytest multiagent_pipeline/tests/test_risk_profiling_agent.py -v
```

13 unit tests cover the five business rules (one per rule, plus one verifying the `br_multi_source` floor), the `br_score` aggregation, the confidence-blend formula, and every cell of the final-risk classification ladder.

---

## 12 — Reproducibility note on figures and tables

**No figure or table in this README was generated by an AI tool.** Every PNG referenced above and every value cited in §6 is produced deterministically by `notebooks/08_report_assets.ipynb`, which reads the canonical CSVs in `data/processed/` and writes both the PNGs (to `images/`) and the underlying numeric summaries (to `images/tables/`). To regenerate the assets, after running the two pipelines:

```bash
PYTHONPATH=. jupyter nbconvert --to notebook --execute notebooks/08_report_assets.ipynb \
    --output 08_report_assets.ipynb
```

The asset inventory is printed at the end of the notebook. The CSVs in `images/tables/` are the single source of truth for every number cited in this README — if you want to audit a claim, the corresponding CSV is the place to look.

---

*Reply × LUISS 2026 — Daniele Giovanardi · Filippo Nannucci · Edoardo Riva*
