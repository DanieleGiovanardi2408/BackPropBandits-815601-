# Airport Risk Intelligence

**Reply × LUISS 2026, Project 2 (Classical vs Multi-Agent)**
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

## 1. Executive summary

The Reply brief asks whether a multi-agent orchestration on LangGraph adds enough operational value to justify its complexity, compared to a classical sequential anomaly-detection pipeline. We implement the same detection logic twice, once as a sequential script and once as a five-agent LangGraph DAG, and compare them on a 567-route Italian border-control dataset.

The main numbers, all reproduced by `notebooks/08_report_assets.ipynb` from the processed CSVs:

| Metric | Value |
|---|---|
| Routes analysed | 567 |
| Anomaly-label distribution (both pipelines) | **17 HIGH / 40 MEDIUM / 510 NORMAL** |
| Final-risk distribution (both pipelines) | **9 CRITICAL / 29 HIGH / 19 MEDIUM / 510 LOW** |
| Per-route label agreement (`anomaly_label`) | **100.00 %** (567 / 567) |
| Per-route label agreement (`final_risk`)    | **100.00 %** (567 / 567) |
| Pearson r on the final scalar score | **0.9999** (float-precision noise from numpy vs pandas reduction order) |
| Spearman ρ on the final scalar score | **0.9999** |
| Business-rule hits (all 5 rules) | **Identical, delta = 0** (r = 1.000 by construction) |
| Top-50 anomalous routes overlap | **49 / 50** |
| Top-10 anomalous routes overlap | **10 / 10** |
| Ensemble weight choice | **Data-driven**, winner of a 4-simplex grid search (Section 4.4.1) |
| Threshold-sensitivity max swing on CRITICAL + HIGH (±10 %) | **2.6 %** (1 route) |

The two architectures produce identical anomaly and final-risk labels on all 567 routes. The residual Pearson r = 0.9999 on the continuous scalar score reflects float-precision differences in reduction order between numpy (classical) and pandas (multi-agent) reductions, not algorithmic divergence. What the multi-agent design adds over the sequential script is operational rather than statistical: an LLM-narrated explanation per HIGH/MEDIUM route, dynamic perimeter filtering at runtime, per-agent failure isolation, and a bounded feedback cycle when the verifier disagrees with the first pass. The classical script does not provide these properties without significant refactoring.

The 100 % label agreement is a property of the code, not a coincidence. The Autoencoder, historically the only stochastic component of the ensemble, is now trained by a single shared module (`shared/autoencoder.py`) called by both pipelines with the rows sorted by route id and `early_stopping=False`. Combined with the already-deterministic IF, LOF, and Z components, and with the business rules now applying the same thresholds on both sides, the end-to-end output of the two pipelines is identical.

---

## 2. Project context

### 2.1 The brief

Reply assigned the **NoiPA validation use case** as the framing for Project 2 of the *Backpropagation Bandits* LUISS course. NoiPA is the digital platform of the Italian Ministry of Economy and Finance (MEF) that ingests heterogeneous datasets from third-party authorities and validates them automatically. The deliverable asks for an anomaly-detection system that could be plugged into NoiPA, accepting heterogeneous tabular records, producing a route-level risk classification and a human-readable explanation per HIGH/MEDIUM route, and surfacing the results to an analyst.

The dataset Reply provided is **not** NoiPA itself: it is a sample of border-control passenger transits at Italian airports, used as an example of the kind of heterogeneous third-party dataset NoiPA would receive. The 567 origin/destination route pairs are the unit of analysis a customs officer would work with operationally.

### 2.2 What we built

Two implementations of the same detection logic, sharing the same preprocessing module, the same `FeatureBuilder` (54 numerical features per route), the same MAD-based baseline, the same four-model ensemble, the same five business rules and the same ensemble weights:

1. **Classical pipeline.** Seven sequential steps: EDA, preprocessing, feature engineering, baseline, ensemble, post-processing, evaluation. Inlined in `main.ipynb` so a reviewer can read it top-to-bottom without leaving the notebook. A standalone `classical_pipeline/main.py` orchestrator is also available for batch runs.
2. **Multi-agent pipeline.** A LangGraph DAG with five specialised agents (`DataAgent`, `BaselineAgent`, `OutlierAgent`, `RiskProfilingAgent`, `ReportAgent`) plus a `SupervisorAgent` verifier wired into the graph as a conditional branch with a bounded feedback cycle. Lives in `multiagent_pipeline/` and is imported by Sections 8 to 13 of `main.ipynb`.

### 2.3 Alignment guarantee

Both pipelines apply the same five business rules with the same thresholds, and the labels they produce share the same English vocabulary: `HIGH / MEDIUM / NORMAL` at the ensemble layer, `CRITICAL / HIGH / MEDIUM / LOW` after the rules. The earlier Italian/English label drift between the two implementations has been removed. The BR hit counts now coincide on every route (zero delta on all five rules), and after the AE alignment described in §4.4 the disagreement on `anomaly_label` is zero.

---

## 3. Dataset

### 3.1 Raw layer

Reply provides two CSV files at the bottom of `data/raw/` (NDA-protected, **not** redistributed in this repository):

| File | Granularity | Rows | Key columns |
|---|---|---|---|
| `ALLARMI.csv` | one row per alarm (Interpol, SDI, NSIS) generated by a border control | 5 080 | route, year-month, motive, outcome (`chiuso` / `respinto` / `fermato` / `segnalato`) |
| `TIPOLOGIA_VIAGGIATORE.csv` | one row per traveller-profile transit on a route-month | 5 095 | nationality, route, year-month, total entered, alarmed, investigated |

### 3.2 Temporal coverage

The panel spans **three months only**: December 2023, January 2024, and February 2024. This is the most consequential constraint on the methodology. It rules out STL-style decomposition, which needs at least 12 observations per series, and forces us to fall back on cross-sectional robust statistics.

![Monthly volume](images/dataset_monthly_volume.png)

*Figure 1. Monthly volume of raw records. The dataset spans Dec-2023 to Feb-2024 with one off-cycle record carrying a `MESE_PARTENZA = 12` value (treated as Dec 2024 by the cleaner).*

### 3.3 Geographic coverage

Routes terminate or originate at Italian airports (FCO, MXP, LIN, BLQ, NAP, …). The 15 most active departure countries by passenger volume:

![Top countries](images/dataset_top_countries.png)

*Figure 2. Top 15 departure countries by passenger volume entered into Italian territory.*

### 3.4 Unit of analysis

After cleaning and aggregating monthly, the unit of analysis is a **route**, that is an `airport_departure -> airport_arrival` pair (e.g. `CMN-FCO` for Casablanca to Rome Fiumicino). The cleaned panel contains **567 unique routes**, each described by **54 numerical features**. The full feature roster is documented in `data/processed/feature_cols.json`. The thirteen features used as inputs to the baseline are reported in §4.3 below.

---

## 4. Methodology

The methodology splits naturally into six layers, identical on both pipelines and implemented through the same shared modules.

### 4.1 Preprocessing

`shared/preprocessing.py` performs cleaning and merging in one pass:

* **Date parsing.** `DATA_PARTENZA` is normalised to UTC; rows with unparseable timestamps are dropped.
* **Country-code normalisation.** ISO-2 codes are mapped to ISO-3 via an embedded lookup table; departure-country labels are stripped of stray casing and whitespace.
* **Gender normalisation.** The raw `GENERE` field carries inconsistent encodings (`M`, `m`, `M.`, `Maschio`); all are collapsed to `M` / `F` / `OTHER`.
* **Sparse-column drop.** Columns with more than 95 % nulls are dropped before the route-level merge.
* **Route-level merge.** Alarms and traveller records are aggregated on `(AREOPORTO_PARTENZA, AREOPORTO_ARRIVO)`. The resulting `dataset_merged.csv` is the input of every downstream layer.

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

with a fallback to the standard deviation when MAD = 0. The MAD = 0 case occurs on sparse features where more than half of the routes share the same value (typically zero on the percentage and rate columns); without the fallback those features would silently produce zero z-scores. The scaling factor 1.4826 makes the MAD a consistent estimator of σ under a normal model.

The 13 features that participate in the baseline are:

`tot_allarmi_log`, `pct_interpol`, `pct_sdi`, `pct_nsis`, `tasso_chiusura`, `tasso_rilevanza`, `tasso_allarme_medio`, `tasso_inv_medio`, `score_rischio_esiti`, `tasso_respinti`, `tasso_fermati`, `false_positive_rate`, `alarm_per_invest`.

The composite `baseline_score` is the mean of the absolute z-scores across the 13 features. In practice the MAD is non-degenerate only on `tot_allarmi_log` (a non-sparse volume feature); on the 12 rate and composition features the median is zero and the fallback to std applies. The hybrid (MAD when defined, std otherwise) handles both cases without an additional configuration knob.

### 4.4 Ensemble anomaly detection

`OutlierAgent` (and the equivalent classical step) trains four independent detectors on the 13-feature matrix, normalises each output to [0, 1], and blends them into a single `ensemble_score`. The four weights below are **data-driven**: they are the winner of a grid search over the 4-simplex (§4.4.1) and supersede the original principled defaults that we initially borrowed from the literature.

| Detector | Weight | Hyper-parameters | Implementation |
|---|---|---|---|
| Isolation Forest | **0.40** | `contamination = 0.10`, `n_estimators = 200`, `random_state = 42` | `sklearn.ensemble.IsolationForest` |
| Local Outlier Factor | **0.15** | `n_neighbors = 20`, `contamination = 0.10` | `sklearn.neighbors.LocalOutlierFactor` |
| Z-score (MAD) | **0.30** | consumes `baseline_score` directly | shared module |
| Autoencoder (MLP) | **0.15** | architecture `13 → 8 → 4 → 8 → 13`, trained on the 510 normal routes via semi-supervision; deterministic single-module implementation (`shared/autoencoder.py`), `early_stopping = False`, sort by route id, no per-run variability | `sklearn.neural_network.MLPRegressor` |

The ensemble degrades gracefully: with fewer than 30 normal routes available, the Autoencoder is excluded from the blend and its 0.15 weight is redistributed proportionally over IsolationForest, LOF and the Z-score.

#### 4.4.1 Data-driven weight choice: ablation and grid search

We validate the choice of weights with two complementary analyses, both implemented in `multiagent_pipeline/src/` and re-runnable from `notebooks/08_report_assets.ipynb`.

**Ablation study** (`ensemble_ablation.py`). We drop one detector at a time, renormalise the remaining weights to sum to one, and compare the resulting top-17 HIGH set against the full ensemble. Results on the 567-route population:

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

Two observations from the table. First, **dropping LOF leaves the top-17 unchanged** and slightly improves the rank correlation with `br_score` (0.558 vs 0.550 for the full ensemble): LOF contributes mostly redundancy with the IF density signal, which is the empirical justification for cutting its weight from 0.30 to 0.15. Second, dropping AE pushes the top-17 overlap down to 0.824 and the BR rank correlation to 0.520: the AE captures non-linear feature combinations that the other three detectors do not, so it stays in the blend at a reduced weight.

![Ensemble ablation](images/ensemble_ablation.png)

*Figure 5. Ablation result. Blue bars show top-17 overlap with the full ensemble; orange bars show the Spearman correlation between the ensemble score and the business-rule score.*

**Grid search** (`ensemble_grid_search.py`). We enumerate the 4-simplex of weight vectors at step 0.05 (all four weights strictly positive, summing to one, around 969 vectors) and score every vector by

```
objective = 0.5 · bootstrap_stability(top-17, 80% subsample) + 0.5 · ((BR_rank_corr + 1) / 2)
```

The objective rewards weight vectors whose top-17 HIGH set survives bootstrap resampling and whose ensemble score rank-correlates with the business-rule score. The two halves act as a sanity check on each other: stability alone would favour degenerate weightings, rule correlation alone would mirror the rules at the expense of the ML signal.

| Weight vector | IF | LOF | Z | AE | Stability | BR rank corr | Objective |
|---|---|---|---|---|---|---|---|
| Initial literature defaults | 0.35 | 0.30 | 0.15 | 0.20 | 0.797 | 0.499 | 0.773 |
| **Grid-search winner (current production)** | **0.40** | **0.15** | **0.30** | **0.15** | **0.833** | **0.550** | **0.804** |
| Gap | +0.05 | −0.15 | +0.15 | −0.05 | +0.036 | +0.051 | **+0.031 (+4.0 %)** |

The grid result agrees with the ablation. IF stays the heaviest weight (+0.05); LOF is halved (−0.15) for the redundancy reason above; Z is doubled (+0.15) because it has the highest individual BR rank correlation; AE is trimmed (−0.05) but retained for non-linear coverage.

![Grid search heatmap](images/ensemble_grid_search_heatmap.png)

*Figure 6. Marginal heatmap of the grid-search objective over (w_IF, w_Z), max over w_LOF and w_AE. The black star marks the current production weights (the grid-search winner). The objective surface is smooth around the winner: small perturbations of the weights do not change the verdict, which is the relevant robustness check.*

**Caveat.** The Z component uses MAD z-scores of the same 13 features that the business rules read, so a high `Z ↔ br_score` correlation is partly mechanical. We chose this objective deliberately, since operational alignment is part of what the brief asks for, but we do not claim the grid optimum is uniquely correct. It is the best weight vector under a stated, reproducible objective, which is the most we can claim in an unsupervised setting.

The 567 routes split into three buckets at **data-driven thresholds**: the p97 of the ensemble score is the boundary between HIGH and MEDIUM, the p90 between MEDIUM and NORMAL.

![Ensemble distribution](images/ensemble_score_distribution.png)

*Figure 4. Distribution of the ensemble anomaly score across 567 routes, with the data-driven p97 (HIGH) and p90 (MEDIUM) thresholds.*

### 4.5 Business rules

The post-processing layer applies five binary business rules that approximate the operational checks an analyst would apply to a flagged route. Both pipelines now apply the same rule set with the same thresholds. The earlier notebook used a different rule set inline; we audited and aligned both sides to a single canonical definition.

| # | Rule id | Condition | Operational interpretation |
|---|---|---|---|
| 1 | `br_high_interpol` | `pct_interpol ≥ 0.30` | INTERPOL alarms dominate the route |
| 2 | `br_high_rejection` | `tasso_respinti ≥ 0.25` | Above-average traveller rejection at the border |
| 3 | `br_low_closure` | `tot_allarmi_log > 3` **and** `tasso_chiusura < 0.10` | Operational backlog: high alarm volume with low closure rate |
| 4 | `br_multi_source` | `pct_interpol ≥ 0.10` **and** `pct_sdi ≥ 0.10` | Multi-database corroboration (route shows up significantly in two distinct intelligence sources) |
| 5 | `br_high_alarm_rate` | `tasso_allarme_medio ≥ 0.50` | One traveller in two on this route triggers an alarm |

Each rule is binary. `br_score = mean(br_*) ∈ [0, 1]` is the aggregate.

**Note on `br_multi_source`.** The earlier implementation used a `pct_interpol > 0 AND pct_sdi > 0` rule. We reviewed the firing rate on the population and tightened it to `≥ 0.10` on both channels. Under the old rule the BR fired on essentially any route where the two databases had a trace presence, which is not the operational signal we want to surface. The tighter rule fires on 152 routes (26.8 % of the population), still material but now corresponding to real multi-source corroboration.

![Business rule hits](images/business_rule_hits.png)

*Figure 7. Business-rule hit frequency on the 567-route population. Both pipelines produce identical counts on every rule (zero delta), so `br_score` Pearson r between the two pipelines is exactly 1.000 by construction.*

### 4.6 Final risk classification

`RiskProfilingAgent` (and the equivalent classical step) collapses the ML signal and the rule signal into a single ordinal label `final_risk ∈ {CRITICAL, HIGH, MEDIUM, LOW}`:

```
CRITICAL : anomaly_label == HIGH   AND br_score ≥ 0.4
HIGH     : anomaly_label == HIGH   OR  (anomaly_label == MEDIUM AND br_score ≥ 0.4)
MEDIUM   : anomaly_label == MEDIUM
LOW      : otherwise
```

The 0.4 boundary on `br_score` corresponds to "at least two of the five rules fired". A blended `confidence` score is also produced for ranking inside each bucket:

```
confidence = 0.60 · ensemble_score + 0.40 · br_score
```

The 60 / 40 ML to rules split is the one specified in the brief: the ML signal carries more weight because it is statistically validated, while the rules add an interpretable layer that can be inspected and adjusted independently.

---

## 5. Multi-agent architecture

### 5.1 The five agents

The graph respects the Reply specification of five visible agents. The `SupervisorAgent` is a verifier wired in as a conditional branch and does not count toward the spec headcount.

| # | Agent | Responsibility |
|---|---|---|
| 1 | `DataAgent` | Loads `ALLARMI.csv` + `TIPOLOGIA_VIAGGIATORE.csv`, applies the user-defined perimeter, and engineers the 54 numerical features per route via `FeatureBuilder`. |
| 2 | `BaselineAgent` | Computes robust MAD z-scores per baseline feature and aggregates them into a single `baseline_score` consumed downstream as the Z-component of the ensemble. |
| 3 | `OutlierAgent` | Trains the four-model weighted ensemble (IF + LOF + Z + AE) and produces `ensemble_score` and `anomaly_label` (HIGH / MEDIUM / NORMAL). |
| 4 | `RiskProfilingAgent` | Applies the five canonical business rules, computes `br_score`, blends ML and rules into `confidence`, and assigns `final_risk` (CRITICAL / HIGH / MEDIUM / LOW). Produces a per-route `risk_drivers` list of textual reason codes consumed by the LLM downstream. |
| 5 | `ReportAgent` (LLM) | Generates a natural-language explanation for each HIGH/MEDIUM route, combining the top z-score drivers from the BaselineAgent with the business rules that fired on that route. Backend pluggable (Claude / local LM Studio / none). |
| ★ | `SupervisorAgent` *(verifier, optional)* | Re-fits Isolation Forest at `contamination = 0.03` on the full population and tags first-pass HIGH routes as `robust_high = True` only if they survive the stricter rule. |

### 5.2 The DAG topology

The graph carries **four data-driven conditional edges** on top of the standard error-stop logic:

1. **after_baseline.** Terminate early when the baseline signal is degenerate (fewer than five features available or `baseline_score` standard deviation below 0.01); the pipeline returns with a clear empty-output diagnostic.
2. **after_outlier.** Route through `SupervisorAgent` only when the first pass produces at least 5 HIGH labels; otherwise short-circuit to the rule layer (refitting Isolation Forest on a tiny subset would be statistically meaningless).
3. **after_supervisor.** Cycle back to `OutlierAgent` when the verifier downgrades more than 50 % of the first-pass HIGH labels, capped at two iterations to guarantee termination. This is the one place where the topology is genuinely non-linear.
4. **after_risk.** Skip the LLM `ReportAgent` when there are no HIGH/MEDIUM routes worth narrating, saving API cost on quiet perimeters.

### 5.3 Operational value of the multi-agent architecture

Three properties of the LangGraph design that the flat sequential script does not provide out of the box:

* **Per-agent failure isolation.** If `BaselineAgent` fails on a degenerate perimeter, the orchestrator returns a partial state containing a `baseline_meta.error` field and a human-readable message. The Streamlit dashboard renders the partial output and tells the analyst what is missing.
* **Supervisor-to-outlier feedback cycle.** When the verifier disagrees with more than 50 % of the first-pass HIGH labels, the orchestrator routes the graph back to OutlierAgent. The classical script runs once and stops; the multi-agent graph can correct its own first pass within a bounded number of retries.
* **Dynamic perimeter filtering.** `DataAgent` accepts a runtime perimeter dict (year, country, airport, zone) and the rest of the graph adapts. The Streamlit dashboard at `streamlit_app/app.py` uses this to let an analyst restrict the analysis interactively.

---

## 6. Results

This section reproduces every claim in the executive summary and adds the residual diagnostics that justify it. Every figure and every table is generated deterministically from the canonical processed CSVs by `notebooks/08_report_assets.ipynb` (see §12 for the asset-reproducibility note).

### 6.1 Distribution convergence

Both pipelines produce **identical anomaly-label distributions** on the 567 routes, and after the AE alignment described in §4.4, also identical post-rule final-risk distributions:

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

*Figure 8. Side-by-side comparison of label distributions. The two pipelines produce identical splits at both the ensemble layer (HIGH/MEDIUM/NORMAL) and the post-rule layer (CRITICAL/HIGH/MEDIUM/LOW), as a direct consequence of the shared AE module and the aligned business-rule layer.*

### 6.2 Top anomalous routes

The 15 routes with the highest ensemble score (multi-agent pipeline). All 15 are above the p97 threshold and labelled HIGH. The maximum score on this dataset is 0.813, on Casablanca to Bologna (`CMN-BLQ`).

![Top routes](images/top_routes_anomaly_score.png)

*Figure 9. Top 15 routes by ensemble anomaly score on the multi-agent pipeline. Colour encodes the anomaly_label.*

### 6.3 Per-route agreement

The two pipelines agree on **567 of 567 anomaly labels (100.00 %)** and on **567 of 567 final-risk labels (100.00 %)**. The confusion matrix is therefore strictly diagonal.

![Confusion matrix](images/anomaly_label_confusion_matrix.png)

*Figure 10. Confusion matrix on anomaly_label (rows = classical, columns = multi-agent). All 567 routes sit on the diagonal; the off-diagonal entries are zero.*

### 6.4 Score correlation

The Pearson correlation between the two pipelines' final scalar scores is 0.9999 (more precisely 0.999987). The Spearman rank correlation is 0.9999 (0.999974). The residual gap below 1.0000 is float-precision noise from reduction order: the classical pipeline accumulates the weighted sum on numpy arrays, the multi-agent on pandas Series, and the two paths iterate memory in marginally different orders. The label assigned to each route does not depend on the difference.

![Score correlation](images/score_correlation_classical_vs_multiagent.png)

*Figure 11. Per-route score correlation. Each point is a route, coloured by the multi-agent anomaly label. Dashed line: y = x. The points sit on the diagonal; the only visible spread lives in the NORMAL band, where small score perturbations do not move the label.*

### 6.5 Business-rule alignment

The five rules produce identical hit counts on every rule (delta = 0, see §4.5). This means that any difference between the two pipelines comes from the ML ensemble, not from drifting business rules.

### 6.6 Bootstrap CI on the agreement metric

To check the agreement against finite-sample uncertainty we resample the merged 567-route DataFrame 1 000 times at 80 % subsample and recompute the row-level agreement on every resample. We report two regimes:

* **Pre-fix** (with the historical stochastic Autoencoder): point estimate 98.24 %, bootstrap mean 98.25 %, 95 % CI [97.79 %, 98.90 %]. Even in the worst-case 80 % subsample the agreement stays above 97.8 %.
* **Post-fix** (after the deterministic AE alignment of §4.4): every resample produces 100 %, so the bootstrap distribution is concentrated on a single point and the 95 % CI is [100 %, 100 %].

The pre-fix CI is the substantive number: it shows the convergence claim is not a small-sample artefact. The post-fix CI follows directly from the AE refactor and contains no additional information. Numerical values are in `images/tables/bootstrap_ci_agreement.csv`.

### 6.7 Threshold sensitivity

We perturb each of the five BR thresholds independently by ±5 % and ±10 % and recompute the final-risk count. The dataset is structurally robust: only three thresholds (`high_alarm_rate`, `high_rejection_rate`, `multi_source_pct`) move the count of CRITICAL + HIGH routes at all, and at most by a single route (2.6 % swing relative to the 38-route baseline).

![Threshold sensitivity](images/threshold_sensitivity.png)

*Figure 12. Sensitivity of the HIGH and CRITICAL counts to ±10 % and ±5 % perturbations of the five BR thresholds. Cells report the number of routes flagged at that level under each perturbation.*

### 6.8 Feature importance

A surrogate Gradient Boosting classifier trained to predict the ensemble flag from the 13 baseline features surfaces the drivers a customs operator would expect to see at the top: total alarm volume, average alarm rate, and the score on the outcome side (`score_rischio_esiti`).

![Feature importance and SHAP](images/feature_importance_shap.png)

*Figure 13. Surrogate feature importance (left) and mean SHAP value (right), top 10 features. The SHAP values are computed against the surrogate model and serve as an interpretability hint, not as a faithful explanation of the ensemble itself.*

---

## 7. Design rationale

Three places where we departed from the literal text of the brief, with the reasoning made explicit.

**STL vs robust z-scores.** The brief mentions a *"historical baseline using rolling averages and seasonal decomposition"*. STL needs at least 12 observations per series; our panel has three months, so STL is not applicable. A 3-month rolling mean is equivalent to the cross-sectional mean we already compute. We use robust z-scores against the population distribution, which is the standard alternative for short panels.

**Four-model ensemble.** The brief lists *"IsolationForest, LOF or Z-score"*. We use all three plus an Autoencoder at weight 0.15. The Autoencoder captures non-linear feature combinations that the density-based detectors do not, and the ensemble degrades gracefully when the perimeter is small: below 30 normal samples the AE is excluded and its weight is redistributed proportionally over the other three.

**Agent count.** The Reply slide lists five agents (`DataAgent`, `BaselineAgent`, `OutlierAgent`, `RiskProfilingAgent`, `ReportAgent`) and we keep the count at five. An earlier design used `FeatureBuilder` as a separate sixth agent; we merged it into `DataAgent` to reduce orchestration overhead without changing the topology shown to a reviewer. The `SupervisorAgent` is documented as an optional verifier rather than a sixth mandatory agent.

---

## 8. Limitations

1. **Single dataset.** The entire evaluation runs on a single Reply-provided dataset. We have not stress-tested either pipeline on a different schema, although `DataAgent` carries an LLM schema-normalisation layer that has not had to fire on this dataset because the canonical columns are all present.
2. **Three-month panel.** The dataset spans only December 2023 to February 2024. A longer panel would unlock STL and rolling means without changing the rest of the pipeline.
3. **LLM narratives are not programmatically validated.** The `ReportAgent` prompt forbids hallucination and is reviewed in spot checks, but we do not prove zero hallucination automatically. The prompt instruments enough structured context (top-3 z-score drivers, fired rules, ensemble score) that gross hallucinations are easy to spot in review, but not impossible to miss.
4. **Autoencoder determinism (historical note).** An earlier iteration of the project relied on `MLPRegressor(..., early_stopping=True)`. The validation split was data-order-dependent and produced run-to-run variability on a handful of MEDIUM/NORMAL boundary routes (about 1.8 % of the population). The current code routes both pipelines through `shared/autoencoder.py`, which sorts the input by route id, disables early stopping, and trains for a fixed `max_iter`. The AE output is now deterministic; the residual ~10⁻⁵ Pearson gap is float-precision noise from reduction order.
5. **No live data.** The Streamlit dashboard runs on the same processed CSVs as the analysis; there is no production ingestion pipeline.

---

## 9. Future work

A `TrendAgent` as an optional sixth node would unlock STL or rolling baselines once the panel covers more than 12 months. The graph already supports optional-agent wiring (see how `ReportAgent` is added conditionally).

The supervisor-to-outlier feedback cycle currently triggers only on first-pass HIGH disagreement. Extending it to MEDIUM borderline routes would tighten the boundary classification.

A multi-locale `ReportAgent` would expose the narrative language as a runtime parameter, so an Italian-speaking operator gets Italian narratives without modifying the prompt. The current prompt is English-only; the change is small in code but enlarges the test surface.

---

## 10. Repository structure

```
.
├── README.md                       This file
├── main.ipynb                      Single-notebook tour of the project
├── Oral_presentation.pdf           Oral defence slides
├── requirements.txt
├── .env.example                    ANTHROPIC_API_KEY template
├── images/                         All PNG figures + tables/ CSV summaries
│   ├── *.png                       (generated by notebooks/08_report_assets.ipynb)
│   └── tables/*.csv
├── notebooks/
│   └── 08_report_assets.ipynb      Reproduces every figure and table in this README
├── shared/
│   ├── preprocessing.py            Cleaning + merge layer used by both pipelines
│   └── autoencoder.py              Deterministic AE, single source of truth
├── multiagent_pipeline/            LangGraph library
│   ├── main.py                     run_pipeline, graph orchestrator
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

## 11. How to reproduce

### 11.1 Requirements

* Python ≥ 3.10
* The two raw CSVs provided by Reply under NDA: `data/raw/ALLARMI.csv` and `data/raw/TIPOLOGIA_VIAGGIATORE.csv`. They are **not** redistributed in this repository.
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
| 10 | Threshold sensitivity |
| 11 | Conclusions |

End-to-end runtime on the 2024 perimeter (567 routes):

* without the LLM: about 2 minutes
* with the LLM: about 7 minutes (Claude generates one narrative per HIGH/MEDIUM route, around 57 calls)

### 11.5 Unit tests

```bash
PYTHONPATH=. python -m pytest multiagent_pipeline/tests/test_risk_profiling_agent.py -v
```

13 unit tests cover the five business rules (one per rule, plus one verifying the `br_multi_source` floor), the `br_score` aggregation, the confidence-blend formula, and every cell of the final-risk classification ladder.

---

## 12. Reproducibility note on figures and tables

Every PNG and every numerical value cited in §6 is generated by `notebooks/08_report_assets.ipynb`, which reads the CSVs in `data/processed/` and writes the PNGs to `images/` and the underlying numeric summaries to `images/tables/`. To regenerate the assets, after running the two pipelines:

```bash
PYTHONPATH=. jupyter nbconvert --to notebook --execute notebooks/08_report_assets.ipynb \
    --output 08_report_assets.ipynb
```

The asset inventory is printed at the end of the notebook. To audit any number cited in this README, the corresponding CSV in `images/tables/` is the source.

---

*Reply × LUISS 2026, Daniele Giovanardi · Filippo Nannucci · Edoardo Riva*
