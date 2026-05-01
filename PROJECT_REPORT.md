# Airport Risk Intelligence — Project Report
**Reply × LUISS 2026 — Project 2 (Classical vs Multi-Agent)**
**Team:** Daniele Giovanardi · Filippo Nannucci · Edoardo Riva

This report walks through `main.ipynb` section by section. For every block of cells we explain **what** it does, **why** we did it that way, and **what we deliberately did not do** (and why). It is a companion document for the notebook: read it alongside, not instead of.

---

## 0 · Title and TL;DR

**Cells:** title markdown + TL;DR table + setup code.

**What we do.** We open with the team list, the question we set out to answer (*"when does multi-agent orchestration buy you something over a classical pipeline?"*), and a TL;DR table with the headline numbers — 567 routes, 17 / 40 / 510 distribution, 97.2 % agreement, ensemble Pearson r = 0.9847. The setup cell anchors `cwd` to the repo root by walking up until it finds `data/raw/ALLARMI.csv`, so the notebook works the same from Jupyter, VSCode and `nbconvert`.

**Why this order.** A reviewer who never gets past the first screen still sees the result. The headline is unambiguous, the table is auditable against any cell below, and the setup cell makes the rest of the notebook portable.

**What we don't do.** We do not duplicate the README's Mermaid diagrams here — `main.ipynb` is the executable narrative, the README is the navigational document. We also do not show a project tree at the top: the section headers do the navigation.

---

## 1 · Exploratory Data Analysis

**Cells:** 30 cells from `classical_pipeline/notebooks/01_EDA.ipynb`, untouched.

**What we do.** Load the two raw CSVs (`ALLARMI`, `TIPOLOGIA_VIAGGIATORE`), profile each schema, count missing values and category coverage, plot distributions, and call out the gaps that drive the cleaning logic in Section 2 (date format inconsistencies, ISO2 country codes, sparse columns, "ND" placeholders).

**Why we keep all 30 cells.** This is the section that justifies the cleaning rules. Cutting it would force the reader to take Section 2 on faith. The EDA also surfaces the 22.8 % missing rate on `MOTIVO_ALLARME` which dictates how the FeatureBuilder later treats that column (one-hot with an "unknown" bucket, no imputation).

**What we don't do.** We do not do exploratory modelling here — no clustering, no dimensionality reduction. EDA is descriptive only; everything inferential happens after preprocessing.

---

## 2 · Data Preprocessing

**Cells:** 1 markdown intro + the entire `shared/preprocessing.py` module (480 lines, inlined) + a single execution cell.

**What we do.** Inline the full source of `shared/preprocessing.py` so the reviewer can audit every constant (the Italian month-abbreviation map, the ISO2→ISO3 country lookup, the `NULL_VARIANTS` regex list, the `NULL_DROP_THRESHOLD = 0.95` for sparse columns) in one place. The execution cell calls `run_preprocessing()` and prints the resulting shapes. Both pipelines depend on this layer: the classical script calls it as Step 0, and the multi-agent `DataAgent` reads the `*_clean.csv` artefacts produced here.

**Why one shared module.** The risk of drift between two cleaning implementations is too high to accept — different cleaning would give different ensemble outputs and the comparative analysis would conflate cleaning bugs with architectural differences. Inlining the source in the notebook makes the dependency visible.

**What we don't do.** We do not run preprocessing again from inside the notebook with different parameters; the constants are deliberately fixed. We also do not apply imputation on `MOTIVO_ALLARME` — it is treated as a categorical with an "unknown" level downstream rather than guessed.

---

## 3 · Feature Engineering

**Cells:** 36 cells from `02_feature_engineering.ipynb`.

**What we do.** Aggregate the cleaned tables into 54 route-level features grouped in six families: alarm-source percentages (`pct_interpol`, `pct_sdi`, `pct_nsis`, …), traveller-volume rates (`tasso_allarme_medio`, `tasso_inv_medio`, …), outcome rates (`tasso_respinti`, `tasso_fermati`, `tasso_chiusura`, …), normalised volumes (`tot_allarmi_log`), interaction features (`alarm_per_invest`), and a composite risk score. The result is the canonical `features` DataFrame used by every downstream stage.

**Why 54 and not more / fewer.** Each feature corresponds to a concrete operational signal a border-control analyst can interpret. We avoided polynomial expansions and embeddings on purpose — we want every score to be back-traceable to a feature an analyst recognises.

**What we don't do.** We do not perform automatic feature selection (e.g. Lasso, mutual information) at this stage; the reduction to 13 BASELINE_FEATURES happens explicitly in Section 4 with documented criteria.

---

## 4 · Baseline Construction

**Cells:** 21 cells from `03_baseline_construction.ipynb`.

**What we do.** Pick 13 BASELINE_FEATURES that capture the operational signal we want to monitor (alarm percentages, rates, volume). For each one we compute Tukey IQR limits and a 2.5σ z-score threshold, then flag a route on a feature when *either* threshold is exceeded. The aggregate `pct_anomalie` (fraction of features flagged) becomes the Z-score signal of the ensemble in Section 5.

**Why hybrid Tukey + z-score.** Each method catches a different shape of outlier: Tukey IQR is robust to a fat-tailed distribution, while the z-score catches mean shifts. Taking the OR is intentionally permissive at the *flag* level so the ensemble downstream can decide.

**What we don't do.** We do not compute a temporal baseline (rolling average / STL) here — Section 12 documents why with empirical evidence, but the short version is: median 2 monthly observations per route, max 3, so STL is mathematically infeasible. We also do not adjust the threshold per route or per country; the baseline is cross-sectional by construction.

---

## 5 · Anomaly Detection — 4-Model Weighted Ensemble

**Cells:** 25 cells from `04_anomaly_detection.ipynb`.

**What we do.** Train four models on the BASELINE_FEATURES (after `StandardScaler`) and blend their normalised scores:

| Model | Weight | Role |
|---|---|---|
| `IsolationForest` (contamination = 0.10, n_estimators = 200, seed = 42) | 0.35 | Global density estimator |
| `LocalOutlierFactor` (k = 20, contamination = 0.10) | 0.30 | Local-density estimator |
| Z-score signal (`pct_anomalie` from Section 4) | 0.15 | Cross-sectional flagging |
| `MLPRegressor` autoencoder, architecture 11 → 8 → 4 → 8 → 11, semi-supervised on IF-normal samples | 0.20 | Non-linear feature interactions |

The autoencoder gracefully degrades: if fewer than 30 normal samples are available it is excluded and the remaining three weights are renormalised to sum to 1. Risk labels are assigned via data-driven thresholds — `p97` of the ensemble score for ALTA, `p90` for MEDIA.

**Why an ensemble and not a single model.** Each method has a different failure mode (IF struggles with axis-aligned outliers, LOF with global outliers, z-score is brittle on multi-modal features). The weighted sum is the simplest pattern that captures all four signals while staying interpretable.

**What we don't do.** We do not retune `contamination` per perimeter; it is fixed at 0.10 for reproducibility. We also do not use deep autoencoders or transformer-based anomaly detection — overkill for 567 × 13 features.

---

## 6 · Post-Processing — Business Rules and Risk Profiles

**Cells:** 27 cells from `05_post_processing.ipynb`.

**What we do.** Apply five binary business rules with thresholds inherited from the classical post-processing layer:

| Rule | Condition |
|---|---|
| `br_high_interpol` | `pct_interpol >= 0.30` |
| `br_high_rejection` | `tasso_respinti >= 0.25` |
| `br_low_closure` | `tot_allarmi_log > 3` AND `tasso_chiusura < 0.10` |
| `br_multi_source` | `pct_interpol > 0` AND `pct_sdi > 0` |
| `br_high_alarm_rate` | `tasso_allarme_medio >= 0.50` |

The five flags are averaged into `br_score ∈ [0, 1]`. The blended `confidence = 0.60·ensemble_score + 0.40·br_score` is what an analyst should look at first. The final ladder is `CRITICO / ALTO / MEDIO / BASSO`, computed from the ML label and the `br_score`:

- `CRITICO` if `risk_label == ALTA` AND `br_score >= 0.4`
- `ALTO` if `risk_label == ALTA` OR (`MEDIA` AND `br_score >= 0.4`)
- `MEDIO` if `risk_label == MEDIA`
- `BASSO` otherwise

**Why blend ML with rules.** A pure-ML score is hard to defend in front of a stakeholder; a pure-rule score misses combinatorial patterns the ML catches. The 60/40 split is conservative — we trust the ML enough to drive the ranking but not enough to ignore explicit operational rules.

**What we don't do.** We do not let the LLM tune the rules. The thresholds and the blend weights are deterministic and live in `RiskProfilingAgent.BR_THRESHOLDS` so they can be unit-tested (`multiagent_pipeline/tests/test_risk_profiling_agent.py` does exactly that).

---

## 7 · Evaluation

**Cells:** 24 cells from `06_evaluation.ipynb`.

**What we do.** Build a quantitative scorecard for the classical pipeline: silhouette score on the binary anomaly labels, Davies-Bouldin and Calinski-Harabasz cluster metrics, 100-iteration bootstrap stability of the IsolationForest flags (a route is "stable" if it gets flagged in ≥ 70 % of bootstrap re-fits), permutation feature importance and SHAP via a GradientBoosting surrogate.

**Why evaluate this way.** Anomaly detection is unsupervised, so we can't rely on accuracy / F1. Cluster-quality metrics + bootstrap stability + feature importance are the standard scorecard in production-grade anomaly systems and they answer different questions: *are the clusters separable? are the flags reproducible? which features drive them?*

**What we don't do.** We do not evaluate the LLM-generated narratives in this section — that requires labelled "good explanation" data we do not have. We accept the LLM output as advisory and document the risk in Section 13 *Limits*.

---

## 8 · Multi-Agent Pipeline

**Cells:** 1 markdown header + 1 code dump of every agent's source + 2 execution cells.

**What we do.** Display the literal source of `multiagent_pipeline/state.py`, the five agents (`data_agent.py`, `baseline_agent.py`, `outlier_agent.py`, `risk_profiling_agent.py`, `report_agent.py`) and the LangGraph orchestrator (`main.py`). Then run `run_pipeline({'anno': 2024}, run_report=True, dry_run=True)` end-to-end and print the resulting risk distributions plus the business-rule hit counts. The display-then-execute structure means the reviewer reads the agent first and then sees it run.

**Why we keep the source as `print(...)` instead of redefining inline.** Redefining the agents inline would force us to also redefine the LangGraph wiring (state graph, conditional edges, error handling) inside the notebook, which would add another ~150 lines of boilerplate and increase the drift risk between the notebook and the production code. Showing the source via `Path(...).read_text()` and importing through `multiagent_pipeline.main` keeps the source of truth in one place — the modules — while the notebook still surfaces every line.

**What we don't do.** We do not use `use_llm=True` in the executed cell to avoid spending Anthropic tokens during examiner runs. The infrastructure is there: pass `use_llm=True` with an `ANTHROPIC_API_KEY` and the ReportAgent calls Claude. We also do not exploit LangGraph branching / supervisor patterns — the graph is linear with conditional edges only for stop-on-error. This is documented honestly in *Limits*.

---

## 9 · Comparative Analysis

**Cells:** 21 cells from `notebooks/07_comparison_classical_vs_multiagent.ipynb`.

**What we do.** Quantify the agreement between the two pipelines on the same 567 routes: label distribution side-by-side, label agreement matrix (confusion matrix on ALTA/MEDIA/NORMALE), Pearson and Spearman correlation between the two final scalar scores, per-model correlation (real `sklearn` IF / LOF / AE in both architectures), Venn diagram of routes flagged ALTA by both pipelines, rank-delta histogram. The Venn labels use f-strings so the numbers are always live.

**Why this is the central deliverable.** The brief asked us to *implement the same anomaly detection system twice and argue which architecture is more convenient*. The agreement metric (97.2 %) is the proof that both implementations are correct; the residual 2.8 % is a window into the deliberate baseline-method differences (Tukey IQR vs MAD).

**What we don't do.** We don't compare runtime here — that's covered in the Conclusions. We also don't redo the EDA on the multi-agent outputs separately; the multi-agent uses the same FeatureBuilder so the inputs are identical.

---

## 10 · Bootstrap Confidence Intervals — *new in this delivery*

**Cells:** 1 markdown + 1 code cell that imports `multiagent_pipeline.src.bootstrap_ci`.

**What we do.** Run a 1 000-iteration bootstrap (80 % subsample, seed = 42) to put a confidence interval on the agreement metric, the Pearson correlation, and the Spearman correlation. Result: agreement = **96.8 % ± 0.7 %** at the 95 % level, Pearson = **0.9820 ± 0.0017**, Spearman = **0.9854 ± 0.0018**.

**Why we added this.** A reviewer of a comparative analysis is right to ask: *what is the confidence interval on the agreement number you printed?* The point estimate alone (97.2 %) does not tell us how much it would move under resampling. The bootstrap answers exactly that.

**What we don't do.** We do not bootstrap the per-model correlations (IF, LOF, AE) — those are 1.0000 by construction up to floating-point, so a CI is meaningless. We also do not adjust for multiple testing — we are reporting CIs, not running hypothesis tests.

---

## 11 · Business-Rule Threshold Sensitivity — *new in this delivery*

**Cells:** 1 markdown + 2 code cells that import `multiagent_pipeline.src.threshold_sensitivity`.

**What we do.** For each of the five business-rule thresholds in turn, sweep ±10 % / ±5 % / 0 % and recompute `final_risk` deterministically. We print the per-threshold maximum swing in the high-risk count (CRITICO + ALTO) and render a heat-map of the route count under each perturbation. Result: only `high_rejection_rate` moves the high-risk count at all under the perturbations, and only by a single route (~2.3 %). The system is structurally robust to the inherited thresholds.

**Why we added this.** A natural follow-up to the comparative analysis is: *if your business rules are inherited from the classical layer, how sensitive is the final risk distribution to the threshold choice?* If the answer were "very sensitive", the credibility of the whole rule layer would be low. The empirical answer is the opposite: the system is stable.

**What we don't do.** We do not perturb the ML weights or the ensemble thresholds (`p97`, `p90`) here — those are evaluated in Section 7 with bootstrap stability. The sensitivity in Section 11 is specifically about the rule layer.

---

## 12 · Temporal Coverage and Trend Slopes — *new in this delivery*

**Cells:** 3 markdown + 2 code cells that import `multiagent_pipeline.src.trend_analysis`.

**What we do.** First, measure the temporal coverage of every route in the dataset (`analyse_temporal_coverage`) and print a verdict: median 2 months / route, max 3 months — STL is mathematically infeasible and a 3-month rolling mean over 3 observations collapses to the cross-sectional mean. Second, for every route with ≥ 2 observations we fit a linear regression of `TOT` (total alarms) vs the time index and classify the slope as RISING / STABLE / DECLINING / INSUFFICIENT. The result on this dataset: 198 STABLE, 60 INSUFFICIENT, 57 RISING, 53 DECLINING.

**Why we added this.** The brief explicitly mentions *historical baseline using rolling averages and seasonal decomposition* (slide 16), but the dataset cannot support either. Rather than ignore the spec mismatch or pretend to run STL on too-short series, we *measure* the limitation and then deliver the best temporal signal the data can support — a per-route trend slope.

**What we don't do.** We do not pretend STL works on 2–3 month panels. We also do not use the trend slope to override the ML/rule classification in this delivery — it is presented as an additional signal that an operator can monitor alongside `final_risk`. A future delivery with longer per-route panels (12+ months) would unlock the spec-suggested techniques without changing the rest of the pipeline (see Section 13 *Future work*).

---

## 13 · Conclusions

**Cells:** 1 markdown synthesis.

**What we do.** Restate the headline numbers from the TL;DR (this is the only place where we deliberately repeat them, because a reader who jumped to the conclusions still needs them). Define `agreement` precisely (row-by-row concordance on `anomaly_label`). Restate the *when to choose which architecture* recommendation in operational terms (classical = batch / audit, multi-agent = analyst-facing / interactive). Explain the 2.8 % residual disagreement (boundary cases between MEDIA and NORMALE under the deliberate Tukey IQR vs MAD baseline difference). List every limit honestly (single dataset, no temporal model, threshold sensitivity now characterised, LangGraph in linear mode, LLM narratives unvalidated, no bootstrap CI on the agreement → now resolved in Section 10) and every future work item.

**Why we close like this.** The brief asked us to *argue which approach is more convenient under what operational conditions*. The Conclusions section is where that argument lives. We avoid declaring a single winner because the answer is genuinely operational, not technical: classical wins on speed and audit, multi-agent wins on interactivity and explanation.

**What we don't do.** We do not summarise the code structure here — that's the README's job. We also do not include a "next steps" project plan with timelines; this is an academic deliverable, not a production roadmap.

---

## Cross-cutting choices we want to flag

1. **One source of truth for cleaning, features, business rules.** Both pipelines call the same `shared/preprocessing.py` and the same `multiagent_pipeline/src/features.py` `FeatureBuilder`. The multi-agent's `RiskProfilingAgent` and the classical's `step_post_processing` use the same `BR_THRESHOLDS` constants — the `br_score` Pearson correlation between the two pipelines is exactly **1.000 by construction**, and we have unit tests that verify it.

2. **Italian risk labels alongside English narrative.** `ALTA / MEDIA / NORMALE` and `CRITICO / ALTO / MEDIO / BASSO` are kept in Italian because the operator domain (NoiPA / border control) is Italian; the rest of the codebase, comments, and LLM narratives are in English so the project is portable.

3. **Spec deviations declared up-front, not hidden.** Three places: the README *Deviations from the Reply spec* table, slide 9 of the PPTX *Spec deviations declared up-front* note, and Section 12 of this notebook (data-driven justification of why STL is infeasible). A reviewer can interrogate the rationale instead of discovering the gap on their own.

4. **Tests are layered.** `multiagent_pipeline/tests/test_risk_profiling_agent.py` has 13 unit tests on the rule layer (one per threshold + aggregate score + final ladder). `multiagent_pipeline/tests/e2e_validation.py` runs the full pipeline on 5 different perimeters and asserts no agent fails. Both are documented in the README *Validation suite* section and runnable in under 5 seconds.

5. **The Streamlit dashboard is a consumer of the same code, not a duplicate.** `streamlit_app/app.py` imports `run_pipeline` and the agents from `multiagent_pipeline/`; it does not redefine any logic. The animated agent graph (`agent_graph.jsx`) is purely visual — no business logic in the JSX.

---

## What an examiner can reproduce locally in under 5 minutes

```bash
git clone https://github.com/DanieleGiovanardi2408/classical-vs-multiagent.git
cd classical-vs-multiagent
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Place the two raw CSVs (from Reply, under NDA)
#   data/raw/ALLARMI.csv
#   data/raw/TIPOLOGIA_VIAGGIATORE.csv

# Run everything
PYTHONPATH=. jupyter lab main.ipynb

# Or just the tests:
PYTHONPATH=. python -m pytest multiagent_pipeline/tests/test_risk_profiling_agent.py -v
PYTHONPATH=. python multiagent_pipeline/tests/e2e_validation.py

# Or the dashboard:
streamlit run streamlit_app/app.py
```

Total wall time on a 2024 MacBook Air M3, no LLM:
- Preprocessing + feature engineering: ~1.5 s
- Classical pipeline (skip-eval): ~3 s
- Multi-agent pipeline: ~1.3 s (`save_outputs=True`)
- Bootstrap CI (1 000 iterations): ~3 s
- Threshold sensitivity sweep: ~0.5 s
- Trend analysis on 368 routes: ~0.8 s
- Notebook end-to-end via `jupyter nbconvert --execute`: ~30 s

---

## Key takeaways for the oral exam

1. **Convergence at 97.2 % is the goal of the brief, not a flaw.** The brief asked to implement the same system twice; convergence proves both are correct.
2. **The 2.8 % residual is at the boundary** — Tukey IQR is slightly more permissive than MAD around the threshold, so a few MEDIA routes flip to NORMALE.
3. **Multi-agent earns its complexity on the operational side**, not the detection side: dynamic perimeter filtering, LLM narratives, modular failure handling.
4. **Spec deviations are declared and justified empirically** (see Section 12: STL needs 12 obs/route, the dataset has 2).
5. **The system is robust to threshold choices** — only one rule moves the high-risk count under ±10 % perturbation, and only by 1 route.

---

*Reply × LUISS 2026 — Daniele Giovanardi · Filippo Nannucci · Edoardo Riva*
