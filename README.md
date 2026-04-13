# Classical vs Multi-Agent: Airport Anomaly Detection

**Reply x LUISS — 2026**

## Overview

This project implements the **same anomaly-detection system twice** using two fundamentally different architectures, then compares them:

1. **Classical Pipeline** — A traditional ML pipeline: preprocessing, feature engineering, statistical baselines, a 4-model ensemble (IsolationForest, LOF, Z-score, Autoencoder), post-processing with business rules, and a full evaluation suite. Runs end-to-end as Jupyter notebooks or as a single Python script.
2. **Multi-Agent Pipeline** — A LangGraph-orchestrated system of 5 specialized agents (Data, Feature, Baseline, Outlier, Report) that replicate the same analytical logic but with autonomous agent coordination, dynamic perimeter filtering, and optional LLM-powered narrative reporting.

The goal is to produce a **comparative analysis** arguing which approach is more convenient and under what operational conditions.

## Context and Problem Statement

Border control authorities at Italian airports manage large volumes of passenger transits daily. Each transit is associated with rich metadata: timestamp, gate, route, nationality, document type, control outcome, and security alerts.

Current systems are **reactive** — they flag anomalies only after incidents occur. This project aims to build a **proactive** system that identifies suspicious route-level patterns before they become operational or security incidents.

The unit of analysis is the **ROUTE** (ROTTA), defined as the pair `departure_airport–arrival_airport` (e.g., `ALG-MXP`). Across 567 unique routes, the system detects which ones exhibit anomalous behavior based on alarm volumes, investigation rates, rejection rates, and other security-relevant indicators.

## Dataset

Two raw CSV files are provided (not tracked by git for confidentiality):

| File | Description | Key columns |
|------|-------------|-------------|
| `data/raw/ALLARMI.csv` | Alert/alarm events per route | Airport codes, date, alarm type (`MOTIVO_ALLARME`), occurrence type, count (`TOT`) |
| `data/raw/TIPOLOGIA_VIAGGIATORE.csv` | Traveler demographics per route | Nationality, gender, age bracket, document type, entry/alarm/investigation counts, control outcomes |

Both files cover approximately 13 months (Dec 2023 – Dec 2024) and require substantial cleaning due to encoding errors, inconsistent null representations (20+ variants), Italian date formats, and mixed ISO2/ISO3 country codes.

## Project Structure

```
classical-vs-multiagent/
├── README.md                       # This file
├── CLAUDE.md                       # Project-specific notes for Claude (architecture, decisions)
├── PROGETTO_RIEPILOGO.md           # Project summary and design choices (Italian)
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variables template
├── .gitignore
│
├── data/
│   ├── raw/                        # Raw CSV datasets (not tracked by git)
│   │   ├── ALLARMI.csv
│   │   └── TIPOLOGIA_VIAGGIATORE.csv
│   └── processed/                  # Pipeline outputs (not tracked by git)
│       ├── allarmi_clean.csv
│       ├── viaggiatori_clean.csv
│       ├── dataset_merged.csv
│       ├── features_classical.csv
│       ├── feature_cols.json
│       ├── features_with_baseline.csv
│       ├── baseline_stats.json
│       ├── anomaly_results.csv
│       ├── anomaly_summary.json
│       ├── final_report.csv
│       ├── risk_profiles.json
│       ├── evaluation_scorecard.json
│       ├── stability_scores.csv
│       ├── feature_importance.csv
│       ├── shap_importance.csv
│       ├── multiagent_report.json
│       └── multiagent_validation_report.json
│
├── shared/
│   ├── __init__.py
│   └── preprocessing.py            # Shared data cleaning (used by both pipelines)
│
├── classical_pipeline/
│   ├── __init__.py
│   ├── main.py                     # Script-based orchestrator (useful for time benchmarking)
│   ├── notebooks/                  # Step-by-step exploratory analysis
│   │   ├── 01_EDA.ipynb
│   │   ├── 02_feature_engineering.ipynb
│   │   ├── 03_baseline_construction.ipynb
│   │   ├── 04_anomaly_detection.ipynb
│   │   ├── 05_post_processing.ipynb
│   │   ├── 06_evaluation.ipynb
│   │   └── 07_comparison.ipynb     # Head-to-head comparison of both pipelines
│   ├── feature_correlation.png
│   ├── feature_distributions.png
│   └── top_routes_risk.png
│
├── multiagent_pipeline/
│   ├── __init__.py
│   ├── main.py                     # LangGraph orchestrator (StateGraph + conditional edges)
│   ├── state.py                    # Shared data contract (AgentState TypedDict)
│   ├── config.py                   # API keys and runtime flags
│   ├── agents/                     # The 5 specialized agents
│   │   ├── __init__.py
│   │   ├── data_agent.py           # Agent 1: Data loading and perimeter filtering
│   │   ├── feature_agent.py        # Agent 2: Feature engineering (54 features)
│   │   ├── baseline_agent.py       # Agent 3: Robust z-scores (MAD-based)
│   │   ├── outlier_agent.py        # Agent 4: Adaptive 4-model ensemble anomaly detection
│   │   └── report_agent.py         # Agent 5: LLM-powered narrative report
│   ├── src/
│   │   ├── __init__.py
│   │   └── features.py             # FeatureBuilder — 6 feature engineering classes
│   ├── tools/
│   │   ├── __init__.py
│   │   └── data_tools.py           # Shared utilities (filter_by_perimeter, etc.)
│   └── tests/
│       └── e2e_validation.py       # End-to-end validation suite (multiple perimeters)
│
├── streamlit_app/
│   ├── __init__.py
│   ├── app.py                      # Interactive web UI with classical vs multi-agent comparison
│   └── agent_graph.jsx             # React component visualizing the 5 agents
│
└── docs/
    ├── DEMO_CHECKLIST.md           # Pre-demo setup and walkthrough steps
    ├── Reply_projects.pdf          # Reply project brief
    └── resoconto_pipeline_classica.docx  # Classical pipeline summary (Italian)
```

## Setup

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

### Installation

```bash
# 1. Clone the repository
git clone <repository-url>
cd classical-vs-multiagent

# 2. Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and add your Anthropic API key (only needed for LLM report generation)
```
data/raw/ALLARMI.csv
data/raw/TIPOLOGIA_VIAGGIATORE.csv
```

These files are not tracked by git for confidentiality reasons.

## Running the Pipelines

### Classical Pipeline (end-to-end)

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | Only for LLM report | — | Anthropic API key for ReportAgent |
| `ANTHROPIC_MODEL` | No | `claude-sonnet-4-5-20250929` | Model used for report generation |
| `USE_LLM` | No | `false` | Enable LLM calls in ReportAgent |
| `DRY_RUN` | No | `true` | If true, generate placeholder reports without LLM calls |

### Placing the Raw Data

The raw CSV files must be placed manually in the `data/raw/` directory:

```
data/raw/ALLARMI.csv
data/raw/TIPOLOGIA_VIAGGIATORE.csv
```

These files are not tracked by git for confidentiality reasons.

## Running the Pipelines

### Shared Preprocessing (run once)

```bash
PYTHONPATH=. python shared/preprocessing.py
```

This produces the cleaned and merged datasets used by both pipelines.

### Classical Pipeline — Notebooks (interactive analysis)

```bash
jupyter notebook classical_pipeline/notebooks/
# Run 01_EDA → 07_comparison in order
```

### Classical Pipeline — Script (end-to-end, useful for time benchmarking)

```bash
# Full pipeline (all steps including evaluation)
PYTHONPATH=. python3 classical_pipeline/main.py

# Skip the evaluation step (faster)
PYTHONPATH=. python3 classical_pipeline/main.py --skip-eval

# Verbose logging
PYTHONPATH=. python3 classical_pipeline/main.py --verbose
```

**Output files** (all saved to `data/processed/`):

| Step | Output file(s) |
|------|----------------|
| Preprocessing | `allarmi_clean.csv`, `viaggiatori_clean.csv`, `dataset_merged.csv` |
| Feature Engineering | `features_classical.csv`, `feature_cols.json` |
| Baseline Construction | `features_with_baseline.csv`, `baseline_stats.json` |
| Anomaly Detection | `anomaly_results.csv`, `anomaly_summary.json` |
| Post-Processing | `final_report.csv`, `risk_profiles.json` |
| Evaluation | `evaluation_scorecard.json`, `stability_scores.csv`, `feature_importance.csv`, `shap_importance.csv` |

### Multi-Agent Pipeline (end-to-end)

```bash
# Full pipeline without LLM report (no API key needed)
PYTHONPATH=. python3 multiagent_pipeline/main.py

# With LLM report (requires ANTHROPIC_API_KEY in .env)
USE_LLM=true PYTHONPATH=. python3 multiagent_pipeline/main.py

# Interactive chain mode (choose perimeter, run agents step by step)
PYTHONPATH=. python3 -m multiagent_pipeline.agents.data_agent
```

### Streamlit Web Interface

The Streamlit app provides an interactive UI to configure filters, run the multi-agent pipeline, and **compare its output side-by-side with the classical pipeline**:

```bash
streamlit run streamlit_app/app.py
# Navigate to http://localhost:8501
```

UI features:
- Sidebar filters: year, departure country, departure/arrival airport, geographic zone
- One-click pipeline execution with dry-run toggle
- Tabs: anomaly results, **classical vs multi-agent comparison**, LLM report, stage details, raw JSON debug
- Live merge between multi-agent results and the static classical report (Pearson r, Spearman r, label agreement, top-N overlap)
- CSV download of results

### End-to-End Validation Tests

```bash
# Run all validation tests (no LLM calls, no cost)
PYTHONPATH=. python3 multiagent_pipeline/tests/e2e_validation.py

# Include optional LLM smoke test (requires API key, small perimeter)
RUN_LLM_SMOKE=true PYTHONPATH=. python3 multiagent_pipeline/tests/e2e_validation.py
```

The validation suite runs multiple test cases with different perimeters (full data, Algeria only, FCO airport only, zone 1 only) and writes results to `data/processed/multiagent_validation_report.json`.

## Pipeline Architecture

### Shared Preprocessing

Both pipelines use the **exact same** preprocessing module (`shared/preprocessing.py`), which handles:

- Automatic CSV separator detection
- Null value normalization (20+ variants including `N.D.`, `N/A`, `??`, `ZZ`, etc.)
- Italian date parsing (`GEN`, `FEB`, ..., `DIC`)
- ISO2-to-ISO3 country code conversion (~65 countries)
- Domain constraint enforcement (`INVESTIGATI <= ENTRATI`, `ALLARMATI <= ENTRATI`)
- Sparse column removal (>50% null threshold)
- Left-join merge of ALLARMI and TIPOLOGIA_VIAGGIATORE on airport pair + date

### Shared Feature Engineering

Both pipelines produce an **identical** set of 54 numerical features per route, organized in 6 classes:

| Class | Features | Source |
|-------|----------|--------|
| `OccurrencePivot` | ~30 occurrence-type columns | ALLARMI |
| `MotivoAllarmeFeatures` | `pct_interpol`, `pct_sdi`, `pct_nsis`, `pct_tsc`, `pct_manuale` | ALLARMI |
| `AllarmiAggregator` | `tot_allarmi_log`, `tasso_chiusura`, `tasso_rilevanza`, `false_positive_rate` | ALLARMI |
| `EsitiPivot` | `n_respinti`, `n_fermati`, `tasso_respinti`, `tasso_fermati`, `score_rischio_esiti` | VIAGGIATORI |
| `ViaggiatoriAggregator` | `tasso_allarme_medio`, `tasso_inv_medio`, `alarm_per_invest` | VIAGGIATORI |
| `FeatureBuilder` | `score_composito` (weighted combination) | Both |

### Adaptive Anomaly Detection Ensemble

Both pipelines use the **same 4-model ensemble**. On the full dataset (567 routes) the weights are:

| Model | Type | Weight | Purpose |
|-------|------|--------|---------|
| IsolationForest | Tree-based isolation | 35% | Global anomalies |
| Local Outlier Factor | Density-based | 30% | Contextual anomalies |
| Z-score Baseline | Statistical (MAD-based) | 15% | Multi-feature deviation |
| Autoencoder (MLPRegressor) | Reconstruction error | 20% | Non-linear patterns |

**Adaptive behaviour on small perimeters:** when a filter produces fewer than 30 normal routes, the Autoencoder is **excluded** (insufficient training samples for a stable reconstruction). The remaining weights are renormalized proportionally:

| Model | Reweighted (small perimeters) |
|-------|------------------------------|
| IsolationForest | ≈ 44% |
| Local Outlier Factor | ≈ 37% |
| Z-score Baseline | ≈ 19% |
| Autoencoder | 0% (excluded) |

This keeps the comparison between classical and multi-agent fair while avoiding noisy autoencoder scores on under-sampled subsets.

Risk labels are assigned using **data-driven thresholds**:
- **ALTA** (High): ensemble score ≥ p97 — top ~3% of routes
- **MEDIA** (Medium): ensemble score ≥ p90 — top ~10% of routes
- **NORMALE** (Normal): everything below

### Shared Constants

All shared constants (ensemble weights, thresholds, baseline features, file paths) are defined in `multiagent_pipeline/state.py` and mirrored in `classical_pipeline/main.py` to ensure a fair comparison.

### What Differs

| Aspect | Classical Pipeline | Multi-Agent Pipeline |
|--------|-------------------|---------------------|
| Orchestration | Single script, sequential function calls | LangGraph `StateGraph` with conditional edges |
| State management | Local variables, CSV files on disk | `AgentState` TypedDict flowing between nodes |
| Error handling | Try/except with early exit | Per-agent `meta["error"]` with optional continue-on-error |
| Report generation | Not included (static CSV output) | Optional LLM-powered narrative explanations (Agent 5) |
| Filtering | Runs on full dataset | Dynamic perimeter filtering (year, country, airport, zone) |
| UI | Notebooks | Streamlit web app |

### Multi-Agent Flow

```
START → DataAgent → [ok?] → FeatureAgent → [ok?] → BaselineAgent → [ok?] → OutlierAgent
              ↓ err              ↓ err                   ↓ err                ↓ ok    ↓ err
             END                END                     END            [anomalies?]  END
                                                                        ↓ yes  ↓ no
                                                                  ReportAgent  END
                                                                        ↓
                                                                       END
```

Agents communicate exclusively through `AgentState` (a shared TypedDict). No agent imports another directly. DataAgent through OutlierAgent are **deterministic** (no LLM) — only the ReportAgent uses an LLM (Anthropic Claude) for generating narrative explanations.

## Evaluation Metrics

The classical pipeline includes a full evaluation suite (Step 5 / Notebook 06):

- **Silhouette Score** — Measures cluster separation between anomalous and normal routes
- **Bootstrap Stability** (100 iterations) — Re-samples 80% of data and re-trains IsolationForest to check which anomalies are consistently detected (stability ≥ 70%)
- **Permutation Feature Importance** — Measures each feature's contribution to IsolationForest predictions
- **SHAP Explainability** — Surrogate GradientBoosting model to approximate SHAP-style feature importance

## Notebooks

The `classical_pipeline/notebooks/` directory contains the step-by-step exploratory analysis that informed the pipeline design:

| Notebook | Purpose |
|----------|---------|
| `01_EDA.ipynb` | Exploratory Data Analysis — data profiling, distributions, null patterns |
| `02_feature_engineering.ipynb` | Feature engineering — builds 54 features per route |
| `03_baseline_construction.ipynb` | Baseline — robust z-scores (MAD), Tukey IQR thresholds, anomaly flags |
| `04_anomaly_detection.ipynb` | Anomaly detection — 4-model ensemble, risk labels |
| `05_post_processing.ipynb` | Post-processing — business rules, confidence scores, risk profiles |
| `06_evaluation.ipynb` | Evaluation — silhouette, bootstrap stability, SHAP, feature importance |
| `07_comparison.ipynb` | Head-to-head comparison of classical vs multi-agent |

## Key Results

Top 5 anomalous routes detected by the ensemble (full dataset, 567 routes):

| Rank | Route | Country | Score | Label |
|------|-------|---------|-------|-------|
| 1 | ALG-MXP | Algeria | 0.5909 | ALTA |
| 2 | RAK-CIA | Morocco | 0.5253 | ALTA |
| 3 | GYD-FCO | Azerbaijan | 0.4608 | ALTA |
| 4 | CMN-BLQ | Morocco | 0.4583 | ALTA |
| 5 | RAK-TSF | Morocco | 0.4560 | ALTA |

Overall distribution: 17 ALTA (3.0%), 40 MEDIA (7.1%), 510 NORMALE (89.9%) out of 567 routes.

## Team

- **Daniele Giovanardi**
- **Filippo Nannucci**
- **Edoardo Riva**

## References

- [IsolationForest — scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
- [Local Outlier Factor — scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html)
- [LangGraph — State Machines](https://langchain-ai.github.io/langgraph/)
- [LangChain — Agents](https://docs.langchain.com/docs/components/agents)
- [SHAP — SHapley Additive exPlanations](https://shap.readthedocs.io/)
