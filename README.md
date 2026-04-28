# Airport Risk Intelligence
### Classical Pipeline vs Multi-Agent System — Reply × LUISS 2026

A head-to-head comparison of two architectures solving the same anomaly detection problem on real airport security data. The same analytical logic is implemented **twice** — once as a classical sequential pipeline, once as a LangGraph multi-agent system — then compared across accuracy, flexibility, and explainability.

---

## What this project does

Border control authorities at Italian airports generate thousands of security alerts per month. This project builds a **proactive** system that identifies which international routes exhibit anomalous security patterns — elevated alarm rates, unusual investigation outcomes, high rejection rates — before they become operational incidents.

The unit of analysis is the **route** (`departure_airport–arrival_airport`, e.g. `CAI-FCO`). Across 567 unique routes the system detects anomalies by combining statistical baselines with a 4-model ensemble.

---

## Two Architectures, One Problem

|  | Classical Pipeline | Multi-Agent Pipeline |
|--|--|--|
| **Paradigm** | Sequential Jupyter notebooks | LangGraph agent graph |
| **Entry point** | `classical_pipeline/main.py` | `multiagent_pipeline/main.py` |
| **Orchestration** | Linear, step-by-step | 5 autonomous agents with conditional edges |
| **Filtering** | Full dataset only | Dynamic perimeter (year / country / airport / zone) |
| **Explainability** | Statistical summaries | LLM narrative (Claude Sonnet) |
| **Output** | `data/processed/final_report.csv` | JSON report + live Streamlit dashboard |

---

## Architecture

### Classical Pipeline

```
ALLARMI.csv ──┐
               ├──▶ Preprocessing ──▶ Feature Engineering ──▶ Baseline Construction
VIAGGIATORI.csv┘       (cleaning)       (54 features/route)    (robust z-scores MAD)
                                                                        │
                                                                        ▼
                                                           Anomaly Detection
                                                           (IsolationForest + LOF
                                                            + Z-score + Autoencoder)
                                                                        │
                                                                        ▼
                                                           Post-processing ──▶ Evaluation
                                                           (business rules,    (scorecard,
                                                            risk labels)        SHAP, stability)
```

### Multi-Agent Pipeline (LangGraph)

```
          START
            │
            ▼
       ┌──────────┐   error
       │DataAgent │──────────────────────────────────────▶ END
       │          │   loads & filters CSVs
       └────┬─────┘
            │ ok
            ▼
      ┌───────────────┐   error
      │FeatureAgent   │─────────────────────────────────▶ END
      │               │   engineers 54 features per route
      └───────┬───────┘
              │ ok
              ▼
      ┌────────────────┐   error
      │BaselineAgent   │──────────────────────────────▶ END
      │                │   robust z-scores (MAD)
      └────────┬───────┘
               │ ok
               ▼
       ┌──────────────┐   error
       │OutlierAgent  │──────────────────────────────▶ END
       │              │   4-model weighted ensemble
       └──────┬───────┘
              │ ok
              ├── no ALTA/MEDIA routes ──────────────▶ END
              │
              │ run_report=True
              ▼
       ┌─────────────┐
       │ReportAgent  │   LLM explanations per anomalous route
       │  (optional) │   Claude Sonnet via Anthropic API
       └──────┬──────┘
              ▼
             END
```

Each agent reads from and writes to a shared `AgentState` TypedDict. Conditional edges implement stop-on-error logic and skip the ReportAgent when not needed.

---

## Dataset

Two raw CSV files (not tracked by git — confidential):

| File | Rows | Description |
|------|------|-------------|
| `data/raw/ALLARMI.csv` | ~8,400 | Security alarm events per route — airport codes, date, alarm type, count |
| `data/raw/TIPOLOGIA_VIAGGIATORE.csv` | ~50,000 | Traveler demographics — nationality, document type, entry/alarm/investigation/outcome counts |

- **567 unique routes** across ~13 months (Dec 2023 – Dec 2024)
- Extensive cleaning required: 20+ null representations, Italian date formats, mixed ISO2/ISO3 country codes
- **54 numerical features** engineered per route across 6 semantic classes
- Output risk labels: `ALTA` (high risk, ~top 3%), `MEDIA` (medium, ~top 10%), `NORMALE`

---

## Project Structure

```
classical-vs-multiagent/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment variables template
├── .gitignore
│
├── data/
│   ├── raw/                           # Source CSVs (gitignored — confidential)
│   │   ├── ALLARMI.csv
│   │   └── TIPOLOGIA_VIAGGIATORE.csv
│   └── processed/                     # Pipeline outputs (gitignored)
│
├── classical_pipeline/                # ── PIPELINE 1 ──────────────────────
│   ├── main.py                        # Runs the full classical pipeline end-to-end
│   └── notebooks/
│       ├── 01_EDA.ipynb               # Exploratory data analysis
│       ├── 02_feature_engineering.ipynb
│       ├── 03_baseline_construction.ipynb
│       ├── 04_anomaly_detection.ipynb
│       ├── 05_post_processing.ipynb
│       └── 06_evaluation.ipynb        # Scorecard, SHAP, stability analysis
│
├── multiagent_pipeline/               # ── PIPELINE 2 ──────────────────────
│   ├── main.py                        # LangGraph orchestrator — run_pipeline()
│   ├── state.py                       # AgentState TypedDict (shared state schema)
│   ├── config.py                      # API key + model configuration
│   ├── agents/
│   │   ├── data_agent.py              # Agent 1 — loads & filters raw CSVs
│   │   ├── feature_agent.py           # Agent 2 — engineers 54 route features
│   │   ├── baseline_agent.py          # Agent 3 — robust z-scores via MAD
│   │   ├── outlier_agent.py           # Agent 4 — 4-model weighted ensemble
│   │   └── report_agent.py            # Agent 5 — LLM narrative explanations
│   ├── src/
│   │   └── features.py                # FeatureBuilder class (shared with classical)
│   ├── tools/
│   │   └── data_tools.py              # Perimeter filtering utilities
│   └── tests/
│       └── e2e_validation.py          # End-to-end validation suite
│
├── shared/
│   └── preprocessing.py               # Data cleaning — used by both pipelines
│
├── streamlit_app/                     # ── DASHBOARD ───────────────────────
│   ├── app.py                         # Main Streamlit application
│   └── agent_graph.jsx                # React animated agent flow visualisation
│
├── notebooks/
│   └── 07_comparison_classical_vs_multiagent.ipynb  # Head-to-head comparison
│
└── docs/
    ├── Reply_projects.pdf             # Competition brief
    └── DEMO_CHECKLIST.md
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/<your-org>/classical-vs-multiagent.git
cd classical-vs-multiagent
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add your data

Place the raw files in:
```
data/raw/ALLARMI.csv
data/raw/TIPOLOGIA_VIAGGIATORE.csv
```

### 3. (Optional) Configure the Anthropic API key

```bash
cp .env.example .env
# Edit .env and add:  ANTHROPIC_API_KEY=sk-ant-...
```

The pipeline works fully without an API key — the LLM report step is optional and can be replaced with a dry-run placeholder.

---

## Running the Pipelines

### Classical Pipeline

```bash
# Option A — explore notebooks step by step (recommended)
jupyter lab classical_pipeline/notebooks/

# Option B — run everything as a single script
python classical_pipeline/main.py
```

Outputs written to `data/processed/`: `final_report.csv`, `anomaly_results.csv`, evaluation scorecard JSON.

### Multi-Agent Pipeline (Python API)

```python
from multiagent_pipeline.main import run_pipeline

# Fast run — no API key needed
state, summary = run_pipeline({"anno": 2024}, run_report=False)
print(summary["stages"])            # per-agent status + elapsed time
print(state["df_anomalies"].shape)  # (n_routes, n_features)

# Full run with LLM explanations (requires ANTHROPIC_API_KEY)
state, summary = run_pipeline(
    perimeter={"anno": 2024, "paese_partenza": "EG"},
    run_report=True,
    use_llm=True,
    save_outputs=True,
)
print(state["report"]["summary"])
```

### Streamlit Dashboard (recommended)

```bash
streamlit run streamlit_app/app.py
```

Opens at **http://localhost:8501** — no extra configuration needed.

---

## Dashboard Features

**🗺️ Route Map**
Interactive Plotly globe with routes coloured by risk level. Click any departure marker to open the route detail panel. Italian arrival airports shown as cyan diamonds. LLM explanation shown inline when the ReportAgent has run.

**⚖️ Classical vs Multi-Agent**
Score correlation scatter plot (Pearson + Spearman r), label agreement matrix, and top-N overlap between both pipelines. Useful for validating that the two architectures agree on which routes are anomalous.

**⚠️ Anomalies**
Sortable table of all routes ranked by ensemble score, plus a risk distribution bar chart and CSV download.

**📋 Report**
Full LLM-generated narrative for each ALTA/MEDIA route, citing the top-3 anomaly drivers with their values and σ deviations from the baseline.

**🔧 Stage Detail**
Per-agent elapsed time, error messages, and run history for the current session.

---

## Key Results

| Metric | Value |
|--------|-------|
| Routes analysed | 567 |
| ALTA risk routes | ~4 (top 1%) |
| MEDIA risk routes | ~53 (top 10%) |
| Pearson r (classical vs MA scores) | ~0.89 |
| Spearman ρ | ~0.91 |
| Label agreement | ~94% |
| Top-20 overlap | 17 / 20 |

The two architectures produce highly correlated risk rankings, validating the multi-agent approach. Key differentiators:

- **Flexibility** — the multi-agent pipeline supports runtime perimeter filters (year, country, airport, zone) without re-running the full pipeline
- **Explainability** — the ReportAgent generates human-readable narratives citing specific statistical evidence (z-scores, feature values)
- **Modularity** — each agent can be tested, swapped, or extended independently; a new data source or detection model only affects the relevant agent

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Data processing | pandas, numpy, scipy |
| Anomaly detection | scikit-learn (IsolationForest, LOF), PyOD (Autoencoder) |
| Statistical baselines | MAD-based robust z-scores, Tukey IQR |
| Agent orchestration | LangGraph (StateGraph), LangChain |
| LLM explanations | Anthropic Claude (`claude-sonnet-4-5`) |
| Dashboard | Streamlit, Plotly, Altair |
| Agent visualisation | React 18 + Babel Standalone (embedded in Streamlit iframe) |
| Feature importance | SHAP |

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Optional | Enables LLM report in ReportAgent |
| `ANTHROPIC_MODEL` | Optional | Override model (default: `claude-sonnet-4-5`) |
| `USE_LLM` | Optional | `true`/`false` — enables LLM globally |
| `DRY_RUN` | Optional | `true` — skips API calls, returns placeholder text |

---

*Reply × LUISS 2026 — Project 2: Classical vs Multi-Agent*
