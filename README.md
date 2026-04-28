# Airport Risk Intelligence
**Reply × LUISS 2026 — Project 2**

---

## Who we are and what this is

We're a team of students competing in the **Reply × LUISS 2026** challenge. For Project 2 we were asked to build an anomaly detection system on real airport security data, and then argue which architectural approach works best.

Rather than just picking one approach and running with it, we decided to **build the same system twice** — once as a classical sequential pipeline (notebooks + Python scripts) and once as a **LangGraph multi-agent system** — so we could compare them properly. Same data, same features, same detection logic, two completely different architectures.

The question we're trying to answer: *when does adding agent orchestration actually buy you something over a well-written classical pipeline?*

---

## The problem

Border control at Italian airports generates a lot of data: every passenger transit, every security alert, every document check. Most of this data sits unused until something goes wrong.

Our system looks at **routes** — pairs of `departure_airport → arrival_airport` (e.g. `CAI-FCO`, Cairo to Rome Fiumicino) — and asks: *is this route behaving anomalously compared to all the others?*

Concretely, we're looking for routes with unusual combinations of:
- High alarm rates (Interpol, SDI, NSIS)
- High investigation and rejection rates
- Low closure rates (alarms that don't get resolved)
- Unusual traveler profiles

We have ~567 unique routes and ~13 months of data. The output is a risk label per route: **ALTA** (top 3%), **MEDIA** (top 10%), **NORMALE**.

---

## Our reasoning

### Why a classical pipeline first

We started classically because it forced us to understand the data properly. Six notebooks, step by step: EDA, feature engineering, baseline construction, anomaly detection, post-processing, evaluation. By the end we had 54 features per route, a robust statistical baseline (MAD-based z-scores rather than mean/std, which are too sensitive to outliers in security data), and a 4-model ensemble — IsolationForest, LOF, Z-score composite, and an Autoencoder.

The classical pipeline works well. It's reproducible, easy to audit, and produces good results. Its main limitation is rigidity: if you want to re-run on a different time window, or filter to a specific country, you re-run everything.

### Why we then built a multi-agent version

The multi-agent version (LangGraph) replicates the exact same logic but as a graph of five specialized agents, each responsible for one stage. The interesting part isn't the detection itself — it's what you gain architecturally:

- **Dynamic filtering**: you can pass a `perimeter` (year, country, airport, zone) at runtime and only the relevant data flows through
- **LLM explanations**: a ReportAgent uses Claude to generate plain-English narratives for each anomalous route, citing the specific z-score drivers
- **Modularity**: each agent can fail, retry, or be swapped independently — you don't re-run the whole thing if the feature step changes

The tradeoff is complexity. A classical pipeline is easier to debug and faster to iterate on. A multi-agent system is more flexible and scales better when requirements change.

### What we found

The two systems agree on ~94% of labels and produce scores with Pearson r ≈ 0.89. The top-20 most anomalous routes overlap 17/20. So the architectures converge on the same answer — the multi-agent version just gets there in a way that's operationally more useful.

---

## Results

![Top anomalous routes](images/top_routes_risk.png)

![Feature distributions](images/feature_distributions.png)

![Feature correlation](images/feature_correlation.png)

---

## Project structure

```
classical-vs-multiagent/
│
├── README.md
├── requirements.txt
├── .env.example
│
├── images/                            # Charts and visualisations
│   ├── top_routes_risk.png
│   ├── feature_distributions.png
│   └── feature_correlation.png
│
├── data/
│   ├── raw/                           # Source CSVs (gitignored — confidential)
│   │   ├── ALLARMI.csv
│   │   └── TIPOLOGIA_VIAGGIATORE.csv
│   └── processed/                     # Pipeline outputs (gitignored)
│
├── classical_pipeline/                # ── Pipeline 1 ──────────────────────
│   ├── main.py                        # Run everything as a single script
│   └── notebooks/
│       ├── 01_EDA.ipynb
│       ├── 02_feature_engineering.ipynb
│       ├── 03_baseline_construction.ipynb
│       ├── 04_anomaly_detection.ipynb
│       ├── 05_post_processing.ipynb
│       └── 06_evaluation.ipynb
│
├── multiagent_pipeline/               # ── Pipeline 2 ──────────────────────
│   ├── main.py                        # LangGraph entry point
│   ├── state.py                       # Shared AgentState schema
│   ├── config.py                      # API key and model config
│   ├── agents/
│   │   ├── data_agent.py              # Loads and filters raw CSVs
│   │   ├── feature_agent.py           # Builds 54 features per route
│   │   ├── baseline_agent.py          # Robust z-scores via MAD
│   │   ├── outlier_agent.py           # 4-model weighted ensemble
│   │   └── report_agent.py            # LLM narrative explanations
│   ├── src/
│   │   └── features.py                # FeatureBuilder (shared with classical)
│   ├── tools/
│   │   └── data_tools.py              # Perimeter filtering helpers
│   └── tests/
│       └── e2e_validation.py          # End-to-end validation
│
├── shared/
│   └── preprocessing.py               # Data cleaning used by both pipelines
│
├── streamlit_app/                     # ── Dashboard ────────────────────────
│   ├── app.py                         # Streamlit application
│   └── agent_graph.jsx                # Animated React agent flow diagram
│
├── notebooks/
│   └── 07_comparison_classical_vs_multiagent.ipynb
│
└── docs/
    ├── Reply_projects.pdf
    └── DEMO_CHECKLIST.md
```

---

## How to run it

### Setup

```bash
git clone https://github.com/<your-org>/classical-vs-multiagent.git
cd classical-vs-multiagent
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Add the raw data files:
```
data/raw/ALLARMI.csv
data/raw/TIPOLOGIA_VIAGGIATORE.csv
```

### Classical pipeline

Open and run the notebooks in order:
```bash
jupyter lab classical_pipeline/notebooks/
```

Or run everything at once:
```bash
python classical_pipeline/main.py
```

### Multi-agent pipeline

```python
from multiagent_pipeline.main import run_pipeline

# Without LLM (no API key needed)
state, summary = run_pipeline({"anno": 2024}, run_report=False)

# With LLM explanations (needs ANTHROPIC_API_KEY in .env)
state, summary = run_pipeline(
    {"anno": 2024},
    run_report=True,
    use_llm=True,
    save_outputs=True,
)
print(state["report"]["summary"])
```

### Dashboard (the nicest way to see everything)

```bash
streamlit run streamlit_app/app.py
```

Opens at `http://localhost:8501`. From here you can:
- Run the multi-agent pipeline with any filter combination
- See the agent graph animate as each stage completes
- Explore the route map — click any route to see its risk details and LLM explanation
- Compare classical vs multi-agent scores side by side

### LLM report (optional)

The ReportAgent calls Claude to generate plain-English explanations for each anomalous route. To enable it:

```bash
cp .env.example .env
# Add your key: ANTHROPIC_API_KEY=sk-ant-...
```

Then check **Enable LLM Report** in the dashboard sidebar before running. Without a key, you can use **Dry run** mode which runs the full pipeline but skips the API calls.

---

## Tech stack

- **Data & ML**: pandas, numpy, scipy, scikit-learn, PyOD
- **Agent orchestration**: LangGraph, LangChain
- **LLM**: Anthropic Claude (`claude-sonnet-4-5`)
- **Dashboard**: Streamlit, Plotly, Altair
- **Agent visualisation**: React 18 + Babel (embedded in Streamlit)
- **Explainability**: SHAP

---

*Reply × LUISS 2026*
