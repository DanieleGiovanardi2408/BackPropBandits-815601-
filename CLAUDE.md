# Classical vs Multi-Agent — Reply × LUISS 2026 Project

## Project description
Comparison between a classical anomaly detection pipeline (sequential Jupyter notebooks)
and a multi-agent system (LangGraph) on the same airport dataset.
Goal: implement the same system twice and produce a comparative analysis
to demonstrate which architecture is more convenient and under which operational conditions.

## Dataset
- `data/raw/ALLARMI.csv` — alarm occurrences per flight/route
- `data/raw/TIPOLOGIA_VIAGGIATORE.csv` — passenger demographic data
- Granularity: 1 row = 1 occurrence per route/date
- ~5080 rows, 567 distinct routes (departure airport → arrival airport)

## Repo structure

```
shared/
  preprocessing.py          # Data cleaning — SHARED between both pipelines

classical_pipeline/
  notebooks/
    01_EDA.ipynb → 07_comparison.ipynb   # Complete classical pipeline
  main.py                  # Script orchestrator (auto-run, useful for time benchmarks)

multiagent_pipeline/
  main.py                 # LangGraph orchestrator (StateGraph + conditional edges)
  state.py                # Data contract (AgentState TypedDict + Pydantic schemas)
  config.py               # API keys and runtime configuration
  agents/
    data_agent.py          # Loads dataset, filters by perimeter
    feature_agent.py       # Feature engineering (wraps FeatureBuilder)
    baseline_agent.py      # Robust z-score (MAD-based)
    outlier_agent.py       # IF + LOF + Z-score + Autoencoder ensemble (sklearn)
    report_agent.py        # Narrative report via LLM (Anthropic Claude)
  src/
    features.py            # FeatureBuilder — 6 feature engineering classes
  tools/
    data_tools.py          # Utilities: filter_by_perimeter, load_last_perimeter
  tests/
    e2e_validation.py      # End-to-end regression on 5 perimeters

streamlit_app/
  app.py                   # Streamlit UI to run the pipeline

docs/                      # Demo checklist, Reply brief, classical summary, agent_graph.jsx (React visualizer)
```

## Multi-agent pipeline — LangGraph architecture

The orchestrator in `main.py` uses `StateGraph(AgentState)` with conditional edges:

```
START → DataAgent → [ok?] → FeatureAgent → [ok?] → BaselineAgent → [ok?] → OutlierAgent
              ↓ err              ↓ err                   ↓ err                ↓     ↓ err
             END                END                     END             [report?]  END
                                                                        ↓ yes ↓ no
                                                                  ReportAgent  END
                                                                        ↓
                                                                       END
```

Each node returns only the fields it writes (delta update). LangGraph merges them into the state.
If `continue_on_error=True`, the conditional edges do not stop the graph on the first error.

## How to run

### Multi-agent pipeline — complete
```python
from multiagent_pipeline.main import run_pipeline
state, summary = run_pipeline(perimeter={"anno": 2024})
```

### Classical pipeline — script (for time benchmarks)
```bash
PYTHONPATH=. python3 classical_pipeline/main.py [--skip-eval] [--verbose]
```

### Multi-agent pipeline — interactive
```bash
python3 -m multiagent_pipeline.agents.data_agent
# Interactive menu: choose filters, then run agents in chain
```

### E2E validation
```bash
PYTHONPATH=. python3 multiagent_pipeline/tests/e2e_validation.py
```

### Streamlit app
```bash
streamlit run streamlit_app/app.py
```

## Important technical notes
- **Preprocessing is shared**: both pipelines use the same cleaned CSVs
- **Fair comparison**: same features, same hyperparameters (contamination=0.10, n_estimators=200, random_state=42)
- **Ensemble weights**: IF=0.35, LOF=0.30, Z=0.15, AE=0.20 — identical to the classical pipeline
- **Data-driven thresholds**: p97 = ALTA, p90 = MEDIA (recomputed at runtime)
- **Adaptive Autoencoder**: below 30 normal routes, the Autoencoder is excluded from the ensemble and weights are redistributed proportionally among IF, LOF and Z-score (IF≈44%, LOF≈37%, Z≈19%)
- **Deterministic agents**: DataAgent, FeatureAgent, BaselineAgent, OutlierAgent do not use LLM — only ReportAgent uses Anthropic Claude
- **Airports**: IATA codes uppercase (e.g. FCO, MXP, TIA)
- **Perimeter**: filters — anno, aeroporto_partenza, aeroporto_arrivo, paese_partenza, zona (1-9)

## Key dependencies
```
pandas, numpy, scikit-learn, langgraph, langchain, langchain-anthropic
streamlit, python-dotenv, pydantic
```

## LLM config (for ReportAgent)
Create a `.env` file in the root with:
```
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-4-5-20250929
```
Without an API key the ReportAgent runs in dry_run mode (placeholders instead of LLM explanations).

## Key results
- 567 routes analysed, 83 countries
- Classical: ALTA=17, MEDIA=40, NORMALE=510
- Multiagent: ALTA=17, MEDIA=40, NORMALE=510 (same distribution)
- Pearson r=0.84, Spearman r=0.86, Label agreement=90.8%
