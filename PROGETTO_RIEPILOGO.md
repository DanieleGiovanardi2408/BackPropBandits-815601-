# Classical vs Multi-Agent — Project Summary
**Reply × LUISS 2026**

---

## Context

The project is commissioned by Reply for LUISS. The real client is represented by border control authorities and airport operators who manage large volumes of passenger transits. Each transit has associated metadata: timestamp, gate, route, nationality, document type, control outcome, and security alarms.

The current problem is that anomaly detection is reactive. The goal is to build a proactive analytical system capable of identifying suspicious patterns before they become operational or security incidents.

The project requires implementing the same anomaly detection system **twice**: once with a classical pipeline and once with a multi-agent architecture. The final output must be a comparative analysis arguing which approach is more convenient and under which operational conditions.

---

## Dataset

The input dataset is a CSV file with Italian airport transit data (source: ALLARMI and TIPOLOGIA_VIAGGIATORE datasets). The main columns are route, departure country, geographic zone, occurrence type, alarm reason, control outcome, and number of passengers entered, alarmed, and investigated.

Preprocessing is **shared** between the classical and multi-agent pipelines, ensuring a fair comparison. It produces three clean files: `allarmi_clean.csv`, `viaggiatori_clean.csv`, `dataset_merged.csv`.

---

## Classical Pipeline (already implemented)

The classical pipeline follows a linear 5-step flow implemented in Jupyter notebooks:

1. **Feature Engineering** — 54 numeric features aggregated per route, built from occurrence pivots, alarm reason percentages, closure/relevance rates and passenger demographic profiles. Key features: `tot_allarmi_log`, `pct_interpol`, `score_rischio_esiti`, `tasso_rilevanza`.

2. **Baseline Construction** — historical baseline for 11 key features, with z-scores to measure each route's deviation from the norm.

3. **Anomaly Detection** — ensemble of 4 models: IsolationForest (weight 35%), LOF (30%), Z-score (15%), Autoencoder (20%). Final composite score from 0 to 1.

4. **Post-Processing** — data-driven thresholds (p97/p90) to classify routes as ALTA, MEDIA, NORMALE. 17 ALTA routes, 40 MEDIA, 510 NORMALE out of 567 total.

5. **Output** — `final_report.csv` with ranked routes, scores and risk metrics. Static report.

---

## Multi-Agent Architecture (in development)

### Why an agent-based architecture

An agent is an LLM combined with a Think → Act → Observe → Reply loop. Tools are Python functions that the LLM can decide to call. The added value over the classical approach lies not in the algorithms (they are the same) but in three aspects: flexibility of the analysis perimeter, autonomous decision-making on which algorithm to use, and a narrative report generated dynamically by an LLM instead of a static output.

### The 5 agents

The chosen architecture is **Supervisor**: a LangGraph orchestrator coordinates 5 specialised agents in sequence.

**Agent 1 — DataAgent**
Loads the dataset and filters it by the perimeter defined by the user (year, airport, country, zone). It is a deterministic agent: it does not use an LLM, it always executes the same steps. Tools: `load_dataset`, `filter_by_perimeter`, `get_dataset_stats`. Output: filtered DataFrame + metadata.

**Agent 2 — FeatureAgent**
Recomputes the same 54 features as the classical pipeline using the classes in `src/features.py`. Ensures the comparison is fair: same feature engineering, different architecture. Tools: `build_features`, `get_feature_cols`. Output: DataFrame aggregated per route.

**Agent 3 — BaselineAgent**
Loads the pre-computed baseline or recomputes it dynamically. Computes z-scores for each route relative to the historical baseline. Tools: `load_baseline`, `compute_zscore`. Output: DataFrame with deviations from the norm.

**Agent 4 — OutlierAgent**
Applies IsolationForest, LOF, Z-score and Autoencoder with the same weights as the classical pipeline (35/30/15/20). Produces the same output format as the classical pipeline, enabling direct comparison. Tools: `run_isolation_forest`, `run_lof`, `run_zscore`, `run_autoencoder`, `ensemble_scores`. Output: routes with scores, ALTA/MEDIA/NORMALE labels and ranking.

**Agent 5 — ReportAgent**
The only agent with a real LLM. Takes the anomalous routes and generates a narrative explanation for each one in natural language. This is the main differentiator from the classical pipeline, which only produces numbers. Tools: `format_route_for_llm`, `generate_explanation`, `export_report`. Output: JSON with textual explanations for each anomaly.

### Important note on LLM usage

In the multi-agent system, not all agents use an LLM. DataAgent, FeatureAgent, BaselineAgent and OutlierAgent are deterministic: they know exactly what to do and do not need autonomous reasoning. Only the ReportAgent uses an LLM, because generating narrative explanations is an ambiguous task that benefits from natural language. This distinction is a deliberate architectural choice, relevant to highlight in the final presentation.

### Technology stack

- **LangGraph** — `StateGraph` with conditional edges to orchestrate the flow between agents
- **LangChain** — integration with Anthropic API for the ReportAgent
- **Anthropic API** — LLM (Claude) for generating narrative explanations
- **scikit-learn** — IsolationForest, LOF, MLPRegressor (autoencoder), StandardScaler
- **pandas, numpy** — feature engineering and baseline
- **Streamlit** — final user interface

---

## Repository structure

```
classical-vs-multiagent/
├── shared/
│   └── preprocessing.py          ← shared between both pipelines
├── classical_pipeline/
│   ├── notebooks/                ← classical pipeline (01-07)
│   └── main.py                   ← script orchestrator (useful for time benchmarks)
├── multiagent_pipeline/
│   ├── main.py                   ← LangGraph orchestrator (StateGraph + conditional edges)
│   ├── state.py                  ← shared data contract (AgentState TypedDict)
│   ├── config.py                 ← API keys and runtime flags
│   ├── agents/
│   │   ├── data_agent.py         ← loads and filters dataset by perimeter
│   │   ├── feature_agent.py      ← 54 features aggregated per route
│   │   ├── baseline_agent.py     ← robust z-score (MAD) on 11 key features
│   │   ├── outlier_agent.py      ← IF + LOF + Z + AE ensemble (real sklearn)
│   │   └── report_agent.py       ← narrative explanations via LLM
│   ├── src/
│   │   └── features.py           ← 6 feature engineering classes
│   ├── tools/
│   │   └── data_tools.py         ← shared utilities (perimeter filter, etc.)
│   └── tests/
│       └── e2e_validation.py     ← end-to-end regression on 5 perimeters
├── data/
│   ├── raw/                      ← ALLARMI.csv, TIPOLOGIA_VIAGGIATORE.csv
│   └── processed/                ← cleaned outputs and pipeline results
├── streamlit_app/
│   └── app.py                    ← interactive UI
├── docs/                         ← demo checklist, Reply brief, summary, agent_graph.jsx (React visualizer)
└── requirements.txt
```

---

## Development order

1. **Data contract** ✅ — `state.py` with `AgentState`, Pydantic schemas, shared constants
2. **src classes** ✅ — `src/features.py` with 6 classes, perfect match with the classical pipeline (567/567 routes, diff = 0.000000)
3. **Python tools** ✅ — pure testable functions in isolation (`tools/data_tools.py`)
4. **Agents in isolation** ✅ — each agent runs and is tested independently
5. **LangGraph orchestrator** ✅ — `StateGraph` with nodes and conditional edges in `main.py`
6. **ReportAgent with LLM** ✅ — generates narrative explanations via Anthropic Claude
7. **Streamlit UI** ✅ — sidebar filters, pipeline execution, anomaly/report/debug tabs
8. **E2E validation** ✅ — 5 regression perimeters, all green

---

## Streamlit interface

The user selects filters (year, airport, country) and sees three sections:

**Anomalous routes table** — sorted by score, with columns for route, country, risk level (ALTA/MEDIA/NORMALE), numeric score and number of models that flagged it.

**Narrative report** — for each ALTA route, a natural language explanation generated by the ReportAgent describing the anomalous pattern, the detected signals and possible causes.

**Classical vs multi-agent comparison** — the same routes with both scores side by side and a scatter plot showing the concordance between the two approaches. This is the most important section for the presentation with Reply.

The final report is saved to disk as `multiagent_report.json`.

---

## Checkpoint questions — CLOSED

- ~~Does every agent need an LLM, or is it enough for the system as a whole to be agentic?~~ → **No**, it is enough for the system to be agentic overall. Only the ReportAgent uses an LLM; the other 4 agents are deterministic by architectural choice.
- ~~How do you want us to evaluate the quality of detected anomalies, given that there is no ground truth?~~ → Concordance between the two approaches (Pearson, Spearman, label agreement) + qualitative evaluation of the narrative reports.
- ~~By "realistic tools" do you mean calls to external APIs/databases, or are well-defined Python functions sufficient?~~ → Well-defined Python functions are sufficient; what matters is that they are testable in isolation.
- ~~Is the Streamlit UI required for the checkpoint or only for the final presentation?~~ → Required for the final presentation, useful for the demo.
- ~~In your Reply experience, under which operational conditions does the classical approach beat the multi-agent?~~ → The classical approach is more convenient when the perimeter is fixed and narrative reporting is not needed; the multi-agent is better for exploratory analysis on variable perimeters.

---

## Comparison metrics

To argue which approach is more convenient, the metrics to measure are:

- **Concordance** between the two approaches on flagged routes (% agreement)
- **End-to-end execution time** on the same dataset
- **Perimeter flexibility** (the multi-agent is more adaptable)
- **Report quality** (qualitative assessment: static numbers vs narrative explanations)
- **Operational cost** (the multi-agent calls a paid LLM API)
