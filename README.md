# Classical vs Multi-Agent: Airport Anomaly Detection

**Reply × LUISS — 2026**

## Overview

This project implements the same anomaly detection system twice:
1. **Classical Pipeline** — Feature engineering + IsolationForest/LOF/Z-score + rule-based post-processing
2. **Multi-Agent Architecture** — LLM-powered agents (Data, Baseline, Outlier Detection, Risk Profiling, Report)

The goal is to produce a comparative analysis arguing which approach is more convenient and under what operational conditions.

## Context

Border control authorities and airport operators manage large volumes of passenger transits daily, each associated with rich metadata: timestamp, gate, route, nationality, document type, control outcome, and security alerts.

## Project Structure

```
classical-vs-multiagent/
├── data/
│   ├── raw/              # Raw CSV/JSON datasets
│   └── processed/        # Dati clean + output ufficiali pipeline
├── classical_pipeline/   # Classical ML pipeline
├── multiagent_pipeline/  # Multi-agent system
│   ├── agents/           # Data/Feature/Baseline/Outlier/Report nodes
│   ├── tests/            # Smoke + E2E validation
│   └── main.py           # Orchestrator run_pipeline
├── streamlit_app/        # Frontend Streamlit
├── scripts/              # Script ad-hoc/runner manuali
├── docs/                 # Checklist demo e note operative
├── notebooks/            # Exploratory analysis
├── reports/              # Generated anomaly reports
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

## Usage

### Classical Pipeline
```bash
python classical_pipeline/main.py --input data/raw/transits.csv
```

### Multi-Agent Pipeline
```bash
PYTHONPATH=. python3 multiagent_pipeline/main.py
```

### Script Runner (ad-hoc)
```bash
PYTHONPATH=. python3 scripts/run_data_agent.py
```

### End-to-End Validation
```bash
# Sempre senza costi LLM
PYTHONPATH=. python3 multiagent_pipeline/tests/e2e_validation.py

# Smoke LLM opzionale (1 perimetro piccolo)
RUN_LLM_SMOKE=true PYTHONPATH=. python3 multiagent_pipeline/tests/e2e_validation.py
```

### Streamlit App
```bash
streamlit run streamlit_app/app.py
```

Interfaccia consigliata:
- sidebar con filtri perimetro (`anno`, `paese_partenza`, aeroporti, `zona`)
- esecuzione orchestrator end-to-end
- tab dedicati a anomalie, report e debug JSON
- toggle dry-run per test low-cost senza chiamate LLM

## Official Outputs

- `data/processed/multiagent_report.json`
- `data/processed/multiagent_validation_report.json`

## Team

- Daniele Giovanardi
- Filippo Nannucci
- Edoardo Riva

## References

- [IsolationForest — scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
- [LangChain Multi-Agent](https://docs.langchain.com/docs/components/agents)
