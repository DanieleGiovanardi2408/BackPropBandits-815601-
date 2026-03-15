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
│   └── processed/        # Cleaned & feature-engineered data
├── classical_pipeline/   # Classical ML pipeline
├── multiagent_pipeline/  # Multi-agent system
├── streamlit_app/        # Streamlit UI
├── notebooks/            # Exploratory analysis
├── reports/              # Generated anomaly reports
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Classical Pipeline
```bash
python classical_pipeline/main.py --input data/raw/transits.csv
```

### Multi-Agent Pipeline
```bash
python multiagent_pipeline/main.py --input data/raw/transits.csv
```

### Streamlit App
```bash
streamlit run streamlit_app/app.py
```

## Team

- Daniele Giovanardi
- Filippo Nannucci
- Edoardo Riva

## References

- [IsolationForest — scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
- [LangChain Multi-Agent](https://docs.langchain.com/docs/components/agents)
