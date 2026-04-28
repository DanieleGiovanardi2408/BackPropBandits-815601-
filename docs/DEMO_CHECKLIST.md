# Demo Checklist

## Pre-demo
- `cp .env.example .env` and set `ANTHROPIC_API_KEY`
- install dependencies: `pip install -r requirements.txt`
- verify smoke test (no LLM): `PYTHONPATH=. python3 multiagent_pipeline/tests/e2e_validation.py`

## Terminal demo
- end-to-end orchestrator, no LLM
  - `PYTHONPATH=. python3 multiagent_pipeline/tests/e2e_validation.py`
- LLM smoke on a small perimeter
  - `RUN_LLM_SMOKE=true PYTHONPATH=. python3 multiagent_pipeline/tests/e2e_validation.py`
- expected output
  - test report at `data/processed/multiagent_validation_report.json`
  - summary shows `n_failed: 0`

## Streamlit UI demo
- start: `streamlit run streamlit_app/app.py`
- show:
  - perimeter filters in sidebar (year, country, airport, zone)
  - run pipeline with `Dry run` mode active
  - **Anomalies tab** — risk table + CSV download
  - **Report tab** — LLM summary + per-route findings + JSON download
  - **Stage detail tab** — agent statuses, timing, errors
  - **Comparison tab** — classical vs multi-agent scores side-by-side, label concordance scatter

## Official outputs to show
- `data/processed/multiagent_report.json`
- `data/processed/multiagent_validation_report.json`
- `notebooks/07_comparison_classical_vs_multiagent.ipynb` — quantitative comparison
