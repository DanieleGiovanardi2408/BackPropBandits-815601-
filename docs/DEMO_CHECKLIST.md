# Demo Checklist

## Pre-demo
- `cp .env.example .env` e configurare `ANTHROPIC_API_KEY`
- installare dipendenze: `pip install -r requirements.txt`
- verificare smoke no-LLM: `PYTHONPATH=. python3 multiagent_pipeline/tests/e2e_validation.py`

## Demo terminale
- orchestrator end-to-end no-LLM
  - `PYTHONPATH=. python3 multiagent_pipeline/tests/e2e_validation.py`
- smoke LLM su perimetro piccolo
  - `RUN_LLM_SMOKE=true PYTHONPATH=. python3 multiagent_pipeline/tests/e2e_validation.py`
- expected output
  - report test in `data/processed/multiagent_validation_report.json`
  - summary con `n_failed: 0`

## Demo UI Streamlit
- avvio: `streamlit run streamlit_app/app.py`
- mostrare:
  - filtri perimetro in sidebar
  - run pipeline con `dry_run` attivo
  - tab anomalie (tabella + download CSV)
  - tab report (summary + findings + download JSON)
  - tab stage detail (stati/tempi/errori)

## Output ufficiali da mostrare
- `data/processed/multiagent_report.json`
- `data/processed/multiagent_validation_report.json`
