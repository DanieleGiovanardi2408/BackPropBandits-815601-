"""ReportAgent — quinto nodo del grafo multi-agent.

Responsabilità:
    Usa un LLM reale per spiegare in linguaggio naturale le rotte anomale
    (ALTA/MEDIA) e costruire il report finale JSON.
"""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
    __package__ = "multiagent_pipeline.agents"

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from langchain_anthropic import ChatAnthropic

from multiagent_pipeline.config import get_anthropic_api_key, get_anthropic_model
from multiagent_pipeline.state import AgentState, PATHS

logger = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def format_route_for_llm(row: dict) -> str:
    """Tool 1 — format_route_for_llm(row)

    Prepara il contesto sintetico di una rotta per il prompt LLM.
    """
    return (
        f"ROTTA={row.get('ROTTA', 'ND')}; "
        f"risk_label={row.get('risk_label', 'ND')}; "
        f"ensemble_score={row.get('ensemble_score', 0):.4f}; "
        f"score_composito={row.get('score_composito', 0):.4f}; "
        f"baseline_score={row.get('baseline_score', 0):.4f}"
    )


def generate_explanation(context: str, llm: ChatAnthropic) -> str:
    """Tool 2 — generate_explanation(context)

    Genera spiegazione narrativa della rotta anomala usando LLM reale.
    """
    prompt = (
        "Sei un analista rischio aeroportuale. "
        "Spiega in massimo 3 frasi perche la rotta puo essere considerata anomala, "
        "citando score e livello di rischio, senza inventare dati.\n\n"
        f"Contesto:\n{context}"
    )
    response = llm.invoke(prompt)
    return response.content.strip()


def build_final_report(findings: list[dict], perimeter: dict, anomaly_meta: dict, n_tot: int) -> dict:
    """Tool 3 — build_final_report(findings)

    Costruisce il JSON finale del report con spiegazioni testuali.
    """
    n_alta = int(anomaly_meta.get("n_alta", 0))
    n_media = int(anomaly_meta.get("n_media", 0))
    n_normale = int(anomaly_meta.get("n_normale", 0))
    scope = ", ".join([f"{k}={v}" for k, v in perimeter.items()]) if perimeter else "nessun filtro"
    summary = (
        f"Analisi completata su {n_tot} rotte ({scope}). "
        f"Distribuzione rischio: ALTA={n_alta}, MEDIA={n_media}, NORMALE={n_normale}. "
        f"Sono state prodotte spiegazioni LLM per {len(findings)} rotte ALTA/MEDIA."
    )
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "perimeter": perimeter,
        "n_rotte_analizzate": n_tot,
        "distribution": {"alta": n_alta, "media": n_media, "normale": n_normale},
        "thresholds": {
            "soglia_alta": anomaly_meta.get("soglia_alta"),
            "soglia_media": anomaly_meta.get("soglia_media"),
        },
        "findings": findings,
        "summary": summary,
    }


def run_report_agent(
    state: AgentState,
    save_output: bool = True,
    output_path: Path | str | None = None,
    use_llm: bool = True,
    dry_run: bool = False,
) -> AgentState:
    """Genera report finale JSON usando LLM e output OutlierAgent."""
    logger.info("ReportAgent -- Avvio")

    try:
        if use_llm and not dry_run and not get_anthropic_api_key():
            raise ValueError("ANTHROPIC_API_KEY non impostata: impossibile usare il ReportAgent LLM.")

        df = state.get("df_anomalies")
        a_meta = state.get("anomaly_meta") or {}
        if a_meta.get("error"):
            raise ValueError(f"anomaly_meta contiene errore: {a_meta['error']}")
        if df is None or not isinstance(df, pd.DataFrame):
            raise ValueError("df_anomalies mancante: esegui prima OutlierAgent.")
        if df.empty:
            raise ValueError("df_anomalies vuoto: impossibile generare report.")

        perimeter = state.get("perimeter") or {}
        n_tot = int(len(df))
        llm = None
        model_name = None
        llm_warning = None
        if use_llm and not dry_run:
            try:
                model_name = get_anthropic_model()
                llm = ChatAnthropic(model=model_name, temperature=0)
            except Exception as e:
                llm_warning = f"LLM init fallita ({e}); uso fallback deterministico."
                logger.warning("ReportAgent -- %s", llm_warning)

        # Rotte da spiegare: tutte le ALTA/MEDIA, ordinate per score decrescente.
        explain_df = (
            df[df["risk_label"].isin(["ALTA", "MEDIA"])]
            .sort_values("ensemble_score", ascending=False)
            .copy()
        )
        explain_cols = [c for c in ["ROTTA", "risk_label", "ensemble_score", "score_composito", "baseline_score"] if c in explain_df.columns]
        explain_rows = explain_df[explain_cols].to_dict(orient="records")

        findings = []
        for row in explain_rows:
            context = format_route_for_llm(row)
            if llm is None:
                explanation = (
                    "Spiegazione LLM non eseguita (modalita dry_run/use_llm=False). "
                    "Verifica score e trend nel frontend."
                )
            else:
                try:
                    explanation = generate_explanation(context, llm)
                except Exception as e:
                    llm_warning = f"Chiamata LLM fallita ({e}); fallback locale attivato."
                    logger.warning("ReportAgent -- %s", llm_warning)
                    explanation = (
                        "Spiegazione LLM non disponibile per errore di connessione/API. "
                        "Analizzare la rotta tramite ensemble_score, baseline_score e risk_label."
                    )
            findings.append({**row, "explanation": explanation})

        report = build_final_report(
            findings=findings,
            perimeter=perimeter,
            anomaly_meta=a_meta,
            n_tot=n_tot,
        )
        if llm_warning:
            report["warning"] = llm_warning

        report_path = None
        if save_output:
            default_out = _PROJECT_ROOT / PATHS["multiagent_report"]
            out_path = Path(output_path) if output_path is not None else default_out
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
            report_path = str(out_path)
            logger.info("ReportAgent output salvato in: %s", report_path)

        logger.info("ReportAgent ✓ Completato — rotte=%d, spiegazioni=%d", n_tot, len(findings))
        return {
            **state,
            "report": report,
            "report_path": report_path,
        }
    except Exception as e:
        logger.error("ReportAgent ✗ Errore: %s", e)
        return {
            **state,
            "report": {"error": str(e), "user_message": "Errore generazione report: verifica configurazione e input."},
            "report_path": None,
        }


if __name__ == "__main__":
    from multiagent_pipeline.agents.data_agent import data_agent_node
    from multiagent_pipeline.agents.feature_agent import run_feature_agent
    from multiagent_pipeline.agents.baseline_agent import run_baseline_agent
    from multiagent_pipeline.agents.outlier_agent import run_outlier_agent

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    s: AgentState = {"perimeter": {"anno": 2024}}
    s = data_agent_node(s)
    s = run_feature_agent(s)
    s = run_baseline_agent(s)
    s = run_outlier_agent(s)
    s = run_report_agent(s)
    print("\n=== RISULTATO ReportAgent ===")
    print(s["report_path"])
    print((s["report"] or {}).get("summary"))
