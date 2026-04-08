from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

# Garantisce import corretti quando si lancia:
# streamlit run streamlit_app/app.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multiagent_pipeline.main import run_pipeline


st.set_page_config(
    page_title="Airport Risk Intelligence",
    page_icon=":airplane_departure:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _inject_style() -> None:
    st.markdown(
        """
        <style>
          .block-container {padding-top: 1.2rem; padding-bottom: 1.2rem; max-width: 1200px;}
          h1, h2, h3 {letter-spacing: 0.2px;}
          .stMetric {
            background: rgba(240,242,246,0.45);
            border: 1px solid rgba(49,51,63,0.15);
            border-radius: 12px;
            padding: 10px 14px;
          }
          .chip {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            border: 1px solid rgba(49,51,63,0.25);
            font-size: 0.82rem;
            margin-right: 6px;
          }
          .ok { background: rgba(44, 182, 125, 0.12); }
          .err { background: rgba(240, 80, 83, 0.12); }
          .section-card {
            border: 1px solid rgba(49,51,63,0.15);
            border-radius: 12px;
            padding: 12px 14px;
            background: rgba(255,255,255,0.02);
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _build_perimeter(
    anno: int | None,
    paese_partenza: str,
    aeroporto_partenza: str,
    aeroporto_arrivo: str,
    zona: int | None,
) -> dict:
    perimeter: dict = {}
    if anno:
        perimeter["anno"] = anno
    if paese_partenza.strip():
        perimeter["paese_partenza"] = paese_partenza.strip()
    if aeroporto_partenza.strip():
        perimeter["aeroporto_partenza"] = aeroporto_partenza.strip().upper()
    if aeroporto_arrivo.strip():
        perimeter["aeroporto_arrivo"] = aeroporto_arrivo.strip().upper()
    if zona:
        perimeter["zona"] = zona
    return perimeter


def _render_stage_badges(summary: dict) -> None:
    stages = summary.get("stages", {})
    if not stages:
        st.info("Nessuno stage eseguito.")
        return

    html = []
    for stage, details in stages.items():
        css = "ok" if details.get("ok") else "err"
        label = f"{stage}: {'OK' if details.get('ok') else 'ERRORE'}"
        html.append(f"<span class='chip {css}'>{label}</span>")
    st.markdown("".join(html), unsafe_allow_html=True)


def _safe_read_report(path: str | None, in_memory: dict | None) -> dict | None:
    if in_memory and not in_memory.get("error"):
        return in_memory
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _load_filter_options() -> dict:
    merged_path = PROJECT_ROOT / "data/processed/dataset_merged.csv"
    if not merged_path.exists():
        return {"anni": [2024], "paesi": [], "apt_dep": [], "apt_arr": [], "zone": list(range(1, 10))}
    try:
        cols = ["ANNO_PARTENZA", "PAESE_PART", "AREOPORTO_PARTENZA", "AREOPORTO_ARRIVO", "ZONA"]
        df = pd.read_csv(merged_path, usecols=[c for c in cols if c in pd.read_csv(merged_path, nrows=1).columns])
        anni = sorted([int(x) for x in df.get("ANNO_PARTENZA", pd.Series(dtype="float")).dropna().unique().tolist()])
        paesi = sorted([str(x) for x in df.get("PAESE_PART", pd.Series(dtype="object")).dropna().astype(str).unique().tolist()])
        apt_dep = sorted([str(x) for x in df.get("AREOPORTO_PARTENZA", pd.Series(dtype="object")).dropna().astype(str).unique().tolist()])
        apt_arr = sorted([str(x) for x in df.get("AREOPORTO_ARRIVO", pd.Series(dtype="object")).dropna().astype(str).unique().tolist()])
        zone = sorted([int(x) for x in df.get("ZONA", pd.Series(dtype="float")).dropna().unique().tolist()])
        return {
            "anni": anni or [2024],
            "paesi": paesi,
            "apt_dep": apt_dep,
            "apt_arr": apt_arr,
            "zone": zone or list(range(1, 10)),
        }
    except Exception:
        return {"anni": [2024], "paesi": [], "apt_dep": [], "apt_arr": [], "zone": list(range(1, 10))}


def _stage_table(summary: dict) -> pd.DataFrame:
    stages = summary.get("stages", {})
    rows = []
    for stage, data in stages.items():
        rows.append(
            {
                "stage": stage,
                "status": "OK" if data.get("ok") else "ERRORE",
                "error": data.get("error") or "",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    _inject_style()
    if "last_run" not in st.session_state:
        st.session_state["last_run"] = None
    if "run_history" not in st.session_state:
        st.session_state["run_history"] = []

    options = _load_filter_options()

    st.title("Airport Risk Intelligence")
    st.caption("Pipeline multi-agent con orchestrator unico e report operativo.")

    with st.sidebar:
        st.header("Configurazione")
        anno = st.selectbox("Anno", options["anni"], index=0)
        paese = st.selectbox("Paese partenza", ["(tutti)"] + options["paesi"], index=0)
        apt_dep = st.selectbox("Aeroporto partenza", ["(tutti)"] + options["apt_dep"], index=0)
        apt_arr = st.selectbox("Aeroporto arrivo", ["(tutti)"] + options["apt_arr"], index=0)
        zona = st.selectbox("Zona", options["zone"], index=0)
        use_zona = st.checkbox("Applica filtro zona", value=False)

        st.divider()
        run_report = st.checkbox(
            "Attiva Report LLM (Anthropic)",
            value=False,
            help="Richiede variabile ambiente ANTHROPIC_API_KEY.",
        )
        dry_run = st.checkbox(
            "Dry run report (no chiamate LLM)",
            value=True,
            help="Genera report senza consumare crediti API.",
        )
        save_outputs = st.checkbox("Salva output su disco", value=True)
        continue_on_error = st.checkbox("Continua se uno stage fallisce", value=False)

        st.divider()
        st.caption("Preset rapidi")
        p1, p2 = st.columns(2)
        with p1:
            preset_all = st.button("Preset: Nazionale", use_container_width=True)
        with p2:
            preset_alg = st.button("Preset: Algeria", use_container_width=True)

        if preset_all:
            st.session_state["preset"] = {"paese": "(tutti)", "use_zona": False}
        if preset_alg:
            st.session_state["preset"] = {"paese": "Algeria", "use_zona": False}

        run = st.button("Esegui pipeline", use_container_width=True, type="primary")

    if run:
        perimeter = _build_perimeter(
            anno=int(anno) if anno else None,
            paese_partenza="" if paese == "(tutti)" else paese,
            aeroporto_partenza="" if apt_dep == "(tutti)" else apt_dep,
            aeroporto_arrivo="" if apt_arr == "(tutti)" else apt_arr,
            zona=int(zona) if use_zona else None,
        )
        if run_report and not os.getenv("ANTHROPIC_API_KEY"):
            st.warning("`ANTHROPIC_API_KEY` non impostata: disattivo automaticamente il report LLM.")
            run_report = False

        with st.spinner("Esecuzione orchestrator in corso..."):
            start = time.perf_counter()
            state, summary = run_pipeline(
                perimeter=perimeter,
                run_report=run_report,
                use_llm=run_report and (not dry_run),
                dry_run=dry_run,
                continue_on_error=continue_on_error,
                save_outputs=save_outputs,
            )
            elapsed_s = round(time.perf_counter() - start, 2)

        st.subheader("Stato Pipeline")
        _render_stage_badges(summary)

        completed = len(summary.get("completed_stages", []))
        failed = len(summary.get("failed_stages", []))
        df_anom = state.get("df_anomalies")
        n_rotte = int(len(df_anom)) if isinstance(df_anom, pd.DataFrame) else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("Stage completati", completed)
        c2.metric("Stage falliti", failed)
        c3.metric("Rotte analizzate", n_rotte, help=f"Runtime: {elapsed_s}s")

        st.markdown(f"<div class='section-card'><b>Runtime:</b> {elapsed_s}s &nbsp; | &nbsp; <b>Perimetro:</b> {perimeter or 'nessun filtro'}</div>", unsafe_allow_html=True)

        st.session_state["last_run"] = {
            "state": state,
            "summary": summary,
            "elapsed_s": elapsed_s,
            "perimeter": perimeter,
        }
        st.session_state["run_history"].append(
            {
                "runtime_s": elapsed_s,
                "completed": completed,
                "failed": failed,
                "perimeter": json.dumps(perimeter, ensure_ascii=False),
            }
        )

    last_run = st.session_state.get("last_run")
    if last_run:
        state = last_run["state"]
        summary = last_run["summary"]
        df_anom = state.get("df_anomalies")
        tab1, tab2, tab3, tab4 = st.tabs(["Anomalie", "Report", "Stage Detail", "Debug JSON"])

        with tab1:
            st.markdown("### Distribuzione rischio")
            if isinstance(df_anom, pd.DataFrame) and not df_anom.empty:
                if "risk_label" in df_anom.columns:
                    counts = (
                        df_anom["risk_label"]
                        .value_counts()
                        .reindex(["ALTA", "MEDIA", "NORMALE"], fill_value=0)
                    )
                    st.bar_chart(counts)
                visible_cols = [
                    c for c in ["ROTTA", "risk_label", "ensemble_score", "baseline_score", "score_composito"]
                    if c in df_anom.columns
                ]
                st.markdown("### Top rotte")
                show_df = df_anom.sort_values(
                    "ensemble_score", ascending=False
                )[visible_cols].head(50)
                st.dataframe(show_df, use_container_width=True)
                st.download_button(
                    "Scarica anomalie (CSV)",
                    data=show_df.to_csv(index=False).encode("utf-8"),
                    file_name="anomalie_top_rotte.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                st.info("Nessun dataframe anomalie disponibile.")

        with tab2:
            report_obj = _safe_read_report(state.get("report_path"), state.get("report"))
            if report_obj:
                st.markdown("### Sommario")
                st.write(report_obj.get("summary", "N/A"))
                findings = report_obj.get("findings", [])
                if findings:
                    st.markdown("### Findings")
                    st.dataframe(pd.DataFrame(findings), use_container_width=True)
                else:
                    st.caption("Nessuna rotta ALTA/MEDIA da spiegare.")
                st.download_button(
                    "Scarica report (JSON)",
                    data=json.dumps(report_obj, indent=2, ensure_ascii=False),
                    file_name="multiagent_report.json",
                    mime="application/json",
                    use_container_width=True,
                )
            else:
                st.info("Report non disponibile (esegui con `run_report=True`).")

        with tab3:
            st.markdown("### Esito stage")
            st_df = _stage_table(summary)
            st.dataframe(st_df, use_container_width=True, hide_index=True)
            if not st_df.empty and (st_df["status"] == "ERRORE").any():
                first_err = st_df[st_df["status"] == "ERRORE"].iloc[0]["error"]
                st.error(first_err or "Stage fallito senza dettaglio errore.")

            hist = st.session_state.get("run_history", [])
            if hist:
                st.markdown("### Storico esecuzioni (sessione corrente)")
                st.dataframe(pd.DataFrame(hist).tail(10), use_container_width=True, hide_index=True)

        with tab4:
            st.markdown("### Summary orchestrator")
            st.json(summary)
            st.markdown("### Meta")
            st.json(
                {
                    "data_meta": state.get("data_meta"),
                    "feature_meta": state.get("feature_meta"),
                    "baseline_meta": state.get("baseline_meta"),
                    "anomaly_meta": state.get("anomaly_meta"),
                    "report_path": state.get("report_path"),
                }
            )
    else:
        st.info("Configura i filtri dalla sidebar e premi **Esegui pipeline**.")


if __name__ == "__main__":
    main()
