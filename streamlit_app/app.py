from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from scipy.stats import pearsonr, spearmanr

# Ensures correct imports when launched with:
# streamlit run streamlit_app/app.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multiagent_pipeline.main import run_pipeline
from multiagent_pipeline.config import get_anthropic_api_key


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
    if anno is not None:
        perimeter["anno"] = anno
    if paese_partenza.strip():
        perimeter["paese_partenza"] = paese_partenza.strip()
    if aeroporto_partenza.strip():
        perimeter["aeroporto_partenza"] = aeroporto_partenza.strip().upper()
    if aeroporto_arrivo.strip():
        perimeter["aeroporto_arrivo"] = aeroporto_arrivo.strip().upper()
    if zona is not None:
        perimeter["zona"] = zona
    return perimeter


def _render_stage_badges(summary: dict) -> None:
    stages = summary.get("stages", {})
    if not stages:
        st.info("No stages executed.")
        return

    html = []
    for stage, details in stages.items():
        css = "ok" if details.get("ok") else "err"
        label = f"{stage}: {'OK' if details.get('ok') else 'ERROR'}"
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
        df = pd.read_csv(merged_path)
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
                "status": "OK" if data.get("ok") else "ERROR",
                "error": data.get("error") or "",
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def _load_classical_report() -> pd.DataFrame | None:
    """Load the classical report (static) once."""
    cl_path = PROJECT_ROOT / "data" / "processed" / "anomaly_results.csv"
    if not cl_path.exists():
        return None
    try:
        return pd.read_csv(cl_path)
    except Exception:
        return None


def main() -> None:
    _inject_style()
    if "last_run" not in st.session_state:
        st.session_state["last_run"] = None
    if "run_history" not in st.session_state:
        st.session_state["run_history"] = []

    options = _load_filter_options()

    st.title("Airport Risk Intelligence")
    st.caption("Multi-agent pipeline with unified orchestrator and operational report.")

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Configuration")

        use_anno = st.checkbox("Filter by year", value=True)
        anno = st.selectbox("Year", options["anni"], index=0, disabled=not use_anno)

        paese = st.selectbox("Departure country", ["(all)"] + options["paesi"], index=0)
        apt_dep = st.selectbox("Departure airport", ["(all)"] + options["apt_dep"], index=0)
        apt_arr = st.selectbox("Arrival airport", ["(all)"] + options["apt_arr"], index=0)

        use_zona = st.checkbox("Filter by zone", value=False)
        zona = st.selectbox("Zone", options["zone"], index=0, disabled=not use_zona)

        st.divider()
        has_api_key = bool(get_anthropic_api_key())
        run_report = st.checkbox(
            "Enable LLM Report (Anthropic)",
            value=has_api_key,
            help="Requires ANTHROPIC_API_KEY environment variable.",
        )
        dry_run = st.checkbox(
            "Dry run report (no LLM calls)",
            value=not has_api_key,
            help="Generate report without consuming API credits.",
        )
        save_outputs = st.checkbox("Save outputs to disk", value=True)
        continue_on_error = st.checkbox("Continue if a stage fails", value=False)

        st.divider()
        run = st.button("Run pipeline", use_container_width=True, type="primary")

    # ── Pipeline execution ───────────────────────────────────────────────────
    if run:
        perimeter = _build_perimeter(
            anno=int(anno) if use_anno else None,
            paese_partenza="" if paese == "(all)" else paese,
            aeroporto_partenza="" if apt_dep == "(all)" else apt_dep,
            aeroporto_arrivo="" if apt_arr == "(all)" else apt_arr,
            zona=int(zona) if use_zona else None,
        )
        if run_report and not get_anthropic_api_key():
            st.warning("`ANTHROPIC_API_KEY` not set: LLM report automatically disabled.")
            run_report = False

        with st.spinner("Running orchestrator..."):
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

        st.subheader("Pipeline Status")
        _render_stage_badges(summary)

        completed = len(summary.get("completed_stages", []))
        failed = len(summary.get("failed_stages", []))
        df_anom = state.get("df_anomalies")
        n_rotte = int(len(df_anom)) if isinstance(df_anom, pd.DataFrame) else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("Completed stages", completed)
        c2.metric("Failed stages", failed)
        c3.metric("Routes analysed", n_rotte, help=f"Runtime: {elapsed_s}s")

        st.markdown(
            f"<div class='section-card'><b>Runtime:</b> {elapsed_s}s &nbsp; | &nbsp; "
            f"<b>Perimeter:</b> {perimeter or 'no filter'}</div>",
            unsafe_allow_html=True,
        )

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

    # ── Result tabs ─────────────────────────────────────────────────────────────
    last_run = st.session_state.get("last_run")
    if last_run:
        state = last_run["state"]
        summary = last_run["summary"]
        df_anom = state.get("df_anomalies")
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Anomalies",
            "Classical vs Multi-Agent",
            "Report",
            "Stage Detail",
            "Debug JSON",
        ])

        # ── Tab 1: Anomalies ───────────────────────────────────────────────────
        with tab1:
            st.markdown("### Risk distribution")
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
                st.markdown("### Top routes")
                show_df = df_anom.sort_values(
                    "ensemble_score", ascending=False
                )[visible_cols].head(50)
                st.dataframe(show_df, use_container_width=True)
                st.download_button(
                    "Download anomalies (CSV)",
                    data=show_df.to_csv(index=False).encode("utf-8"),
                    file_name="top_routes_anomalies.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                st.info("No anomaly data available.")

        # ── Tab 2: Classical vs Multi-Agent comparison ─────────────────────────
        with tab2:
            st.markdown("### Classical vs Multi-Agent Comparison")
            cl = _load_classical_report()

            if cl is None:
                st.warning(
                    "Classical `anomaly_results.csv` not found in `data/processed/`. "
                    "Run the classical pipeline notebooks first (01→06)."
                )
            elif not isinstance(df_anom, pd.DataFrame) or df_anom.empty:
                st.info("Run the multi-agent pipeline to enable the comparison.")
            else:
                cl_cols = ["ROTTA", "anomaly_score", "anomaly_label"]
                missing_cl = [c for c in cl_cols if c not in cl.columns]
                if missing_cl:
                    st.error(f"Missing columns in anomaly_results.csv: {missing_cl}")
                else:
                    df_cmp = cl[cl_cols].merge(
                        df_anom[["ROTTA", "ensemble_score", "risk_label"]],
                        on="ROTTA", how="inner",
                    )

                    if df_cmp.empty:
                        st.warning("No routes in common between the classical report and the current run.")
                    else:
                        df_cmp["label_concorde"] = df_cmp["anomaly_label"] == df_cmp["risk_label"]
                        df_cmp["delta_score"] = (
                            df_cmp["ensemble_score"] - df_cmp["anomaly_score"]
                        ).round(4)

                        n_rotte_cmp = len(df_cmp)
                        perimeter_info = last_run.get("perimeter", {})
                        scope_label = (
                            "full dataset"
                            if not perimeter_info
                            else f"perimeter: {perimeter_info}"
                        )
                        st.caption(f"Comparison on **{n_rotte_cmp}** routes ({scope_label})")

                        # ── KPI ──────────────────────────────────────────────
                        pr, _ = pearsonr(df_cmp["anomaly_score"], df_cmp["ensemble_score"])
                        sr, _ = spearmanr(df_cmp["anomaly_score"], df_cmp["ensemble_score"])
                        agree = df_cmp["label_concorde"].mean()
                        top_n = min(20, len(df_cmp))
                        top_cl = set(df_cmp.nlargest(top_n, "anomaly_score")["ROTTA"])
                        top_ma = set(df_cmp.nlargest(top_n, "ensemble_score")["ROTTA"])

                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Pearson r", f"{pr:.3f}", help="Linear score correlation")
                        c2.metric("Spearman r", f"{sr:.3f}", help="Rank correlation")
                        c3.metric("Label agreement", f"{agree:.1%}", help="% routes with same risk label")
                        c4.metric(
                            f"Top-{top_n} overlap",
                            f"{len(top_cl & top_ma)}/{top_n}",
                            help=f"Routes in common in top-{top_n} of both pipelines",
                        )

                        # ── Scatter plot ─────────────────────────────────────
                        chart = (
                            alt.Chart(df_cmp)
                            .mark_circle(size=55, opacity=0.7)
                            .encode(
                                x=alt.X("anomaly_score:Q", title="Classical score"),
                                y=alt.Y("ensemble_score:Q", title="Multi-Agent score"),
                                color=alt.Color(
                                    "anomaly_label:N",
                                    scale=alt.Scale(
                                        domain=["ALTA", "MEDIA", "NORMALE"],
                                        range=["#e05252", "#e0a852", "#5285e0"],
                                    ),
                                    legend=alt.Legend(title="Classical label"),
                                ),
                                tooltip=[
                                    "ROTTA",
                                    "anomaly_label",
                                    "risk_label",
                                    "anomaly_score",
                                    "ensemble_score",
                                    "delta_score",
                                ],
                            )
                            .properties(
                                title=f"Score correlation (Pearson r={pr:.3f} | Spearman r={sr:.3f})",
                                width=600,
                                height=400,
                            )
                        )

                        max_val = max(
                            float(df_cmp["anomaly_score"].max()),
                            float(df_cmp["ensemble_score"].max()),
                            0.6,
                        )
                        diagonal = (
                            alt.Chart(pd.DataFrame({"x": [0, max_val], "y": [0, max_val]}))
                            .mark_line(color="gray", strokeDash=[4, 4], opacity=0.5)
                            .encode(x="x:Q", y="y:Q")
                        )

                        st.altair_chart(chart + diagonal, use_container_width=True)

                        # ── Gold standard: ALTA in both ────────────────────────
                        gold = df_cmp[
                            (df_cmp["anomaly_label"] == "ALTA")
                            & (df_cmp["risk_label"] == "ALTA")
                        ]
                        st.markdown(f"### ALTA routes agreed by both pipelines ({len(gold)} routes)")
                        if not gold.empty:
                            st.dataframe(
                                gold[["ROTTA", "anomaly_score", "ensemble_score", "delta_score"]]
                                .sort_values("anomaly_score", ascending=False),
                                use_container_width=True,
                                hide_index=True,
                            )
                        else:
                            st.info("No routes classified as ALTA by both pipelines.")

                        # ── Full comparison table ───────────────────────────────
                        with st.expander(f"Full comparison table ({n_rotte_cmp} routes)"):
                            st.dataframe(
                                df_cmp.sort_values("anomaly_score", ascending=False),
                                use_container_width=True,
                                hide_index=True,
                            )
                            st.download_button(
                                "Download comparison (CSV)",
                                data=df_cmp.to_csv(index=False).encode("utf-8"),
                                file_name="comparison_classical_vs_multiagent.csv",
                                mime="text/csv",
                                use_container_width=True,
                            )

        # ── Tab 3: Report ────────────────────────────────────────────────────
        with tab3:
            raw_report = state.get("report") or {}
            report_error = raw_report.get("error") if isinstance(raw_report, dict) else None
            report_obj = _safe_read_report(state.get("report_path"), state.get("report"))
            if report_obj:
                st.markdown("### Summary")
                st.write(report_obj.get("summary", "N/A"))
                findings = report_obj.get("findings", [])
                if findings:
                    st.markdown("### Findings")
                    st.dataframe(pd.DataFrame(findings), use_container_width=True)
                else:
                    st.caption("No HIGH/MEDIUM risk routes to explain.")
                st.download_button(
                    "Download report (JSON)",
                    data=json.dumps(report_obj, indent=2, ensure_ascii=False),
                    file_name="multiagent_report.json",
                    mime="application/json",
                    use_container_width=True,
                )
            else:
                stages = (summary or {}).get("stages", {})
                report_stage = stages.get("report")
                if report_error:
                    st.error(f"ReportAgent error: {report_error}")
                elif report_stage is None:
                    st.info(
                        "Report not executed in this run. Check 'Enable LLM Report' "
                        "and run the pipeline again."
                    )
                else:
                    st.info("Report not available for this run.")

        # ── Tab 4: Stage Detail ──────────────────────────────────────────────
        with tab4:
            st.markdown("### Stage Results")
            st_df = _stage_table(summary)
            st.dataframe(st_df, use_container_width=True, hide_index=True)
            if not st_df.empty and (st_df["status"] == "ERRORE").any():
                first_err = st_df[st_df["status"] == "ERRORE"].iloc[0]["error"]
                st.error(first_err or "Stage failed with no error detail.")

            hist = st.session_state.get("run_history", [])
            if hist:
                st.markdown("### Run History (current session)")
                st.dataframe(pd.DataFrame(hist).tail(10), use_container_width=True, hide_index=True)

        # ── Tab 5: Debug JSON ────────────────────────────────────────────────
        with tab5:
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
        st.info("Configure filters from the sidebar and click **Run pipeline**.")


if __name__ == "__main__":
    main()
