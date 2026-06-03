"""ReportAgent — fifth node of the multi-agent graph.

Responsibilities:
    Uses a real LLM to explain anomalous routes (HIGH/MEDIUM) in natural
    language and build the final JSON report.
"""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
    __package__ = "multiagent_pipeline.agents"

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from langchain_anthropic import ChatAnthropic

from multiagent_pipeline.config import (
    get_anthropic_api_key, get_anthropic_model,
    get_llm_backend, get_llm_base_url, get_llm_model, get_llm_api_key,
    get_llm_concurrency, get_llm_narrate_levels, get_llm_dedup_threshold,
)
from multiagent_pipeline.state import AgentState, PATHS

logger = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_llm():
    """Backend factory for the ReportAgent narration LLM.

    Selected via the ``LLM_BACKEND`` env var (config.get_llm_backend):
        * ``anthropic``          → ChatAnthropic (cloud Claude) — DEFAULT, i.e.
                                   unchanged behaviour when no env is set.
        * ``openai_compatible``  → ChatOpenAI against ``LLM_BASE_URL``
                                   (LM Studio / Ollama / vLLM — local, zero-cost).
        * ``none``               → no LLM (deterministic placeholders).

    Returns a LangChain chat model exposing ``.invoke([...])``, or ``None``.
    The rest of the agent only calls ``.invoke`` so the backend is swappable
    without touching the orchestration.
    """
    backend = get_llm_backend()
    if backend == "none":
        return None
    if backend == "openai_compatible":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            base_url=get_llm_base_url(),
            api_key=get_llm_api_key(),
            model=get_llm_model(),
            temperature=0,
        )
    # default: cloud Claude (the colleague's canonical choice)
    return ChatAnthropic(model=get_anthropic_model(), temperature=0)


# Mapping z-score column → (absolute value column, human-readable label)
# These are the 11 baseline features computed by BaselineAgent.
_Z_COL_MAP: dict[str, tuple[str, str]] = {
    "z_pct_interpol":        ("pct_interpol",         "% allarmi Interpol"),
    "z_pct_sdi":             ("pct_sdi",              "% allarmi SDI"),
    "z_pct_nsis":            ("pct_nsis",             "% allarmi NSIS"),
    "z_tasso_rilevanza":     ("tasso_rilevanza",       "tasso rilevanza"),
    "z_tasso_inv_medio":     ("tasso_inv_medio",       "tasso investigazione"),
    "z_tasso_allarme_medio": ("tasso_allarme_medio",   "tasso allarme medio"),
    "z_tasso_chiusura":      ("tasso_chiusura",        "tasso chiusura"),
    "z_tasso_fermati":       ("tasso_fermati",         "tasso fermati"),
    "z_tasso_respinti":      ("tasso_respinti",        "tasso respinti"),
    "z_score_rischio_esiti": ("score_rischio_esiti",   "score rischio esiti"),
    "z_tot_allarmi_log":     ("tot_allarmi_sum",       "volume allarmi totali"),
}


def _fmt(v: object) -> str:
    """Formats a numeric value for the LLM prompt, handles NaN/None."""
    try:
        return f"{float(v):.3f}"  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "N/D"


def format_route_for_llm(row: dict) -> str:
    """Tool 1 — format_route_for_llm(row)

    Builds a structured context for the LLM by:
    1. Identifying the top-3 anomaly drivers (features with highest |z-score|
       relative to the historical baseline computed by BaselineAgent).
    2. Adding absolute values, individual model scores, and route context.

    This gives the LLM concrete, cited evidence instead of bare aggregate scores.
    """
    # ── Top-3 z-score drivers ──────────────────────────────────────────────
    z_scores: dict[str, float] = {}
    for z_col in _Z_COL_MAP:
        val = row.get(z_col)
        if val is not None:
            try:
                z_scores[z_col] = abs(float(val))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                pass

    top3 = sorted(z_scores, key=z_scores.__getitem__, reverse=True)[:3]

    driver_lines = []
    for z_col in top3:
        abs_col, label = _Z_COL_MAP[z_col]
        z_val = float(row.get(z_col, 0))  # type: ignore[arg-type]
        abs_val = row.get(abs_col)
        driver_lines.append(
            f"  - {label}: valore={_fmt(abs_val)}, deviazione={z_val:+.2f}σ dalla baseline"
        )

    drivers_str = (
        "\n".join(driver_lines) if driver_lines
        else "  - z-score non disponibili per questa rotta"
    )

    # ── Individual model scores ────────────────────────────────────────────
    models_str = (
        f"IsolationForest={_fmt(row.get('score_if'))}, "
        f"LOF={_fmt(row.get('score_lof'))}, "
        f"Autoencoder={_fmt(row.get('score_ae'))}"
    )

    # ── Business-rule drivers + final risk (RiskProfilingAgent layer) ───────
    # These are consumed by the ⑤ prompt (cite a rule + the final risk) and were
    # previously missing from the context, so the prompt asked for data it could
    # not see.
    rd = row.get("risk_drivers")
    rules_str = ("; ".join(map(str, rd)) if isinstance(rd, (list, tuple)) and len(rd)
                 else "none fired")

    # ── Route context ──────────────────────────────────────────────────────
    return (
        f"ROUTE: {row.get('ROTTA', 'N/A')} | "
        f"Departure country: {row.get('PAESE_PART', 'N/A')} | "
        f"Zone: {row.get('ZONA', 'N/A')}\n"
        f"Anomaly level: {row.get('anomaly_label', 'N/A')} | "
        f"Final risk classification: {row.get('final_risk', 'N/A')} | "
        f"Ensemble score: {_fmt(row.get('ensemble_score', 0))}\n"
        f"Score per model: {models_str}\n"
        f"Volumes: total_alarms={_fmt(row.get('tot_allarmi_sum'))}, "
        f"passengers_entered={_fmt(row.get('tot_entrati'))}\n"
        f"Business rules fired: {rules_str}\n"
        f"Top 3 anomaly drivers (features furthest from historical baseline):\n"
        f"{drivers_str}"
    )


def generate_explanation(context: str, llm) -> str:
    """Tool 2 — generate_explanation(context) — optimisation ⑤.

    Constrained narration. Every figure (drivers + sigma, business rules, final
    risk) is already in ``context`` (built by format_route_for_llm). The model is
    told to COPY the numbers verbatim and stay within 3 sentences, so it writes
    prose only. The caller then applies a faithfulness guardrail as a structural
    backstop against any invented number — which makes even a small local model
    reliable. Backend-agnostic (cloud Claude or local), only uses ``.invoke``.
    """
    from langchain_core.messages import SystemMessage, HumanMessage
    system = (
        "You are an airport border-control risk analyst writing for operators. "
        "Write in English only. Use ONLY the figures in the data below and copy "
        "each number EXACTLY as written; never compute, re-round, or invent a number."
    )
    user = (
        "Write at most 3 short sentences explaining why this route is anomalous and "
        "what its risk profile means for operators. Your explanation MUST cite at least "
        "TWO statistical drivers (with value and sigma), reference the final risk "
        "classification, and mention the business rules that fired (or state none fired). "
        "No bullet points, no headings.\n\n"
        f"Route data:\n{context}"
    )
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return response.content.strip()


def build_final_report(findings: list[dict], perimeter: dict, anomaly_meta: dict, n_tot: int) -> dict:
    """Tool 3 — build_final_report(findings)

    Builds the final JSON report with textual explanations.
    """
    n_high = int(anomaly_meta.get("n_high", 0))
    n_medium = int(anomaly_meta.get("n_medium", 0))
    n_normal = int(anomaly_meta.get("n_normal", 0))
    scope = ", ".join([f"{k}={v}" for k, v in perimeter.items()]) if perimeter else "no filter"
    summary = (
        f"Analysis completed on {n_tot} routes ({scope}). "
        f"Anomaly distribution: HIGH={n_high}, MEDIUM={n_medium}, NORMAL={n_normal}. "
        f"LLM explanations were generated for {len(findings)} HIGH/MEDIUM routes."
    )
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "perimeter": perimeter,
        "n_routes_analysed": n_tot,
        "distribution": {"high": n_high, "medium": n_medium, "normal": n_normal},
        "thresholds": {
            "threshold_high":   anomaly_meta.get("threshold_high"),
            "threshold_medium": anomaly_meta.get("threshold_medium"),
        },
        "findings": findings,
        "summary": summary,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Faithfulness guardrail + narration cache (optimisation ⑤ + cache/concurrency)
# ══════════════════════════════════════════════════════════════════════════════

_NUM_RE = re.compile(r"-?\d+\.?\d*")
_CACHE_PATH = _PROJECT_ROOT / "data" / "processed" / "llm_report_cache.json"


def _matches_context(n: float, ctx_nums, tol: float = 0.01) -> bool:
    """True if n appears in the context, allowing percent/fraction equivalence
    (output '46%' is faithful to a context value of 0.46)."""
    return any(abs(n - c) <= tol or abs(n / 100 - c) <= tol or abs(n * 100 - c) <= tol
               for c in ctx_nums)


def _ctx_numbers(context: str) -> set[float]:
    out: set[float] = set()
    for tok in _NUM_RE.findall(context or ""):
        try:
            out.add(round(float(tok), 2))
        except ValueError:
            pass
    return out


def _guardrail(text: str, allowed: set[float], tol: float = 0.01) -> tuple[str, int]:
    """⑤ backstop: replace any figure-like number absent from the context with
    the nearest real one → faithfulness by construction. Small bare integers
    (counts/ordinals like 'two drivers') are left untouched."""
    allowed_sorted = sorted(allowed)
    fixes = 0

    def _repl(m: "re.Match") -> str:
        nonlocal fixes
        tok = m.group(0)
        try:
            v = round(float(tok), 2)
        except ValueError:
            return tok
        if _matches_context(v, allowed_sorted, tol):
            return tok
        if not (("." in tok) or (abs(v) >= 100)) or not allowed_sorted:
            return tok
        nearest = min(allowed_sorted, key=lambda a: abs(a - v))
        fixes += 1
        return f"{int(nearest)}" if float(nearest).is_integer() else f"{nearest:.2f}"

    return _NUM_RE.sub(_repl, text), fixes


def _cache_signature(row: dict) -> str:
    """Stable cache key — only deterministic business fields, NEVER the model
    scores (ensemble/AE can drift run-to-run), so re-runs of the same perimeter
    hit the cache. Keyed on route, labels, top-3 drivers (value+sigma) and the
    business-rule drivers."""
    parts = [str(row.get("ROTTA", "")),
             str(row.get("anomaly_label", "")),
             str(row.get("final_risk", ""))]
    z_scores: dict[str, float] = {}
    for z_col in _Z_COL_MAP:
        v = row.get(z_col)
        if v is not None:
            try:
                z_scores[z_col] = abs(float(v))
            except (TypeError, ValueError):
                pass
    for z_col in sorted(z_scores, key=z_scores.__getitem__, reverse=True)[:3]:
        abs_col, label = _Z_COL_MAP[z_col]
        try:
            zv = round(float(row.get(z_col, 0)), 2)
        except (TypeError, ValueError):
            zv = 0.0
        av = row.get(abs_col)
        try:
            av = round(float(av), 3)
        except (TypeError, ValueError):
            pass
        parts.append(f"{label}|{av}|{zv}")
    rd = row.get("risk_drivers")
    parts.append("|".join(sorted(map(str, rd))) if isinstance(rd, (list, tuple)) else str(rd))
    return "\n".join(parts)


def _fingerprint(row: dict) -> tuple:
    """Operational risk pattern for dedup — keyed on the business rules that fired
    (the qualitative 'story') plus the final-risk tier. The exact z-driver values
    are carried by each route's own template, so routes sharing a rule profile
    reuse one representative LLM narration.

    Rationale (measured on real data): a driver-based key was far too granular —
    38 narrated routes collapsed to only 32 patterns (and route-order alone split
    ~5 spurious ones). Keying on (rules, final_risk) gives ~20 patterns, the
    meaningful operational grain, halving LLM calls with no loss of per-route
    detail (templates keep each route's figures)."""
    rd = row.get("risk_drivers")
    rules = tuple(sorted(map(str, rd))) if isinstance(rd, (list, tuple)) else (str(rd),)
    return (rules, str(row.get("final_risk", "")))


def _load_cache() -> dict:
    try:
        return json.loads(_CACHE_PATH.read_text())
    except Exception:
        return {}


def _save_cache(cache: dict) -> None:
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2))
    except Exception as e:  # noqa: BLE001
        logger.warning("ReportAgent -- cache save failed: %s", e)


def _template_explanation(row: dict) -> str:
    """Deterministic explanation (no LLM): used for anomaly levels outside
    get_llm_narrate_levels() and as the no-LLM / failure fallback."""
    z_scores: dict[str, float] = {}
    for z_col in _Z_COL_MAP:
        v = row.get(z_col)
        if v is not None:
            try:
                z_scores[z_col] = abs(float(v))
            except (TypeError, ValueError):
                pass
    drivers = []
    for z_col in sorted(z_scores, key=z_scores.__getitem__, reverse=True)[:2]:
        abs_col, label = _Z_COL_MAP[z_col]
        try:
            zv = float(row.get(z_col, 0))
        except (TypeError, ValueError):
            zv = 0.0
        drivers.append(f"{label}={_fmt(row.get(abs_col))} ({zv:+.2f}σ)")
    rd = row.get("risk_drivers")
    rules = "; ".join(map(str, rd)) if isinstance(rd, (list, tuple)) and len(rd) else "no business rule fired"
    final = row.get("final_risk") or row.get("anomaly_label") or "N/A"
    drv = ", ".join(drivers) if drivers else "n/a"
    return (f"Route {row.get('ROTTA', 'N/A')} classified {final}. "
            f"Main drivers: {drv}. Business rules: {rules}.")


def run_report_agent(
    state: AgentState,
    save_output: bool = True,
    output_path: Path | str | None = None,
    use_llm: bool = True,
    dry_run: bool = False,
) -> AgentState:
    """Generates the final JSON report using LLM and OutlierAgent output."""
    logger.info("ReportAgent -- Starting")

    try:
        if (use_llm and not dry_run
                and get_llm_backend() == "anthropic" and not get_anthropic_api_key()):
            raise ValueError(
                "ANTHROPIC_API_KEY not set: cannot use the anthropic LLM backend "
                "(set LLM_BACKEND=openai_compatible for a local model, or =none)."
            )

        # Prefer df_risk (contains business-rule columns, final_risk and
        # risk_drivers from the RiskProfilingAgent) and fall back to
        # df_anomalies for backward compatibility.
        df = state.get("df_risk")
        upstream_meta = state.get("risk_meta") or {}
        used_risk_layer = isinstance(df, pd.DataFrame) and not df.empty

        if not used_risk_layer:
            df = state.get("df_anomalies")
            upstream_meta = state.get("anomaly_meta") or {}
            logger.info("ReportAgent -- df_risk missing/empty, falling back to df_anomalies")

        if upstream_meta.get("error"):
            raise ValueError(f"upstream meta contains error: {upstream_meta['error']}")
        if df is None or not isinstance(df, pd.DataFrame):
            raise ValueError("df_risk/df_anomalies missing: run upstream agents first.")
        if df.empty:
            raise ValueError("upstream DataFrame is empty: cannot generate report.")
        # ``a_meta`` is what the existing summary builder expects (counts of
        # HIGH/MEDIUM/NORMAL + thresholds). Always populate it from anomaly_meta
        # when present, falling back to the upstream meta otherwise.
        a_meta = state.get("anomaly_meta") or upstream_meta

        perimeter = state.get("perimeter") or {}
        n_tot = int(len(df))
        llm = None
        model_name = None
        llm_warning = None
        if use_llm and not dry_run:
            try:
                llm = build_llm()
                backend = get_llm_backend()
                model_name = (get_llm_model() if backend == "openai_compatible"
                              else get_anthropic_model() if backend == "anthropic"
                              else "none")
            except Exception as e:
                llm_warning = f"LLM init failed ({e}); using deterministic fallback."
                logger.warning("ReportAgent -- %s", llm_warning)

        # Routes to explain: all HIGH/MEDIUM, sorted by descending score.
        # Full rows are passed to format_route_for_llm so it can extract
        # z-score drivers. The final findings dict only stores key fields.
        explain_df = (
            df[df["anomaly_label"].isin(["HIGH", "MEDIUM"])]
            .sort_values("ensemble_score", ascending=False)
            .copy()
        )
        full_rows = explain_df.to_dict(orient="records")
        # Output columns include the RiskProfilingAgent layer so consumers
        # of the JSON report (Streamlit, downstream auditors) see the full
        # picture, not just the ML ensemble.
        _output_cols = ["ROTTA", "PAESE_PART", "ZONA", "anomaly_label",
                        "ensemble_score", "score_composito", "baseline_score",
                        "final_risk", "confidence", "br_score", "risk_drivers"]

        # ── Per-route narration: cache → LLM (⑤ + guardrail) → template ─────
        # The cache key is perimeter-stable (no scores), so re-runs of the same
        # perimeter are instant. Cache-miss LLM calls run concurrently.
        cache = _load_cache()
        new_entries: dict[str, str] = {}
        narrate_levels = get_llm_narrate_levels()

        def _is_narrated(row: dict) -> bool:
            # Gate on final_risk (operational severity = ML + business rules),
            # falling back to anomaly_label.
            severity = str(row.get("final_risk") or row.get("anomaly_label") or "").upper()
            return severity in narrate_levels

        # Adaptive pattern-dedup: when MANY routes qualify for narration, the LLM
        # writes ONE example per risk pattern (fingerprint) and the rest get a
        # deterministic template → bounded LLM cost on large perimeters. Small
        # perimeters (≤ threshold) keep a dedicated LLM narration per route.
        rep_rotte: set = set()
        dedup_active = False
        n_patterns = 0
        if llm is not None:
            narrate_rows = [r for r in full_rows if _is_narrated(r)]
            if len(narrate_rows) > get_llm_dedup_threshold():
                dedup_active = True
                groups: dict[tuple, list] = {}
                for r in narrate_rows:
                    groups.setdefault(_fingerprint(r), []).append(r)
                n_patterns = len(groups)
                for grp in groups.values():
                    rep = max(grp, key=lambda r: float(r.get("confidence")
                                                       or r.get("ensemble_score") or 0))
                    rep_rotte.add(rep.get("ROTTA"))
                logger.info("ReportAgent -- pattern-dedup ON: %d narrated routes -> "
                            "%d LLM calls (patterns)", len(narrate_rows), n_patterns)

        def _explain(row: dict) -> tuple[str, str]:
            sig = _cache_signature(row)
            cached = cache.get(sig)
            if cached is not None:
                return cached, "cache"
            if llm is None:
                return _template_explanation(row), "no_llm"
            if not _is_narrated(row):
                return _template_explanation(row), "template"
            if dedup_active and row.get("ROTTA") not in rep_rotte:
                # follower of a pattern already narrated by its representative
                return _template_explanation(row), "dedup_template"
            context = format_route_for_llm(row)
            try:
                raw = generate_explanation(context, llm)
                text, _fixes = _guardrail(raw, _ctx_numbers(context))
                new_entries[sig] = text  # GIL-atomic dict insert → thread-safe
                return text, "llm"
            except Exception as e:  # noqa: BLE001
                logger.warning("ReportAgent -- LLM call failed (%s); fallback used", e)
                return _template_explanation(row), "llm_fail"

        concurrency = get_llm_concurrency()
        if llm is not None and concurrency > 1 and len(full_rows) > 1:
            with ThreadPoolExecutor(max_workers=concurrency) as ex:
                outcomes = list(ex.map(_explain, full_rows))
        else:
            outcomes = [_explain(r) for r in full_rows]

        if new_entries:
            cache.update(new_entries)
            _save_cache(cache)
        if not llm_warning and any(src == "llm_fail" for _, src in outcomes):
            llm_warning = "One or more LLM calls failed; deterministic fallback used for those routes."
        src_counts: dict[str, int] = {}
        for _txt, _src in outcomes:
            src_counts[_src] = src_counts.get(_src, 0) + 1

        findings = []
        for row, (explanation, _src) in zip(full_rows, outcomes):
            # Store only key fields in the output JSON (not all 65 columns)
            finding_record = {k: row.get(k) for k in _output_cols if k in row}
            finding_record["explanation"] = explanation
            findings.append(finding_record)

        report = build_final_report(
            findings=findings,
            perimeter=perimeter,
            anomaly_meta=a_meta,
            n_tot=n_tot,
        )
        if llm_warning:
            report["warning"] = llm_warning
        report["narration"] = {
            "mode": "pattern-dedup" if dedup_active else "per-route",
            "patterns": n_patterns,
            "sources": src_counts,
        }

        report_path = None
        if save_output:
            default_out = _PROJECT_ROOT / PATHS["multiagent_report"]
            out_path = Path(output_path) if output_path is not None else default_out
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
            report_path = str(out_path)
            logger.info("ReportAgent output saved to: %s", report_path)

        logger.info("ReportAgent ✓ Completed — routes=%d, explanations=%d", n_tot, len(findings))
        return {
            **state,
            "report": report,
            "report_path": report_path,
        }
    except Exception as e:
        logger.error("ReportAgent ✗ Error: %s", e)
        return {
            **state,
            "report": {"error": str(e), "user_message": "Report generation error: check configuration and inputs."},
            "report_path": None,
        }


if __name__ == "__main__":
    from multiagent_pipeline.agents.data_agent import data_agent_node
    
    from multiagent_pipeline.agents.baseline_agent import run_baseline_agent
    from multiagent_pipeline.agents.outlier_agent import run_outlier_agent

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    s: AgentState = {"perimeter": {"anno": 2024}}
    s = data_agent_node(s)
    s = run_baseline_agent(s)
    s = run_outlier_agent(s)
    s = run_report_agent(s)
    print("\n=== ReportAgent RESULT ===")
    print(s["report_path"])
    print((s["report"] or {}).get("summary"))
