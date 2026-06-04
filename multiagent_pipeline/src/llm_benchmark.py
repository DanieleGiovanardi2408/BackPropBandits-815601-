"""LLM backend benchmark for the ReportAgent narratives.

Compares LLM backends on the *real* report-generation task across the axes that
matter for the cost-efficiency decision (Project #2, "classical vs multi-agent"):

  - latency        : seconds per route on the target hardware (avg + p50 + p95);
  - throughput     : end-to-end output tokens / second (hardware-normalised);
  - cost           : USD per route and projected per full run (local = $0);
  - faithfulness   : do the numbers cited in the narrative actually appear in
                     the context, or did the model invent them?
  - constraints    : English-only, no <think> trace, ~2-4 sentences, cites the
                     route's top anomaly drivers.

Design notes
------------
* Self-contained: synthetic but realistic routes (no confidential data), so the
  benchmark is reproducible and shareable in the repo / slides.
* Schema is aligned with the *canonical* pipeline: ``anomaly_label`` ∈
  {HIGH, MEDIUM, NORMAL}, ``final_risk`` ∈ {CRITICAL, HIGH, MEDIUM, LOW} — the
  same English schema ``format_route_for_llm`` reads.
* Real token counts are captured from the model response (LangChain
  ``usage_metadata`` / OpenAI ``token_usage``); falls back to a char-based
  estimate (~4 chars/token) only if the server omits usage, flagged in output.
* Results accumulate into ``data/processed/llm_benchmark.json`` keyed by model,
  so local vs cloud rows build a single cross-backend comparison table.

Usage
-----
    # local (LM Studio / Ollama / vLLM) — benchmarks the *loaded* model:
    python3 -m multiagent_pipeline.src.llm_benchmark
    python3 -m multiagent_pipeline.src.llm_benchmark --model qwen2.5-7b-instruct --repeats 5

    # cloud Claude row (needs ANTHROPIC_API_KEY) — same prompts, measured:
    python3 -m multiagent_pipeline.src.llm_benchmark --backend anthropic

    # plumbing check, no model needed (validates metrics/percentiles/cost):
    python3 -m multiagent_pipeline.src.llm_benchmark --backend mock --repeats 3

Knobs
-----
    BENCH_MAX_TOKENS  output token cap (default 400) — lower it to A/B the
                      "shorter output = lower CPU latency" optimisation.
    BENCH_MODEL       override the local model id (same as --model).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request
from pathlib import Path

import numpy as np
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI

from multiagent_pipeline.agents.report_agent import format_route_for_llm, generate_explanation

BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
NATIVE_MODELS_API = "http://localhost:1234/api/v0/models"
RESULTS_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "llm_benchmark.json"
MAX_TOKENS = int(os.getenv("BENCH_MAX_TOKENS", "400"))
_NUM_RE = re.compile(r"-?\d+\.?\d*")
_ITALIAN_MARKERS = ["rotta", "allarmi", "rischio", "soglia", "respinti", "frontiera", "controlli", "viaggiatori"]

# Full-run sizes used to project per-route cost to a whole report.
# ~57 = HIGH+MEDIUM (canonical narrates both); ~17 = HIGH-only (optimisation).
_RUN_SIZE_FULL = 57
_RUN_SIZE_HIGH_ONLY = 17

# ── Pricing (USD per 1K tokens) ───────────────────────────────────────────────
# List prices as of 2026 — VERIFY before quoting in the slides. Matched by
# substring against the model id; anything not matched (local / self-hosted)
# costs $0 marginal.
PRICING_USD_PER_1K: dict[str, tuple[float, float]] = {
    "claude-opus":       (0.015, 0.075),
    "claude-sonnet-4-5": (0.003, 0.015),
    "claude-sonnet":     (0.003, 0.015),
    "claude-haiku":      (0.001, 0.005),
}


def _price_for(model_id: str) -> tuple[float, float]:
    """(input, output) USD per 1K tokens; (0, 0) for local/self-hosted."""
    mid = (model_id or "").lower()
    for key, price in PRICING_USD_PER_1K.items():
        if key in mid:
            return price
    return 0.0, 0.0


# ── Token-usage capture ───────────────────────────────────────────────────────
class _UsageCollector(BaseCallbackHandler):
    """Captures real (input, output) token counts per LLM call.

    Attached to the chat model, so it works without touching
    ``generate_explanation`` in report_agent. Reads LangChain's
    ``usage_metadata`` first, then the OpenAI-style ``token_usage`` fallback.
    """

    def __init__(self) -> None:
        self.calls: list[dict | None] = []

    def on_llm_end(self, response, **kwargs) -> None:  # noqa: ANN001
        usage = None
        try:
            msg = response.generations[0][0].message
            um = getattr(msg, "usage_metadata", None)
            if um:
                usage = {"in": int(um.get("input_tokens") or 0),
                         "out": int(um.get("output_tokens") or 0)}
        except Exception:
            pass
        if usage is None:
            lo = getattr(response, "llm_output", None) or {}
            tu = lo.get("token_usage") or lo.get("usage") or {}
            if tu:
                usage = {"in": int(tu.get("prompt_tokens") or 0),
                         "out": int(tu.get("completion_tokens") or 0)}
        self.calls.append(usage)


# ── Mock backend (plumbing validation, no model needed) ───────────────────────
class _MockMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _MockLLM:
    """Echoes a few real numbers from the prompt so faithfulness/coverage paths
    exercise; no token usage (falls back to char estimate)."""

    def invoke(self, messages):  # noqa: ANN001
        ctx = messages[-1].content if messages else ""
        nums = _NUM_RE.findall(ctx)[:3]
        joined = ", ".join(nums) if nums else "no figures"
        return _MockMessage(
            f"This route is classified HIGH risk. Key drivers cited: {joined}. "
            f"No business rule fired in this synthetic mock generation."
        )


# ── Representative synthetic routes (no confidential data) ─────────────────────
# Schema aligned with the canonical pipeline (English labels). Each route carries
# ``_driver_values``: absolute values of the three features format_route_for_llm
# surfaces as top drivers (highest |z|). Keys prefixed with ``_`` are ignored.
ROUTES = [
    {  # R1 — HIGH, Interpol-driven
        "ROTTA": "CMN-FCO", "PAESE_PART": "MAR", "ZONA": 4,
        "anomaly_label": "HIGH", "ensemble_score": 0.91,
        "score_if": 0.88, "score_lof": 0.93, "score_ae": 0.85,
        "tot_allarmi_sum": 142.0, "tot_entrati": 5300.0,
        "z_pct_interpol": 3.4, "z_tasso_respinti": 2.9, "z_tot_allarmi_log": 2.2,
        "z_score_rischio_esiti": 1.8, "z_tasso_allarme_medio": 1.1, "z_pct_sdi": 0.7,
        "z_pct_nsis": -0.3, "z_tasso_rilevanza": 1.4, "z_tasso_inv_medio": 0.9,
        "z_tasso_chiusura": -1.2, "z_tasso_fermati": 1.0,
        "pct_interpol": 0.46, "tasso_respinti": 0.31, "score_rischio_esiti": 0.40,
        "tasso_allarme_medio": 0.22, "pct_sdi": 0.18, "pct_nsis": 0.05,
        "tasso_rilevanza": 0.55, "tasso_inv_medio": 0.34, "tasso_chiusura": 0.08,
        "tasso_fermati": 0.12,
        "_driver_values": [0.46, 0.31, 142.0],
        "final_risk": "CRITICAL",
        "risk_drivers": ["High INTERPOL alarm rate", "High rejection rate",
                         "Multi-source alarms (INTERPOL + SDI)"],
    },
    {  # R2 — HIGH, outcome/rejection-driven
        "ROTTA": "TIA-FCO", "PAESE_PART": "ALB", "ZONA": 3,
        "anomaly_label": "HIGH", "ensemble_score": 0.85,
        "score_if": 0.82, "score_lof": 0.86, "score_ae": 0.80,
        "tot_allarmi_sum": 88.0, "tot_entrati": 4100.0,
        "z_pct_interpol": 1.2, "z_tasso_respinti": 3.6, "z_tot_allarmi_log": 1.5,
        "z_score_rischio_esiti": 3.1, "z_tasso_allarme_medio": 1.0, "z_pct_sdi": 0.9,
        "z_pct_nsis": 0.2, "z_tasso_rilevanza": 1.1, "z_tasso_inv_medio": 0.6,
        "z_tasso_chiusura": -0.8, "z_tasso_fermati": 2.4,
        "pct_interpol": 0.21, "tasso_respinti": 0.42, "score_rischio_esiti": 0.55,
        "tasso_allarme_medio": 0.19, "pct_sdi": 0.20, "pct_nsis": 0.06,
        "tasso_rilevanza": 0.48, "tasso_inv_medio": 0.28, "tasso_chiusura": 0.11,
        "tasso_fermati": 0.20,
        "_driver_values": [0.42, 0.55, 0.20],
        "final_risk": "CRITICAL",
        "risk_drivers": ["High rejection rate", "Multi-source alarms (INTERPOL + SDI)"],
    },
    {  # R3 — MEDIUM, borderline
        "ROTTA": "IST-MXP", "PAESE_PART": "TUR", "ZONA": 5,
        "anomaly_label": "MEDIUM", "ensemble_score": 0.31,
        "score_if": 0.30, "score_lof": 0.33, "score_ae": 0.29,
        "tot_allarmi_sum": 24.0, "tot_entrati": 6900.0,
        "z_pct_interpol": 0.6, "z_tasso_respinti": 0.7, "z_tot_allarmi_log": 0.5,
        "z_score_rischio_esiti": 0.8, "z_tasso_allarme_medio": 1.8, "z_pct_sdi": 1.5,
        "z_pct_nsis": 0.4, "z_tasso_rilevanza": 0.9, "z_tasso_inv_medio": 1.3,
        "z_tasso_chiusura": 0.3, "z_tasso_fermati": 0.5,
        "pct_interpol": 0.09, "tasso_respinti": 0.12, "score_rischio_esiti": 0.18,
        "tasso_allarme_medio": 0.18, "pct_sdi": 0.22, "pct_nsis": 0.04,
        "tasso_rilevanza": 0.31, "tasso_inv_medio": 0.30, "tasso_chiusura": 0.27,
        "tasso_fermati": 0.07,
        "_driver_values": [0.18, 0.22, 0.30],
        "final_risk": "MEDIUM",
        "risk_drivers": ["Multi-source alarms (INTERPOL + SDI)"],
    },
]


def detect_loaded_model() -> str | None:
    """Return the id of the chat model to benchmark.

    Honors BENCH_MODEL when set; otherwise returns the first loaded chat model
    via the LM Studio native API, falling back to the first non-embedding model
    from the OpenAI-compatible endpoint.
    """
    override = os.getenv("BENCH_MODEL")
    if override:
        return override
    try:
        with urllib.request.urlopen(NATIVE_MODELS_API, timeout=10) as r:
            for m in json.load(r).get("data", []):
                if m.get("state") == "loaded" and m.get("type") in ("llm", "vlm"):
                    return m["id"]
    except Exception:
        pass
    try:
        with urllib.request.urlopen(BASE_URL + "/models", timeout=10) as r:
            for m in json.load(r).get("data", []):
                if "embed" not in m["id"].lower():
                    return m["id"]
    except Exception:
        pass
    return None


def _nums(text: str) -> list[float]:
    out = []
    for tok in _NUM_RE.findall(text or ""):
        try:
            out.append(round(float(tok), 2))
        except ValueError:
            pass
    return out


def _pct(values: list[float], q: float) -> float:
    """Percentile with linear interpolation; 0.0 on empty input."""
    return float(np.percentile(values, q)) if values else 0.0


def _matches_context(n: float, ctx_nums, tol: float = 0.01) -> bool:
    """True if n appears in the context, allowing percent/fraction equivalence
    (output '46%' is faithful to a context value of 0.46, and vice-versa)."""
    return any(abs(n - c) <= tol or abs(n / 100 - c) <= tol or abs(n * 100 - c) <= tol
               for c in ctx_nums)


def evaluate(route: dict, context: str, explanation: str, latency: float) -> dict:
    """Compute faithfulness + constraint metrics for one generation (text only;
    latency/tokens/cost are handled by the caller)."""
    out_nums = _nums(explanation)
    ctx_nums = set(_nums(context))
    unmatched = [n for n in out_nums if not _matches_context(n, ctx_nums)]
    faithfulness = 1.0 - (len(unmatched) / len(out_nums)) if out_nums else 1.0

    driver_vals = route.get("_driver_values", [])
    covered = sum(1 for v in driver_vals if any(abs(round(v, 2) - o) <= 0.01 for o in out_nums))

    sentences = [s for s in re.split(r"[.!?]+\s+", (explanation or "").strip()) if s.strip()]
    italian = [w for w in _ITALIAN_MARKERS if w in (explanation or "").lower()]

    return {
        "latency_s": round(latency, 1),
        "empty": len((explanation or "").strip()) == 0,
        "chars": len(explanation or ""),
        "sentences": len(sentences),
        "has_think": "<think>" in (explanation or "").lower(),
        "italian_leak": italian,
        "numbers_cited": len(out_nums),
        "hallucinated_count": len(unmatched),
        "hallucinated_values": sorted(set(unmatched)),
        "faithfulness": round(faithfulness, 3),
        "driver_coverage": f"{covered}/{len(driver_vals)}",
    }


# ── Optimisation ⑤ — constrained narration ────────────────────────────────────
# Inject the verified figures, let the model write ONLY prose, then guardrail any
# number it still invents → faithfulness 1.00 by construction, for ANY model.
def _build_facts(route: dict) -> str:
    """Canonical context PLUS final_risk + business rules — the enrichment the
    canonical ``format_route_for_llm`` is currently missing (finding ②)."""
    base = format_route_for_llm(route)
    final_risk = route.get("final_risk", "N/A")
    rd = route.get("risk_drivers") or []
    rules = "; ".join(map(str, rd)) if rd else "none fired"
    return (f"{base}\n"
            f"Final risk classification: {final_risk}\n"
            f"Business rules fired: {rules}")


def generate_constrained(facts: str, llm) -> str:  # noqa: ANN001
    """⑤ prompt: copy figures verbatim, ≤3 sentences, prose only."""
    from langchain_core.messages import SystemMessage, HumanMessage
    system = ("You are a border-control risk analyst. Write in English only. "
              "Use ONLY the figures in the facts and copy each number EXACTLY as written. "
              "Never compute, re-round, or introduce a number that is not in the facts.")
    user = ("Write at most 3 short sentences for a border-control operator explaining why "
            "this route is flagged and what its risk level means. Cite at least two drivers "
            "(value and sigma) and the final risk classification. No bullet points, no headings.\n\n"
            f"Verified facts:\n{facts}")
    resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return resp.content.strip()


def _guardrail(text: str, allowed: set[float], tol: float = 0.01) -> tuple[str, int]:
    """Replace any figure-like number absent from the context with the nearest
    real one (faithfulness 1.00 by construction). Small bare integers
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
            return tok                       # already a real figure (incl. % form)
        is_figure = ("." in tok) or (abs(v) >= 100)
        if not is_figure or not allowed_sorted:
            return tok                       # a count/ordinal — leave it alone
        nearest = min(allowed_sorted, key=lambda a: abs(a - v))
        fixes += 1
        return f"{int(nearest)}" if float(nearest).is_integer() else f"{nearest:.2f}"

    return _NUM_RE.sub(_repl, text), fixes


def build_backend(backend: str, model_override: str | None,
                  base_url: str | None = None,
                  api_key: str | None = None) -> tuple[object, str, str, _UsageCollector]:
    """Returns (llm, model_id, backend_label, usage_collector)."""
    collector = _UsageCollector()
    if backend == "mock":
        return _MockLLM(), "mock-echo", "mock", collector
    if backend == "anthropic":
        from langchain_anthropic import ChatAnthropic
        from multiagent_pipeline.config import get_anthropic_api_key, get_anthropic_model
        if not get_anthropic_api_key():
            raise RuntimeError("ANTHROPIC_API_KEY not set — cannot benchmark the Claude backend.")
        model_id = model_override or get_anthropic_model()
        llm = ChatAnthropic(model=model_id, temperature=0, max_tokens=MAX_TOKENS,
                            callbacks=[collector])
        return llm, model_id, "anthropic", collector
    # default: local OpenAI-compatible (LM Studio / Ollama / vLLM)
    model_id = model_override or detect_loaded_model()
    if not model_id:
        raise RuntimeError("No loaded chat model found. Run `lms load <id>` on the server first.")
    llm = ChatOpenAI(base_url=(base_url or BASE_URL), api_key=(api_key or "lm-studio"),
                     model=model_id, temperature=0, max_tokens=MAX_TOKENS, callbacks=[collector])
    return llm, model_id, backend, collector


def benchmark(backend: str, model_override: str | None, repeats: int,
              base_url: str | None = None, api_key: str | None = None,
              mode: str = "free") -> dict:
    llm, model_id, backend_label, collector = build_backend(backend, model_override, base_url, api_key)
    price_in, price_out = _price_for(model_id)
    print(f"Benchmarking [{backend_label}] {model_id}  | mode={mode} | repeats={repeats} | max_tokens={MAX_TOKENS}\n")

    samples: list[dict] = []          # one per (route, repeat): latency + tokens
    quality: list[dict] = []          # text-quality eval per generation
    token_estimated = False

    for rep in range(repeats):
        for route in ROUTES:
            # ⑤ constrained: facts block (with final_risk + rules), prose only.
            context = _build_facts(route) if mode == "constrained" else format_route_for_llm(route)
            before = len(collector.calls)
            t0 = time.time()
            try:
                raw = (generate_constrained(context, llm) if mode == "constrained"
                       else generate_explanation(context, llm))
            except Exception as e:  # noqa: BLE001 — record the failure, keep going
                raw = ""
                print(f"  {route['ROTTA']:8s} ERROR: {e}")
            dt = time.time() - t0

            usage = collector.calls[-1] if len(collector.calls) > before else None
            if usage and (usage["out"] or usage["in"]):
                in_tok, out_tok = usage["in"], usage["out"]
            else:
                # Fallback: estimate ~4 chars/token (flagged in the output).
                token_estimated = True
                in_tok = max(1, len(context) // 4)
                out_tok = max(1, len(raw) // 4)

            # ⑤ guardrail: snap invented figures to the nearest real one.
            if mode == "constrained":
                expl, n_fixes = _guardrail(raw, set(_nums(context)))
            else:
                expl, n_fixes = raw, 0

            ev = evaluate(route, context, expl, dt)
            if mode == "constrained":
                ev["faith_raw"] = evaluate(route, context, raw, dt)["faithfulness"]
                ev["guardrail_fixes"] = n_fixes
            tok_s = round(out_tok / dt, 1) if dt > 0 else 0.0
            cost = (in_tok / 1000.0) * price_in + (out_tok / 1000.0) * price_out
            ev.update(route=route["ROTTA"], anomaly_label=route["anomaly_label"],
                      in_tokens=in_tok, out_tokens=out_tok, tok_per_s=tok_s,
                      cost_usd=round(cost, 6), text=expl)
            samples.append({"latency_s": dt, "in": in_tok, "out": out_tok, "cost": cost})
            quality.append(ev)
            tag = "EST" if usage is None else "   "
            extra = (f" | raw {ev['faith_raw']:.2f}->{ev['faithfulness']:.2f} fix {n_fixes}"
                     if mode == "constrained" else "")
            print(f"  r{rep+1} {route['ROTTA']:8s} | {dt:6.1f}s | {tok_s:5.1f} tok/s {tag} "
                  f"| faith {ev['faithfulness']:.2f} | halluc {ev['hallucinated_count']} "
                  f"| drivers {ev['driver_coverage']} | sent {ev['sentences']:2d} "
                  f"| {'EMPTY' if ev['empty'] else 'ok'}{extra}")

    n = len(quality)
    lat = [s["latency_s"] for s in samples]
    total_lat = sum(lat)
    total_out = sum(s["out"] for s in samples)
    cost_per_route = sum(s["cost"] for s in samples) / n if n else 0.0

    agg = {
        "model": model_id,
        "backend": backend_label,
        "mode": mode,
        "n_routes": len(ROUTES),
        "repeats": repeats,
        "n_samples": n,
        "max_tokens_cap": MAX_TOKENS,
        "tokens_estimated": token_estimated,
        "latency_s": {
            "avg": round(total_lat / n, 1) if n else 0.0,
            "p50": round(_pct(lat, 50), 1),
            "p95": round(_pct(lat, 95), 1),
            "min": round(min(lat), 1) if lat else 0.0,
            "max": round(max(lat), 1) if lat else 0.0,
        },
        "throughput_tok_s_e2e": round(total_out / total_lat, 1) if total_lat else 0.0,
        "avg_in_tokens": round(sum(s["in"] for s in samples) / n) if n else 0,
        "avg_out_tokens": round(total_out / n) if n else 0,
        "avg_faithfulness": round(sum(q["faithfulness"] for q in quality) / n, 3) if n else 0.0,
        "total_hallucinated": sum(q["hallucinated_count"] for q in quality),
        "empty_outputs": sum(1 for q in quality if q["empty"]),
        "avg_sentences": round(sum(q["sentences"] for q in quality) / n, 1) if n else 0.0,
        "cost": {
            "price_in_per_1k": price_in,
            "price_out_per_1k": price_out,
            "per_route_usd": round(cost_per_route, 6),
            f"run_{_RUN_SIZE_HIGH_ONLY}_usd": round(cost_per_route * _RUN_SIZE_HIGH_ONLY, 4),
            f"run_{_RUN_SIZE_FULL}_usd": round(cost_per_route * _RUN_SIZE_FULL, 4),
        },
        "per_route": quality,
    }
    if mode == "constrained":
        agg["avg_faith_raw"] = round(sum(q.get("faith_raw", 1.0) for q in quality) / n, 3) if n else 0.0
        agg["guardrail_fixes_total"] = sum(q.get("guardrail_fixes", 0) for q in quality)
    return agg


def _fmt_usd(x: float) -> str:
    return "free" if x == 0 else f"${x:.4f}"


def print_comparison(allres: dict) -> None:
    print("\n=== CROSS-BACKEND COMPARISON (lower latency / cost, higher faith & tok/s = better) ===")
    header = (f"{'model':26s} {'backend':12s} {'p50':>6s} {'p95':>6s} {'avg':>6s} "
              f"{'tok/s':>6s} {'faith':>6s} {'hall':>5s} {'empty':>6s} {'sent':>5s} {'$/17rt':>9s}")
    print(header)
    print("-" * len(header))
    for a in allres.values():
        lat = a.get("latency_s", {})
        cost = a.get("cost", {})
        run17 = cost.get(f"run_{_RUN_SIZE_HIGH_ONLY}_usd", 0.0)
        name = (a.get("label") or a.get("model", "?"))
        print(f"{name[:26]:26s} {a.get('backend','?'):12s} "
              f"{lat.get('p50',0):>6.1f} {lat.get('p95',0):>6.1f} {lat.get('avg',0):>6.1f} "
              f"{a.get('throughput_tok_s_e2e',0):>6.1f} {a.get('avg_faithfulness',0):>6.2f} "
              f"{a.get('total_hallucinated',0):>5d} {a.get('empty_outputs',0):>6d} "
              f"{a.get('avg_sentences',0):>5.1f} {_fmt_usd(run17):>9s}")
    print(f"\n  $/17rt = projected cost for a {_RUN_SIZE_HIGH_ONLY}-route (HIGH-only) run; "
          f"local/self-hosted = free. p50/p95 in seconds/route.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark LLM backends for the ReportAgent.")
    ap.add_argument("--backend", choices=["local", "anthropic", "mock"], default="local")
    ap.add_argument("--model", default=None, help="override the model id")
    ap.add_argument("--repeats", type=int, default=1, help="generations per route (for p50/p95)")
    ap.add_argument("--label", default=None,
                    help="display name / JSON key — distinguishes tuning variants of the same model")
    ap.add_argument("--base-url", default=None,
                    help="OpenAI-compatible endpoint override (e.g. Ollama: http://localhost:11434/v1)")
    ap.add_argument("--api-key", default=None, help="API key for the local endpoint (default 'lm-studio')")
    ap.add_argument("--mode", choices=["free", "constrained"], default="free",
                    help="free = canonical generation; constrained = ⑤ (inject figures + guardrail)")
    args = ap.parse_args()

    agg = benchmark(args.backend, args.model, max(1, args.repeats), args.base_url, args.api_key, args.mode)
    if args.label:
        agg["label"] = args.label

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    allres = {}
    if RESULTS_PATH.exists():
        try:
            allres = json.loads(RESULTS_PATH.read_text())
        except json.JSONDecodeError:
            allres = {}
    allres[agg.get("label") or agg["model"]] = agg
    RESULTS_PATH.write_text(json.dumps(allres, indent=2, ensure_ascii=False))
    print(f"\nSaved -> {RESULTS_PATH}")

    if agg["tokens_estimated"]:
        print("  NOTE: token counts estimated (~4 chars/token) — server did not return usage.")

    print_comparison(allres)


if __name__ == "__main__":
    main()
