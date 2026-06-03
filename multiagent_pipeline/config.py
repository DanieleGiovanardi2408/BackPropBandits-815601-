"""Centralised configuration for the multi-agent pipeline."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_PROJECT_ROOT / ".env")


def _to_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def get_anthropic_api_key() -> str | None:
    return os.getenv("ANTHROPIC_API_KEY")


def get_anthropic_model() -> str:
    return os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")


def get_use_llm(default: bool = False) -> bool:
    return _to_bool(os.getenv("USE_LLM"), default)


def get_dry_run(default: bool = False) -> bool:
    return _to_bool(os.getenv("DRY_RUN"), default)


# ── Pluggable LLM backend (ReportAgent) ───────────────────────────────────────
def get_llm_backend() -> str:
    """Engine the ReportAgent uses: 'anthropic' (DEFAULT — cloud Claude, i.e. the
    colleague's behaviour when no env is set), 'openai_compatible' (local LM
    Studio / Ollama / vLLM), or 'none' (deterministic templates, no LLM)."""
    return (os.getenv("LLM_BACKEND") or "anthropic").strip().lower()


def get_llm_base_url() -> str:
    """OpenAI-compatible endpoint for the local backend (LM Studio default)."""
    return os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")


def get_llm_model() -> str:
    """Model id for the local backend (exact id from GET /v1/models)."""
    return os.getenv("LLM_MODEL", "qwen2.5-3b-instruct")


def get_llm_api_key() -> str:
    """API key for the local endpoint (LM Studio / Ollama ignore the value)."""
    return os.getenv("LLM_API_KEY", "lm-studio")


def get_llm_concurrency(default: int = 1) -> int:
    """Parallel LLM calls for cache-miss routes (ThreadPoolExecutor workers)."""
    try:
        return max(1, int(os.getenv("LLM_CONCURRENCY", str(default))))
    except (TypeError, ValueError):
        return default


def get_llm_narrate_levels() -> set[str]:
    """final_risk tiers the LLM narrates; the rest get a deterministic template.
    Matched against the RiskProfilingAgent's final_risk (CRITICAL/HIGH/MEDIUM/LOW)
    — operational severity (ML blended with business rules), not the raw ML label.
    Default {CRITICAL, HIGH} focuses the LLM on the routes that matter and keeps
    cost bounded on large perimeters; set LLM_NARRATE_LEVELS=CRITICAL,HIGH,MEDIUM to
    narrate every anomalous route, or =CRITICAL for the leanest mode."""
    raw = os.getenv("LLM_NARRATE_LEVELS", "CRITICAL,HIGH")
    return {x.strip().upper() for x in raw.split(",") if x.strip()}


def get_llm_dedup_threshold(default: int = 15) -> int:
    """Above this many LLM-narrated routes, the ReportAgent switches to pattern
    dedup: one LLM narration per risk pattern (fingerprint) + a deterministic
    template for the rest — bounds LLM cost on large perimeters. At or below it,
    every narrated route gets its own LLM explanation (full quality). Set
    LLM_DEDUP_THRESHOLD=0 to always dedup, or a very large value to never dedup."""
    try:
        return max(0, int(os.getenv("LLM_DEDUP_THRESHOLD", str(default))))
    except (TypeError, ValueError):
        return default

