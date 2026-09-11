"""
NF-07 — Generation Failure Taxonomy (GFT)

Typed failure classification for every failed generation job.
Replaces raw stack traces with structured failure_record.json files
containing a failure code, category, and suggested action.

Gap closed: No diffusion pipeline implements a typed failure taxonomy.
Failures are silent black images or raw exceptions — no structured routing.
"""
from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path


class FailureCode(str, Enum):
    SAFETY_REFUSAL         = "SAFETY_REFUSAL"
    PROMPT_CONFLICT        = "PROMPT_CONFLICT"
    COHERENCE_FAILURE      = "COHERENCE_FAILURE"
    LOW_ADHERENCE          = "LOW_ADHERENCE"
    BLEED_WARNING          = "BLEED_WARNING"
    COMPUTE_TIMEOUT        = "COMPUTE_TIMEOUT"
    MODEL_OOM              = "MODEL_OOM"
    MODEL_LOAD_ERROR       = "MODEL_LOAD_ERROR"
    CONSENT_CHAIN_INVALID  = "CONSENT_CHAIN_INVALID"
    WATERMARK_FAILURE      = "WATERMARK_FAILURE"
    SPEC_AMBIGUOUS         = "SPEC_AMBIGUOUS"
    UNKNOWN                = "UNKNOWN"


_TAXONOMY: dict[FailureCode, dict] = {
    FailureCode.SAFETY_REFUSAL: {
        "category": "Safety",
        "trigger":  "Safety checker blocked the output or prompt",
        "action":   "Revise negative_controls in spec or adjust prompt_builder config",
    },
    FailureCode.PROMPT_CONFLICT: {
        "category": "Spec",
        "trigger":  "AAG detected a BLOCK-level attribute contradiction",
        "action":   "Run `avatar-gen validate-spec` and fix the reported issues before retrying",
    },
    FailureCode.COHERENCE_FAILURE: {
        "category": "Quality",
        "trigger":  "Generated image fails dimension/format validation",
        "action":   "Retry with a different seed; check model checkpoint integrity",
    },
    FailureCode.LOW_ADHERENCE: {
        "category": "Quality",
        "trigger":  "overall_adherence_pct < 70 in spec-delta report",
        "action":   "Review prompt_builder attribute mapping; increase inference steps",
    },
    FailureCode.BLEED_WARNING: {
        "category": "Quality",
        "trigger":  "Orthogonality score < 0.85 on a non-target attribute",
        "action":   "Check attribute weighting in prompt_builder; isolate the bleeding attribute",
    },
    FailureCode.COMPUTE_TIMEOUT: {
        "category": "Infrastructure",
        "trigger":  "Inference exceeded the configured time budget",
        "action":   "Switch compute_route or reduce inference steps / resolution",
    },
    FailureCode.MODEL_OOM: {
        "category": "Infrastructure",
        "trigger":  "Out-of-memory on the accelerator",
        "action":   "Reduce resolution, batch size, or enable attention slicing",
    },
    FailureCode.MODEL_LOAD_ERROR: {
        "category": "Infrastructure",
        "trigger":  "Failed to load model checkpoint or missing dependencies",
        "action":   "Install required dependencies (pip install 'avatar-gen[gpu]') or check model path",
    },
    FailureCode.CONSENT_CHAIN_INVALID: {
        "category": "Compliance",
        "trigger":  "Consent chain verification failed or consent token absent",
        "action":   "Re-execute consent flow via `avatar-gen consent-init`",
    },
    FailureCode.WATERMARK_FAILURE: {
        "category": "Compliance",
        "trigger":  "Watermark embedding or verification failed",
        "action":   "Check image format (must be RGB PNG/JPEG); verify invisible-watermark install",
    },
    FailureCode.SPEC_AMBIGUOUS: {
        "category": "Spec",
        "trigger":  "Required spec field missing or underspecified",
        "action":   "Ensure all required fields are present; run `avatar-gen validate-spec`",
    },
    FailureCode.UNKNOWN: {
        "category": "Unknown",
        "trigger":  "Unclassified exception",
        "action":   "Inspect failure_record.json error_detail for the raw traceback",
    },
}


@dataclass
class FailureRecord:
    failure_code: FailureCode
    category: str
    timestamp: str
    spec_hash: str | None
    model: str | None
    seed: int | None
    error_detail: str
    suggested_action: str
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "failure_code":     self.failure_code.value,
            "category":         self.category,
            "timestamp":        self.timestamp,
            "spec_hash":        self.spec_hash,
            "model":            self.model,
            "seed":             self.seed,
            "error_detail":     self.error_detail,
            "suggested_action": self.suggested_action,
            **self.extra,
        }


# ── Classifier ────────────────────────────────────────────────────────────────

def classify_exception(exc: Exception) -> FailureCode:
    """Map a Python exception to the closest FailureCode."""
    msg = str(exc).lower()
    type(exc).__name__

    if "consent" in msg or "consent_chain" in msg:
        return FailureCode.CONSENT_CHAIN_INVALID
    if "watermark" in msg:
        return FailureCode.WATERMARK_FAILURE
    if "timeout" in msg or "timed out" in msg:
        return FailureCode.COMPUTE_TIMEOUT
    if "out of memory" in msg or "oom" in msg or "cuda out" in msg:
        return FailureCode.MODEL_OOM
    if "diffusers" in msg or "gpu" in msg or "no module named" in msg or "load" in msg:
        return FailureCode.MODEL_LOAD_ERROR
    if "block" in msg and "grammar" in msg:
        return FailureCode.PROMPT_CONFLICT
    if "safety" in msg or "nsfw" in msg:
        return FailureCode.SAFETY_REFUSAL
    if "validation" in msg and ("spec" in msg or "schema" in msg):
        return FailureCode.SPEC_AMBIGUOUS
    if "dimension" in msg or "format" in msg or "corrupt" in msg:
        return FailureCode.COHERENCE_FAILURE
    return FailureCode.UNKNOWN


def classify_safety_flags(flags: list[str]) -> FailureCode:
    """Map safety check flags to a FailureCode."""
    if not flags:
        return FailureCode.UNKNOWN
    flag_lower = " ".join(flags).lower()
    if "consent" in flag_lower:
        return FailureCode.CONSENT_CHAIN_INVALID
    if "watermark" in flag_lower:
        return FailureCode.WATERMARK_FAILURE
    if "nsfw" in flag_lower or "safety" in flag_lower:
        return FailureCode.SAFETY_REFUSAL
    if "blank" in flag_lower or "dimension" in flag_lower:
        return FailureCode.COHERENCE_FAILURE
    return FailureCode.UNKNOWN


# ── Public API ────────────────────────────────────────────────────────────────

def record_failure(
    code: FailureCode,
    error_detail: str,
    output_path: Path,
    spec_hash: str | None = None,
    model: str | None = None,
    seed: int | None = None,
    extra: dict | None = None,
) -> FailureRecord:
    """Create and write a failure_record.json.

    Args:
        code:         Typed failure code.
        error_detail: Raw error message or traceback string.
        output_path:  Path to write failure_record.json.
        spec_hash:    SHA-256 of the spec (for traceability).
        model:        Model identifier.
        seed:         Generation seed.
        extra:        Additional context key-values.

    Returns:
        FailureRecord written to disk.
    """
    taxonomy = _TAXONOMY.get(code, _TAXONOMY[FailureCode.UNKNOWN])
    record = FailureRecord(
        failure_code=code,
        category=taxonomy["category"],
        timestamp=datetime.now(timezone.utc).isoformat(),
        spec_hash=spec_hash,
        model=model,
        seed=seed,
        error_detail=error_detail[:2000],  # cap length
        suggested_action=taxonomy["action"],
        extra=extra or {},
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")
    return record


def record_exception(
    exc: Exception,
    output_path: Path,
    spec_hash: str | None = None,
    model: str | None = None,
    seed: int | None = None,
) -> FailureRecord:
    """Classify an exception and write failure_record.json."""
    code   = classify_exception(exc)
    detail = traceback.format_exc()
    return record_failure(code, detail, output_path, spec_hash, model, seed)


def summarize_failures(batch_dir: Path) -> dict[str, int]:
    """Count failure codes across all failure_record.json files in batch_dir."""
    from collections import Counter
    counts: Counter = Counter()
    for fp in batch_dir.rglob("failure_record.json"):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            counts[data.get("failure_code", "UNKNOWN")] += 1
        except Exception:
            counts["UNKNOWN"] += 1
    return dict(counts)
