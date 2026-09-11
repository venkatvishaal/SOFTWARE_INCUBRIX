"""
provenance_writer.py — Emit avatar_manifest.json for every generated avatar.

Combines the spec, inference result, safety result, and optional novel-feature
outputs into a complete, schema-validated provenance record.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import jsonschema

from avatar_gen import __version__
from avatar_gen.inference_client import InferenceResult
from avatar_gen.safety_checker import SafetyResult
from avatar_gen.spec_parser import AvatarSpec

_SCHEMA_PATH = Path(__file__).parent / "schemas" / "avatar_manifest.schema.json"
_SCHEMA: dict | None = None


def _get_schema() -> dict:
    global _SCHEMA
    if _SCHEMA is None:
        _SCHEMA = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    return _SCHEMA


def write_manifest(
    spec: AvatarSpec,
    result: InferenceResult,
    safety: SafetyResult,
    image_path: Path,
    job_dir: Path,
    *,
    orthogonality_scores: dict | None = None,
    watermark: dict | None = None,
    consent_chain: dict | None = None,
    rung_progression: list[str] | None = None,
    compute_saved_pct: float | None = None,
    spec_delta_path: Path | None = None,
    failure_record_path: Path | None = None,
) -> Path:
    """Build and write avatar_manifest.json to job_dir.

    Args:
        spec:                  Source AvatarSpec.
        result:                InferenceResult from the compute client.
        safety:                SafetyResult from the safety checker.
        image_path:            Path to the saved output image.
        job_dir:               Directory to write the manifest into.
        orthogonality_scores:  Optional NF-01 scores dict.
        watermark:             Optional NF-05 watermark metadata dict.
        consent_chain:         Optional NF-04 consent chain dict.
        rung_progression:      Optional NF-08 resolution ladder list.
        compute_saved_pct:     Optional NF-08 compute savings estimate.
        spec_delta_path:       Optional path to spec_delta.json.
        failure_record_path:   Optional path to failure_record.json.

    Returns:
        Path to the written manifest file.
    """
    from PIL import Image as PILImage

    img = PILImage.open(image_path)
    w, h = img.size
    ar = _aspect_ratio_str(w, h)

    manifest: dict = {
        "avatar_id":        spec.avatar_id,
        "spec":             _spec_to_dict(spec),
        "seed":             spec.seed,
        "model":            result.model,
        "model_revision":   result.model_revision,
        "checkpoint":       None,
        "prompt_generated": "",   # will be filled by pipeline coordinator
        "negative_prompt":  "",
        "provenance": {
            "pipeline_version": __version__,
            "generated_at":     datetime.now(timezone.utc).isoformat(),
            "compute_route":    result.compute_route,
            "runtime_sec":      round(result.runtime_sec, 3),
            "steps":            result.steps,
            "guidance_scale":   result.guidance_scale,
        },
        "safety":     safety.to_dict(),
        "dimensions": {
            "width":        w,
            "height":       h,
            "aspect_ratio": ar,
        },
        "orthogonality_scores": orthogonality_scores,
        "watermark":            watermark,
        "consent_chain":        consent_chain,
        "rung_progression":     rung_progression,
        "compute_saved_pct":    compute_saved_pct,
        "output_files": {
            "image":          str(image_path.resolve()),
            "spec_delta":     str(spec_delta_path.resolve()) if spec_delta_path else None,
            "failure_record": str(failure_record_path.resolve()) if failure_record_path else None,
        },
    }

    _validate_manifest(manifest)

    out_path = job_dir / "avatar_manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_path


def patch_manifest(manifest_path: Path, updates: dict) -> None:
    """Merge updates into an existing manifest (used by pipeline coordinator)."""
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    _deep_merge(data, updates)
    _validate_manifest(data)
    manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_manifest(path: Path) -> dict:
    """Load and return a manifest dictionary."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ── Internal helpers ──────────────────────────────────────────────────────────

def _spec_to_dict(spec: AvatarSpec) -> dict:
    return spec.model_dump()


def _aspect_ratio_str(w: int, h: int) -> str:
    from math import gcd
    d = gcd(w, h)
    return f"{w // d}:{h // d}"


def _validate_manifest(data: dict) -> None:
    schema = _get_schema()
    try:
        jsonschema.validate(data, schema)
    except jsonschema.ValidationError as exc:
        raise ValueError(f"Manifest schema validation failed: {exc.message}") from exc


def _deep_merge(base: dict, updates: dict) -> None:
    for k, v in updates.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
