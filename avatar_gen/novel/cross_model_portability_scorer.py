"""
NF-09 — Cross-Model Prompt Portability Score (CMPP)

Measures how well a structured avatar spec translates between two or more
open models, reporting per-attribute portability with named recommendations.

Gap closed: HPSv2/GenAI-Bench compare models globally. No tool decomposes
portability per-attribute and produces actionable tuning recommendations.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ModelAdherenceProfile:
    model: str
    avatar_id: str
    attribute_adherence: dict[str, float]   # attribute → adherence ∈ [0,1]
    overall_adherence: float


@dataclass
class PortabilityReport:
    spec_hash: str
    model_a: str
    model_b: str
    cmpp_overall: float
    cmpp_per_attribute: dict[str, float]
    least_portable_attribute: str
    most_portable_attribute: str
    recommendation: str

    def to_dict(self) -> dict:
        return {
            "spec_hash":                self.spec_hash,
            "model_a":                  self.model_a,
            "model_b":                  self.model_b,
            "cmpp_overall":             round(self.cmpp_overall, 4),
            "cmpp_per_attribute":       {
                k: round(v, 4) for k, v in self.cmpp_per_attribute.items()
            },
            "least_portable_attribute": self.least_portable_attribute,
            "most_portable_attribute":  self.most_portable_attribute,
            "recommendation":           self.recommendation,
        }

    def summary(self) -> str:
        lines = [
            f"Cross-Model Portability: {self.model_a} -> {self.model_b}",
            f"  Overall CMPP: {self.cmpp_overall:.2%}",
            "  Per-attribute:",
        ]
        for attr, score in sorted(self.cmpp_per_attribute.items(), key=lambda x: x[1]):
            bar = "#" * int(score * 20) + "-" * (20 - int(score * 20))
            lines.append(f"    {attr:20s} [{bar}] {score:.2%}")
        lines.append(f"  [!] Least portable: {self.least_portable_attribute}")
        lines.append(f"  [*] {self.recommendation}")
        return "\n".join(lines)


# ── Profile builder (from spec-delta JSON files) ───────────────────────────────

def _match_to_score(match: str) -> float:
    return {"FULL": 1.0, "PARTIAL": 0.5, "MISSING": 0.0}.get(match, 0.0)


def profile_from_spec_delta(spec_delta_path: Path, model: str) -> ModelAdherenceProfile:
    """Build a ModelAdherenceProfile from a spec_delta.json file."""
    data = json.loads(spec_delta_path.read_text(encoding="utf-8"))
    verdicts = data.get("attribute_verdicts", {})
    attr_adherence = {
        attr: _match_to_score(v["match"])
        for attr, v in verdicts.items()
    }
    overall = float(np.mean(list(attr_adherence.values()))) if attr_adherence else 0.0
    return ModelAdherenceProfile(
        model=model,
        avatar_id=data.get("avatar_id", "unknown"),
        attribute_adherence=attr_adherence,
        overall_adherence=round(overall, 4),
    )


def profile_from_dict(
    attribute_adherence: dict[str, float],
    model: str,
    avatar_id: str = "manual",
) -> ModelAdherenceProfile:
    """Build a profile directly from a dict of attribute→adherence scores."""
    overall = float(np.mean(list(attribute_adherence.values()))) if attribute_adherence else 0.0
    return ModelAdherenceProfile(
        model=model,
        avatar_id=avatar_id,
        attribute_adherence=attribute_adherence,
        overall_adherence=round(overall, 4),
    )


# ── Public API ────────────────────────────────────────────────────────────────

def compute_portability(
    profile_a: ModelAdherenceProfile,
    profile_b: ModelAdherenceProfile,
    spec_hash: str = "unknown",
    output_path: Path | None = None,
) -> PortabilityReport:
    """Compute the Cross-Model Prompt Portability Score between two profiles.

    CMPP_attribute = 1 - |adherence_A - adherence_B|
    CMPP_overall   = mean(CMPP_attribute for all shared attributes)

    Args:
        profile_a:    Adherence profile for model A.
        profile_b:    Adherence profile for model B.
        spec_hash:    SHA-256 of the spec used (for traceability).
        output_path:  Optional path to write portability_report.json.

    Returns:
        PortabilityReport with per-attribute CMPP scores and recommendation.
    """
    shared_attrs = set(profile_a.attribute_adherence) & set(profile_b.attribute_adherence)
    if not shared_attrs:
        raise ValueError("No shared attributes between the two profiles — cannot compute CMPP.")

    cmpp_per_attr: dict[str, float] = {}
    for attr in shared_attrs:
        a_val = profile_a.attribute_adherence[attr]
        b_val = profile_b.attribute_adherence[attr]
        cmpp_per_attr[attr] = round(1.0 - abs(a_val - b_val), 4)

    cmpp_overall = round(float(np.mean(list(cmpp_per_attr.values()))), 4)

    least_attr = min(cmpp_per_attr, key=cmpp_per_attr.get)  # type: ignore
    most_attr  = max(cmpp_per_attr, key=cmpp_per_attr.get)  # type: ignore

    least_score = cmpp_per_attr[least_attr]
    recommendation = (
        f"'{least_attr}' attribute prompts (portability={least_score:.2%}) "
        f"require model-specific tuning when switching from "
        f"{profile_a.model} to {profile_b.model}. "
        f"Consider model-specific token synonyms for this attribute."
    )

    report = PortabilityReport(
        spec_hash=spec_hash,
        model_a=profile_a.model,
        model_b=profile_b.model,
        cmpp_overall=cmpp_overall,
        cmpp_per_attribute=cmpp_per_attr,
        least_portable_attribute=least_attr,
        most_portable_attribute=most_attr,
        recommendation=recommendation,
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    return report


def compute_portability_from_dirs(
    job_dir_a: Path,
    model_a: str,
    job_dir_b: Path,
    model_b: str,
    spec_hash: str = "unknown",
    output_path: Path | None = None,
) -> PortabilityReport:
    """High-level: load spec-delta reports from two job dirs and compute CMPP."""
    delta_a = job_dir_a / "spec_delta.json"
    delta_b = job_dir_b / "spec_delta.json"

    if not delta_a.exists():
        raise FileNotFoundError(f"spec_delta.json not found in {job_dir_a}")
    if not delta_b.exists():
        raise FileNotFoundError(f"spec_delta.json not found in {job_dir_b}")

    profile_a = profile_from_spec_delta(delta_a, model_a)
    profile_b = profile_from_spec_delta(delta_b, model_b)
    return compute_portability(profile_a, profile_b, spec_hash, output_path)
