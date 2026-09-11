"""
NF-03 — Bias Surface Auditor (BSA)

Audits a candidate-assembled batch of avatars for demographic representation
gaps using Shannon entropy per attribute dimension.

Gap closed: OpenBias and DQA audit models in aggregate; no tool audits a
specific candidate-assembled batch with actionable per-attribute recommendations.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

_ENTROPY_THRESHOLD = 1.5   # bits — below this → GAP_WARNING

_MAX_DISTINCT: dict[str, int] = {
    "skin_tone":  10,
    "age_band":    5,
    "hair_colour": 11,
    "attire":     10,
    "background": 10,
    "pose":        7,
}


@dataclass
class AttributeEntropyRecord:
    attribute: str
    entropy_bits: float
    distinct_values: list[str]
    status: str   # OK | GAP_WARNING
    recommendation: str | None = None


@dataclass
class BiasAuditReport:
    batch_id: str
    n_avatars: int
    attribute_entropy: list[AttributeEntropyRecord]
    clone_pairs: list[tuple[int, int]]    # (index_a, index_b) sharing >2 attrs
    overall_status: str                   # OK | GAP_WARNING
    recommendations: list[str]

    def to_dict(self) -> dict:
        return {
            "batch_id":  self.batch_id,
            "n_avatars": self.n_avatars,
            "attribute_entropy": {
                r.attribute: {
                    "entropy_bits":    round(r.entropy_bits, 3),
                    "distinct_values": r.distinct_values,
                    "status":          r.status,
                    "recommendation":  r.recommendation,
                }
                for r in self.attribute_entropy
            },
            "clone_pairs":   list(self.clone_pairs),
            "overall_status": self.overall_status,
            "recommendations": self.recommendations,
        }


def _shannon_entropy(values: list[str]) -> float:
    """Compute Shannon entropy in bits for a list of categorical values."""
    if not values:
        return 0.0
    n = len(values)
    from collections import Counter
    counts = Counter(values)
    entropy = 0.0
    for cnt in counts.values():
        p = cnt / n
        entropy -= p * math.log2(p)
    return entropy


def _extract_attr(spec: dict, attribute: str) -> str:
    if attribute == "hair_colour":
        hair = spec.get("hair", {})
        if isinstance(hair, dict):
            return hair.get("colour", "unknown")
        return "unknown"
    return str(spec.get(attribute, "unknown"))


def audit_batch(
    manifests: Sequence[dict],
    batch_id: str = "batch",
    entropy_threshold: float = _ENTROPY_THRESHOLD,
    clone_max_shared: int = 2,
    output_path: Path | None = None,
) -> BiasAuditReport:
    """Audit a batch of avatar manifests for representation gaps.

    Args:
        manifests:          List of avatar_manifest.json dicts.
        batch_id:           Identifier for this batch.
        entropy_threshold:  Minimum Shannon entropy bits before GAP_WARNING.
        clone_max_shared:   Two avatars sharing more than this many attrs trigger a clone warning.
        output_path:        Optional path to write bias_audit.json.

    Returns:
        BiasAuditReport with per-attribute entropy scores and recommendations.
    """
    specs = [m.get("spec", {}) for m in manifests]
    n = len(specs)

    entropy_records: list[AttributeEntropyRecord] = []
    recommendations: list[str] = []

    for attr in _MAX_DISTINCT:
        values = [_extract_attr(s, attr) for s in specs]
        entropy = _shannon_entropy(values)
        distinct = sorted(set(values))
        status = "OK" if entropy >= entropy_threshold else "GAP_WARNING"

        rec = None
        if status == "GAP_WARNING":
            needed = max(0, 3 - len(distinct))
            rec = (
                f"Add at least {needed} more distinct '{attr}' value(s) to improve coverage. "
                f"Currently: {distinct}"
            )
            recommendations.append(rec)

        entropy_records.append(AttributeEntropyRecord(
            attribute=attr,
            entropy_bits=round(entropy, 3),
            distinct_values=distinct,
            status=status,
            recommendation=rec,
        ))

    # Clone pair detection
    clone_pairs: list[tuple[int, int]] = []
    _attr_list = list(_MAX_DISTINCT.keys())
    for i in range(n):
        for j in range(i + 1, n):
            shared = sum(
                1 for a in _attr_list
                if _extract_attr(specs[i], a) == _extract_attr(specs[j], a)
            )
            if shared > clone_max_shared:
                clone_pairs.append((i, j))

    overall = "OK" if not recommendations else "GAP_WARNING"

    report = BiasAuditReport(
        batch_id=batch_id,
        n_avatars=n,
        attribute_entropy=entropy_records,
        clone_pairs=clone_pairs,
        overall_status=overall,
        recommendations=recommendations,
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    return report


def audit_batch_from_dir(
    batch_dir: Path,
    output_path: Path | None = None,
) -> BiasAuditReport:
    """Load all avatar_manifest.json files from a batch directory and audit."""
    batch_dir = Path(batch_dir)
    manifest_paths = sorted(batch_dir.rglob("avatar_manifest.json"))
    if not manifest_paths:
        raise FileNotFoundError(f"No avatar_manifest.json files found in {batch_dir}")
    manifests = [json.loads(p.read_text(encoding="utf-8")) for p in manifest_paths]
    return audit_batch(manifests, batch_id=batch_dir.name, output_path=output_path)
