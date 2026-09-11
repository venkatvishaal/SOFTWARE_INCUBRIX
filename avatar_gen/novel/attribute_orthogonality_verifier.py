"""
NF-01 — Attribute Orthogonality Verifier (AOV)

Detects *attribute bleed*: unintended change in a held-constant attribute
when only one attribute was requested to change between two avatar jobs.

Gap closed: No existing pipeline (SD, InstantID, IP-Adapter-FaceID) measures
per-attribute isolation. Orthogonality Score (OS) = 1 - max(Δ_unchanged).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

_BLEED_THRESHOLD = 0.85   # OS below this → BLEED_WARNING


@dataclass
class OrthogonalityResult:
    changed_attr: str
    scores: dict[str, float]         # attribute → OS ∈ [0,1]
    bleed_warnings: list[str]        # attributes with OS < threshold
    overall_ok: bool

    def to_dict(self) -> dict:
        return {
            "changed_attr":   self.changed_attr,
            "orthogonality_scores": self.scores,
            "bleed_warnings": self.bleed_warnings,
            "overall_ok":     self.overall_ok,
        }


# ── Attribute descriptor extractors ──────────────────────────────────────────

def _dominant_colour(img: Image.Image, region: str = "full") -> np.ndarray:
    """Return mean RGB of image (or a region) as a 3-vector."""
    rgb = img.convert("RGB")
    if region == "top_half":
        rgb = rgb.crop((0, 0, rgb.width, rgb.height // 2))
    elif region == "bottom_half":
        rgb = rgb.crop((0, rgb.height // 2, rgb.width, rgb.height))
    elif region == "centre":
        w, h = rgb.width, rgb.height
        rgb = rgb.crop((w // 4, h // 4, 3 * w // 4, 3 * h // 4))
    arr = np.array(rgb, dtype=float)
    return arr.mean(axis=(0, 1)) / 255.0


def _texture_score(img: Image.Image) -> float:
    """Laplacian variance as a texture complexity proxy."""
    grey = np.array(img.convert("L"), dtype=float)
    laplacian = np.array([
        [0,  1, 0],
        [1, -4, 1],
        [0,  1, 0],
    ])
    from scipy.ndimage import convolve
    conv = convolve(grey, laplacian)
    return float(conv.var())


def _extract_descriptors(img: Image.Image) -> dict[str, np.ndarray | float]:
    """Extract lightweight per-attribute descriptors from a PIL image."""
    return {
        "skin_tone":  _dominant_colour(img, "centre"),
        "hair":       _dominant_colour(img, "top_half"),
        "background": _dominant_colour(img, "full"),
        "attire":     _dominant_colour(img, "bottom_half"),
        "texture":    np.array([_texture_score(img)]),
    }


def _descriptor_distance(a: np.ndarray | float, b: np.ndarray | float) -> float:
    """Normalised Euclidean distance in [0,1] between two descriptors."""
    va = np.atleast_1d(np.array(a, dtype=float))
    vb = np.atleast_1d(np.array(b, dtype=float))
    max_dist = np.sqrt(len(va))  # max possible L2 for unit-range vectors
    return float(np.linalg.norm(va - vb)) / max(max_dist, 1e-9)


# ── Public API ────────────────────────────────────────────────────────────────

def verify_orthogonality(
    baseline_image: Image.Image | Path,
    variant_image: Image.Image | Path,
    changed_attr: str,
    threshold: float = _BLEED_THRESHOLD,
) -> OrthogonalityResult:
    """Compare two images: one is the baseline, one has exactly one attribute changed.

    Args:
        baseline_image: PIL Image or path — the un-modified avatar.
        variant_image:  PIL Image or path — the avatar with changed_attr modified.
        changed_attr:   Name of the attribute that was intentionally changed.
        threshold:      Orthogonality Score below which BLEED_WARNING fires.

    Returns:
        OrthogonalityResult with per-attribute OS scores and warnings.
    """
    if isinstance(baseline_image, Path):
        baseline_image = Image.open(baseline_image).convert("RGB")
    if isinstance(variant_image, Path):
        variant_image = Image.open(variant_image).convert("RGB")

    desc_a = _extract_descriptors(baseline_image)
    desc_b = _extract_descriptors(variant_image)

    scores: dict[str, float] = {}
    bleed_warnings: list[str] = []

    for attr, val_a in desc_a.items():
        val_b = desc_b[attr]
        dist = _descriptor_distance(val_a, val_b)
        os = round(1.0 - dist, 4)
        scores[attr] = os
        if attr != changed_attr and os < threshold:
            bleed_warnings.append(attr)

    return OrthogonalityResult(
        changed_attr=changed_attr,
        scores=scores,
        bleed_warnings=bleed_warnings,
        overall_ok=len(bleed_warnings) == 0,
    )


def verify_orthogonality_jobs(
    baseline_job_dir: Path,
    variant_job_dir: Path,
    changed_attr: str,
    output_path: Path | None = None,
) -> OrthogonalityResult:
    """High-level wrapper: load images from job dirs and run verify_orthogonality."""
    def _find_image(job_dir: Path) -> Path:
        manifest_path = job_dir / "avatar_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return Path(manifest["output_files"]["image"])

    baseline_img = Image.open(_find_image(baseline_job_dir)).convert("RGB")
    variant_img  = Image.open(_find_image(variant_job_dir)).convert("RGB")
    result = verify_orthogonality(baseline_img, variant_img, changed_attr)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    return result
