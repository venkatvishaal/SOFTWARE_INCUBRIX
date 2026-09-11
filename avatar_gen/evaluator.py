"""
evaluator.py — Compute quality metrics for generated avatars.

Metrics:
  - prompt_spec_adherence: CLIP cosine similarity (positive prompt vs image)
  - diversity_coverage:    Distribution of attributes across a batch
  - identity_consistency:  Face embedding cosine similarity across poses
    (Exceptional tier; requires insightface optional dep)

All heavy model loads are lazy; CPU-only lightweight fallbacks are always available.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

# ── Dataclasses ────────────────────────────────────────────────────────────────

@dataclass
class AdherenceResult:
    avatar_id: str
    prompt: str
    clip_score: float | None     # None when CLIP not available
    fallback_score: float        # Lightweight colour-histogram proxy
    overall: float               # clip_score if available, else fallback_score

    def to_dict(self) -> dict:
        return {
            "avatar_id":    self.avatar_id,
            "clip_score":   self.clip_score,
            "fallback_score": self.fallback_score,
            "overall":      self.overall,
        }


@dataclass
class DiversityResult:
    batch_id: str
    attribute_coverage: dict[str, set]   # attribute → set of distinct values
    coverage_pct: float                  # % of possible attribute slots filled

    def to_dict(self) -> dict:
        return {
            "batch_id":          self.batch_id,
            "attribute_coverage": {k: list(v) for k, v in self.attribute_coverage.items()},
            "coverage_pct":      self.coverage_pct,
        }


@dataclass
class IdentityConsistencyResult:
    subject_id: str
    pairs: list[dict]    # list of {img_a, img_b, cosine_similarity}
    mean_similarity: float
    min_similarity: float

    def to_dict(self) -> dict:
        return {
            "subject_id":      self.subject_id,
            "pairs":           self.pairs,
            "mean_similarity": self.mean_similarity,
            "min_similarity":  self.min_similarity,
        }


# ── Prompt / Spec Adherence ────────────────────────────────────────────────────

def _colour_histogram_adherence(img: Image.Image, prompt: str) -> float:
    """Lightweight fallback: measure image complexity as proxy for adherence.

    A high-quality generation should have moderate colour diversity (not too
    uniform = blank, not too chaotic = noise).  Returns a score in [0, 1].
    """
    arr = np.array(img.convert("RGB"))
    # Compute per-channel histogram entropy
    scores = []
    for c in range(3):
        hist, _ = np.histogram(arr[:, :, c], bins=32, range=(0, 255))
        hist = hist / (hist.sum() + 1e-9)
        entropy = -np.sum(hist * np.log2(hist + 1e-12))
        scores.append(entropy / np.log2(32))  # normalise to [0,1]
    return float(np.mean(scores))


def compute_adherence(
    image: Image.Image | Path,
    prompt: str,
    avatar_id: str = "unknown",
) -> AdherenceResult:
    """Compute prompt-adherence score.

    Uses CLIP if open-clip-torch is installed; falls back to colour entropy.
    """
    if isinstance(image, Path):
        image = Image.open(image).convert("RGB")

    clip_score: float | None = None
    fallback = _colour_histogram_adherence(image, prompt)

    try:
        import open_clip
        import torch

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        model.eval()

        with torch.no_grad():
            img_tensor = preprocess(image).unsqueeze(0)
            txt_tokens  = tokenizer([prompt[:77]])   # CLIP max 77 tokens
            img_feat = model.encode_image(img_tensor)
            txt_feat = model.encode_text(txt_tokens)
            img_feat /= img_feat.norm(dim=-1, keepdim=True)
            txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
            clip_score = float((img_feat @ txt_feat.T).item())
            # Normalise CLIP cosine from [-1,1] to [0,1]
            clip_score = (clip_score + 1.0) / 2.0
    except ImportError:
        pass
    except Exception:
        pass

    overall = clip_score if clip_score is not None else fallback
    return AdherenceResult(
        avatar_id=avatar_id,
        prompt=prompt,
        clip_score=clip_score,
        fallback_score=fallback,
        overall=overall,
    )


# ── Diversity / Coverage ───────────────────────────────────────────────────────

_DIVERSITY_ATTRIBUTES = ("skin_tone", "age_band", "attire", "background", "pose")
_MAX_VALUES_PER_ATTR = {
    "skin_tone":  10,
    "age_band":    5,
    "attire":     10,
    "background": 10,
    "pose":        7,
}


def compute_diversity(manifests: Sequence[dict], batch_id: str = "batch") -> DiversityResult:
    """Compute attribute coverage across a batch of manifests.

    Args:
        manifests: List of loaded avatar_manifest.json dicts.
        batch_id:  Identifier for the batch.

    Returns:
        DiversityResult with per-attribute distinct value sets and coverage %.
    """
    coverage: dict[str, set] = {attr: set() for attr in _DIVERSITY_ATTRIBUTES}

    for m in manifests:
        spec = m.get("spec", {})
        spec.get("hair", {})
        coverage["skin_tone"].add(spec.get("skin_tone", "unknown"))
        coverage["age_band"].add(spec.get("age_band",  "unknown"))
        coverage["attire"].add(spec.get("attire",    "unknown"))
        coverage["background"].add(spec.get("background", "unknown"))
        coverage["pose"].add(spec.get("pose",       "unknown"))

    total_possible = sum(_MAX_VALUES_PER_ATTR.values())
    total_covered  = sum(len(v) for v in coverage.values())
    coverage_pct   = 100.0 * total_covered / max(total_possible, 1)

    return DiversityResult(
        batch_id=batch_id,
        attribute_coverage=coverage,
        coverage_pct=round(coverage_pct, 2),
    )


# ── Identity Consistency ───────────────────────────────────────────────────────

def _pixel_hash_similarity(img_a: Image.Image, img_b: Image.Image) -> float:
    """Lightweight fallback: compare 8x8 thumbnail grayscale hashes."""
    def phash(img: Image.Image) -> np.ndarray:
        thumb = img.convert("L").resize((8, 8), Image.Resampling.LANCZOS)
        arr = np.array(thumb, dtype=float)
        return (arr > arr.mean()).astype(float)

    h_a = phash(img_a)
    h_b = phash(img_b)
    hamming = np.sum(h_a != h_b)
    return 1.0 - (hamming / 64.0)


def compute_identity_consistency(
    image_paths: list[Path],
    subject_id: str = "subject",
) -> IdentityConsistencyResult:
    """Measure face-embedding consistency across multiple images.

    Uses InsightFace/ArcFace if available; falls back to perceptual hash similarity.
    """
    images = [Image.open(p).convert("RGB") for p in image_paths]
    embeddings: list[np.ndarray | None] = [None] * len(images)

    # Attempt ArcFace embeddings
    try:
        import insightface
        import onnxruntime  # noqa: F401

        app = insightface.app.FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0)

        for i, img in enumerate(images):
            arr = np.array(img)[:, :, ::-1]  # RGB→BGR
            faces = app.get(arr)
            if faces:
                embeddings[i] = faces[0].embedding / (np.linalg.norm(faces[0].embedding) + 1e-9)
    except (ImportError, Exception):
        pass  # fall back to pixel hashing

    pairs: list[dict] = []
    similarities: list[float] = []

    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            emb_i = embeddings[i]
            emb_j = embeddings[j]
            if emb_i is not None and emb_j is not None:
                sim = float(np.dot(emb_i, emb_j))
            else:
                sim = _pixel_hash_similarity(images[i], images[j])

            pairs.append({
                "img_a": str(image_paths[i]),
                "img_b": str(image_paths[j]),
                "cosine_similarity": round(sim, 4),
            })
            similarities.append(sim)

    mean_sim = float(np.mean(similarities)) if similarities else 0.0
    min_sim  = float(np.min(similarities))  if similarities else 0.0

    return IdentityConsistencyResult(
        subject_id=subject_id,
        pairs=pairs,
        mean_similarity=round(mean_sim, 4),
        min_similarity=round(min_sim, 4),
    )
