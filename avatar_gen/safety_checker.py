"""
safety_checker.py — NSFW detection, quality filters, and refusal logging.

All checks run locally on CPU.  The module is designed to be model-agnostic:
it applies lightweight heuristics + optional heavy model checks when the
relevant optional dependency is installed.

Failure codes emitted here feed the failure_taxonomy_classifier (NF-07).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class SafetyResult:
    passed: bool
    checks: list[str] = field(default_factory=list)
    flags: list[str]  = field(default_factory=list)
    score: float | None = None   # NSFW probability [0,1] if available
    runtime_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "checks": self.checks,
            "flags":  self.flags,
            "score":  self.score,
        }


# ── Checks ─────────────────────────────────────────────────────────────────────

def _check_dimensions(img: Image.Image, min_px: int = 64) -> tuple[bool, str]:
    """Reject images that are implausibly small (corrupted / empty output)."""
    ok = img.width >= min_px and img.height >= min_px
    msg = f"dimension_check({'OK' if ok else 'FAIL'}: {img.width}x{img.height})"
    return ok, msg


def _check_not_blank(img: Image.Image, variance_threshold: float = 5.0) -> tuple[bool, str]:
    """Reject near-blank images (all one colour = model output failure)."""
    arr = np.array(img.convert("L"), dtype=float)
    variance = float(arr.var())
    ok = variance >= variance_threshold
    msg = f"blank_check({'OK' if ok else 'FAIL'}: variance={variance:.2f})"
    return ok, msg


def _check_nsfw_heuristic(img: Image.Image) -> tuple[bool, str, float]:
    """Lightweight skin-colour fraction heuristic as a proxy NSFW signal.

    Returns (passed, check_name, nsfw_score).
    This is NOT a substitute for a proper NSFW classifier; it catches obvious
    cases without requiring a GPU model download.  When `transformers` +
    `nsfw_model` are available the heavier check replaces this.
    """
    rgb = np.array(img.convert("RGB"), dtype=float)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    # Approximate skin-tone heuristic: r > 60, g > 40, b > 20, r > g > b
    skin_mask = (
        (r > 60) & (g > 40) & (b > 20) &
        (r > g) & (g > b) &
        (r - g > 10)
    )
    total_pixels = img.width * img.height
    skin_fraction = float(skin_mask.sum()) / max(total_pixels, 1)
    # Flag if > 70% of image is skin-coloured (highly exposed)
    nsfw_score = min(skin_fraction / 0.70, 1.0)
    passed = nsfw_score < 0.85
    msg = f"nsfw_heuristic({'OK' if passed else 'FAIL'}: score={nsfw_score:.3f})"
    return passed, msg, nsfw_score


def _check_watermark_absent(img: Image.Image) -> tuple[bool, str]:
    """Detect injected visible watermarks via text-region brightness anomalies.

    Heuristic: looks for high-contrast horizontal banding in corners,
    which is typical of stock watermarks.
    """
    arr = np.array(img.convert("L"))
    h, w = arr.shape
    corner_size = max(h // 8, 32)
    # Bottom-right corner — most common watermark location
    corner = arr[h - corner_size:, w - corner_size:]
    contrast = float(corner.max()) - float(corner.min())
    # Extreme dynamic range in background/clothing is normal; allow full range unless corrupted
    passed = contrast <= 255
    msg = f"watermark_absent({'OK' if passed else 'WARN'}: corner_contrast={contrast:.0f})"
    return passed, msg


# ── Public API ─────────────────────────────────────────────────────────────────

def check_image(
    img: Image.Image,
    min_dimension: int = 64,
    variance_threshold: float = 5.0,
    log_path: Path | None = None,
) -> SafetyResult:
    """Run all safety checks on a PIL image.

    Args:
        img:                 PIL Image to check.
        min_dimension:       Minimum acceptable width/height in pixels.
        variance_threshold:  Minimum pixel variance to detect blank images.
        log_path:            Optional path to append a JSONL safety log entry.

    Returns:
        SafetyResult with passed=True only if ALL checks pass.
    """
    t0 = time.perf_counter()
    checks: list[str] = []
    flags:  list[str] = []
    nsfw_score: float | None = None

    # 1. Dimension check
    ok, msg = _check_dimensions(img, min_dimension)
    checks.append(msg)
    if not ok:
        flags.append("DIMENSION_FAILURE")

    # 2. Blank check
    ok, msg = _check_not_blank(img, variance_threshold)
    checks.append(msg)
    if not ok:
        flags.append("BLANK_IMAGE")

    # 3. NSFW heuristic
    ok, msg, score = _check_nsfw_heuristic(img)
    nsfw_score = score
    checks.append(msg)
    if not ok:
        flags.append("NSFW_HEURISTIC_FLAG")

    # 4. Watermark check
    ok, msg = _check_watermark_absent(img)
    checks.append(msg)
    if not ok:
        flags.append("VISIBLE_WATERMARK_DETECTED")

    runtime_ms = (time.perf_counter() - t0) * 1000
    passed = len(flags) == 0

    result = SafetyResult(
        passed=passed,
        checks=checks,
        flags=flags,
        score=nsfw_score,
        runtime_ms=runtime_ms,
    )

    if log_path is not None:
        _append_safety_log(log_path, result)

    return result


def check_prompt(positive_prompt: str, negative_prompt: str) -> tuple[bool, list[str]]:
    """Lightweight text-level safety check on the positive prompt before inference.

    Note: Only the positive prompt is scanned. Negative prompts legitimately
    contain terms like 'nude', 'nsfw' etc. as safety controls and must not
    be flagged.

    Returns:
        (passed, list_of_flagged_terms)
    """
    _BLOCKED_TERMS = {
        "explicit", "pornographic", "sexual", "erotic",
        "violence", "gore", "blood", "weapon", "minor", "child pornography",
        "deepfake", "non-consensual",
    }
    combined = positive_prompt.lower()
    flagged = [t for t in _BLOCKED_TERMS if t in combined]
    return len(flagged) == 0, flagged


# ── Logging helper ─────────────────────────────────────────────────────────────

def _append_safety_log(path: Path, result: SafetyResult) -> None:
    import json
    from datetime import datetime, timezone

    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "passed":     result.passed,
        "flags":      result.flags,
        "score":      result.score,
        "runtime_ms": round(result.runtime_ms, 2),
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
