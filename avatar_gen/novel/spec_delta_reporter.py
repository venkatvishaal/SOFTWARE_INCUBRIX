"""
NF-02 — Spec-Delta Reporter

Produces a machine-readable per-attribute verdict showing which requested
attributes were FULL / PARTIAL / MISSING in the generated image.

Gap closed: CLIP scores prompts holistically. No pipeline produces a named
per-attribute verdict table with specific mismatch descriptions.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

from avatar_gen.spec_parser import AvatarSpec

Match = Literal["FULL", "PARTIAL", "MISSING"]

_SKIN_TONE_RGB: dict[str, tuple[int, int, int]] = {
    "very_light_fair": (255, 224, 196),
    "light_fair":      (240, 200, 168),
    "light_medium":    (220, 175, 140),
    "medium":          (190, 150, 110),
    "warm_medium":     (180, 135, 100),
    "olive":           (160, 128, 95),
    "tan":             (145, 110, 80),
    "brown":           (120, 85,  60),
    "dark_brown":      (90,  60,  40),
    "deep_dark":       (55,  35,  20),
}

_HAIR_COLOUR_RGB: dict[str, tuple[int, int, int]] = {
    "black":       (20,  20,  20),
    "dark_brown":  (50,  30,  15),
    "medium_brown":(90,  55,  30),
    "auburn":      (110, 55,  25),
    "blonde":      (200, 175, 110),
    "red":         (160, 50,  30),
    "grey":        (150, 150, 150),
    "white":       (230, 230, 230),
    "dyed_blue":   (40,  80,  180),
    "dyed_red":    (200, 30,  30),
    "dyed_green":  (30,  160, 60),
}

_POSE_LABELS = {
    "front_facing":        "facing forward",
    "three_quarter_left":  "three quarter left",
    "three_quarter_right": "three quarter right",
    "profile_left":        "profile left",
    "profile_right":       "profile right",
    "slight_tilt_up":      "looking up",
    "slight_tilt_down":    "looking down",
}


@dataclass
class AttributeVerdict:
    attribute: str
    requested: str
    detected: str
    match: Match
    delta: str | None = None


@dataclass
class SpecDeltaReport:
    avatar_id: str
    attribute_verdicts: list[AttributeVerdict]
    overall_adherence_pct: float
    low_adherence: bool

    def to_dict(self) -> dict:
        return {
            "avatar_id": self.avatar_id,
            "attribute_verdicts": {
                v.attribute: {
                    "requested": v.requested,
                    "detected":  v.detected,
                    "match":     v.match,
                    "delta":     v.delta,
                }
                for v in self.attribute_verdicts
            },
            "overall_adherence_pct": self.overall_adherence_pct,
            "low_adherence": self.low_adherence,
        }


# ── Attribute-specific detectors ──────────────────────────────────────────────

def _rgb_distance(c1: tuple, c2: tuple) -> float:
    return float(np.linalg.norm(np.array(c1, float) - np.array(c2, float)))


def _nearest_colour(region_mean: np.ndarray, palette: dict[str, tuple]) -> str:
    best, best_dist = "", float("inf")
    for name, rgb in palette.items():
        d = _rgb_distance(tuple(region_mean.astype(int)), rgb)
        if d < best_dist:
            best_dist, best = d, name
    return best


def _check_skin_tone(img: Image.Image, requested: str) -> AttributeVerdict:
    rgb = np.array(img.convert("RGB"))
    h, w = rgb.shape[:2]
    face_region = rgb[h // 6:h // 2, w // 4:3 * w // 4]
    mean_colour = face_region.mean(axis=(0, 1))
    detected = _nearest_colour(mean_colour, _SKIN_TONE_RGB)

    if detected == requested:
        match = "FULL"
        delta = None
    else:
        # Check if adjacent tone
        keys = list(_SKIN_TONE_RGB.keys())
        req_idx = keys.index(requested) if requested in keys else -1
        det_idx = keys.index(detected)  if detected  in keys else -1
        if abs(req_idx - det_idx) <= 1:
            match = "PARTIAL"
            delta = f"detected={detected} (adjacent tone)"
        else:
            match = "MISSING"
            delta = f"detected={detected}"

    return AttributeVerdict("skin_tone", requested, detected, match, delta)  # type: ignore[arg-type]


def _check_hair(img: Image.Image, hair_spec) -> AttributeVerdict:
    rgb = np.array(img.convert("RGB"))
    h, w = rgb.shape[:2]
    top_region = rgb[: h // 5, :]
    mean_colour = top_region.mean(axis=(0, 1))
    detected_colour = _nearest_colour(mean_colour, _HAIR_COLOUR_RGB)
    requested_colour = hair_spec.colour if hasattr(hair_spec, "colour") else str(hair_spec)

    if detected_colour == requested_colour:
        return AttributeVerdict("hair", requested_colour, detected_colour, "FULL")  # type: ignore
    keys = list(_HAIR_COLOUR_RGB.keys())
    ri = keys.index(requested_colour) if requested_colour in keys else -1
    di = keys.index(detected_colour)  if detected_colour  in keys else -1
    if abs(ri - di) <= 1:
        return AttributeVerdict("hair", requested_colour, detected_colour, "PARTIAL",  # type: ignore
                                f"colour close ({detected_colour})")
    return AttributeVerdict("hair", requested_colour, detected_colour, "MISSING",  # type: ignore
                            f"detected_colour={detected_colour}")


def _check_background(img: Image.Image, requested: str) -> AttributeVerdict:
    rgb = np.array(img.convert("RGB"))
    # Sample corners as background proxy
    h, w = rgb.shape[:2]
    corners = np.concatenate([
        rgb[:30, :30].reshape(-1, 3),
        rgb[:30, w-30:].reshape(-1, 3),
        rgb[h-30:, :30].reshape(-1, 3),
        rgb[h-30:, w-30:].reshape(-1, 3),
    ])
    mean = corners.mean(axis=0)
    # Simple heuristic: high brightness → neutral/white; low → dark/outdoor
    brightness = mean.mean()
    saturation = mean.std()

    neutral_bg = {"neutral_grey", "neutral_white", "gradient_blue"}
    if requested in neutral_bg and brightness > 160 and saturation < 40:
        detected, match = "neutral_background", "FULL"
    elif requested in {"office", "conference_room", "classroom"} and brightness > 100:
        detected, match = "indoor_generic", "PARTIAL"
    elif requested in {"outdoor_park", "outdoor_urban"} and saturation > 30:
        detected, match = "outdoor_generic", "PARTIAL"
    else:
        detected, match = "unknown", "MISSING"

    delta = None if match == "FULL" else f"detected={detected}"
    return AttributeVerdict("background", requested, detected, match, delta)  # type: ignore


def _check_pose(img: Image.Image, requested: str) -> AttributeVerdict:
    """Lightweight pose check using face centroid horizontal offset."""
    rgb = np.array(img.convert("L"))
    # Naive: centre of mass of bright region approximates face centroid
    h, w = rgb.shape
    face_region = rgb[h // 6: h // 2, :]
    col_means = face_region.mean(axis=0)
    centroid_x = float(np.average(np.arange(w), weights=col_means + 1))
    offset = (centroid_x - w / 2) / (w / 2)  # -1=left, 0=centre, +1=right

    if requested == "front_facing" and abs(offset) < 0.2:
        return AttributeVerdict("pose", requested, "front_facing", "FULL")  # type: ignore
    if "left" in requested and offset < -0.1:
        return AttributeVerdict("pose", requested, requested, "FULL")  # type: ignore
    if "right" in requested and offset > 0.1:
        return AttributeVerdict("pose", requested, requested, "FULL")  # type: ignore
    return AttributeVerdict("pose", requested, "indeterminate", "PARTIAL",  # type: ignore
                            f"centroid_offset={offset:.2f}")


# ── Public API ────────────────────────────────────────────────────────────────

def compute_spec_delta(
    spec: AvatarSpec,
    image: Image.Image | Path,
    adherence_threshold: float = 70.0,
    output_path: Path | None = None,
) -> SpecDeltaReport:
    """Compute per-attribute verdict between spec and generated image.

    Args:
        spec:                AvatarSpec (requested attributes).
        image:               PIL Image or path to the generated avatar.
        adherence_threshold: overall_adherence_pct below this → low_adherence=True.
        output_path:         Optional path to write spec_delta.json.

    Returns:
        SpecDeltaReport with per-attribute verdicts and overall adherence %.
    """
    if isinstance(image, Path):
        image = Image.open(image).convert("RGB")

    verdicts: list[AttributeVerdict] = [
        _check_skin_tone(image, spec.skin_tone),
        _check_hair(image, spec.hair),
        _check_background(image, spec.background),
        _check_pose(image, spec.pose),
    ]

    score_map = {"FULL": 100.0, "PARTIAL": 50.0, "MISSING": 0.0}
    adherence = float(np.mean([score_map[v.match] for v in verdicts]))

    report = SpecDeltaReport(
        avatar_id=spec.avatar_id,
        attribute_verdicts=verdicts,
        overall_adherence_pct=round(adherence, 2),
        low_adherence=adherence < adherence_threshold,
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    return report
