"""
output_validator.py — Post-generation output validation.

Verifies that:
  - The image file exists and is a valid PNG/JPEG.
  - Dimensions match the requested resolution.
  - The avatar_manifest.json is present and schema-valid.
  - The safety log records a passed check.
  - The watermark is present if watermarking is enabled.
  - The spec_delta.json is present (if expected).

Returns a typed ValidationReport that the CLI renders and the benchmark records.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, UnidentifiedImageError


@dataclass
class ValidationIssue:
    code: str
    severity: str   # ERROR | WARN
    detail: str


@dataclass
class ValidationReport:
    job_dir: str
    passed: bool
    issues: list[ValidationIssue] = field(default_factory=list)

    def add(self, code: str, severity: str, detail: str) -> None:
        self.issues.append(ValidationIssue(code, severity, detail))
        if severity == "ERROR":
            self.passed = False

    def to_dict(self) -> dict:
        return {
            "job_dir": self.job_dir,
            "passed":  self.passed,
            "issues": [
                {"code": i.code, "severity": i.severity, "detail": i.detail}
                for i in self.issues
            ],
        }

    def summary(self) -> str:
        status = "[OK] PASSED" if self.passed else "[FAIL] FAILED"
        lines = [f"{status} - {self.job_dir}"]
        for issue in self.issues:
            icon = "[WARN]" if issue.severity == "WARN" else "[ERROR]"
            lines.append(f"  {icon} [{issue.code}] {issue.detail}")
        return "\n".join(lines)


def validate_job(
    job_dir: str | Path,
    expected_width: int | None = None,
    expected_height: int | None = None,
    expect_watermark: bool = True,
    expect_spec_delta: bool = False,
) -> ValidationReport:
    """Validate all outputs in a job directory.

    Args:
        job_dir:          Path to the job output directory.
        expected_width:   Expected image width in pixels (optional).
        expected_height:  Expected image height in pixels (optional).
        expect_watermark: Whether to check for watermark metadata.
        expect_spec_delta: Whether spec_delta.json must be present.

    Returns:
        ValidationReport — passed=True only when no ERROR-level issues found.
    """
    job_dir = Path(job_dir)
    report = ValidationReport(job_dir=str(job_dir), passed=True)

    # 1. Manifest existence
    manifest_path = job_dir / "avatar_manifest.json"
    if not manifest_path.exists():
        report.add("MISSING_MANIFEST", "ERROR", "avatar_manifest.json not found")
        return report  # can't continue without manifest

    # 2. Load and schema-validate manifest
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        report.add("MANIFEST_PARSE_ERROR", "ERROR", str(exc))
        return report

    _validate_manifest_fields(manifest, report)

    # 3. Image file existence + validity
    image_path_str = manifest.get("output_files", {}).get("image")
    if not image_path_str:
        report.add("MISSING_IMAGE_PATH", "ERROR", "output_files.image not set in manifest")
    else:
        image_path = Path(image_path_str)
        _validate_image_file(image_path, expected_width, expected_height, report)

    # 4. Safety check
    safety = manifest.get("safety", {})
    if not safety.get("passed", False):
        flags = safety.get("flags", [])
        report.add("SAFETY_FAILED", "ERROR", f"Safety check failed: {flags}")

    # 5. Watermark
    if expect_watermark:
        wm = manifest.get("watermark")
        if not wm:
            report.add("MISSING_WATERMARK", "WARN", "Watermark metadata absent in manifest")
        elif not wm.get("survives_jpeg_q70"):
            report.add("WATERMARK_WEAK", "WARN", "Watermark may not survive JPEG re-encoding")

    # 6. Spec-delta
    if expect_spec_delta:
        delta_path_str = manifest.get("output_files", {}).get("spec_delta")
        if not delta_path_str or not Path(delta_path_str).exists():
            report.add("MISSING_SPEC_DELTA", "WARN", "spec_delta.json not found")

    # 7. Provenance completeness
    prov = manifest.get("provenance", {})
    for req in ("pipeline_version", "generated_at", "compute_route", "runtime_sec"):
        if prov.get(req) is None and req not in prov:
            report.add("INCOMPLETE_PROVENANCE", "WARN", f"provenance.{req} missing")

    return report


# ── Internal helpers ──────────────────────────────────────────────────────────

def _validate_manifest_fields(manifest: dict, report: ValidationReport) -> None:
    required = ["avatar_id", "spec", "seed", "model", "model_revision",
                "prompt_generated", "negative_prompt", "provenance", "safety", "dimensions"]
    for field_name in required:
        if field_name not in manifest or manifest[field_name] is None:
            report.add(
                "MANIFEST_FIELD_MISSING", "ERROR",
                f"Required field '{field_name}' missing or null in manifest"
            )


def _validate_image_file(
    image_path: Path,
    expected_width: int | None,
    expected_height: int | None,
    report: ValidationReport,
) -> None:
    if not image_path.exists():
        report.add("IMAGE_NOT_FOUND", "ERROR", f"Image file not found: {image_path}")
        return
    if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        report.add("INVALID_FORMAT", "ERROR", f"Unsupported image format: {image_path.suffix}")
        return
    try:
        img = Image.open(image_path)
        img.verify()
        img = Image.open(image_path)  # re-open after verify
        w, h = img.size
    except UnidentifiedImageError as exc:
        report.add("CORRUPT_IMAGE", "ERROR", f"Cannot open image: {exc}")
        return
    except Exception as exc:
        report.add("IMAGE_ERROR", "ERROR", str(exc))
        return

    if expected_width and w != expected_width:
        report.add("DIMENSION_MISMATCH", "WARN",
                   f"Width {w} != expected {expected_width}")
    if expected_height and h != expected_height:
        report.add("DIMENSION_MISMATCH", "WARN",
                   f"Height {h} != expected {expected_height}")
    if img.mode not in {"RGB", "RGBA"}:
        report.add("UNEXPECTED_MODE", "WARN", f"Image mode is {img.mode}, expected RGB/RGBA")
