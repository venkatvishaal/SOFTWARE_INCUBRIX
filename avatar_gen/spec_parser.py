"""
spec_parser.py — Load, validate, and normalise avatar specifications.

Reads JSON or YAML spec files, validates against avatar_spec.schema.json,
assigns a UUID if absent, and returns a typed AvatarSpec pydantic model.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import jsonschema
import yaml
from pydantic import BaseModel, Field, model_validator

# ── Schema path ───────────────────────────────────────────────────────────────
_SCHEMA_PATH = Path(__file__).parent / "schemas" / "avatar_spec.schema.json"
_SCHEMA: dict | None = None


def _get_schema() -> dict:
    global _SCHEMA
    if _SCHEMA is None:
        _SCHEMA = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    return _SCHEMA


# ── Pydantic models ───────────────────────────────────────────────────────────

class HairSpec(BaseModel):
    length: str
    colour: str
    texture: str
    style: str | None = None


class AvatarSpec(BaseModel):
    avatar_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    age_band: str
    presentation: str
    skin_tone: str
    hair: HairSpec
    attire: str
    background: str
    pose: str
    seed: int
    negative_controls: list[str] = Field(
        default=["blurry", "watermark", "nsfw", "deformed", "low_quality"]
    )
    aspect_ratio: str = "1:1"
    resolution_hint: int = 512
    reference_images: list[str] = Field(default_factory=list)
    consent_token: str | None = None
    model_override: str | None = None
    extra_prompt_tokens: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _consent_required_with_references(self) -> "AvatarSpec":
        if self.reference_images and not self.consent_token:
            raise ValueError(
                "consent_token is required when reference_images are provided. "
                "Run `avatar-gen consent-init` to create a Consent Event Record."
            )
        return self


# ── Public API ────────────────────────────────────────────────────────────────

def load_spec(path: str | Path) -> AvatarSpec:
    """Load and validate a spec from a JSON or YAML file.

    Args:
        path: Path to the spec file (.json or .yaml/.yml).

    Returns:
        Validated AvatarSpec instance.

    Raises:
        FileNotFoundError: If the spec file does not exist.
        jsonschema.ValidationError: If the raw spec fails JSON Schema validation.
        ValueError: If Pydantic model validation fails.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Spec file not found: {path}")

    raw = _read_raw(path)
    _validate_schema(raw)
    return AvatarSpec(**raw)


def load_spec_dict(data: dict) -> AvatarSpec:
    """Load a spec from an already-parsed dictionary (for testing / programmatic use)."""
    _validate_schema(data)
    return AvatarSpec(**data)


def spec_to_dict(spec: AvatarSpec) -> dict:
    """Serialise a spec to a plain dictionary (JSON-safe)."""
    return spec.model_dump()


def spec_to_file(spec: AvatarSpec, path: str | Path, fmt: str = "yaml") -> None:
    """Write a spec to disk.

    Args:
        spec: AvatarSpec instance.
        path: Destination file path.
        fmt:  "yaml" or "json".
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = spec_to_dict(spec)
    with path.open("w", encoding="utf-8") as fh:
        if fmt == "json":
            json.dump(data, fh, indent=2)
        else:
            yaml.dump(data, fh, allow_unicode=True, sort_keys=False)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _read_raw(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        return yaml.safe_load(text) or {}
    return json.loads(text)


def _validate_schema(data: dict) -> None:
    schema = _get_schema()
    validator = jsonschema.Draft7Validator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda e: e.path)
    if errors:
        messages = "; ".join(
            f"{'.'.join(str(p) for p in e.absolute_path) or 'root'}: {e.message}"
            for e in errors[:5]  # show up to 5 errors
        )
        raise jsonschema.ValidationError(f"Spec validation failed: {messages}")
