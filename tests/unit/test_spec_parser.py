"""Unit tests — spec_parser"""
import json
import tempfile
from pathlib import Path

import pytest

from avatar_gen.spec_parser import load_spec, load_spec_dict, spec_to_file

VALID_SPEC = {
    "age_band":    "young_adult",
    "presentation":"feminine",
    "skin_tone":   "warm_medium",
    "hair":        {"length": "medium", "colour": "dark_brown", "texture": "wavy"},
    "attire":      "business_casual",
    "background":  "office",
    "pose":        "front_facing",
    "seed":        42,
}


def test_load_valid_spec_dict():
    spec = load_spec_dict(VALID_SPEC)
    assert spec.age_band == "young_adult"
    assert spec.seed == 42
    assert isinstance(spec.avatar_id, str) and len(spec.avatar_id) == 36


def test_auto_uuid_assigned():
    s1 = load_spec_dict(VALID_SPEC)
    s2 = load_spec_dict(VALID_SPEC)
    assert s1.avatar_id != s2.avatar_id


def test_load_from_yaml_file():
    with tempfile.TemporaryDirectory() as td:
        import yaml
        p = Path(td) / "spec.yaml"
        p.write_text(yaml.dump(VALID_SPEC), encoding="utf-8")
        spec = load_spec(p)
        assert spec.skin_tone == "warm_medium"


def test_load_from_json_file():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "spec.json"
        p.write_text(json.dumps(VALID_SPEC), encoding="utf-8")
        spec = load_spec(p)
        assert spec.attire == "business_casual"


def test_missing_required_field():
    import jsonschema
    bad = {k: v for k, v in VALID_SPEC.items() if k != "age_band"}
    with pytest.raises(jsonschema.ValidationError):
        load_spec_dict(bad)


def test_invalid_enum_value():
    import jsonschema
    bad = {**VALID_SPEC, "age_band": "infant"}
    with pytest.raises(jsonschema.ValidationError):
        load_spec_dict(bad)


def test_consent_token_required_with_references():
    bad = {**VALID_SPEC, "reference_images": ["photo.jpg"]}
    with pytest.raises(ValueError, match="consent_token"):
        load_spec_dict(bad)


def test_spec_roundtrip_yaml():
    with tempfile.TemporaryDirectory() as td:
        spec = load_spec_dict(VALID_SPEC)
        p = Path(td) / "out.yaml"
        spec_to_file(spec, p, fmt="yaml")
        spec2 = load_spec(p)
        assert spec2.seed == spec.seed
        assert spec2.avatar_id == spec.avatar_id


def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_spec(Path("/nonexistent/spec.yaml"))
