"""Unit tests — prompt_builder"""
from avatar_gen.prompt_builder import build_negative_prompt, build_positive_prompt, build_prompts
from avatar_gen.spec_parser import load_spec_dict

SPEC_DATA = {
    "age_band":    "middle_aged",
    "presentation":"masculine",
    "skin_tone":   "dark_brown",
    "hair":        {"length": "short", "colour": "black", "texture": "coily"},
    "attire":      "formal_suit",
    "background":  "office",
    "pose":        "three_quarter_left",
    "seed":        7,
}


def test_positive_prompt_contains_attributes():
    spec = load_spec_dict(SPEC_DATA)
    prompt = build_positive_prompt(spec)
    assert "dark brown skin tone" in prompt
    assert "formal business suit" in prompt
    assert "three-quarter view facing left" in prompt
    assert "photorealistic portrait" in prompt


def test_negative_prompt_contains_safety_tokens():
    spec = load_spec_dict(SPEC_DATA)
    neg = build_negative_prompt(spec)
    assert "nsfw" in neg
    assert "watermark" in neg
    assert "blurry" in neg


def test_no_nationality_labels_in_prompt():
    spec = load_spec_dict(SPEC_DATA)
    prompt = build_positive_prompt(spec)
    banned = ["african", "asian", "european", "caucasian", "indian", "chinese"]
    for term in banned:
        assert term not in prompt.lower(), f"Nationality label found: {term}"


def test_bald_hair_token():
    data = {**SPEC_DATA, "hair": {"length": "bald", "colour": "black", "texture": "straight"}}
    spec = load_spec_dict(data)
    prompt = build_positive_prompt(spec)
    assert "shaved bald head" in prompt


def test_extra_prompt_tokens_included():
    data = {**SPEC_DATA, "extra_prompt_tokens": ["award winning", "dramatic lighting"]}
    spec = load_spec_dict(data)
    prompt = build_positive_prompt(spec)
    assert "award winning" in prompt
    assert "dramatic lighting" in prompt


def test_negative_prompt_deduplicated():
    data = {**SPEC_DATA, "negative_controls": ["blurry", "watermark"]}
    spec = load_spec_dict(data)
    neg = build_negative_prompt(spec)
    tokens = [t.strip() for t in neg.split(",")]
    assert len(tokens) == len(set(tokens)), "Duplicate tokens found in negative prompt"


def test_build_prompts_returns_tuple():
    spec = load_spec_dict(SPEC_DATA)
    pos, neg = build_prompts(spec)
    assert isinstance(pos, str) and len(pos) > 20
    assert isinstance(neg, str) and len(neg) > 20
