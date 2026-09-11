"""
prompt_builder.py — Convert AvatarSpec into positive and negative prompt strings.

Maps structured attribute values to natural-language prompt tokens suitable
for Stable Diffusion (and compatible) models. No nationality labels are used;
all attributes are represented as neutral visual descriptors.
"""
from __future__ import annotations

from avatar_gen.spec_parser import AvatarSpec

# ── Attribute → token maps ────────────────────────────────────────────────────

_AGE_BAND_TOKENS: dict[str, str] = {
    "child":       "young child, approximately 8 years old",
    "teenager":    "teenager, approximately 16 years old",
    "young_adult": "young adult, approximately 25 years old",
    "middle_aged": "middle-aged adult, approximately 45 years old",
    "senior":      "senior adult, approximately 65 years old",
}

_PRESENTATION_TOKENS: dict[str, str] = {
    "feminine":    "feminine-presenting person",
    "masculine":   "masculine-presenting person",
    "androgynous": "androgynous-presenting person",
}

_SKIN_TONE_TOKENS: dict[str, str] = {
    "very_light_fair": "very light fair skin tone",
    "light_fair":      "light fair skin tone",
    "light_medium":    "light medium skin tone",
    "medium":          "medium skin tone",
    "warm_medium":     "warm medium skin tone",
    "olive":           "olive skin tone",
    "tan":             "tan skin tone",
    "brown":           "brown skin tone",
    "dark_brown":      "dark brown skin tone",
    "deep_dark":       "deep dark skin tone",
}

_HAIR_LENGTH_TOKENS: dict[str, str] = {
    "bald":       "shaved bald head",
    "very_short": "very short hair",
    "short":      "short hair",
    "medium":     "medium-length hair",
    "long":       "long hair",
    "very_long":  "very long hair",
}

_HAIR_COLOUR_TOKENS: dict[str, str] = {
    "black":      "black",
    "dark_brown": "dark brown",
    "medium_brown":"medium brown",
    "auburn":     "auburn",
    "blonde":     "blonde",
    "red":        "red",
    "grey":       "grey",
    "white":      "white",
    "dyed_blue":  "blue-dyed",
    "dyed_red":   "bright red-dyed",
    "dyed_green": "green-dyed",
}

_HAIR_TEXTURE_TOKENS: dict[str, str] = {
    "straight": "straight",
    "wavy":     "wavy",
    "curly":    "curly",
    "coily":    "coily",
    "afro":     "afro-textured",
    "locs":     "dreadlocked",
}

_ATTIRE_TOKENS: dict[str, str] = {
    "formal_suit":        "wearing a formal business suit and tie",
    "business_casual":    "wearing business casual attire",
    "casual_tshirt":      "wearing a casual t-shirt",
    "casual_shirt":       "wearing a casual button-down shirt",
    "traditional_attire": "wearing traditional cultural attire",
    "uniform_medical":    "wearing medical scrubs",
    "uniform_corporate":  "wearing a corporate uniform",
    "sportswear":         "wearing athletic sportswear",
    "creative_casual":    "wearing creative casual clothing",
    "winter_coat":        "wearing a winter coat",
}

_BACKGROUND_TOKENS: dict[str, str] = {
    "neutral_grey":    "neutral grey background",
    "neutral_white":   "neutral white background",
    "gradient_blue":   "soft blue gradient background",
    "office":          "modern office environment background",
    "outdoor_park":    "outdoor park setting background",
    "outdoor_urban":   "outdoor urban street background",
    "studio_lit":      "professional studio lighting background",
    "home_bookshelf":  "home bookshelf background",
    "conference_room": "conference room background",
    "classroom":       "classroom background",
}

_POSE_TOKENS: dict[str, str] = {
    "front_facing":        "facing directly forward, portrait orientation",
    "three_quarter_left":  "three-quarter view facing left",
    "three_quarter_right": "three-quarter view facing right",
    "profile_left":        "left profile view",
    "profile_right":       "right profile view",
    "slight_tilt_up":      "slightly tilted upward gaze",
    "slight_tilt_down":    "slightly downward gaze",
}

# Quality boosters appended to every positive prompt
_QUALITY_TOKENS = (
    "photorealistic portrait, high detail, sharp focus, professional photography, "
    "8k resolution, cinematic lighting"
)

# Default negative tokens (extended from spec defaults)
_DEFAULT_NEGATIVE = (
    "blurry, watermark, text overlay, logo, signature, deformed anatomy, "
    "extra limbs, fused fingers, bad hands, ugly, low quality, disfigured, "
    "nsfw, nude, violent, offensive content, cartoon, anime, painting, "
    "illustration, 3d render, cgi"
)


# ── Public API ─────────────────────────────────────────────────────────────────

def build_positive_prompt(spec: AvatarSpec) -> str:
    """Build a positive prompt string from an AvatarSpec.

    Returns:
        A comma-separated prompt string ready for model inference.
    """
    tokens: list[str] = []

    # Subject
    tokens.append(_PRESENTATION_TOKENS.get(spec.presentation, spec.presentation))
    tokens.append(_AGE_BAND_TOKENS.get(spec.age_band, spec.age_band))

    # Skin tone
    tokens.append(_SKIN_TONE_TOKENS.get(spec.skin_tone, spec.skin_tone))

    # Hair
    hair = spec.hair
    if hair.length == "bald":
        tokens.append(_HAIR_LENGTH_TOKENS["bald"])
    else:
        colour_tok = _HAIR_COLOUR_TOKENS.get(hair.colour, hair.colour)
        texture_tok = _HAIR_TEXTURE_TOKENS.get(hair.texture, hair.texture)
        length_tok = _HAIR_LENGTH_TOKENS.get(hair.length, hair.length)
        tokens.append(f"{length_tok}, {colour_tok} {texture_tok} hair")
        if hair.style:
            tokens.append(hair.style)

    # Attire
    tokens.append(_ATTIRE_TOKENS.get(spec.attire, spec.attire))

    # Background
    tokens.append(_BACKGROUND_TOKENS.get(spec.background, spec.background))

    # Pose
    tokens.append(_POSE_TOKENS.get(spec.pose, spec.pose))

    # Extra caller tokens
    tokens.extend(spec.extra_prompt_tokens)

    # Quality suffix
    tokens.append(_QUALITY_TOKENS)

    return ", ".join(t.strip() for t in tokens if t.strip())


def build_negative_prompt(spec: AvatarSpec) -> str:
    """Build a negative prompt string from an AvatarSpec."""
    extra = ", ".join(spec.negative_controls)
    combined = f"{_DEFAULT_NEGATIVE}, {extra}" if extra else _DEFAULT_NEGATIVE
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for tok in (t.strip() for t in combined.split(",")):
        if tok and tok not in seen:
            seen.add(tok)
            unique.append(tok)
    return ", ".join(unique)


def build_prompts(spec: AvatarSpec) -> tuple[str, str]:
    """Return (positive_prompt, negative_prompt) tuple."""
    return build_positive_prompt(spec), build_negative_prompt(spec)
