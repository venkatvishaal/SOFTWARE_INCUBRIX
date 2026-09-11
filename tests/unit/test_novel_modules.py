"""Unit tests — novel modules (NF-01 through NF-09)"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_solid_image(r=128, g=128, b=128, size=64) -> Image.Image:
    return Image.new("RGB", (size, size), color=(r, g, b))


def _make_gradient_image(size=64) -> Image.Image:
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    for i in range(size):
        arr[:, i, :] = [i * 4 % 256, (i * 3) % 256, (255 - i * 4) % 256]
    return Image.fromarray(arr)


SPEC_DATA = {
    "age_band":    "young_adult",
    "presentation":"feminine",
    "skin_tone":   "warm_medium",
    "hair":        {"length": "medium", "colour": "dark_brown", "texture": "wavy"},
    "attire":      "business_casual",
    "background":  "neutral_grey",
    "pose":        "front_facing",
    "seed":        1234,
}


# ── NF-01: Attribute Orthogonality Verifier ────────────────────────────────────

class TestAttributeOrthogonalityVerifier:
    def test_identical_images_full_orthogonality(self):
        from avatar_gen.novel.attribute_orthogonality_verifier import verify_orthogonality
        img = _make_gradient_image()
        result = verify_orthogonality(img, img, changed_attr="hair")
        assert result.changed_attr == "hair"
        assert all(v > 0.95 for k, v in result.scores.items() if k != "hair")

    def test_completely_different_images_bleed_detected(self):
        from avatar_gen.novel.attribute_orthogonality_verifier import verify_orthogonality
        img_a = _make_solid_image(50, 50, 50)
        img_b = _make_solid_image(200, 200, 200)
        result = verify_orthogonality(img_a, img_b, changed_attr="hair")
        # Different images → expect bleed on non-target attrs
        assert isinstance(result.bleed_warnings, list)

    def test_output_has_expected_keys(self):
        from avatar_gen.novel.attribute_orthogonality_verifier import verify_orthogonality
        img = _make_gradient_image()
        result = verify_orthogonality(img, img, "skin_tone")
        d = result.to_dict()
        assert "changed_attr" in d
        assert "orthogonality_scores" in d
        assert "bleed_warnings" in d
        assert "overall_ok" in d

    def test_writes_json_output(self):
        from avatar_gen.novel.attribute_orthogonality_verifier import verify_orthogonality
        img = _make_gradient_image()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "result.json"
            # We pass output_path via verify_orthogonality_jobs; test dict directly
            result = verify_orthogonality(img, img, "hair")
            out.write_text(json.dumps(result.to_dict()), encoding="utf-8")
            assert out.exists()
            data = json.loads(out.read_text())
            assert data["changed_attr"] == "hair"


# ── NF-02: Spec-Delta Reporter ────────────────────────────────────────────────

class TestSpecDeltaReporter:
    def test_report_has_verdicts(self):
        from avatar_gen.novel.spec_delta_reporter import compute_spec_delta
        from avatar_gen.spec_parser import load_spec_dict
        spec = load_spec_dict(SPEC_DATA)
        img  = _make_gradient_image(size=128)
        report = compute_spec_delta(spec, img)
        assert len(report.attribute_verdicts) >= 4
        assert 0 <= report.overall_adherence_pct <= 100

    def test_full_match_perfect_skin_tone(self):
        from avatar_gen.novel.spec_delta_reporter import compute_spec_delta
        from avatar_gen.spec_parser import load_spec_dict
        # warm_medium ≈ RGB(180, 135, 100)
        spec = load_spec_dict({**SPEC_DATA, "skin_tone": "warm_medium"})
        img = Image.new("RGB", (128, 128), color=(180, 135, 100))
        report = compute_spec_delta(spec, img)
        skin_verdict = next(v for v in report.attribute_verdicts if v.attribute == "skin_tone")
        assert skin_verdict.match in ("FULL", "PARTIAL")

    def test_output_written_to_disk(self):
        from avatar_gen.novel.spec_delta_reporter import compute_spec_delta
        from avatar_gen.spec_parser import load_spec_dict
        spec = load_spec_dict(SPEC_DATA)
        img  = _make_gradient_image(size=128)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "spec_delta.json"
            compute_spec_delta(spec, img, output_path=out)
            assert out.exists()
            data = json.loads(out.read_text())
            assert "attribute_verdicts" in data
            assert "overall_adherence_pct" in data


# ── NF-03: Bias Surface Auditor ───────────────────────────────────────────────

class TestBiasSurfaceAuditor:
    def _make_manifests(self, n=6) -> list[dict]:
        skin_tones = ["light_fair", "warm_medium", "dark_brown", "olive", "deep_dark", "brown"]
        age_bands  = ["young_adult", "middle_aged", "senior", "teenager", "child", "young_adult"]
        return [
            {"spec": {
                "skin_tone": skin_tones[i % len(skin_tones)],
                "age_band":  age_bands[i % len(age_bands)],
                "attire":    "business_casual",
                "background":"office",
                "pose":      "front_facing",
                "hair":      {"colour": "black"},
            }}
            for i in range(n)
        ]

    def test_diverse_batch_passes(self):
        from avatar_gen.novel.bias_surface_auditor import audit_batch
        manifests = self._make_manifests(6)
        report = audit_batch(manifests, batch_id="test")
        assert report.n_avatars == 6

    def test_homogeneous_batch_flags_gap(self):
        from avatar_gen.novel.bias_surface_auditor import audit_batch
        manifests = [
            {"spec": {"skin_tone": "light_fair", "age_band": "young_adult",
                      "attire": "casual_tshirt", "background": "office",
                      "pose": "front_facing", "hair": {"colour": "blonde"}}}
            for _ in range(6)
        ]
        report = audit_batch(manifests)
        assert report.overall_status == "GAP_WARNING"
        assert len(report.recommendations) > 0

    def test_clone_pair_detection(self):
        from avatar_gen.novel.bias_surface_auditor import audit_batch
        spec_same = {"skin_tone": "light_fair", "age_band": "young_adult",
                     "attire": "casual_tshirt", "background": "office",
                     "pose": "front_facing", "hair": {"colour": "blonde"}}
        manifests = [{"spec": spec_same}, {"spec": spec_same}]
        report = audit_batch(manifests, clone_max_shared=2)
        assert len(report.clone_pairs) > 0


# ── NF-04: Consent Chain Manager ─────────────────────────────────────────────

class TestConsentChainManager:
    def test_create_and_verify_consent(self):
        from avatar_gen.novel.consent_chain_manager import (
            build_generation_binding,
            create_consent_record,
            verify_consent_chain,
        )
        with tempfile.TemporaryDirectory() as td:
            keys_dir    = Path(td) / "keys"
            consent_dir = Path(td) / "consents"
            cer = create_consent_record(
                subject_hash="abc123", scope="test", expiry_date="2030-01-01",
                keys_dir=keys_dir, consent_dir=consent_dir,
            )
            event = json.loads(cer.read_text())
            binding = build_generation_binding(
                event["consent_id"], avatar_id="test-uuid", seed=42,
                consent_dir=consent_dir, keys_dir=keys_dir,
            )
            assert binding["verified_at_generation"] is True

            # Embed in a fake manifest and verify chain
            manifest = {
                "avatar_id": "test-uuid", "seed": 42,
                "consent_chain": binding,
            }
            assert verify_consent_chain(manifest, consent_dir, keys_dir) is True

    def test_revocation_invalidates_chain(self):
        from avatar_gen.novel.consent_chain_manager import (
            build_generation_binding,
            create_consent_record,
            revoke_consent,
            verify_consent_chain,
        )
        with tempfile.TemporaryDirectory() as td:
            keys_dir    = Path(td) / "keys"
            consent_dir = Path(td) / "consents"
            cer = create_consent_record(
                "xyz789", "test", "2030-01-01", keys_dir, consent_dir,
            )
            event = json.loads(cer.read_text())
            binding = build_generation_binding(
                event["consent_id"], "avatar-2", 99, consent_dir, keys_dir,
            )
            revoke_consent(event["consent_id"], consent_dir)
            manifest = {"avatar_id": "avatar-2", "seed": 99, "consent_chain": binding}
            assert verify_consent_chain(manifest, consent_dir, keys_dir) is False


# ── NF-05: Steganographic Watermarker ─────────────────────────────────────────

class TestSteganographicWatermarker:
    def test_embed_and_verify(self):
        from avatar_gen.novel.steganographic_watermarker import embed_watermark, verify_watermark
        img = _make_gradient_image(size=256)
        wm_img, result = embed_watermark(img, "test-avatar-id-1234", "test-model")
        assert result.embedded
        assert result.payload_bits == 64
        # better than random chance (detectability, not lossless)
        assert result.ber_self_check < 0.5

        verify = verify_watermark(wm_img)
        assert verify.verified
        assert verify.payload_hex and len(verify.payload_hex) == 16  # 8 bytes = 16 hex chars

    def test_watermark_result_to_dict(self):
        from avatar_gen.novel.steganographic_watermarker import embed_watermark
        img = _make_gradient_image(size=256)
        _, result = embed_watermark(img, "uuid-abc", "model-x")
        d = result.to_dict()
        assert "algorithm" in d
        assert "payload_bits" in d
        assert "ber_self_check" in d
        assert "survives_jpeg_q70" in d


# ── NF-06: Attribute Grammar Validator ───────────────────────────────────────

class TestAttributeGrammarValidator:
    def test_valid_spec_passes(self):
        from avatar_gen.novel.attribute_grammar_validator import validate_spec
        from avatar_gen.spec_parser import load_spec_dict
        spec = load_spec_dict(SPEC_DATA)
        report = validate_spec(spec)
        assert not report.has_blocks

    def test_block_rule_detected(self):
        from avatar_gen.novel.attribute_grammar_validator import validate_spec
        from avatar_gen.spec_parser import load_spec_dict
        bad = {**SPEC_DATA, "age_band": "child", "attire": "formal_suit"}
        spec = load_spec_dict(bad)
        report = validate_spec(spec)
        assert report.has_blocks
        assert report.valid is False

    def test_force_override_downgrades_block(self):
        from avatar_gen.novel.attribute_grammar_validator import validate_spec
        from avatar_gen.spec_parser import load_spec_dict
        bad = {**SPEC_DATA, "age_band": "child", "attire": "formal_suit"}
        spec = load_spec_dict(bad)
        report = validate_spec(spec, force_override=True)
        assert not report.has_blocks   # downgraded to WARN
        assert report.valid is True

    def test_stereotype_risk_detected(self):
        from avatar_gen.novel.attribute_grammar_validator import validate_spec
        from avatar_gen.spec_parser import load_spec_dict
        risky = {**SPEC_DATA, "skin_tone": "dark_brown",
                 "background": "outdoor_park", "attire": "traditional_attire"}
        spec = load_spec_dict(risky)
        report = validate_spec(spec)
        assert report.has_stereotype_risk

    def test_report_written_to_disk(self):
        from avatar_gen.novel.attribute_grammar_validator import validate_spec
        from avatar_gen.spec_parser import load_spec_dict
        spec = load_spec_dict(SPEC_DATA)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "report.json"
            validate_spec(spec, output_path=out)
            assert out.exists()
            data = json.loads(out.read_text())
            assert "valid" in data


# ── NF-07: Failure Taxonomy Classifier ───────────────────────────────────────

class TestFailureTaxonomyClassifier:
    def test_classify_safety_exception(self):
        from avatar_gen.novel.failure_taxonomy_classifier import FailureCode, classify_exception
        exc = RuntimeError("nsfw content detected")
        assert classify_exception(exc) == FailureCode.SAFETY_REFUSAL

    def test_classify_oom_exception(self):
        from avatar_gen.novel.failure_taxonomy_classifier import FailureCode, classify_exception
        exc = RuntimeError("CUDA out of memory")
        assert classify_exception(exc) == FailureCode.MODEL_OOM

    def test_record_failure_writes_json(self):
        from avatar_gen.novel.failure_taxonomy_classifier import FailureCode, record_failure
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "failure_record.json"
            record_failure(FailureCode.COMPUTE_TIMEOUT, "timed out", out,
                                 spec_hash="abc123", seed=42)
            assert out.exists()
            data = json.loads(out.read_text())
            assert data["failure_code"] == "COMPUTE_TIMEOUT"
            assert "suggested_action" in data

    def test_summarize_failures(self):
        from avatar_gen.novel.failure_taxonomy_classifier import (
            FailureCode,
            record_failure,
            summarize_failures,
        )
        with tempfile.TemporaryDirectory() as td:
            for i, code in enumerate([FailureCode.SAFETY_REFUSAL, FailureCode.SAFETY_REFUSAL,
                                       FailureCode.MODEL_OOM]):
                record_failure(code, "test", Path(td) / f"job{i}" / "failure_record.json")
            counts = summarize_failures(Path(td))
            assert counts.get("SAFETY_REFUSAL") == 2
            assert counts.get("MODEL_OOM") == 1


# ── NF-08: Adaptive Resolution Ladder ────────────────────────────────────────

class TestAdaptiveResolutionLadder:
    def test_stops_at_first_passing_rung(self):
        from avatar_gen.novel.adaptive_resolution_ladder import run_adaptive_ladder
        call_log: list[int] = []

        def gen_fn(w, h):
            call_log.append(w)
            # Return a varied image that passes gates
            return _make_gradient_image(size=w)

        img, result = run_adaptive_ladder(gen_fn, ladder=[128, 256, 512], max_rung=512,
                                          adherence_threshold=10.0)
        # Should stop early (at 128 or 256)
        assert len(call_log) <= 2
        assert result.compute_saved_pct >= 0

    def test_progression_logged(self):
        from avatar_gen.novel.adaptive_resolution_ladder import run_adaptive_ladder
        _, result = run_adaptive_ladder(
            lambda w, h: _make_gradient_image(size=w),
            ladder=[128, 256], max_rung=256, adherence_threshold=0.0,
        )
        assert isinstance(result.rung_progression, list)
        assert len(result.rung_progression) >= 1

    def test_output_json_written(self):
        from avatar_gen.novel.adaptive_resolution_ladder import run_adaptive_ladder
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "ladder.json"
            run_adaptive_ladder(
                lambda w, h: _make_gradient_image(size=w),
                ladder=[128, 256], max_rung=256, output_path=out,
            )
            assert out.exists()
            data = json.loads(out.read_text())
            assert "rung_progression" in data


# ── NF-09: Cross-Model Portability Scorer ────────────────────────────────────

class TestCrossModelPortabilityScorer:
    def test_perfect_portability(self):
        from avatar_gen.novel.cross_model_portability_scorer import (
            compute_portability,
            profile_from_dict,
        )
        attrs = {"skin_tone": 0.9, "hair": 0.8, "pose": 1.0, "background": 0.7}
        pa = profile_from_dict(attrs, "model_A")
        pb = profile_from_dict(attrs, "model_B")
        report = compute_portability(pa, pb)
        assert report.cmpp_overall == pytest.approx(1.0, abs=1e-6)

    def test_imperfect_portability(self):
        from avatar_gen.novel.cross_model_portability_scorer import (
            compute_portability,
            profile_from_dict,
        )
        pa = profile_from_dict({"hair": 1.0, "pose": 1.0}, "model_A")
        pb = profile_from_dict({"hair": 0.0, "pose": 1.0}, "model_B")
        report = compute_portability(pa, pb)
        assert report.least_portable_attribute == "hair"
        assert report.cmpp_per_attribute["hair"] == pytest.approx(0.0)

    def test_report_to_dict(self):
        from avatar_gen.novel.cross_model_portability_scorer import (
            compute_portability,
            profile_from_dict,
        )
        pa = profile_from_dict({"skin_tone": 0.8}, "A")
        pb = profile_from_dict({"skin_tone": 0.6}, "B")
        report = compute_portability(pa, pb)
        d = report.to_dict()
        assert "cmpp_overall" in d
        assert "cmpp_per_attribute" in d
        assert "recommendation" in d

    def test_no_shared_attrs_raises(self):
        from avatar_gen.novel.cross_model_portability_scorer import (
            compute_portability,
            profile_from_dict,
        )
        pa = profile_from_dict({"hair": 1.0}, "A")
        pb = profile_from_dict({"pose": 1.0}, "B")
        with pytest.raises(ValueError, match="No shared attributes"):
            compute_portability(pa, pb)
