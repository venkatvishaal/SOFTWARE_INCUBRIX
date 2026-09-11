"""End-to-end test — full avatar generation pipeline"""
import json
import tempfile
from pathlib import Path

SPEC = {
    "age_band":    "middle_aged",
    "presentation":"masculine",
    "skin_tone":   "dark_brown",
    "hair":        {"length": "short", "colour": "black", "texture": "coily"},
    "attire":      "formal_suit",
    "background":  "office",
    "pose":        "front_facing",
    "seed":        7777,
}


def test_end_to_end_avatar_generation():
    """Full pipeline: spec → grammar check → prompt → inference → safety →
    watermark → spec-delta → manifest → validate.
    This is the mandatory end-to-end test required by the assessment."""
    from avatar_gen.inference_client import LocalCPUClient
    from avatar_gen.novel.attribute_grammar_validator import validate_spec
    from avatar_gen.novel.spec_delta_reporter import compute_spec_delta
    from avatar_gen.novel.steganographic_watermarker import embed_watermark
    from avatar_gen.output_validator import validate_job
    from avatar_gen.prompt_builder import build_prompts
    from avatar_gen.provenance_writer import patch_manifest, write_manifest
    from avatar_gen.safety_checker import check_image, check_prompt
    from avatar_gen.spec_parser import load_spec_dict

    with tempfile.TemporaryDirectory() as td:
        job_dir = Path(td) / "e2e_job"
        job_dir.mkdir()

        # 1. Load spec
        spec = load_spec_dict(SPEC)
        assert spec.seed == 7777

        # 2. Grammar check
        grammar_report = validate_spec(spec)
        assert grammar_report.valid, grammar_report.summary()
        assert not grammar_report.has_blocks

        # 3. Build prompts
        pos_prompt, neg_prompt = build_prompts(spec)
        assert "dark brown skin tone" in pos_prompt
        assert "formal business suit" in pos_prompt

        # 4. Prompt safety
        safe, flagged = check_prompt(pos_prompt, neg_prompt)
        assert safe, f"Prompt flagged: {flagged}"

        # 5. Inference (CPU stub)
        client = LocalCPUClient()
        result = client.generate(pos_prompt, neg_prompt, spec.seed, 512, 512)
        assert result.image.size == (512, 512)
        assert result.compute_route == "local_cpu_stub"

        # 6. Safety check on output
        safety = check_image(result.image)
        assert safety.passed, f"Safety failed: {safety.flags}"

        # 7. Watermark
        wm_img, wm_result = embed_watermark(result.image, spec.avatar_id, result.model)
        assert wm_result.embedded
        assert wm_result.ber_self_check < 0.5   # detectability threshold (better than random)

        # 8. Save image
        img_path = job_dir / f"avatar_{spec.avatar_id[:8]}.png"
        wm_img.save(str(img_path))
        assert img_path.exists()

        # 9. Spec-delta
        delta_path = job_dir / "spec_delta.json"
        delta = compute_spec_delta(spec, img_path, output_path=delta_path)
        assert delta_path.exists()
        assert 0 <= delta.overall_adherence_pct <= 100

        # 10. Write manifest
        manifest_path = write_manifest(
            spec=spec, result=result, safety=safety,
            image_path=img_path, job_dir=job_dir,
            watermark=wm_result.to_dict(),
            spec_delta_path=delta_path,
        )
        patch_manifest(manifest_path, {
            "prompt_generated": pos_prompt,
            "negative_prompt":  neg_prompt,
        })
        assert manifest_path.exists()

        manifest = json.loads(manifest_path.read_text())
        assert manifest["avatar_id"] == spec.avatar_id
        assert manifest["seed"] == 7777
        assert manifest["safety"]["passed"] is True
        assert manifest["watermark"]["embedded"] is True

        # 11. Validate
        report = validate_job(str(job_dir), expect_watermark=True, expect_spec_delta=True)
        assert report.passed, report.summary()

        print("✅ E2E test PASSED")
        print(f"   Avatar ID:  {spec.avatar_id}")
        print(f"   Image:      {img_path}")
        print(f"   Adherence:  {delta.overall_adherence_pct:.1f}%")
        print(f"   WM BER:     {wm_result.ber_self_check}")
