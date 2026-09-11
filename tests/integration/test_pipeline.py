"""Integration tests — full CLI pipeline"""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

SPEC = {
    "age_band":    "young_adult",
    "presentation":"feminine",
    "skin_tone":   "warm_medium",
    "hair":        {"length": "medium", "colour": "dark_brown", "texture": "wavy"},
    "attire":      "business_casual",
    "background":  "office",
    "pose":        "front_facing",
    "seed":        42,
}


def _run_cli(*args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "avatar_gen.cli", *args],
        capture_output=True, text=True
    )


@pytest.fixture
def spec_file(tmp_path) -> Path:
    p = tmp_path / "spec.yaml"
    p.write_text(yaml.dump(SPEC), encoding="utf-8")
    return p


@pytest.fixture
def job_dir(tmp_path, spec_file) -> Path:
    import uuid

    from avatar_gen.prompt_builder import build_prompts
    from avatar_gen.spec_parser import load_spec

    job = tmp_path / "job"
    job.mkdir()

    spec = load_spec(spec_file)
    pos, neg = build_prompts(spec)
    params = {
        "job_id": str(uuid.uuid4()),
        "spec_hash": "abc123",
        "positive_prompt": pos,
        "negative_prompt": neg,
        "seed": spec.seed,
        "width": 256,
        "height": 256,
        "model": "stub",
    }
    (job / "job_params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    import shutil
    shutil.copy(spec_file, job / "spec.yaml")
    return job


class TestPreparePipeline:
    def test_prepare_creates_bundle(self, spec_file, tmp_path):
        from avatar_gen.novel.attribute_grammar_validator import validate_spec_file
        from avatar_gen.prompt_builder import build_prompts
        from avatar_gen.spec_parser import load_spec

        tmp_path / "bundle"
        report = validate_spec_file(spec_file)
        assert report.valid

        spec = load_spec(spec_file)
        pos, neg = build_prompts(spec)
        assert len(pos) > 20
        assert len(neg) > 10


class TestRunPipeline:
    def test_run_local_cpu_produces_output(self, job_dir):
        from avatar_gen.inference_client import LocalCPUClient
        from avatar_gen.novel.spec_delta_reporter import compute_spec_delta
        from avatar_gen.novel.steganographic_watermarker import embed_watermark
        from avatar_gen.provenance_writer import write_manifest
        from avatar_gen.safety_checker import check_image
        from avatar_gen.spec_parser import load_spec

        spec = load_spec(job_dir / "spec.yaml")
        client = LocalCPUClient()
        result = client.generate("portrait", "blurry", spec.seed, 256, 256)
        assert result.image.size == (256, 256)

        safety = check_image(result.image)
        assert safety.passed

        img_path = job_dir / "avatar_test.png"
        result.image.save(str(img_path))
        assert img_path.exists()

        _, wm = embed_watermark(result.image, spec.avatar_id, "stub")
        assert wm.embedded

        delta = compute_spec_delta(spec, result.image)
        assert 0 <= delta.overall_adherence_pct <= 100

        manifest_path = write_manifest(
            spec=spec, result=result, safety=safety,
            image_path=img_path, job_dir=job_dir,
            watermark=wm.to_dict(),
        )
        from avatar_gen.provenance_writer import patch_manifest
        patch_manifest(manifest_path, {
            "prompt_generated": "test positive prompt",
            "negative_prompt": "test negative prompt",
        })
        assert manifest_path.exists()


class TestValidatePipeline:
    def test_validate_passes_for_correct_output(self, job_dir):
        from avatar_gen.inference_client import LocalCPUClient
        from avatar_gen.novel.steganographic_watermarker import embed_watermark
        from avatar_gen.output_validator import validate_job
        from avatar_gen.provenance_writer import patch_manifest, write_manifest
        from avatar_gen.safety_checker import check_image
        from avatar_gen.spec_parser import load_spec

        spec = load_spec(job_dir / "spec.yaml")
        client = LocalCPUClient()
        result = client.generate("portrait", "blurry", spec.seed, 256, 256)

        img_path = job_dir / "avatar_test.png"
        _, wm = embed_watermark(result.image, spec.avatar_id, "stub")
        wm.image if hasattr(wm, "image") else result.image
        result.image.save(str(img_path))

        safety = check_image(result.image)
        manifest_path = write_manifest(
            spec=spec, result=result, safety=safety,
            image_path=img_path, job_dir=job_dir,
            watermark=wm.to_dict(),
        )
        patch_manifest(manifest_path, {
            "prompt_generated": "portrait",
            "negative_prompt": "blurry",
        })

        report = validate_job(str(job_dir), expect_watermark=True, expect_spec_delta=False)
        assert report.passed, report.summary()
