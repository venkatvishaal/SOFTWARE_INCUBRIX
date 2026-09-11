# avatar-gen — Open-Source AI Human-Avatar Generation Pipeline

> **IncuBrix SASTRA 2027 · Graduate Hiring · Track 02 Submission**

[![Tests](https://img.shields.io/badge/tests-50%20passed-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![Model](https://img.shields.io/badge/model-SD%20v1.5-orange)]()
[![Revision](https://img.shields.io/badge/revision-451f4fe-purple)]()
[![Budget](https://img.shields.io/badge/cost-%240-success)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey)]()

---

## Executive Summary

`avatar-gen` is a zero-budget, fully open-source AI human-avatar generation pipeline engineered for full reproducibility, cryptographic provenance, safety enforcement, and synthetic media transparency.

The pipeline maps structured, typed YAML/JSON avatar specifications into photorealistic image outputs using Stable Diffusion v1.5, accompanied by cryptographic audit trails, steganographic watermarking, attribute drift analysis, and bias surface monitoring.

---

## Pinned Model Specification & Provenance

To guarantee 100% deterministic reproducibility from clean environments, `avatar-gen` explicitly pins model identifiers, weights, and revision commit SHAs:

| Parameter | Pinned Value | Verification Note |
|---|---|---|
| **Model Repository** | `stable-diffusion-v1-5/stable-diffusion-v1-5` | Maintained Hugging Face org path |
| **Model Revision (SHA)** | `451f4fe16113bff5a5d2269ed5ad43b0592e9a14` | Canonical `main` HEAD commit hash |
| **Checkpoint File** | `v1-5-pruned-emaonly.safetensors` | Explicit safetensors weights selection |
| **Compute Routes** | `huggingface_zerogpu`, `kaggle`, `lightning`, `diffusers_cuda`, `local_cpu` | Approved free-tier compute providers |
| **Relative Artifact Paths** | `jobs/job_001/avatar_550e8400.png` | Cross-platform portable paths (no absolute drive letters) |

```bash
# Verify pinned model revision hash via Hugging Face REST API
curl -s https://huggingface.co/api/models/stable-diffusion-v1-5/stable-diffusion-v1-5 | jq -r '.sha'
# Expected output: 451f4fe16113bff5a5d2269ed5ad43b0592e9a14
```

---

## Key Technical Features

1. **Strict Spec Parsing & Validation**: Pydantic-backed validation enforcing Fitzpatrick-anchored skin tones, neutral attributes, and prohibiting nationality/ethnicity descriptors.
2. **Cryptographic Provenance Manifest (`avatar_manifest.json`)**: Every job emits an immutable provenance manifest logging full specs, seeds, exact model commit SHAs, compute routes, runtime metrics, and safety scores.
3. **Safety & Heuristic Verification**: Multi-layered output validation including dimension verification, variance blank-checking, NSFW heuristic thresholds (`score < 0.5`, `lower_is_safer`), and watermark detection.
4. **Steganographic Watermarking**: DCT/DWT steganographic payload embedding into generated images for early compliance with synthetic media labeling requirements.
5. **9 Novel Engineering Modules**: Modular extensions addressing attribute bleed, bias surface auditability, consent management, adaptive resolution, and cross-model portability.

---

## 9 Novel Feature Modules (NF-01 – NF-09)

| Module ID | Name | CLI Subcommand | Functionality |
|---|---|---|---|
| **NF-01** | Attribute Orthogonality Verifier | `audit-orthogonality` | Detects attribute bleed across prompt mutations (e.g. hair length altering perceived age). |
| **NF-02** | Spec-Delta Reporter | `spec-delta` | Calculates per-attribute drift scores comparing target spec vs. generated output features. |
| **NF-03** | Bias Surface Auditor | `audit-bias` | Computes Shannon entropy across demographic outputs to flag representation gaps. |
| **NF-04** | Cryptographic Consent Chain (C3) | `consent-init`, `verify-consent`, `revoke-consent` | Issues cryptographic Consent Event Records required for reference-image generation. |
| **NF-05** | Steganographic Watermarker | `verify-watermark` | Embeds and decodes 64-bit DWT/DCT watermarks for synthetic content provenance. |
| **NF-06** | Attribute Grammar Validator | `validate-spec` | Pre-generation validation rule engine blocking invalid or stereotype-risk feature combinations. |
| **NF-07** | Failure Taxonomy Classifier | `summarize-failures` | Categorizes pipeline exceptions (OOM, safety flag, timeout) into structured taxonomy reports. |
| **NF-08** | Adaptive Resolution Ladder | `--adaptive-resolution` | Dynamically steps generation resolution up/down based on available hardware VRAM. |
| **NF-09** | Cross-Model Portability Scorer | `portability-score` | Evaluates prompt portability across model checkpoints (e.g. SD 1.5 vs SD 2.1 vs SDXL). |

---

## Quick Start Guide

### Installation

```bash
# Clone repository
git clone https://github.com/incubrix/avatar-gen.git
cd avatar-gen

# Install in editable mode (CPU orchestration)
pip install -e "."

# Install with development & GPU support
pip install -e ".[dev,gpu]"
```

### End-to-End Execution Workflow

```bash
# Step 1: Validate input specification against grammar rules
avatar-gen validate-spec --spec examples/spec_young_adult_f.yaml

# Step 2: Prepare job bundle (creates spec & output structures)
avatar-gen prepare --spec examples/spec_young_adult_f.yaml --output-dir jobs/job_001

# Step 3: Run pipeline inference (CPU stub for local testing, or remote GPU)
avatar-gen run --job-dir jobs/job_001 --compute-route local_cpu

# Step 4: Run safety & output validator
avatar-gen validate --job-dir jobs/job_001

# Step 5: Generate spec-delta report
avatar-gen spec-delta --job-dir jobs/job_001

# Step 6: Verify steganographic watermark
avatar-gen verify-watermark --image jobs/job_001/avatar_550e8400.png

# Step 7: Run automated pipeline benchmarks
avatar-gen bench --spec examples/spec_young_adult_f.yaml --repetitions 3
```

---

## Architecture & Project Structure

```
avatar-gen/
├── avatar_gen/
│   ├── cli.py                     # 13-subcommand CLI entry point
│   ├── spec_parser.py             # Spec loader + Pydantic validation
│   ├── prompt_builder.py          # Attribute → natural-language prompt synthesis
│   ├── inference_client.py        # Backend routes (local_cpu, diffusers_cuda, zeroGPU, etc.)
│   ├── safety_checker.py          # NSFW score, blank detection, dimension checks
│   ├── provenance_writer.py       # avatar_manifest.json emission
│   ├── output_validator.py        # Post-generation compliance validation
│   ├── evaluator.py               # Adherence & identity evaluation metrics
│   ├── benchmark.py               # Memory & runtime benchmarking
│   ├── schemas/
│   │   ├── avatar_spec.schema.json
│   │   └── avatar_manifest.schema.json
│   └── novel/                     # 9 Novel Feature Modules
│       ├── attribute_orthogonality_verifier.py   # NF-01
│       ├── spec_delta_reporter.py                # NF-02
│       ├── bias_surface_auditor.py               # NF-03
│       ├── consent_chain_manager.py              # NF-04
│       ├── steganographic_watermarker.py         # NF-05
│       ├── attribute_grammar_validator.py        # NF-06
│       ├── failure_taxonomy_classifier.py        # NF-07
│       ├── adaptive_resolution_ladder.py         # NF-08
│       └── cross_model_portability_scorer.py     # NF-09
├── config/
│   ├── default_config.yaml
│   └── attribute_grammar.yaml
├── examples/
│   ├── spec_young_adult_f.yaml
│   └── spec_middle_aged_m.yaml
├── jobs/
│   └── job_001/                   # Sample job directory with manifest & outputs
├── tests/
│   ├── unit/                      # Unit test suite (46 tests)
│   ├── integration/               # Pipeline integration test suite (3 tests)
│   └── e2e/                       # End-to-end pipeline test (1 test)
└── pyproject.toml
```

---

## Provenance Manifest Schema Example

Generated manifests (`jobs/job_001/avatar_manifest.json`) record deterministic generation parameters:

```json
{
  "avatar_id": "550e8400-e29b-41d4-a716-446655440000",
  "seed": 42,
  "model": "stable-diffusion-v1-5/stable-diffusion-v1-5",
  "model_revision": "451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
  "checkpoint": "v1-5-pruned-emaonly.safetensors",
  "provenance": {
    "pipeline_version": "1.0.0",
    "generated_at": "2026-09-10T02:36:08.158753+00:00",
    "compute_route": "huggingface_zerogpu",
    "runtime_sec": 0.904,
    "timing_scope": "denoising_loop_only",
    "steps": 30,
    "guidance_scale": 7.5
  },
  "safety": {
    "passed": true,
    "score": 0.2964510236467634,
    "score_direction": "lower_is_safer",
    "threshold": 0.5,
    "checks": [
      "dimension_check(OK: 512x512)",
      "blank_check(OK: variance=3137.04)",
      "nsfw_heuristic(OK: score=0.296 < threshold=0.5)",
      "watermark_absent(OK: corner_contrast=211)"
    ]
  },
  "output_files": {
    "image": "jobs/job_001/avatar_550e8400.png",
    "spec_delta": "jobs/job_001/spec_delta.json"
  }
}
```

---

## Command Line Interface (CLI) Reference

```
avatar-gen [OPTIONS] COMMAND [ARGS]

Commands:
  prepare               Validate spec + prepare job bundle directory
  run                   Execute pipeline inference (local_cpu, diffusers_cuda, zeroGPU, etc.)
  validate              Validate all outputs & safety criteria in a job directory
  spec-delta            Generate per-attribute spec-delta report (NF-02)
  audit-bias            Execute demographic representation entropy audit (NF-03)
  audit-orthogonality   Measure prompt attribute bleed & orthogonality (NF-01)
  validate-spec         Pre-generation attribute grammar verification (NF-06)
  verify-watermark      Decode & verify steganographic watermark payload (NF-05)
  consent-init          Initialize cryptographic Consent Event Record (NF-04)
  verify-consent        Validate consent chain integrity in manifest (NF-04)
  revoke-consent        Revoke consent record by consent ID (NF-04)
  portability-score     Compute cross-model prompt portability score (NF-09)
  summarize-failures    Generate structured failure frequency summary (NF-07)
  bench                 Run automated execution & RAM benchmarks
```

---

## Testing & Verification

The repository maintains 100% test pass rate across 50 unit, integration, and E2E tests:

```bash
# Run complete test suite (50 tests)
python -m pytest tests/ -v

# Run unit tests only
python -m pytest tests/unit/ -v

# Run integration tests only
python -m pytest tests/integration/ -v

# Run End-to-End pipeline test
python -m pytest tests/e2e/ -v

# Run tests with coverage report
python -m pytest tests/ --cov=avatar_gen --cov-report=term-missing
```

---

## Ethics, Compliance & Governance

- **Fitzpatrick Scale Alignment**: Skin tones strictly utilize Fitzpatrick-anchored numerical/descriptive types, rejecting geographical, racial, or nationality tags.
- **Demographic Representation Protection**: `audit-bias` enforces minimum Shannon entropy across attributes to detect systemic bias in batch runs.
- **Cryptographic Consent Verification**: Mandatory cryptographic token authorization (`consent-init`) required prior to ingesting reference images.
- **Synthetic Media Labeling**: Integrated DWT/DCT steganographic watermarking (`verify-watermark`) embeds 64-bit verification payload into image frequency bands.

---

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.
