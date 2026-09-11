"""
cli.py — avatar-gen command-line interface

Entry point: `avatar-gen`

Subcommands:
  prepare            Build a job bundle from a spec file
  run                Execute a local job (CPU stub or GPU)
  validate           Validate outputs in a job directory
  spec-delta         Compute per-attribute spec-delta report
  audit-bias         Run Bias Surface Auditor on a batch
  audit-orthogonality  Run Attribute Orthogonality Verifier
  validate-spec      Pre-generation spec grammar check
  verify-watermark   Decode and verify steganographic watermark
  verify-consent     Verify consent chain in a manifest
  consent-init       Create a new Consent Event Record
  revoke-consent     Revoke a consent record by ID
  portability-score  Compute Cross-Model Portability Score
  summarize-failures Print failure-type frequency table for a batch
  bench              Run automated benchmark
"""
from __future__ import annotations

import hashlib
import json
import sys
import uuid
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _spec_hash(spec_path: Path) -> str:
    return hashlib.sha256(spec_path.read_bytes()).hexdigest()[:16]


def _load_config(config_path: str | None) -> dict:
    import yaml
    if config_path:
        return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    default = Path(__file__).parent.parent / "config" / "default_config.yaml"
    if default.exists():
        return yaml.safe_load(default.read_text(encoding="utf-8"))
    return {}


# ── Main group ────────────────────────────────────────────────────────────────

@click.group()
@click.version_option(package_name="avatar-gen")
@click.option("--config", "config_path", default=None, envvar="AVATAR_GEN_CONFIG",
              help="Path to config YAML (default: config/default_config.yaml)")
@click.pass_context
def main(ctx: click.Context, config_path: str | None) -> None:
    """avatar-gen — Open-Source AI Human-Avatar Generation Pipeline\n
    IncuBrix SASTRA 2027 · Track 02"""
    ctx.ensure_object(dict)
    ctx.obj["config"] = _load_config(config_path)


# ── prepare ───────────────────────────────────────────────────────────────────

@main.command("prepare")
@click.option("--spec", "spec_path", required=True, type=click.Path(exists=True),
              help="Path to avatar spec YAML/JSON")
@click.option("--output-dir", "output_dir", required=True, type=click.Path(),
              help="Directory to write the job bundle")
@click.option("--force", is_flag=True, default=False,
              help="Skip BLOCK-level grammar issues (downgrade to WARN)")
@click.pass_context
def cmd_prepare(ctx, spec_path: str, output_dir: str, force: bool) -> None:
    """Validate a spec and prepare a portable job bundle."""
    from avatar_gen.novel.attribute_grammar_validator import validate_spec_file
    from avatar_gen.prompt_builder import build_prompts
    from avatar_gen.spec_parser import load_spec

    spec_path_obj = Path(spec_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Grammar check
    console.print(f"[cyan]Validating spec:[/cyan] {spec_path}")
    grammar_report = validate_spec_file(spec_path_obj, force_override=force,
                                        output_path=out / "spec_validation.json")
    if grammar_report.has_blocks and not force:
        console.print(f"[red]BLOCKED:[/red]\n{grammar_report.summary()}")
        sys.exit(2)
    if grammar_report.issues:
        console.print(f"[yellow]Grammar warnings:[/yellow]\n{grammar_report.summary()}")

    # Load spec
    spec = load_spec(spec_path_obj)
    pos_prompt, neg_prompt = build_prompts(spec)

    # Write job bundle
    import yaml

    yaml_dump = yaml.dump(spec.model_dump(), allow_unicode=True)
    (out / "spec.yaml").write_text(yaml_dump, encoding="utf-8")

    job_params = {
        "job_id":           str(uuid.uuid4()),
        "spec_hash":        _spec_hash(spec_path_obj),
        "positive_prompt":  pos_prompt,
        "negative_prompt":  neg_prompt,
        "seed":             spec.seed,
        "width":            spec.resolution_hint,
        "height":           spec.resolution_hint,
        "model":            spec.model_override or (
            ctx.obj["config"].get("pipeline", {}).get("default_model", "stub")
        ),
    }
    (out / "job_params.json").write_text(json.dumps(job_params, indent=2), encoding="utf-8")

    console.print(f"[green][OK] Job bundle prepared:[/green] {out}")
    console.print(f"   Positive prompt: {pos_prompt[:120]}...")


# ── run ───────────────────────────────────────────────────────────────────────

@main.command("run")
@click.option("--job-dir", "job_dir", required=True, type=click.Path(exists=True))
@click.option("--compute-route", "compute_route", default="local_cpu",
              help="local_cpu | free_ai | diffusers_cuda | kaggle | lightning | hf_zerogpu")
@click.option("--steps", default=30, show_default=True)
@click.option("--guidance-scale", "guidance_scale", default=7.5, show_default=True)
@click.option("--adaptive-resolution", "adaptive_res", is_flag=True, default=False,
              help="Enable adaptive resolution ladder (NF-08)")
@click.pass_context
def cmd_run(ctx, job_dir: str, compute_route: str, steps: int,
            guidance_scale: float, adaptive_res: bool) -> None:
    """Execute a prepared job bundle and generate an avatar."""
    from avatar_gen.inference_client import get_client
    from avatar_gen.novel.failure_taxonomy_classifier import (
        FailureCode,
        record_exception,
        record_failure,
    )
    from avatar_gen.novel.spec_delta_reporter import compute_spec_delta
    from avatar_gen.novel.steganographic_watermarker import embed_watermark
    from avatar_gen.provenance_writer import patch_manifest, write_manifest
    from avatar_gen.safety_checker import check_image, check_prompt
    from avatar_gen.spec_parser import load_spec

    job = Path(job_dir)
    params = json.loads((job / "job_params.json").read_text(encoding="utf-8"))
    spec   = load_spec(job / "spec.yaml")

    pos_prompt = params["positive_prompt"]
    neg_prompt = params["negative_prompt"]
    seed       = params["seed"]
    width      = params.get("width", 512)
    height     = params.get("height", 512)
    model_id   = params.get("model", "stub")

    # Prompt safety check
    safe, flagged = check_prompt(pos_prompt, neg_prompt)
    if not safe:
        fr_path = job / "failure_record.json"
        record_failure(FailureCode.SAFETY_REFUSAL, f"Flagged terms: {flagged}", fr_path,
                       spec_hash=params.get("spec_hash"), seed=seed)
        console.print(f"[red]SAFETY_REFUSAL:[/red] Prompt flagged: {flagged}")
        sys.exit(3)

    # Inference
    client = get_client(compute_route, model_id=model_id) if compute_route != "local_cpu" \
             else get_client("local_cpu")

    try:
        if adaptive_res:
            from avatar_gen.novel.adaptive_resolution_ladder import run_adaptive_ladder
            cfg = ctx.obj["config"].get("adaptive_resolution", {})
            ladder = cfg.get("ladder", [256, 384, 512, 768])
            max_rung = cfg.get("max_rung", width)

            def _gen_fn(w, h):
                res = client.generate(pos_prompt, neg_prompt, seed, w, h, steps, guidance_scale)
                return res.image

            img, ladder_result = run_adaptive_ladder(
                _gen_fn, spec_summary=pos_prompt[:50],
                ladder=ladder, max_rung=max_rung,
                output_path=job / "ladder_result.json",
            )

            from avatar_gen.inference_client import InferenceResult

            result = InferenceResult(
                image=img,
                model=model_id,
                model_revision="auto",
                compute_route=compute_route,
                runtime_sec=0.0,
                steps=steps,
                guidance_scale=guidance_scale,
                seed=seed,
            )
            ladder_fields = ladder_result.to_manifest_fields()
        else:
            result = client.generate(
                pos_prompt, neg_prompt, seed, width, height, steps, guidance_scale
            )
            img = result.image
            ladder_fields = {}

    except Exception as exc:
        fr = record_exception(exc, job / "failure_record.json",
                              spec_hash=params.get("spec_hash"), seed=seed, model=model_id)
        console.print(f"[red]Inference failed:[/red] {fr.failure_code.value} - {exc}")
        sys.exit(1)

    # Safety check on output
    safety_result = check_image(img, log_path=job / "safety_log.jsonl")
    if not safety_result.passed:
        from avatar_gen.novel.failure_taxonomy_classifier import classify_safety_flags
        code = classify_safety_flags(safety_result.flags)
        record_failure(code, f"Safety flags: {safety_result.flags}", job / "failure_record.json",
                       spec_hash=params.get("spec_hash"), seed=seed, model=model_id)
        console.print(f"[red]Safety check failed:[/red] {safety_result.flags}")
        sys.exit(3)

    # Watermark
    wm_metadata: dict | None = None
    wm_result = None
    img_path = (job / f"avatar_{spec.avatar_id[:8]}.png").resolve()
    try:
        img_wm, wm_result = embed_watermark(img, spec.avatar_id, model_id)
        img_wm.save(img_path)
        wm_metadata = wm_result.to_dict()
    except Exception:
        img.save(img_path)

    # Spec-delta
    spec_delta_path = job / "spec_delta.json"
    compute_spec_delta(spec, img_path, output_path=spec_delta_path)

    # Manifest
    manifest_path = write_manifest(
        spec=spec, result=result, safety=safety_result,
        image_path=img_path, job_dir=job,
        watermark=wm_metadata,
        **ladder_fields,
        spec_delta_path=spec_delta_path,
    )
    patch_manifest(manifest_path, {
        "prompt_generated": pos_prompt,
        "negative_prompt":  neg_prompt,
    })

    console.print(f"[green][OK] Avatar generated:[/green] {img_path}")
    console.print(f"   Manifest: {manifest_path}")
    if wm_result:
        ber = wm_result.ber_self_check
        survives = wm_result.survives_jpeg_q70
        console.print(f"   Watermark BER: {ber} | survives JPEG Q70: {survives}")


# ── validate ──────────────────────────────────────────────────────────────────

@main.command("validate")
@click.option("--job-dir", "job_dir", required=True, type=click.Path(exists=True))
@click.option("--expect-watermark/--no-watermark", "expect_watermark", default=True)
@click.option("--expect-spec-delta/--no-spec-delta", "expect_spec_delta", default=True)
def cmd_validate(job_dir: str, expect_watermark: bool, expect_spec_delta: bool) -> None:
    """Validate all outputs in a job directory."""
    from avatar_gen.output_validator import validate_job

    report = validate_job(job_dir, expect_watermark=expect_watermark,
                          expect_spec_delta=expect_spec_delta)
    console.print(report.summary())
    sys.exit(0 if report.passed else 1)


# ── spec-delta ────────────────────────────────────────────────────────────────

@main.command("spec-delta")
@click.option("--job-dir", "job_dir", required=True, type=click.Path(exists=True))
def cmd_spec_delta(job_dir: str) -> None:
    """Compute per-attribute spec-delta report for a generated avatar."""
    import json

    from avatar_gen.novel.spec_delta_reporter import compute_spec_delta
    from avatar_gen.spec_parser import load_spec

    job = Path(job_dir)
    manifest = json.loads((job / "avatar_manifest.json").read_text(encoding="utf-8"))
    spec = load_spec(job / "spec.yaml")
    img_path = Path(manifest["output_files"]["image"])
    out_path = job / "spec_delta.json"

    report = compute_spec_delta(spec, img_path, output_path=out_path)

    table = Table(title="Spec-Delta Report", show_lines=True)
    table.add_column("Attribute")
    table.add_column("Requested")
    table.add_column("Detected")
    table.add_column("Match")
    table.add_column("Delta")

    for v in report.attribute_verdicts:
        colour = {"FULL": "green", "PARTIAL": "yellow", "MISSING": "red"}.get(v.match, "white")
        table.add_row(v.attribute, v.requested, v.detected,
                      f"[{colour}]{v.match}[/{colour}]", v.delta or "-")

    console.print(table)
    status = "[yellow]LOW ADHERENCE[/yellow]" if report.low_adherence else "[green]OK[/green]"
    console.print(f"Overall adherence: {report.overall_adherence_pct:.1f}% - {status}")


# ── audit-bias ────────────────────────────────────────────────────────────────

@main.command("audit-bias")
@click.option("--batch-dir", "batch_dir", required=True, type=click.Path(exists=True))
@click.option("--output", "output_path", default=None, type=click.Path())
def cmd_audit_bias(batch_dir: str, output_path: str | None) -> None:
    """Run Bias Surface Auditor on a batch directory."""
    from avatar_gen.novel.bias_surface_auditor import audit_batch_from_dir

    out = Path(output_path) if output_path else Path(batch_dir) / "bias_audit.json"
    report = audit_batch_from_dir(Path(batch_dir), output_path=out)

    title = f"Bias Audit - {report.batch_id} ({report.n_avatars} avatars)"
    table = Table(title=title, show_lines=True)
    table.add_column("Attribute")
    table.add_column("Entropy (bits)")
    table.add_column("Distinct Values")
    table.add_column("Status")

    for rec in report.attribute_entropy:
        colour = "green" if rec.status == "OK" else "yellow"
        table.add_row(
            rec.attribute,
            f"{rec.entropy_bits:.3f}",
            ", ".join(rec.distinct_values[:5]) + ("…" if len(rec.distinct_values) > 5 else ""),
            f"[{colour}]{rec.status}[/{colour}]",
        )

    console.print(table)
    if report.recommendations:
        console.print("[yellow]Recommendations:[/yellow]")
        for r in report.recommendations:
            console.print(f"  • {r}")
    if report.clone_pairs:
        console.print(f"[yellow]Clone pairs detected:[/yellow] {report.clone_pairs}")


# ── audit-orthogonality ───────────────────────────────────────────────────────

@main.command("audit-orthogonality")
@click.option("--baseline", "baseline_dir", required=True, type=click.Path(exists=True))
@click.option("--variant", "variant_dir", required=True, type=click.Path(exists=True))
@click.option("--changed-attr", "changed_attr", required=True)
@click.option("--output", "output_path", default=None, type=click.Path())
def cmd_audit_orthogonality(baseline_dir: str, variant_dir: str,
                             changed_attr: str, output_path: str | None) -> None:
    """Compare two job dirs; detect attribute bleed (NF-01)."""
    from avatar_gen.novel.attribute_orthogonality_verifier import verify_orthogonality_jobs

    out = Path(output_path) if output_path else None
    result = verify_orthogonality_jobs(Path(baseline_dir), Path(variant_dir),
                                       changed_attr, output_path=out)

    table = Table(title=f"Orthogonality - changed: {changed_attr}", show_lines=True)
    table.add_column("Attribute")
    table.add_column("Score")
    table.add_column("Status")

    for attr, score in result.scores.items():
        is_target = attr == changed_attr
        warn      = not is_target and score < 0.85
        colour    = "cyan" if is_target else ("red" if warn else "green")
        status    = "TARGET" if is_target else ("BLEED_WARNING" if warn else "OK")
        table.add_row(attr, f"{score:.4f}", f"[{colour}]{status}[/{colour}]")

    console.print(table)
    console.print("[green]No bleed detected.[/green]" if result.overall_ok
                  else f"[red]Bleed in:[/red] {result.bleed_warnings}")


# ── validate-spec ─────────────────────────────────────────────────────────────

@main.command("validate-spec")
@click.option("--spec", "spec_path", required=True, type=click.Path(exists=True))
@click.option("--force", is_flag=True, default=False, help="Downgrade BLOCK → WARN")
@click.option("--output", "output_path", default=None, type=click.Path())
def cmd_validate_spec(spec_path: str, force: bool, output_path: str | None) -> None:
    """Pre-generation attribute grammar check (NF-06)."""
    from avatar_gen.novel.attribute_grammar_validator import validate_spec_file

    out = Path(output_path) if output_path else None
    report = validate_spec_file(Path(spec_path), force_override=force, output_path=out)
    console.print(report.summary())
    sys.exit(0 if report.valid else 2)


# ── verify-watermark ──────────────────────────────────────────────────────────

@main.command("verify-watermark")
@click.option("--image", "image_path", default=None, type=click.Path(),
              help="Single image to verify")
@click.option("--batch-dir", "batch_dir", default=None, type=click.Path(),
              help="Batch directory — verify all images")
@click.option("--registry", "registry_dir", default=None, type=click.Path(),
              help="Directory to search for matching manifests")
def cmd_verify_watermark(image_path: str | None, batch_dir: str | None,
                          registry_dir: str | None) -> None:
    """Decode and verify steganographic watermarks (NF-05)."""
    from PIL import Image

    from avatar_gen.novel.steganographic_watermarker import verify_watermark

    registry = Path(registry_dir) if registry_dir else None
    paths: list[Path] = []

    if image_path:
        paths.append(Path(image_path))
    if batch_dir:
        paths.extend(sorted(Path(batch_dir).rglob("*.png")) +
                     sorted(Path(batch_dir).rglob("*.jpg")))

    if not paths:
        console.print("[red]Provide --image or --batch-dir[/red]")
        sys.exit(1)

    table = Table(title="Watermark Verification", show_lines=True)
    table.add_column("Image")
    table.add_column("Verified")
    table.add_column("Payload (hex)")
    table.add_column("Matched Avatar ID")

    for p in paths:
        img = Image.open(p).convert("RGB")
        r = verify_watermark(img, manifest_registry_dir=registry)
        colour = "green" if r.verified else "red"
        table.add_row(
            p.name,
            f"[{colour}]{'YES' if r.verified else 'NO'}[/{colour}]",
            r.payload_hex or "-",
            r.matched_avatar_id or "-",
        )

    console.print(table)


# ── consent-init ──────────────────────────────────────────────────────────────

@main.command("consent-init")
@click.option("--subject-hash", "subject_hash", required=True,
              help="SHA-256 hex of subject identifier (NOT raw PII)")
@click.option("--scope", default="avatar_generation_individual_mode", show_default=True)
@click.option("--expiry", "expiry_date", required=True,
              help="Expiry date in YYYY-MM-DD format")
@click.option("--witness", "witness_note", default="", help="Optional witness note")
@click.option("--keys-dir", "keys_dir", default="keys", type=click.Path(), show_default=True)
@click.option("--consent-dir", "consent_dir", default="consents",
              type=click.Path(), show_default=True)
def cmd_consent_init(subject_hash: str, scope: str, expiry_date: str,
                     witness_note: str, keys_dir: str, consent_dir: str) -> None:
    """Create a new cryptographic Consent Event Record (NF-04)."""
    from avatar_gen.novel.consent_chain_manager import create_consent_record

    cer_path = create_consent_record(
        subject_hash=subject_hash, scope=scope, expiry_date=expiry_date,
        keys_dir=Path(keys_dir), consent_dir=Path(consent_dir), witness_note=witness_note,
    )
    console.print(f"[green][OK] Consent record created:[/green] {cer_path}")
    import json
    event = json.loads(cer_path.read_text(encoding="utf-8"))
    console.print(f"   Consent ID: {event['consent_id']}")
    console.print(f"   Expires:    {event['expiry_date']}")


# ── verify-consent ────────────────────────────────────────────────────────────

@main.command("verify-consent")
@click.option("--manifest", "manifest_path", required=True, type=click.Path(exists=True))
@click.option("--keys-dir", "keys_dir", default="keys", type=click.Path(), show_default=True)
@click.option("--consent-dir", "consent_dir", default="consents",
              type=click.Path(), show_default=True)
def cmd_verify_consent(manifest_path: str, keys_dir: str, consent_dir: str) -> None:
    """Verify the consent chain embedded in an avatar manifest (NF-04)."""
    from avatar_gen.novel.consent_chain_manager import verify_consent_chain

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    ok = verify_consent_chain(manifest, Path(consent_dir), Path(keys_dir))
    if ok:
        console.print("[green][OK] Consent chain VALID[/green]")
    else:
        console.print("[red][FAIL] Consent chain INVALID[/red]")
        sys.exit(4)


# ── revoke-consent ────────────────────────────────────────────────────────────

@main.command("revoke-consent")
@click.option("--consent-id", "consent_id", required=True)
@click.option("--consent-dir", "consent_dir", default="consents",
              type=click.Path(), show_default=True)
def cmd_revoke_consent(consent_id: str, consent_dir: str) -> None:
    """Revoke a consent record by ID (NF-04)."""
    from avatar_gen.novel.consent_chain_manager import revoke_consent

    ok = revoke_consent(consent_id, Path(consent_dir))
    if ok:
        console.print(f"[green][OK] Consent {consent_id} revoked.[/green]")
    else:
        console.print(f"[red]Consent record {consent_id} not found.[/red]")
        sys.exit(1)


# ── portability-score ─────────────────────────────────────────────────────────

@main.command("portability-score")
@click.option("--job-a", "job_a", required=True, type=click.Path(exists=True))
@click.option("--model-a", "model_a", required=True)
@click.option("--job-b", "job_b", required=True, type=click.Path(exists=True))
@click.option("--model-b", "model_b", required=True)
@click.option("--output", "output_path", default=None, type=click.Path())
def cmd_portability_score(job_a: str, model_a: str, job_b: str,
                           model_b: str, output_path: str | None) -> None:
    """Compute Cross-Model Prompt Portability Score (NF-09)."""
    from avatar_gen.novel.cross_model_portability_scorer import compute_portability_from_dirs

    out = Path(output_path) if output_path else Path(job_a) / "portability_report.json"
    report = compute_portability_from_dirs(Path(job_a), model_a, Path(job_b), model_b,
                                           output_path=out)
    console.print(report.summary())


# ── summarize-failures ────────────────────────────────────────────────────────

@main.command("summarize-failures")
@click.option("--batch-dir", "batch_dir", required=True, type=click.Path(exists=True))
def cmd_summarize_failures(batch_dir: str) -> None:
    """Print failure-type frequency table for a batch (NF-07)."""
    from avatar_gen.novel.failure_taxonomy_classifier import summarize_failures

    counts = summarize_failures(Path(batch_dir))
    if not counts:
        console.print("[green]No failures found.[/green]")
        return

    table = Table(title=f"Failure Summary - {batch_dir}", show_lines=True)
    table.add_column("Failure Code")
    table.add_column("Count")

    for code, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        table.add_row(code, str(cnt))

    console.print(table)


# ── bench ─────────────────────────────────────────────────────────────────────

@main.command("bench")
@click.option("--spec", "spec_path", required=True, type=click.Path(exists=True))
@click.option("--repetitions", default=3, show_default=True)
@click.option("--warmup", "warmup_runs", default=1, show_default=True)
@click.option("--compute-route", "compute_route", default="local_cpu", show_default=True)
@click.option("--output", "output_path", default="benchmark_result.json", show_default=True)
@click.pass_context
def cmd_bench(ctx, spec_path: str, repetitions: int, warmup_runs: int,
              compute_route: str, output_path: str) -> None:
    """Run automated benchmark for a spec file."""

    from avatar_gen.benchmark import run_benchmark
    from avatar_gen.inference_client import get_client
    from avatar_gen.prompt_builder import build_prompts
    from avatar_gen.safety_checker import check_image
    from avatar_gen.spec_parser import load_spec

    spec = load_spec(Path(spec_path))
    pos_prompt, neg_prompt = build_prompts(spec)
    client = get_client(compute_route)

    def _generate_one() -> dict:
        result = client.generate(
            pos_prompt, neg_prompt, spec.seed,
            spec.resolution_hint, spec.resolution_hint,
        )
        safety = check_image(result.image)
        return {"success": safety.passed, "runtime_sec": result.runtime_sec}

    console.print(f"[cyan]Benchmarking:[/cyan] {spec_path} × {repetitions} reps "
                  f"(+{warmup_runs} warmup) on [yellow]{compute_route}[/yellow]")

    bench_result = run_benchmark(
        _generate_one,
        repetitions=repetitions,
        warmup_runs=warmup_runs,
        output_path=Path(output_path),
    )
    console.print(bench_result.summary_table())
    console.print(f"[green]Results written to:[/green] {output_path}")


if __name__ == "__main__":
    main()
