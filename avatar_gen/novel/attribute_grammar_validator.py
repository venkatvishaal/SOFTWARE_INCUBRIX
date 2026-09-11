"""
NF-06 — Attribute Grammar Validator

Pre-generation semantic validator that checks avatar specs against the
Avatar Attribute Grammar (AAG) defined in config/attribute_grammar.yaml.

Blocks contradictory specs (BLOCK), flags stereotype-risk combinations
(STEREOTYPE_RISK), and warns on improbable combinations (WARN).

Gap closed: No existing pipeline validates semantic coherence of specs
before spending GPU budget. ComfyUI/SD/ControlNet accept any text prompt.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

from avatar_gen.spec_parser import AvatarSpec

Severity = Literal["BLOCK", "STEREOTYPE_RISK", "WARN"]

_DEFAULT_GRAMMAR = Path(__file__).parent.parent.parent / "config" / "attribute_grammar.yaml"
_GRAMMAR: dict | None = None


def _load_grammar(grammar_path: Path | None = None) -> dict:
    global _GRAMMAR
    path = grammar_path or _DEFAULT_GRAMMAR
    if _GRAMMAR is None or grammar_path:
        _GRAMMAR = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _GRAMMAR


@dataclass
class GrammarIssue:
    rule_id: str
    severity: Severity
    detail: str
    suggestion: str | None = None

    def to_dict(self) -> dict:
        d = {"rule": self.rule_id, "severity": self.severity, "detail": self.detail}
        if self.suggestion:
            d["suggestion"] = self.suggestion
        return d


@dataclass
class SpecValidationReport:
    spec_file: str
    valid: bool    # True only when zero BLOCK-level issues
    issues: list[GrammarIssue] = field(default_factory=list)

    @property
    def has_blocks(self) -> bool:
        return any(i.severity == "BLOCK" for i in self.issues)

    @property
    def has_stereotype_risk(self) -> bool:
        return any(i.severity == "STEREOTYPE_RISK" for i in self.issues)

    def to_dict(self) -> dict:
        return {
            "spec_file": self.spec_file,
            "valid":     self.valid,
            "issues":    [i.to_dict() for i in self.issues],
        }

    def summary(self) -> str:
        if self.valid and not self.issues:
            return "✅ Spec is valid — no issues found."
        lines = ["Spec validation issues:"]
        for issue in self.issues:
            icon = {"BLOCK": "🚫", "STEREOTYPE_RISK": "⚠️", "WARN": "⚠"}.get(issue.severity, "?")
            lines.append(f"  {icon} [{issue.severity}] {issue.rule_id}: {issue.detail}")
            if issue.suggestion:
                lines.append(f"       💡 {issue.suggestion}")
        return "\n".join(lines)


# ── Rule evaluators ───────────────────────────────────────────────────────────

def _get_field(spec: AvatarSpec, field_path: str) -> str:
    """Retrieve a (possibly nested) field value from AvatarSpec."""
    parts = field_path.split(".")
    obj: object = spec
    for p in parts:
        if hasattr(obj, p):
            obj = getattr(obj, p)
        elif isinstance(obj, dict):
            obj = obj.get(p, "")
        else:
            return ""
    return str(obj) if obj is not None else ""


def _condition_matches(spec: AvatarSpec, condition: dict) -> bool:
    """Check if a condition dict matches the spec."""
    field_val = _get_field(spec, condition["field"])
    expected  = condition["value"]
    if isinstance(expected, list):
        return field_val in expected
    return field_val == expected


def _check_rule(spec: AvatarSpec, rule: dict) -> bool:
    """Return True if ALL conditions in the rule match."""
    return all(_condition_matches(spec, cond) for cond in rule["conditions"])


# ── Public API ────────────────────────────────────────────────────────────────

def validate_spec(
    spec: AvatarSpec,
    spec_file: str = "<programmatic>",
    grammar_path: Path | None = None,
    force_override: bool = False,
    output_path: Path | None = None,
) -> SpecValidationReport:
    """Validate a spec against the Attribute Grammar.

    Args:
        spec:           AvatarSpec to validate.
        spec_file:      Display name for the spec (used in report).
        grammar_path:   Path to custom grammar YAML (uses default if None).
        force_override: If True, downgrade BLOCK → WARN (allows generation).
        output_path:    Optional path to write spec_validation.json.

    Returns:
        SpecValidationReport — valid=False if any BLOCK-level issue found
        (unless force_override=True).
    """
    grammar = _load_grammar(grammar_path)
    issues: list[GrammarIssue] = []

    # BLOCK rules
    for rule in grammar.get("block_rules", []):
        if _check_rule(spec, rule):
            sev: Severity = "WARN" if force_override else "BLOCK"
            issues.append(GrammarIssue(
                rule_id=rule["id"],
                severity=sev,
                detail=rule["description"],
            ))

    # WARN rules
    for rule in grammar.get("warn_rules", []):
        if _check_rule(spec, rule):
            issues.append(GrammarIssue(
                rule_id=rule["id"],
                severity="WARN",
                detail=rule["description"],
            ))

    # STEREOTYPE_RISK rules
    for rule in grammar.get("stereotype_risk_rules", []):
        if _check_rule(spec, rule):
            issues.append(GrammarIssue(
                rule_id=rule["id"],
                severity="STEREOTYPE_RISK",
                detail=rule["description"].strip(),
                suggestion=rule.get("suggestion"),
            ))

    has_block = any(i.severity == "BLOCK" for i in issues)
    report = SpecValidationReport(
        spec_file=spec_file,
        valid=not has_block,
        issues=issues,
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    return report


def validate_spec_file(
    spec_path: Path,
    grammar_path: Path | None = None,
    force_override: bool = False,
    output_path: Path | None = None,
) -> SpecValidationReport:
    """Convenience: load a spec file and validate it."""
    from avatar_gen.spec_parser import load_spec
    spec = load_spec(spec_path)
    return validate_spec(spec, str(spec_path), grammar_path, force_override, output_path)
