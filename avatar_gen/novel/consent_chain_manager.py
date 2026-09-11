"""
NF-04 — Cryptographic Consent Chain (C3)

Creates and verifies tamper-evident consent records using Ed25519 signatures.
Every individual-mode generation is cryptographically bound to a Consent Event
Record (CER) via a hash chain.

Gap closed: C2PA signs content after generation. No pipeline mathematically
binds a pre-generation consent record to the specific generation event.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

# ── Exit codes ────────────────────────────────────────────────────────────────
EXIT_CONSENT_CHAIN_INVALID = 4


# ── Key management ────────────────────────────────────────────────────────────

def generate_keypair(keys_dir: Path) -> tuple[Path, Path]:
    """Generate an Ed25519 keypair and write to keys_dir.

    Returns:
        (private_key_path, public_key_path)
    """
    keys_dir.mkdir(parents=True, exist_ok=True)
    private_key = Ed25519PrivateKey.generate()
    public_key  = private_key.public_key()

    priv_path = keys_dir / "private.pem"
    pub_path  = keys_dir / "public.pem"

    priv_path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    pub_path.write_bytes(
        public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    return priv_path, pub_path


def _load_private_key(priv_path: Path) -> Ed25519PrivateKey:
    return serialization.load_pem_private_key(priv_path.read_bytes(), password=None)  # type: ignore


def _load_public_key(pub_path: Path) -> Ed25519PublicKey:
    return serialization.load_pem_public_key(pub_path.read_bytes())  # type: ignore


# ── Consent Event Record ──────────────────────────────────────────────────────

def create_consent_record(
    subject_hash: str,          # SHA-256 of subject identifier (never the raw PII)
    scope: str,                 # e.g. "avatar_generation_individual_mode"
    expiry_date: str,           # ISO-8601 date
    keys_dir: Path,
    consent_dir: Path,
    witness_note: str = "",
) -> Path:
    """Create and sign a Consent Event Record (CER).

    Args:
        subject_hash:  SHA-256 hex of the subject's identifier (not raw PII).
        scope:         Permitted use scope description.
        expiry_date:   ISO-8601 date string when consent expires.
        keys_dir:      Directory containing private.pem / public.pem.
        consent_dir:   Directory to write the CER and signature.
        witness_note:  Optional human witness note.

    Returns:
        Path to consent_event.json.
    """
    consent_id = str(uuid.uuid4())
    event: dict = {
        "consent_id":   consent_id,
        "subject_hash": subject_hash,
        "scope":        scope,
        "created_at":   datetime.now(timezone.utc).isoformat(),
        "expiry_date":  expiry_date,
        "witness_note": witness_note,
        "revoked":      False,
    }

    priv_path = keys_dir / "private.pem"

    # Auto-generate keypair if absent
    if not priv_path.exists():
        generate_keypair(keys_dir)

    private_key = _load_private_key(priv_path)
    payload_bytes = json.dumps(event, sort_keys=True).encode("utf-8")
    signature_bytes = private_key.sign(payload_bytes)

    consent_dir.mkdir(parents=True, exist_ok=True)
    cer_path = consent_dir / f"consent_event_{consent_id}.json"
    sig_path = consent_dir / f"consent_event_{consent_id}.json.sig"

    cer_path.write_text(json.dumps(event, indent=2), encoding="utf-8")
    sig_path.write_bytes(signature_bytes)

    return cer_path


def _sha256_hex(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def build_generation_binding(
    consent_id: str,
    avatar_id: str,
    seed: int,
    consent_dir: Path,
    keys_dir: Path,
) -> dict:
    """Build the consent_chain dict to embed in avatar_manifest.json.

    Raises:
        SystemExit(4): If the consent record cannot be verified.
    """
    import base64

    # Find the CER
    cer_candidates = list(consent_dir.glob(f"consent_event_{consent_id}.json"))
    if not cer_candidates:
        raise SystemExit(EXIT_CONSENT_CHAIN_INVALID)
    cer_path = cer_candidates[0]
    sig_path = cer_path.with_suffix(".json.sig")

    if not sig_path.exists():
        raise SystemExit(EXIT_CONSENT_CHAIN_INVALID)

    # Verify signature
    pub_path = keys_dir / "public.pem"
    try:
        pub_key = _load_public_key(pub_path)
        payload_bytes = cer_path.read_bytes()
        sig_bytes = sig_path.read_bytes()
        # Reload JSON to normalise
        event = json.loads(payload_bytes)
        canonical = json.dumps(event, sort_keys=True).encode("utf-8")
        pub_key.verify(sig_bytes, canonical)
    except (InvalidSignature, Exception):
        raise SystemExit(EXIT_CONSENT_CHAIN_INVALID)

    # Check expiry
    expiry = datetime.fromisoformat(event["expiry_date"]).replace(tzinfo=timezone.utc)
    if datetime.now(timezone.utc) > expiry:
        raise SystemExit(EXIT_CONSENT_CHAIN_INVALID)

    # Check not revoked
    if event.get("revoked", False):
        raise SystemExit(EXIT_CONSENT_CHAIN_INVALID)

    event_hash   = _sha256_hex(json.dumps(event, sort_keys=True))
    binding_hash = _sha256_hex(f"{event_hash}:{avatar_id}:{seed}")

    return {
        "consent_event_id":   consent_id,
        "consent_event_hash": f"sha256:{event_hash}",
        "consent_event_sig":  f"ed25519:{base64.b64encode(sig_bytes).decode()}",
        "public_key_ref":     str(pub_path),
        "binding_hash":       f"sha256:{binding_hash}",
        "verified_at_generation": True,
    }


def verify_consent_chain(manifest: dict, consent_dir: Path, keys_dir: Path) -> bool:
    """Verify the consent chain in an avatar manifest.

    Returns:
        True if chain is valid, False otherwise.
    """
    import base64

    chain = manifest.get("consent_chain")
    if not chain:
        return False

    consent_id = chain.get("consent_event_id")
    if not consent_id:
        return False

    cer_candidates = list(consent_dir.glob(f"consent_event_{consent_id}.json"))
    if not cer_candidates:
        return False

    cer_path = cer_candidates[0]
    sig_path = cer_path.with_suffix(".json.sig")
    if not sig_path.exists():
        return False

    try:
        event    = json.loads(cer_path.read_text(encoding="utf-8"))
        canonical = json.dumps(event, sort_keys=True).encode("utf-8")
        pub_key   = _load_public_key(keys_dir / "public.pem")
        sig_b64   = chain["consent_event_sig"].replace("ed25519:", "")
        sig_bytes = base64.b64decode(sig_b64)
        pub_key.verify(sig_bytes, canonical)
    except Exception:
        return False

    # Verify binding hash
    event_hash    = _sha256_hex(json.dumps(event, sort_keys=True))
    avatar_id     = manifest.get("avatar_id", "")
    seed          = manifest.get("seed", 0)
    expected_bind = f"sha256:{_sha256_hex(f'{event_hash}:{avatar_id}:{seed}')}"
    return chain.get("binding_hash") == expected_bind


def revoke_consent(consent_id: str, consent_dir: Path) -> bool:
    """Mark a consent record as revoked.

    Returns:
        True if revocation succeeded.
    """
    cer_candidates = list(consent_dir.glob(f"consent_event_{consent_id}.json"))
    if not cer_candidates:
        return False
    cer_path = cer_candidates[0]
    event = json.loads(cer_path.read_text(encoding="utf-8"))
    event["revoked"] = True
    event["revoked_at"] = datetime.now(timezone.utc).isoformat()
    cer_path.write_text(json.dumps(event, indent=2), encoding="utf-8")
    return True
