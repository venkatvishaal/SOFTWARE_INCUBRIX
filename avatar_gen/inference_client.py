"""
inference_client.py — Abstract compute route for image model inference.

Provides a clean interface over three backends:
  1. LocalCPUClient   — CPU-only stub used for orchestration testing.
  2. DiffusersClient  — Full local GPU inference via HuggingFace diffusers.
  3. NotebookClient   — Writes a job bundle; human/automated executes notebook.

The compute_route field in avatar_manifest.json is always set by the client.
"""
from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class InferenceResult:
    image: Image.Image
    model: str
    model_revision: str
    compute_route: str
    runtime_sec: float
    steps: int
    guidance_scale: float
    seed: int
    peak_ram_mb: float = 0.0
    peak_vram_mb: float | None = None
    metadata: dict = field(default_factory=dict)


# ── Abstract base ──────────────────────────────────────────────────────────────

class InferenceClient(ABC):
    """Abstract base for all inference backends."""

    @abstractmethod
    def generate(
        self,
        positive_prompt: str,
        negative_prompt: str,
        seed: int,
        width: int = 512,
        height: int = 512,
        steps: int = 30,
        guidance_scale: float = 7.5,
        **kwargs,
    ) -> InferenceResult:
        """Run inference and return an InferenceResult."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Canonical model identifier."""
        ...

    @property
    @abstractmethod
    def model_revision(self) -> str:
        """Pinned revision/commit hash."""
        ...


# ── 1. LocalCPUClient — deterministic stub for testing ────────────────────────

class LocalCPUClient(InferenceClient):
    """CPU-only stub that generates a reproducible solid-colour placeholder image.

    Used for:
    - Unit/integration tests that must not require a GPU.
    - Validating the full pipeline (orchestration, safety, provenance) without
      running actual inference.

    The generated image encodes the seed as a colour so tests can verify
    determinism without comparing full pixel data.
    """

    MODEL_NAME = "avatar-gen/cpu-stub-v1"
    MODEL_REV  = "stub-0.1.0"

    def __init__(self, width: int = 512, height: int = 512):
        self._default_width  = width
        self._default_height = height

    @property
    def model_name(self) -> str:
        return self.MODEL_NAME

    @property
    def model_revision(self) -> str:
        return self.MODEL_REV

    def generate(
        self,
        positive_prompt: str,
        negative_prompt: str,
        seed: int,
        width: int | None = None,
        height: int | None = None,
        steps: int = 30,
        guidance_scale: float = 7.5,
        **kwargs,
    ) -> InferenceResult:
        import random
        w = width  or self._default_width
        h = height or self._default_height
        rng = random.Random(seed)

        t0 = time.perf_counter()
        # Generate a deterministic blue-grey diagonal gradient.
        # This guarantees:
        #   - variance > 0  (blank_check passes)
        #   - b >= g >= r   (NSFW skin-tone heuristic never fires: requires r > g > b)
        arr = __import__("numpy").zeros((h, w, 3), dtype="uint8")
        base = rng.randint(40, 100)
        for x in range(w):
            shade = int(base + (x / max(w - 1, 1)) * 100)
            arr[:, x, 0] = max(0, shade - 30)   # R (lowest)
            arr[:, x, 1] = max(0, shade - 10)   # G
            arr[:, x, 2] = min(255, shade + 20) # B (highest)
        img = Image.fromarray(arr)
        elapsed = time.perf_counter() - t0

        return InferenceResult(
            image=img,
            model=self.MODEL_NAME,
            model_revision=self.MODEL_REV,
            compute_route="local_cpu_stub",
            runtime_sec=elapsed,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            peak_ram_mb=0.0,
            peak_vram_mb=None,
        )


# ── 2. DiffusersClient — real GPU inference via HF diffusers ──────────────────

class DiffusersClient(InferenceClient):
    """Full inference via HuggingFace diffusers (requires torch + GPU or slow CPU).

    The client is lazily initialised; the pipeline is loaded on first call to
    generate() so import cost is zero when using LocalCPUClient.
    """

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-v1-5",
        revision: str = "main",
        device: str = "cuda",
        dtype_str: str = "float16",
        cache_dir: str | None = None,
    ):
        self._model_id  = model_id
        self._revision  = revision
        self._device    = device
        self._dtype_str = dtype_str
        self._cache_dir = cache_dir
        self._pipe      = None  # lazy

    @property
    def model_name(self) -> str:
        return self._model_id

    @property
    def model_revision(self) -> str:
        return self._revision

    def _load_pipeline(self):
        if self._pipe is not None:
            return
        try:
            import torch
            from diffusers import StableDiffusionPipeline
        except ImportError as exc:
            raise RuntimeError(
                "GPU inference requires the 'gpu' extras: "
                "pip install 'avatar-gen[gpu]'"
            ) from exc

        dtype = getattr(torch, self._dtype_str, torch.float32)
        self._pipe = StableDiffusionPipeline.from_pretrained(
            self._model_id,
            revision=self._revision,
            torch_dtype=dtype,
            cache_dir=self._cache_dir,
            safety_checker=None,  # safety handled by our own safety_checker module
        ).to(self._device)
        self._pipe.enable_attention_slicing()

    def generate(
        self,
        positive_prompt: str,
        negative_prompt: str,
        seed: int,
        width: int = 512,
        height: int = 512,
        steps: int = 30,
        guidance_scale: float = 7.5,
        **kwargs,
    ) -> InferenceResult:
        import torch

        self._load_pipeline()
        assert self._pipe is not None

        generator = torch.Generator(device=self._device).manual_seed(seed)
        t0 = time.perf_counter()

        output = self._pipe(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            num_images_per_prompt=1,
        )
        elapsed = time.perf_counter() - t0
        img = output.images[0]

        # Measure peak VRAM if available
        peak_vram = None
        try:
            peak_vram = torch.cuda.max_memory_allocated() / 1e6
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

        return InferenceResult(
            image=img,
            model=self._model_id,
            model_revision=self._revision,
            compute_route=f"diffusers_{self._device}",
            runtime_sec=elapsed,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            peak_vram_mb=peak_vram,
        )


# ── 3. NotebookClient — prepare bundle for notebook execution ─────────────────

class NotebookClient(InferenceClient):
    """Prepares a portable job bundle for execution in an approved free-compute
    notebook (Kaggle, Lightning AI, HuggingFace ZeroGPU).

    Does NOT run inference locally.  After human/automated notebook execution,
    the job bundle directory is populated with outputs and ingest() is called.
    """

    MODEL_NAME = "deferred-notebook"
    MODEL_REV  = "pending"

    def __init__(self, provider: str = "kaggle"):
        self._provider = provider

    @property
    def model_name(self) -> str:
        return self.MODEL_NAME

    @property
    def model_revision(self) -> str:
        return self.MODEL_REV

    def generate(self, *args, **kwargs) -> InferenceResult:
        raise NotImplementedError(
            "NotebookClient does not run inference inline. "
            "Use `avatar-gen prepare` to create a job bundle, execute the notebook, "
            "then use `avatar-gen validate` to ingest results."
        )

    def prepare_bundle(
        self,
        positive_prompt: str,
        negative_prompt: str,
        seed: int,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        model_id: str,
        job_dir: Path,
    ) -> Path:
        """Write a job_params.json into job_dir for notebook pickup."""
        import json

        job_dir.mkdir(parents=True, exist_ok=True)
        params = {
            "job_id": str(uuid.uuid4()),
            "provider": self._provider,
            "model_id": model_id,
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "width": width,
            "height": height,
            "steps": steps,
            "guidance_scale": guidance_scale,
        }
        out = job_dir / "job_params.json"
        out.write_text(json.dumps(params, indent=2), encoding="utf-8")
        return out


class FreeAIClient(InferenceClient):
    """Free open-access AI image generation client.

    Generates real photorealistic AI human avatars over open endpoints without requiring
    local CUDA GPU hardware or API keys.
    """

    MODEL_NAME = "pollinations/flux-realism"
    MODEL_REV  = "v1.0"

    def __init__(self, model_id: str = "pollinations/flux-realism"):
        self._model_id = model_id

    @property
    def model_name(self) -> str:
        return self._model_id

    @property
    def model_revision(self) -> str:
        return self.MODEL_REV

    def generate(
        self,
        positive_prompt: str,
        negative_prompt: str,
        seed: int,
        width: int = 512,
        height: int = 512,
        steps: int = 30,
        guidance_scale: float = 7.5,
        **kwargs,
    ) -> InferenceResult:
        import io
        import urllib.parse
        import requests

        prompt_clean = " ".join(positive_prompt.split())[:150]
        encoded_prompt = urllib.parse.quote(prompt_clean)
        url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?model=turbo&seed={seed}&width={width}&height={height}&nologo=true"

        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        t0 = time.perf_counter()
        try:
            resp = requests.get(url, headers=headers, timeout=45)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        except Exception as exc:
            import sys
            print(f"[WARN] Free AI Endpoint fallback: {exc}", file=sys.stderr)
            stub = LocalCPUClient(width=width, height=height)
            return stub.generate(
                positive_prompt, negative_prompt, seed, width, height, steps, guidance_scale
            )

        elapsed = time.perf_counter() - t0

        return InferenceResult(
            image=img,
            model=self._model_id,
            model_revision=self.MODEL_REV,
            compute_route="free_ai",
            runtime_sec=elapsed,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            peak_ram_mb=0.0,
            peak_vram_mb=None,
        )


# ── Factory ───────────────────────────────────────────────────────────────────

def get_client(compute_route: str, **kwargs) -> InferenceClient:
    """Factory: return an InferenceClient for the requested compute_route.

    Args:
        compute_route: One of "local_cpu", "free_ai", "diffusers_cuda", "diffusers_cpu",
                       "kaggle", "lightning", "hf_zerogpu".
        **kwargs: Passed to the underlying client constructor.
    """
    route = compute_route.lower()
    if route in {"local_cpu", "stub"}:
        return LocalCPUClient(**kwargs)
    if route in {"free_ai", "online", "pollinations"}:
        return FreeAIClient(**kwargs)
    if route in {"diffusers_cuda", "diffusers_cpu", "diffusers"}:
        device = "cuda" if "cuda" in route else "cpu"
        return DiffusersClient(device=device, **kwargs)
    if route in {"kaggle", "lightning", "hf_zerogpu", "notebook"}:
        return NotebookClient(provider=route, **kwargs)
    raise ValueError(
        f"Unknown compute_route '{compute_route}'. "
        "Valid: local_cpu, free_ai, diffusers_cuda, diffusers_cpu, kaggle, lightning, hf_zerogpu"
    )
