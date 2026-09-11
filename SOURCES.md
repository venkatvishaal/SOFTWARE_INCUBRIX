# SOURCES.md — External Sources and Acknowledgements

All sources used in the design and implementation of this project are listed below,
with the date accessed and the specific use.

## Models and Libraries

| Source | Use |
|---|---|
| [Stable Diffusion v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) (RunwayML, CreativeML Open RAIL-M) | Primary inference model reference |
| [HuggingFace diffusers](https://github.com/huggingface/diffusers) (Apache 2.0) | GPU inference backend |
| [invisible-watermark](https://github.com/ShieldMnt/invisible-watermark) (MIT) | DWT-DCT steganographic watermarking |
| [open-clip-torch](https://github.com/mlfoundations/open_clip) (MIT) | Optional CLIP adherence scoring |
| [insightface](https://github.com/deepinsight/insightface) (MIT) | Optional face embedding (identity consistency) |
| [cryptography](https://cryptography.io/) (Apache 2.0) | Ed25519 consent chain signing |
| [Pydantic](https://docs.pydantic.dev/) (MIT) | Structured spec model validation |
| [jsonschema](https://python-jsonschema.readthedocs.io/) (MIT) | JSON Schema validation |
| [Click](https://click.palletsprojects.com/) (BSD) | CLI framework |
| [Rich](https://rich.readthedocs.io/) (MIT) | Terminal table/colour output |

## Research and Design References

| Reference | Accessed | Specific Use |
|---|---|---|
| C2PA Technical Specification v2.1 (Coalition for Content Provenance) | 2026-09-01 | Provenance metadata design (NF-04 consent chain) |
| OpenBias (Otterbacher et al., 2024) | 2026-09-01 | Bias audit methodology (NF-03) |
| Shannon, C. E. (1948). "A Mathematical Theory of Communication" | — | Shannon entropy for bias measurement (NF-03) |
| DWT-DCT watermarking paper (Cox et al., 1997) | — | Watermark algorithm basis (NF-05) |
| Rombach et al. (2022). "High-Resolution Image Synthesis with Latent Diffusion Models" | 2026-09-01 | Model architecture understanding |
| HPSv2 (Wu et al., 2023) | 2026-09-01 | CLIP-based adherence scoring (NF-09) |
| Fitzpatrick, T. B. (1988). "The validity and practicality of sun-reactive skin types" | — | Skin tone taxonomy basis |

## Datasets (Referenced Only — Not Downloaded)

| Dataset | Reason Not Downloaded |
|---|---|
| FFHQ (NVIDIA) | Requires NVIDIA agreement; used only as architectural reference |
| FairFace | Reviewed for attribute taxonomy design only |

## AI Tool Disclosure

This project was built with AI coding assistance (Google Antigravity).
All generated code was reviewed, tested, and validated against the
assessment requirements by the candidate.

See [AI_USE.md](AI_USE.md) for full disclosure.
