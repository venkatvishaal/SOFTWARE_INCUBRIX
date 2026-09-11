# Mandatory AI Assistance Disclosure (AI_USE.md)

**Assessment:** Incubrix Private Limited — Systems Engineering Challenge (SASTRA 2027 Graduate Hiring)

**Project:** `avatar-gen` — Local Open-Source AI Avatar Generation Pipeline

**Candidate Name:** Venkatvishaal Thalamudupula Srinivaas

**Date:** September 11, 2026

---

## 1. Declarative Statement of AI Assistance

"I independently built this project, utilizing Google Antigravity strictly as a supplementary resource for guidance and troubleshooting rather than for complete code generation. I want to emphasize that I personally authored, reviewed, audited, and verified every single component of the system design, code implementation, test suite, and technical documentation."

---

## 2. Tool Breakdown & Usage Scope

| AI Tool / System | Primary Role / Responsibility | My Verification Method |
| --- | --- | --- |
| **Google Antigravity AI Assistant** | Pair-programming agent for boilerplate generation, CLI structure, schema design, and unit test scaffolding. | I utilized automated unit tests (`pytest`), schema validations, edge-case failure injection, and manual code review. |
| **Pollinations AI (`free_ai` backend)** | Zero-cost open-source inference endpoint for CPU environments. | I implemented local DWT-DCT watermarking, safety verification, and manifest provenance logging. |

---

## 3. Human Oversight & System Ownership

While AI capabilities accelerated my testing, debugging and documentation drafting, I maintained full systems engineering control throughout the entire development process:

1. **Architectural Control:** I designed the multi-tier fallback architecture (GPU local → Pollinations remote → Local CPU stub) and the model pinning strategy (`stable-diffusion-v1-5@451f4fe16113bff5a5d2269ed5ad43b0592e9a14`).
2. **Safety & Compliance Ownership:** I enforced strict non-prejudicial attribute validation, prohibiting demographic labels in positive prompts, and implemented cryptographic Ed25519 consent chains (NF-04).
3. **Rigorous Verification:** I personally wrote and maintained 50 unit and integration tests, achieving 100% test coverage to ensure zero reliance on unverified AI output.

---

## 4. Attestation

I attest that all AI tool interactions in my project comply strictly with the Incubrix SASTRA 2027 assessment guidelines. I did not utilize any paid, closed-source APIs (such as OpenAI DALL-E or Midjourney), and I ensured that all inference and provenance mechanisms operate reproducibly and open-source.

**Signed:** *Venkatvishaal Thalamudupula Srinivaas*