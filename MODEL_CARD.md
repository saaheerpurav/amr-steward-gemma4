---
base_model: google/gemma-4-e2b-it
library_name: peft
pipeline_tag: text-generation
license: apache-2.0
tags:
- base_model:adapter:google/gemma-4-e2b-it
- grpo
- lora
- transformers
- trl
- reinforcement-learning
- medical
- antimicrobial-resistance
- clinical-decision-support
- gemma
---

# AMR-Steward — Antibiotic Prescribing Agent (Gemma 4)

**Gemma 4 (`gemma-4-e2b-it`) + LoRA trained with multi-head GRPO** to prescribe the correct antibiotic for drug-resistant bacterial infections. Reward is fully verifiable: seven pure-function components against EUCAST v16.0 breakpoints and IDSA 2022/2023 clinical guidelines. **No LLM-as-judge anywhere.**

| | |
|---|---|
| **Base model** | [google/gemma-4-e2b-it](https://huggingface.co/google/gemma-4-e2b-it) |
| **Fine-tuning** | LoRA (r=16, α=32, targets: q/k/v/o/gate/up/down projections) |
| **Algorithm** | Multi-head GRPO (TRL + PEFT, bf16) |
| **Hardware** | A10G GPU — HuggingFace Spaces |
| **Live demo** | [saaheerpurav-amr-steward.hf.space/demo](https://saaheerpurav-amr-steward.hf.space/demo) |
| **Environment** | [github.com/saaheerpurav/amr-steward](https://github.com/saaheerpurav/amr-steward) |

---

## Training Results

Three curriculum stages — susceptible organisms → MDR + severe renal failure + allergy constraints:

| Stage | Organisms | Budget | Peak | Mean |
|-------|-----------|--------|------|------|
| **1 — Susceptible** | *K. pneumoniae*, *E. coli*, *S. aureus* | 5 tools | **0.923** | 0.840 |
| **2 — Resistant/MDR** | + ESBL, MRSA, VRE | 4 tools | **0.840** | 0.790 |
| **3 — MDR + Renal + Allergies** | + CRE, XDR Pseudomonas | 3 tools | **0.988** | 0.707 |

Broad-empiric baseline: **0/10** adversarial cases passed. Trained Gemma 4: **10/10**.

![Reward curves — all 3 curriculum stages](reward_curves.png)

---

## What This Model Does

The agent receives a clinical patient case and must investigate, then prescribe:

```
Patient: 67F, ICU, K. pneumoniae bacteremia, meropenem MIC=8.0, CrCl=35, no allergies

Gemma 4 investigates:
  → interpret_resistance("meropenem")   → "MIC 8.0 → EUCAST: Resistant"
  → check_guideline("bacteremia")       → "IDSA: CRE K. pneumoniae → ceftazidime-avibactam"
  → assess_patient_factors()            → "CrCl 35: reduce to 1.25g IV q12h"

Gemma 4 prescribes:
  → ceftazidime-avibactam 1.25g IV q12h, 14 days
  → reward: 0.92
```

Without training (broad-empiric): prescribes meropenem → **reward ~0.11** (drug has zero effect on resistant strain).

---

## JEPA World Model

The training environment includes a **JEPA (Joint Embedding Predictive Architecture) world model** — applying Meta AI's I-JEPA pattern ([Assran et al., CVPR 2023](https://arxiv.org/abs/2301.08243)) inside a clinical RL environment.

The world model (≈50K params) predicts in latent space how each tool call would change the known clinical state, using an **EMA-stabilised target encoder** (τ=0.99). It guides Gemma 4's investigation strategy via observation hints, reward shaping, and latent consistency bonuses.

---

## Reward Design

All components are pure functions — deterministic, verifiable, zero subjectivity:

| Component | What it measures | Range |
|-----------|-----------------|-------|
| **R0** Allergy gate | Prescribing an allergen → total = 0.0 | {0, 1} |
| **R1** Microbiologic activity | EUCAST MIC classification | {0, 1} |
| **R2** Guideline concordance | IDSA first-line=1.0, alternative=0.5 | {0, 0.5, 1} |
| **R3** Stewardship (gated on R1) | Narrowest active spectrum | [0, 1] |
| **R4** Dose correctness | Renal-adjusted dose match | [0, 1] |
| **R5** Tool efficiency | Systematic investigation vs guessing | [0, 1] |
| **R6** Format | Clean single COMMIT line | [0, 1] |

```python
quality_ratio = agent_score / compute_optimal_prescription(patient)
total         = 0.90 × quality_ratio + 0.10 × R5
```

---

## Usage

This is a PEFT LoRA adapter — load on top of `google/gemma-4-e2b-it`:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained("google/gemma-4-e2b-it")
model = PeftModel.from_pretrained(base_model, "saaheerpurav/amr-steward-gemma4")
tokenizer = AutoTokenizer.from_pretrained("saaheerpurav/amr-steward-gemma4")
```

To use inside the AMR-Steward environment:

```bash
git clone https://github.com/saaheerpurav/amr-steward
pip install -r requirements.txt
uvicorn app:app --port 7860
```

---

## Scope and Limitations

- **Not approved for clinical use.** Research artefact only.
- Covers the five WHO critical-priority pathogens: *K. pneumoniae*, *E. coli*, *P. aeruginosa*, *S. aureus*, *Enterococcus* spp.
- Single-organism, single-drug episodes. No polymicrobial cases or combination therapy.
- Trained on synthetic patient cases, not real EHR data.

---

## Training Infrastructure

| | |
|---|---|
| **GPU** | NVIDIA A10G (24 GB) via HuggingFace Spaces |
| **Precision** | bf16 |
| **LoRA rank** | r=16, α=32 |
| **GRPO generations** | 4 per step |
| **Max completion length** | 512 tokens |
| **Framework** | TRL · PEFT · HuggingFace Transformers |

---

*Not approved for clinical use.*
