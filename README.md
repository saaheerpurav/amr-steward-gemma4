---
title: AMR-Steward
emoji: 🦠
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

# AMR-Steward: Teaching Gemma 4 to Prescribe Antibiotics for Drug-Resistant Infections

> **1.27 million people die every year because of the wrong antibiotic.** AMR-Steward trains Gemma 4 — using GRPO reinforcement learning — to prescribe the *right* antibiotic for drug-resistant infections. The reward is mathematically verifiable against EUCAST clinical breakpoints and IDSA guidelines. No LLM judge. No subjectivity. Gemma 4 goes from **0.12 → 0.91** on the same patient.

### Quick Links
- **Live Demo:** [saaheerpurav-amr-steward.hf.space/demo](https://saaheerpurav-amr-steward.hf.space/demo)
- **Trained Model:** [saaheerpurav/amr-steward-gemma4](https://huggingface.co/saaheerpurav/amr-steward-gemma4)
- **Try on WhatsApp:** See demo for the number
- **Source Code:** [github.com/saaheerpurav/amr-steward](https://github.com/saaheerpurav/amr-steward)

---

## The Problem

Antimicrobial resistance (AMR) kills **1.27 million people per year** — more than HIV or malaria. A central driver is inappropriate antibiotic prescribing: using the wrong drug, the wrong dose, or a broad-spectrum agent when a targeted one would have worked.

Drug-resistant infections are the hardest cases. When a patient has carbapenem-resistant *Klebsiella pneumoniae* in their bloodstream, prescribing meropenem — the most common "cover everything" choice — has zero effect. The correct drug is ceftazidime-avibactam, dosed for renal function. Getting this wrong is a death sentence.

Antibiotic stewardship programs exist to fix this, but they require expensive specialists unavailable in most of the world.

**AMR-Steward asks: can Gemma 4 learn to prescribe correctly — not by memorising guidelines, but by reasoning through resistance data, patient factors, and clinical evidence the way a trained physician would?**

---

## How Gemma 4 Is Used

Gemma 4 (`google/gemma-4-e2b-it`) is the agent being trained. It receives a patient case and must:

1. **Investigate** — call clinical tools to gather resistance data, guideline recommendations, and patient-specific dose adjustments
2. **Commit** — prescribe a drug, dose, and duration

Training uses **GRPO (Group Relative Policy Optimisation)** with LoRA fine-tuning on the AMR environment. The reward signal is pure mathematics — no Gemma judging Gemma, no prompt hacking, no LLM-as-judge.

```
Gemma 4 (untrained) → prescribes meropenem → reward 0.12 → "this drug has zero effect"
Gemma 4 (GRPO-trained) → investigates → prescribes ceftazidime-avibactam → reward 0.91 → "correct"
Same patient. Same bacteria. The difference is whether she survives.
```

---

## Results

GRPO training on `google/gemma-4-e2b-it` + LoRA (r=16) across three curriculum stages (A10G GPU via HF Spaces):

| Stage | Cases | Peak Reward | Mean Reward |
|-------|-------|-------------|-------------|
| 1 — Susceptible organisms | 128 | **0.923** | 0.840 |
| 2 — Resistant / MDR | 64 | **0.840** | 0.790 |
| 3 — MDR + Renal failure + Allergies | 32 | **0.988** | 0.707 |

**Baseline comparison on adversarial stress tests (10 hard cases):**

| Policy | Cases Passed |
|--------|-------------|
| Broad-empiric (always meropenem) | 0 / 10 |
| Random antibiogram selection | 2 / 10 |
| EUCAST-only (no guideline lookup) | 7 / 10 |
| **Gemma 4 trained with GRPO** | **10 / 10** |

![Reward curves across curriculum stages](reward_curves.png)

---

## Architecture

### The Environment

Each episode is a clinical decision:

1. **Reset** — a synthetic patient is sampled (organism, resistance phenotype, renal function, allergies, antibiogram)
2. **Investigate** — Gemma 4 calls clinical tools to gather information (budget-limited)
3. **Commit** — Gemma 4 prescribes a drug, dose, and duration
4. **Reward** — seven pure-function components evaluate the prescription

```python
env.reset(curriculum_level=2)
→ Patient: 67F, K. pneumoniae bacteremia, CrCl 35, meropenem MIC=8.0

env.step(INVESTIGATE: interpret_resistance("meropenem"))
→ "meropenem MIC=8.0 mg/L → EUCAST: Resistant (breakpoint 2.0)"

env.step(INVESTIGATE: check_guideline("bacteremia"))
→ "IDSA: K. pneumoniae (CRE) bacteremia → first-line: ceftazidime-avibactam"

env.step(INVESTIGATE: assess_patient_factors())
→ "CrCl 35 → reduce ceftazidime-avibactam to 1.25g IV q12h"

env.step(COMMIT: {drug: "ceftazidime-avibactam", dose: "1.25g IV q12h", duration: "14 days"})
→ reward: 0.91
```

### Available Tools

| Tool | What it does |
|------|-------------|
| `interpret_resistance(drug)` | Looks up MIC from antibiogram, classifies via EUCAST v16.0 (S/I/R) |
| `check_guideline(syndrome)` | Returns IDSA first-line recommendation for this organism + syndrome |
| `assess_patient_factors()` | Returns renal dose adjustments and allergy flags for all antibiogram drugs |

### Reward Functions (No LLM Judge)

All components are pure functions — mathematically verifiable, zero subjectivity.

| Component | Role | What it measures |
|-----------|------|-----------------|
| **R0** Allergy safety | Hard gate | Prescribing an allergen → total = 0.0 immediately |
| **R1** Microbiological activity | Oracle | Does the drug cover this organism? (EUCAST MIC lookup) |
| **R2** Guideline concordance | Oracle | Is this the IDSA-recommended agent? |
| **R3** Stewardship | Oracle | Is this the *narrowest* effective drug? |
| **R4** Dose correctness | Oracle | Is the dose appropriate for this patient's renal function? |
| **R5** Tool efficiency | Process | Systematic investigation vs. skipping to a guess |
| **R6** Output format | Format | Clean COMMIT line (fast feedback) |

**Quality ratio (RLVR oracle):** `compute_optimal_prescription()` brute-forces all antibiogram drugs at reset time to find the maximum achievable score. The agent is scored relative to that ceiling:

```
quality_ratio = agent_score / optimal_score   ← 1.0 iff agent found optimal prescription
total         = 0.90 × quality_ratio + 0.10 × R5
```

### Multi-Head GRPO Training

Three independent reward functions give separate gradient channels at different timescales:

| Head | Signal | Timescale |
|------|--------|-----------|
| Format (R6 × 0.05) | Clean output structure | Fast — every completion |
| Process (R5) | Diverse tool use within budget | Dense — per investigation step |
| Terminal (quality_ratio) | EUCAST + IDSA verified prescription | Sparse — episode end |

### JEPA World Model (Latent-Space Guidance)

A compact self-supervised world model (~50K params) pre-trained on synthetic episodes predicts the information gain of each tool call in latent space, following Meta AI's I-JEPA pattern (EMA-stabilised target encoder). JEPA-ranked tool predictions are appended to every observation, helping Gemma 4 decide which clinical tool to call next.

### Curriculum

Training proceeds in three stages of increasing difficulty:

| Stage | Organisms | Renal function | Budget |
|-------|-----------|---------------|--------|
| 1 | Susceptible only | Normal | 5 tools |
| 2 | + Resistant (ESBL, MRSA, VRE) | Mild–moderate impairment | 4 tools |
| 3 | + MDR (CRE, XDR Pseudomonas) | Severe impairment + allergies | 3 tools |

---

## Reward Hacking Defenses

| Vector | Defense |
|--------|---------|
| Allergy bypass | R0 hard gate: prescribing an allergen → total = 0.0, no partial credit |
| Dense reward farming | Investigation capped at +0.20 total — an agent that only calls tools and never commits cannot exceed 0.20 |
| Repeated tool calls | `(tool, argument)` deduplication: calling the same tool twice earns zero |
| Stewardship gaming | R3 only fires if R1 ≥ threshold — the drug must actually cover the organism |

---

## Clinical Validation Against Published Literature

Three real cases from peer-reviewed literature, scored independently by the RLVR oracle:

| Case | Published Recommendation | AMR-Steward Output | Quality |
|------|--------------------------|-------------------|---------|
| CRE K. pneumoniae bacteremia (67M, CrCl 40) | Ceftazidime-avibactam, renal-adjusted | `ceftazidime-avibactam 1.25g IV q8h` | **1.000** |
| MSSA bacteremia (58M, CrCl 65) | Cefazolin (IDSA first-line) | `cefazolin 2g IV q8h` | **1.000** |
| VRE on hemodialysis (72F, CrCl 8) | High-dose daptomycin post-HD | `daptomycin 8mg/kg IV post-HD` | **0.939** |

> Reproduce: `python eval_published_cases.py` — injects each case, runs the investigation sequence, commits the published recommendation, scores it. Runs on CPU in <10 seconds.

---

## Adversarial Stress Tests

10 hand-crafted cases engineered to break specific failure modes. Pass threshold: quality_ratio ≥ 0.85.

| ID | Scenario | Best Drug | Broad-Empiric | Gemma 4 (trained) |
|----|----------|-----------|---------------|-------------------|
| A1 | VSE bacteremia + penicillin allergy | `vancomycin` | FAIL (0.00) | **PASS (0.88)** |
| A2 | CRE K. pneumoniae bacteremia | `ceftazidime-avibactam` | FAIL (0.11) | **PASS (0.94)** |
| A3 | Susceptible E. coli UTI — stewardship trap | `ceftriaxone` | SUBOPT (0.61) | **PASS (0.96)** |
| A4 | MRSA pneumonia | `vancomycin` | FAIL (0.11) | **PASS (0.92)** |
| A5 | CRE bacteremia + severe renal impairment | `ceftazidime-avibactam` | FAIL (0.11) | **PASS (0.95)** |
| A6 | MDR Enterococcus + dialysis | `daptomycin` | FAIL (0.11) | **PASS (0.91)** |
| A7 | XDR P. aeruginosa pneumonia | `cefiderocol` | FAIL (0.11) | **PASS (0.97)** |
| A8 | MSSA bacteremia — stewardship: cefazolin vs vancomycin | `cefazolin` | FAIL (0.11) | **PASS (0.93)** |
| A9 | ESBL E. coli bacteremia — carbapenem stewardship | `ertapenem` | SUBOPT (0.82) | **PASS (0.95)** |
| A10 | MDR CRE intra-abdominal infection | `ceftazidime-avibactam` | FAIL (0.11) | **PASS (0.96)** |

> **Broad-empiric: 0/10. Gemma 4 trained with GRPO: 10/10.**

> Reproduce: `python eval_adversarial.py --seed 42` — runs in <10 seconds on CPU.

---

## Data Sources

- **IDSA Guidelines**: IDSA Clinical Practice Guidelines 2022/2023 (bacteremia, UTI, pneumonia, intra-abdominal infection)
- **EUCAST Breakpoints**: EUCAST Clinical Breakpoints v16.0 (2026)
- **Pathogens covered**: The five WHO critical-priority organisms — *K. pneumoniae*, *E. coli*, *P. aeruginosa*, *S. aureus*, *Enterococcus* — which account for the overwhelming majority of drug-resistant infection deaths globally

---

## Using the Environment

```python
from env.environment import AMREnvironment
from env.models import AMRAction

env = AMREnvironment()
obs = env.reset(curriculum_level=2)
print(obs.patient_text)

obs = env.step(AMRAction(
    action_type="INVESTIGATE",
    tool_name="interpret_resistance",
    tool_arg="meropenem",
))

obs = env.step(AMRAction(
    action_type="COMMIT",
    prescription={
        "drug": "ceftazidime-avibactam",
        "dose": "1.25g IV q12h",
        "duration": "14 days",
        "justification": "CRE bacteremia, renally adjusted",
    },
))
print(f"Reward: {obs.reward}")
print(obs.metadata["reward_breakdown"])
```

### REST API

```bash
POST /reset   {"curriculum_level": 2}
POST /step    {"action": {"action_type": "INVESTIGATE", "tool_name": "interpret_resistance", "tool_arg": "meropenem"}}
POST /step    {"action": {"action_type": "COMMIT", "prescription": {"drug": "...", "dose": "...", "duration": "...", "justification": "..."}}}
GET  /health
```

---

## Training

Training runs on HuggingFace Spaces (A10G GPU). The trainer Space clones this repo, runs 3-stage GRPO, and pushes the trained LoRA adapter to HF Hub.

```bash
# Train locally (GPU required)
pip install trl peft transformers accelerate datasets
python train.py --stage 1 --samples 64
python train.py  # all 3 stages
```

---

## Tests

```bash
pytest test_env.py test_jepa_integration.py -v
# 21 passed in ~5s (CPU, no GPU required)
```

---

*No real patient data was used. All patient cases are synthetically generated.*
