# AMR-Steward: Teaching Gemma 4 to Prescribe Antibiotics for Drug-Resistant Infections

*Kaggle Gemma 4 Good Hackathon, Health & Sciences track.*

> Live demo: [saaheerpurav-amr-steward.hf.space/demo](https://saaheerpurav-amr-steward.hf.space/demo) · Trained model: [saaheerpurav/amr-steward-gemma4](https://huggingface.co/saaheerpurav/amr-steward-gemma4) · Code: [github.com/saaheerpurav/amr-steward-gemma4](https://github.com/saaheerpurav/amr-steward-gemma4)

---

## The Patient Who Didn't Have to Die

A 67-year-old man is admitted with a bloodstream infection. Blood cultures come back: *Klebsiella pneumoniae* - carbapenem-resistant. The on-call clinician has three minutes between patients. The right drug is ceftazidime-avibactam, but getting there requires checking his creatinine clearance, verifying the EUCAST MIC breakpoint, confirming no beta-lactam allergies, and cross-referencing the 2023 IDSA bacteremia guideline - simultaneously, under time pressure.

So she prescribes meropenem. The most common default. It has zero effect on a carbapenem-resistant organism. The patient dies.

This happens every day. Across 5,000 hospitals in low- and middle-income countries, where 70% of AMR deaths occur and where there is no infectious disease specialist on call, not today, not tomorrow, not ever.

**Antimicrobial resistance kills 1.27 million people per year.** By 2050 it will surpass cancer as the world's leading cause of death. And every wrong antibiotic prescription makes the next one harder - resistant organisms survive, spread, and render our last-resort drugs useless.

The knowledge to prevent this exists. The EUCAST breakpoint tables exist. The IDSA guidelines exist. The renal dosing calculators exist. **The problem is that no clinician can synthesize all of it correctly, under pressure, every time - especially not in the places that need it most.**

That is the problem AMR-Steward solves.

---

## The Solution: A Stewardship Pharmacist in 2 Billion Parameters

AMR-Steward is `google/gemma-4-e2b-it`, fine-tuned via GRPO reinforcement learning to act as an antibiotic stewardship agent. It answers one specific clinical question:

> *Given this patient's pathogen, resistance profile, renal function, and allergies - what antibiotic should I prescribe, at what dose, for how long?*

Not a general medical chatbot. Not a PDF summarizer. One workflow, one decision, done correctly every time.

The model operates in two phases for every patient case:

**Phase 1: Investigation.** Gemma 4 calls clinical tools to gather what it needs before prescribing. It has a limited tool budget and must use it intelligently - the same way an experienced pharmacist already has a hypothesis before ordering tests.

| Tool | What It Does |
|---|---|
| `interpret_resistance(drug)` | MIC lookup, EUCAST S/I/R classification |
| `check_guideline(syndrome)` | IDSA first-line recommendations |
| `assess_patient_factors()` | Renal dose adjustments and allergy flags |

**Phase 2: Commitment.** After investigation, the model commits to a specific drug, dose, route, and duration. No hedging. A recommendation a clinician can act on immediately.

---

## Before and After

Here is the same carbapenem-resistant *K. pneumoniae* case, two ways:

**Without AMR-Steward (broad-empiric default):**
The clinician prescribes meropenem. MIC = 8 mg/L. EUCAST classification: Resistant. The drug does nothing. Score: **0.11 / 1.00**.

**With AMR-Steward:**
The model calls `interpret_resistance('meropenem')`, sees resistance, then calls `check_guideline('bacteremia')`, identifies ceftazidime-avibactam as IDSA-preferred for CRE, then calls `assess_patient_factors()`, adjusts the dose for CrCl 40. It commits: `ceftazidime-avibactam 1.25g IV q8h`. Score: **1.000 / 1.00**.

That difference - 0.11 to 1.00 - is the difference between a drug that does nothing and the drug that saves the patient.

Across 10 adversarial cases designed to break the most common prescribing defaults:

| Policy | Cases Passed (quality >= 0.85) |
|---|---|
| Broad-empiric (always meropenem) | **0 / 10** |
| Random selection | **2 / 10** |
| EUCAST-only (no guideline knowledge) | **7 / 10** |
| AMR-Steward | **9 / 10** |

The base model - `gemma-4-e2b-it` with no fine-tuning - scores **0.12** on the hardest cases. The trained AMR-Steward scores **0.91**. That is what reinforcement learning from clinical evidence looks like.

---

## Why This Had to Be Gemma 4

Four reasons this only works with an open model:

1. **Native function calling.** The Investigate-then-Commit workflow requires a model that reliably invokes tools, parses responses, and reasons about them. Gemma 4's function calling is structural, not patched on.
2. **2B parameters runs anywhere.** Hospitals in LMICs - where AMR kills the most people - cannot run 70B-parameter models. The `e2b` variant runs without specialized infrastructure, making real deployment possible in the places that actually need it.
3. **Open weights mean safe fine-tuning.** Clinical training data cannot leave the institution. Proprietary closed APIs make that impossible. Gemma 4's open weights mean fine-tuning happens locally, with full data control, no PHI exposure.
4. **It actually fine-tunes.** GRPO with LoRA on a single A10G GPU took the model from 0.12 to 0.91. That is not a marginal improvement. That is the model learning to reason about medicine.

---

## How It Learned: Curriculum Training

The model didn't start with XDR *Pseudomonas*. It learned the same way a medical resident does - easy cases first, then complexity added progressively:

| Stage | Pathogens | Renal Complexity | Tool Budget | Result |
|-------|-----------|-----------------|-------------|--------|
| 1 | Susceptible only | Normal | 5 tools | Peak 0.842, Mean 0.555 |
| 2 | + ESBL, MRSA, VRE | Mild-moderate impairment | 4 tools | Peak 0.800, Mean 0.631 |
| 3 | + CRE, XDR Pseudomonas | Severe + allergies | 3 tools | Peak 0.900, Mean 0.740 |

By Stage 3, the model handles the hardest cases with the fewest tool calls. It became efficient because the budget forced it to - no wasted investigations, no redundant checks.

![Reward curves across all three curriculum stages](reward_curves.png)

---

## The Reward System: Seven Rules, Zero Subjectivity

Every prescription is scored by seven pure mathematical functions. No LLM judge. No rubric interpretation. The same objective criteria a human clinician should apply:

| | Component | What it measures |
|---|---|---|
| R0 | Allergy safety | Hard gate - prescribe an allergen and the total score is 0.00 |
| R1 | Microbiologic activity | Does the drug actually cover this pathogen per EUCAST? |
| R2 | Guideline concordance | Does it follow IDSA first-line recommendations? |
| R3 | Stewardship | Is it the *narrowest* effective drug, or unnecessary broad-spectrum? |
| R4 | Dose correctness | Is the dose adjusted for this patient's renal function? |
| R5 | Tool efficiency | Did the model investigate systematically, or guess? |
| R6 | Output format | Is the prescription clean and parseable? |

R3 only fires if R1 is satisfied first - you cannot claim stewardship credit without proving microbiological activity. R0 is a hard gate with no override - an allergen prescription zeroes the entire score, no matter what else was right.

This is how real antibiotic stewardship works. The reward system encodes it exactly.

---

## The JEPA World Model: Learning to Ask the Right Questions

Inside the training environment, a ~50K parameter JEPA (Joint Embedding Predictive Architecture) world model predicts - in latent space - how much information each tool call will provide before the model makes it.

This is the equivalent of an experienced clinician already knowing which test to order. Before Gemma 4 decides what to investigate, it sees JEPA-ranked tool predictions showing which tools are most likely to be informative given the current patient state.

The JEPA model uses an EMA-stabilised target encoder (the I-JEPA pattern from Assran et al., CVPR 2023) - the critical mechanism that prevents representational collapse:

```
ctx_repr  = context_encoder(s_before)
pred_repr = predictor(concat(ctx_repr, tool))
tgt_repr  = target_encoder(s_after)    # EMA-stabilised, stop-gradient
Loss      = MSE(pred_repr, tgt_repr)
```

During training, JEPA contributes in three ways: ranked tool suggestions in every observation, scaled investigation bonuses (0.5x-1.5x) by predicted information gain, and a curiosity bonus for tool calls that genuinely change the known clinical state.

---

## Clinical Validation Against Published Literature

Three real cases from peer-reviewed infectious disease journals. The model's output scored against the expert recommendation:

| Case | Patient | Output | Quality |
|---|---|---|---|
| CRE *K. pneumoniae* bacteremia | 67M, CrCl 40 | `ceftazidime-avibactam 1.25g IV q8h` | **1.000** |
| MSSA bacteremia | 58M, CrCl 65 | `cefazolin 2g IV q8h` | **1.000** |
| VRE on hemodialysis | 72F, CrCl 8 | `daptomycin 8mg/kg IV post-HD` | **0.939** |

The MSSA case is the stewardship trap most clinicians fail: default instinct is vancomycin, but IDSA first-line for susceptible *S. aureus* is cefazolin. AMR-Steward chose cefazolin.

The 0.939 on Case 3 is correct: IDSA lists linezolid as first-line for VRE, but Britt et al. recommends high-dose daptomycin for this specific profile (dialysis, high bacterial burden). The model chose the clinically appropriate alternative.

Reproduce: `python eval_published_cases.py` (CPU, under 10 seconds)

---

## Who This Is For

Every hospital that faces a drug-resistant infection but cannot staff a 24/7 infectious disease consultation service. That is most hospitals on earth.

**The workflow it replaces:** A clinician searching EUCAST tables, IDSA PDFs, renal dosing calculators, and allergy records simultaneously, under time pressure, after a 14-hour shift. AMR-Steward does this in one agent loop.

**The systemic effect:** Every correctly-narrowed prescription is a pathogen that does not learn to resist a last-resort drug. Antibiotic stewardship is not just about the patient in front of you. It is about preserving the drugs that the next patient will need.

*No real patient data was used. All training cases are synthetically generated from EUCAST v16.0 breakpoints and IDSA 2022/2023 guidelines. AMR-Steward is designed for clinical decision support, not autonomous prescribing.*

---

## What We'd Add Given More Time

- **Polymicrobial cases:** currently single-organism. Real ICU patients often have 2-3 pathogens.
- **Combination therapy:** endocarditis and severe MDR cases need combos.
- **Allergy nuance:** current R0 fires on substring match. A graded R0 with cross-reactivity weights would be more clinically realistic.
- **Live antibiogram integration:** connecting `interpret_resistance` to real hospital antibiogram databases is the critical path to clinical deployment.

---

## Reproducing the Results

```bash
git clone https://github.com/saaheerpurav/amr-steward-gemma4
cd amr-steward-gemma4
pip install -r requirements.txt

# Validation against published clinical cases (CPU, ~10 seconds)
python eval_published_cases.py

# Adversarial stress test (CPU, ~10 seconds)
python eval_adversarial.py --seed 42

# Run the environment locally
uvicorn app:app --port 7860
```

---

*AMR-Steward is a research artefact and is not approved for clinical use.*
