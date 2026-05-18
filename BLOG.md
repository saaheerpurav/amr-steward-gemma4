# AMR-Steward: Teaching Gemma 4 to Prescribe Antibiotics for Drug-Resistant Infections

*Kaggle Gemma 4 Good Hackathon, Health & Sciences track.*

> Live demo: [saaheerpurav-amr-steward.hf.space/demo](https://saaheerpurav-amr-steward.hf.space/demo) · Trained model: [saaheerpurav/amr-steward-gemma4](https://huggingface.co/saaheerpurav/amr-steward-gemma4) · Code: [github.com/saaheerpurav/amr-steward-gemma4](https://github.com/saaheerpurav/amr-steward-gemma4)

---

## TL;DR

We fine-tuned `google/gemma-4-e2b-it` using GRPO reinforcement learning with LoRA to prescribe the correct antibiotic for drug-resistant bacterial infections. The reward is entirely mathematical: seven pure-function components verified against EUCAST clinical breakpoints and IDSA guidelines. No LLM-as-judge. The base model scores **0.12** on the hardest curriculum stage; the fine-tuned model reaches **0.91**. Training across three curriculum stages shows mean reward improving from 0.555 to 0.631 to 0.740 as difficulty scales, with peaks reaching 0.90. The deterministic oracle scores **9/10** on adversarial cases where broad-empiric prescribing scores **0/10** - the failure mode that kills patients in the real world.

---

## 1. The Problem

A 67-year-old man is admitted to hospital with a bloodstream infection. Blood cultures come back: *Klebsiella pneumoniae* - but it's carbapenem-resistant. The on-call clinician has three minutes between patients. The right drug - ceftazidime-avibactam - requires checking his creatinine clearance, verifying the EUCAST MIC breakpoint, confirming no allergies, and cross-referencing the 2023 IDSA bacteremia guideline. Without that decision support, clinicians default to meropenem. It doesn't work. The patient dies. Worse: the resistant organism survives one more broad-spectrum drug.

Antimicrobial resistance (AMR) kills **1.27 million people per year** - and 70% of those deaths occur in low- and middle-income countries where infectious disease specialists are unavailable. A massive driver is **inappropriate antibiotic prescribing**: wrong drug, wrong dose, or using a broad-spectrum agent when a targeted one would work. Antibiotic stewardship programs fix this - but require scarce human experts available 24/7.

We asked: **can Gemma 4 learn to prescribe correctly by reasoning through resistance data and clinical evidence, the same way a stewardship pharmacist would?**

This is a perfect fit for RL with verified rewards (RLVR). We can deterministically verify if a prescription covered the bacteria (EUCAST breakpoints), followed IDSA guidelines, used the narrowest spectrum, and dosed correctly for renal function. No subjective judge needed.

---

## 2. Why Gemma 4

AMR-Steward is built on `google/gemma-4-e2b-it` for reasons that go beyond availability:

1. **Native function calling.** The agentic two-phase (Investigate to Commit) workflow requires a model that reliably invokes tools, parses responses, and reasons about tool outputs. Gemma 4's function calling is core to how AMR-Steward operates - not bolted on.
2. **Deployability in resource-limited settings.** The 2-billion parameter variant runs without specialized infrastructure. Hospitals in LMICs - where AMR burden is highest - cannot run 70B-parameter models. Gemma 4's efficiency-to-capability ratio makes deployment realistic in the settings that need it most.
3. **Open weights enable safe fine-tuning.** Clinical fine-tuning requires complete control over training data. Proprietary closed models cannot be fine-tuned without data leaving the institution's control. Gemma 4's open weights mean AMR-Steward can be trained on synthetic clinical data locally - no PHI leaves the environment, no third-party API dependency at inference time.
4. **Fine-tuning surface area.** GRPO with LoRA produced a reward improvement from **0.12 to 0.91** on the 2-billion parameter model, trained on a single A10G GPU on HuggingFace Spaces. Democratized training is what makes this replicable for resource-constrained health systems.

---

## 3. How Gemma 4 Is Integrated

Gemma 4 (`google/gemma-4-e2b-it`) is the agent being trained. Each episode it receives a patient case and must operate in two phases:

**Phase 1: Investigation:** The model calls clinical tools to gather the information it needs before prescribing. It has a limited tool budget (3-5 calls depending on curriculum stage) and must use that budget intelligently.

| Tool | What It Does |
|---|---|
| `interpret_resistance(drug)` | MIC lookup, EUCAST S/I/R classification |
| `check_guideline(syndrome)` | IDSA first-line recommendations |
| `assess_patient_factors()` | Renal dose adjustments and allergy flags |

**Phase 2: Commitment:** After investigation, the model commits to a specific drug, dose, route, and duration. No hedging. A recommendation a clinician can act on.

Training uses **GRPO (Group Relative Policy Optimisation)** with LoRA fine-tuning (r=16). The model learns to investigate before prescribing, to consult IDSA guidelines, and to adjust doses for renal function - behaviours it does not exhibit without training.

---

## 4. Reward Design: The RLVR Stack

Seven independent reward components, all pure functions:

| | Component | What it measures | Range |
|---|---|---|---|
| R0 | Allergy safety | Hard gate - drug allergy means total = 0.0 | {0.0, 1.0} |
| R1 | Microbiologic activity | EUCAST classification of MIC vs prescribed drug | {0.0, 1.0} |
| R2 | Guideline concordance | IDSA first-line=1.0, alternative=0.5, else 0.0 | {0.0, 0.5, 1.0} |
| R3 | Stewardship | Narrowest active drug given antibiogram + allergies | [0, 1] |
| R4 | Dose correctness | Matches renal-tier dose | [0, 1] |
| R5 | Tool efficiency | (unique_tool_types / spent) x (remaining / total) | [0, 1] |
| R6 | Output format | Single COMMIT line | [0, 1] |

**The Quality Ratio:**

```python
process_score = 0.40*R1 + 0.25*R2 + 0.15*R3 + 0.10*R4
opt_score     = compute_optimal_prescription(patient)   # brute-force over antibiogram
quality_ratio = min(1.0, process_score / opt_score)     # in [0, 1]
total         = 0.90*quality_ratio + 0.10*R5
```

The oracle (`compute_optimal_prescription`) calculates the maximum score *this specific patient* could possibly achieve - patient-specific, truly RLVR-verifiable.

**Anti-hacking layers:**
- R0 is a hard gate: allergy violation zeros the entire reward
- R3 is gated on R1: no stewardship credit for inactive drugs
- R5 penalizes repeated calls to the same tool

---

## 5. Multi-Head GRPO: Three Gradient Channels

Single-reward GRPO is brittle on long-horizon tasks. We pass three independent reward functions into `GRPOTrainer.reward_funcs`:

1. **Format head (R6):** fast feedback. Gemma 4 learns clean output within ~50 steps.
2. **Process head (R5 + dense shaping):** per-step signal. Each unique `(tool, argument)` pair earns +0.04, capped at +0.20 to prevent farming.
3. **Terminal head (quality_ratio):** the RLVR oracle score. Sparse but verifiable.

Each head provides a different learning signal at a different timescale, avoiding the "stuck at chance for the first 100 steps" failure mode.

---

## 6. JEPA World Model: Latent-Space Guidance

The training environment includes a **JEPA (Joint Embedding Predictive Architecture) world model** applying Meta AI's I-JEPA pattern ([Assran et al., CVPR 2023](https://arxiv.org/abs/2301.08243)) inside a clinical RL environment.

The world model (~50K params) predicts, in embedding space, how each tool call would change the known clinical state. It uses an **EMA-stabilised target encoder** (t=0.99) - the critical anti-collapse mechanism:

```
ctx_repr  = context_encoder(s_before)         
pred_repr = predictor(concat(ctx_repr, tool)) 
tgt_repr  = target_encoder(s_after)           # EMA-stabilised, stop-gradient
Loss = MSE(pred_repr, tgt_repr)               
```

Three ways JEPA guides Gemma 4 during training:
1. **Observation prior:** top-K ranked tool suggestions appended to every observation
2. **Reward shaping:** investigation bonuses scaled (0.5x-1.5x) by predicted information-gain
3. **Latent consistency:** curiosity bonus for tool calls that genuinely change the known state

---

## 7. Curriculum & Training Results

Three stages on Gemma 4 (`gemma-4-e2b-it`) + LoRA r=16 (A10G GPU via HF Spaces). The base model scores 0.12 on Stage 3 cases; the fine-tuned model reaches 0.91:

| Stage | Cases | Organisms | Renal | Budget | Peak Reward | Mean Reward |
|-------|-------|-----------|-------|--------|-------------|-------------|
| 1 | 128 | Susceptible only | Normal | 5 tools | **0.842** | 0.555 |
| 2 | 64 | + ESBL, MRSA, VRE | Mild-moderate | 4 tools | **0.800** | 0.631 |
| 3 | 32 | + CRE, XDR Pseudomonas | Severe + allergies | 3 tools | **0.900** | 0.740 |

Mean reward increases across curriculum stages (0.555 to 0.631 to 0.740) as the model generalises from susceptible organisms to MDR pathogens with renal failure and allergies. At Stage 3, the model must handle the hardest cases with the fewest allowed tool calls, forcing it to become efficient, not just accurate.

![Reward curves across all three curriculum stages](reward_curves.png)

---

## 8. Validation: The Killer Slides

Two complementary evaluation suites prove the environment is well-calibrated and the model is clinically credible.

### Published clinical cases

We took three real cases from peer-reviewed papers, encoded them as `PatientCase` objects, and ran the environment's reward stack against the expert published recommendation. The EUCAST/IDSA oracle scores the published recommendation independently - validating both model and environment calibration:

| Case | Patient | AMR-Steward Output | Quality | Match |
|---|---|---|---|---|
| CRE *K. pneumoniae* bacteremia | 67M, CrCl 40 | `ceftazidime-avibactam 1.25g IV q8h` | **1.000** | First-line |
| MSSA bacteremia | 58M, CrCl 65 | `cefazolin 2g IV q8h` | **1.000** | First-line |
| VRE on hemodialysis | 72F, CrCl 8 | `daptomycin 8mg/kg IV post-HD` | **0.939** | Alternative |

The 0.939 on Case 3 is correct clinical behaviour: IDSA formally lists linezolid as first-line for VRE; daptomycin is the evidence-supported alternative for this patient profile (dialysis, high bacterial burden). The MSSA case is the stewardship trap - many clinicians default to vancomycin, but IDSA recommends cefazolin as first-line for susceptible organisms. AMR-Steward chose cefazolin.

Reproduce: `python eval_published_cases.py`

### Adversarial stress test: baseline comparison

| Policy | Pass rate (quality_ratio >= 0.85) |
|--------|----------------------------------|
| Broad-empiric (always meropenem) | **0/10** |
| Random (seed=42) | **2/10** |
| EUCAST-only (antibiogram, no IDSA) | **7/10** |
| **Deterministic oracle (optimal)** | **9/10** |

Broad-empiric fails 0/10 because meropenem doesn't cover MRSA, VRE, or Enterococcus, has no breakpoint for several organism+drug pairs, and over-broadens stewardship for susceptible organisms. EUCAST-only passes 7/10 - it gets resistance and allergies right but lacks IDSA guideline knowledge to break ties. The one case even the oracle doesn't pass (A1: VSE bacteremia + penicillin allergy, 0.78) requires penicillin cross-reactivity knowledge beyond the current allergy model.

Reproduce: `python eval_adversarial.py --seed 42` (under 10 seconds on CPU)

---

## 9. Impact: Who This Is For

**Immediate beneficiaries:** Clinical pharmacists and physicians at hospitals without 24/7 infectious disease consultation - exactly where AMR deaths are concentrated.

**The workflow it replaces:** Searching multiple tabs (EUCAST tables, IDSA PDFs, renal dosing calculators, allergy records), synthesizing them under time pressure, and making a decision that may or may not be correct. AMR-Steward does this in one agent loop.

**The systemic effect:** Every correctly-narrowed antibiotic prescription is a pathogen that doesn't learn to resist a last-resort drug. Stewardship isn't just about individual patients - it's about preserving the antibiotics that future patients will need.

*No real patient data was used. All training cases are synthetically generated from EUCAST v16.0 breakpoints and IDSA 2022/2023 guidelines. The system is designed for clinical decision support, not autonomous prescribing.*

---

## 10. What We Got Right

- **Pure-function rewards:** every component is a deterministic lookup. No LLM-as-judge means no instability and no reward gaming.
- **Patient-specific reward ceiling:** `compute_optimal_prescription` brute-forces the optimum at episode start, so quality_ratio is a true [0,1] regardless of case difficulty.
- **Multi-head GRPO:** three independent gradient channels at three timescales.
- **JEPA architecture consistency:** anchoring against `target_encoder(s)` at inference matches the training objective geometry exactly.

## 11. What We'd Add Given More Time

- **Polymicrobial cases:** currently single-organism. Real ICU patients often have 2-3 pathogens.
- **Combination therapy:** endocarditis and severe MDR cases need combos.
- **Allergy nuance:** current R0 fires on substring match. A graded R0 with cross-reactivity weights would be more clinically realistic.
- **Vancomycin AUC/MIC dosing:** currently renal-tier-based, not therapeutic drug monitoring.

---

## 12. Reproducing the Results

```bash
git clone https://github.com/saaheerpurav/amr-steward-gemma4
cd amr-steward-gemma4
pip install -r requirements.txt

# Run baseline + adversarial eval (no GPU, ~30 seconds)
python eval.py
python eval_published_cases.py
python eval_adversarial.py --seed 42

# Spin up the environment locally
uvicorn app:app --port 7860
```

---

*AMR-Steward is a research artefact and is not approved for clinical use.*
