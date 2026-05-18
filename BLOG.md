# AMR-Steward — Teaching Gemma 4 to Prescribe Antibiotics Correctly

> Live environment: [saaheerpurav-amr-steward.hf.space](https://saaheerpurav-amr-steward.hf.space) · Trained model: [saaheerpurav/amr-steward-gemma4](https://huggingface.co/saaheerpurav/amr-steward-gemma4) · Code: [github.com/saaheerpurav/amr-steward](https://github.com/saaheerpurav/amr-steward)

---

## TL;DR

We built an RL environment that teaches Gemma 4 to prescribe the correct antibiotic for drug-resistant bacterial infections using GRPO. The reward is entirely mathematical — seven pure-function components verified against EUCAST clinical breakpoints and IDSA guidelines. No LLM-as-judge. Training across three curriculum stages shows mean reward improving from 0.555 → 0.631 → 0.740 as difficulty scales, with peaks reaching 0.90. The deterministic oracle scores **9/10** on adversarial cases where broad-empiric prescribing scores **0/10** — the failure mode that kills patients in the real world.

---

## 1. The Problem

Antimicrobial resistance (AMR) kills **1.27 million people per year** — more than HIV or malaria. A massive driver is **inappropriate antibiotic prescribing**: wrong drug, wrong dose, or using a broad-spectrum agent when a targeted one would work.

When a patient has carbapenem-resistant *Klebsiella pneumoniae*, prescribing meropenem — the most common "cover everything" default — has zero effect on the bacteria. The correct drug is ceftazidime-avibactam, dosed for renal function. Getting this wrong is a death sentence.

We asked: **can Gemma 4 learn to prescribe correctly by reasoning through resistance data and clinical evidence, the way a trained physician would?**

This is a perfect fit for RL with verified rewards (RLVR). We can deterministically verify if a prescription covered the bacteria (EUCAST breakpoints), followed IDSA guidelines, used the narrowest spectrum, and dosed correctly. No subjective judge needed.

---

## 2. How Gemma 4 Is Integrated

Gemma 4 (`google/gemma-4-e2b-it`) is the agent being trained. Each episode it receives a patient case and must:

1. **Investigate** — call clinical tools to gather resistance data, guideline recommendations, and patient-specific adjustments (budget-limited)
2. **Commit** — prescribe a drug, dose, and duration in a structured JSON format

Training uses **GRPO (Group Relative Policy Optimisation)** with LoRA fine-tuning (r=16). The model learns to investigate before prescribing, to consult IDSA guidelines, and to adjust doses for renal function — behaviours it does not exhibit without training.

---

## 3. Reward Design — The RLVR Stack

Seven independent reward components, all pure functions:

| | Component | What it measures | Range |
|---|---|---|---|
| R0 | Allergy safety | Hard gate — drug allergy → total = 0.0 | {0.0, 1.0} |
| R1 | Microbiologic activity | EUCAST classification of MIC vs prescribed drug | {0.0, 1.0} |
| R2 | Guideline concordance | IDSA first-line=1.0, alternative=0.5, else 0.0 | {0.0, 0.5, 1.0} |
| R3 | Stewardship | Narrowest active drug given antibiogram + allergies | [0, 1] |
| R4 | Dose correctness | Matches renal-tier dose | [0, 1] |
| R5 | Tool efficiency | (unique_tool_types / spent) × (remaining / total) | [0, 1] |
| R6 | Output format | Single COMMIT line | [0, 1] |

**The Quality Ratio:**

```python
process_score = 0.40·R1 + 0.25·R2 + 0.15·R3 + 0.10·R4
opt_score     = compute_optimal_prescription(patient)   # brute-force over antibiogram
quality_ratio = min(1.0, process_score / opt_score)     # ∈ [0, 1]
total         = 0.90·quality_ratio + 0.10·R5
```

The oracle (`compute_optimal_prescription`) calculates the maximum score *this specific patient* could possibly achieve — patient-specific, truly RLVR-verifiable.

**Anti-hacking layers**:
- R0 is a hard gate — allergy violation zeros the entire reward
- R3 is gated on R1 — no stewardship credit for inactive drugs
- R5 penalizes repeated calls to the same tool

---

## 4. Multi-Head GRPO — Three Gradient Channels

Single-reward GRPO is brittle on long-horizon tasks. We pass three independent reward functions into `GRPOTrainer.reward_funcs`:

1. **Format head (R6)** — fast feedback. Gemma 4 learns clean output within ~50 steps.
2. **Process head (R5 + dense shaping)** — per-step signal. Each unique `(tool, argument)` pair earns +0.04, capped at +0.20 to prevent farming.
3. **Terminal head (quality_ratio)** — the RLVR oracle score. Sparse but verifiable.

Each head provides a different learning signal at a different timescale, avoiding the "stuck at chance for the first 100 steps" failure mode.

---

## 5. JEPA World Model — Latent-Space Guidance

The training environment includes a **JEPA (Joint Embedding Predictive Architecture) world model** applying Meta AI's I-JEPA pattern ([Assran et al., CVPR 2023](https://arxiv.org/abs/2301.08243)) inside a clinical RL environment.

The world model (≈50K params) predicts, in embedding space, how each tool call would change the known clinical state. It uses an **EMA-stabilised target encoder** (τ=0.99) — the critical anti-collapse mechanism:

```
ctx_repr  = context_encoder(s_before)         
pred_repr = predictor(concat(ctx_repr, tool)) 
tgt_repr  = target_encoder(s_after)           # EMA-stabilised, stop-gradient
Loss = MSE(pred_repr, tgt_repr)               
```

Three ways JEPA guides Gemma 4 during training:
1. **Observation prior** — top-K ranked tool suggestions appended to every observation
2. **Reward shaping** — investigation bonuses scaled (0.5×–1.5×) by predicted information-gain
3. **Latent consistency** — curiosity bonus for tool calls that genuinely change the known state

---

## 6. Curriculum & Training Results

Three stages on Gemma 4 (`gemma-4-e2b-it`) + LoRA r=16 (A10G GPU via HF Spaces):

| Stage | Cases | Organisms | Renal | Budget | Result |
|-------|-------|-----------|-------|--------|--------|
| 1 | 128 | Susceptible only | Normal | 5 tools | 0.475 → **peak 0.842** (mean 0.555) |
| 2 | 64 | + ESBL, MRSA, VRE | Mild–moderate | 4 tools | 0.600 → **peak 0.800** (mean 0.631) |
| 3 | 32 | + CRE, XDR, VISA | Severe + allergies | 3 tools | 0.800 → **peak 0.900** (mean 0.740) |

Mean reward increases across curriculum stages (0.555 → 0.631 → 0.740) as the model generalises from susceptible organisms to MDR pathogens with renal failure and allergies.

![Reward curves across all three curriculum stages](reward_curves.png)

---

## 7. Validation

### Published clinical cases — 3/3 match expert recommendations

| Case | Citation | Expert prescription | Quality |
|------|----------|---------------------|---------|
| CRE bacteremia, post-renal-transplant | Tamma PD et al. *Clin Infect Dis.* 2023 | Ceftazidime-avibactam 1.25g IV q8h | **1.000** |
| MSSA bacteremia | Maraolo AE et al. *Open Forum Infect Dis.* 2018 | Cefazolin 2g IV q8h | **1.000** |
| VRE on hemodialysis | Britt NS et al. *Clin Infect Dis.* 2015 | Daptomycin 8mg/kg post-HD | **0.939** |

### Adversarial stress test — baseline comparison

| Policy | Pass rate (quality_ratio ≥ 0.85) |
|--------|----------------------------------|
| Broad-empiric (always meropenem) | **0/10** |
| Random (seed=42) | **2/10** |
| EUCAST-only (antibiogram, no IDSA) | **7/10** |
| **Deterministic oracle (optimal)** | **9/10** |

---

## 8. What We Got Right

- **Pure-function rewards** — every component is a deterministic lookup. No LLM-as-judge means no instability and no reward gaming.
- **Patient-specific reward ceiling** — `compute_optimal_prescription` brute-forces the optimum at episode start, so quality_ratio is a true [0,1] regardless of case difficulty.
- **Multi-head GRPO** — three independent gradient channels at three timescales.
- **JEPA architecture consistency** — anchoring against `target_encoder(s)` at inference matches the training objective geometry exactly.

## 9. What We'd Add Given More Time

- **Polymicrobial cases** — currently single-organism
- **Combination therapy** — endocarditis and severe MDR cases need combos
- **Vancomycin AUC/MIC dosing** — currently renal-tier-based, not therapeutic drug monitoring

---

## 10. Reproducing the Results

```bash
git clone https://github.com/saaheerpurav/amr-steward
cd amr-steward
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
