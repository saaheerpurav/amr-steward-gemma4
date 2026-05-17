# Failure Mode Analysis — AMR-Steward

> **Source data:** `adversarial_results.json` — all scores are deterministic, reproducible with `python eval_adversarial.py --seed 42`.

This document explains *why* each policy succeeded or failed on the 10 adversarial cases. Understanding failure modes is what separates a research system from a demo: it shows we know exactly where the boundary is and why it exists.

---

## How to Read This

Each case has:
- **What happened:** the policy's prescription and reward breakdown (R0–R5).
- **Root cause:** the specific component that caused failure or success.
- **Insight:** what the agent (or environment) needs to handle this correctly.

---

## 5 Failure Cases

### A1 — VSE Bacteremia + Penicillin Allergy

**Policy:** Broad-Empiric (always meropenem) | **Score:** 0.00 | **Verdict:** FAIL

| Component | Score | Reason |
|-----------|-------|--------|
| R0 Allergy | 0.0 | Patient has penicillin allergy; meropenem is a beta-lactam → R0 hard gate fires |
| R1–R5 | 0.0 | Irrelevant — R0 zeros the entire reward |

**Root cause:** R0 fires the moment a drug with any beta-lactam flag is prescribed to a penicillin-allergic patient. Meropenem's allergy flag is `penicillin_cross_reactivity_low_risk`, which matches the patient's documented `penicillin` allergy string. The environment enforces zero-tolerance: wrong allergy = treatment failure = reward 0.

**Insight:** The model must call `assess_patient_factors()` *before* committing. The allergy flag is visible in the tool output. An agent that skips investigation and prescribes its default drug will fail any allergy case, regardless of how pharmacologically sound the default is.

---

### A3 — Susceptible E. coli UTI (Stewardship Trap)

**Policy:** Broad-Empiric | **Score:** 0.545 | **Verdict:** SUBOPT

| Component | Score | Reason |
|-----------|-------|--------|
| R0 Allergy | 1.0 | No allergy conflict |
| R1 Activity | 1.0 | Meropenem covers susceptible E. coli |
| R2 Guideline | 0.0 | IDSA first-line for susceptible E. coli UTI is ceftriaxone, not meropenem |
| R3 Stewardship | 0.1 | Meropenem is a carbapenem; using it for a simple susceptible UTI is unnecessary escalation |
| R4 Dose | 1.0 | Dose is correct |

**Root cause:** R2 (guideline concordance) is the discriminating component here. Meropenem *works* — R1 is 1.0 — but IDSA explicitly reserves carbapenems for drug-resistant organisms. Using meropenem for a susceptible UTI is medically wasteful and encourages resistance development. The environment correctly penalizes this.

**Insight:** This is the stewardship trap: the agent knows a broad drug "works" but must learn to prefer the narrowest active drug that follows guidelines. The EUCAST-only policy also fails this case (score 0.76) because it picks cefazolin (narrowest susceptible drug in the antibiogram) rather than ceftriaxone (the IDSA-recommended agent for UTI). Only the oracle policy, which checks `check_guideline(UTI)`, finds ceftriaxone.

---

### A8 — MSSA Bacteremia (Stewardship: Cefazolin vs Vancomycin)

**Policy:** Broad-Empiric | **Score:** 0.10 | **Verdict:** FAIL

| Component | Score | Reason |
|-----------|-------|--------|
| R0 Allergy | 1.0 | No allergy |
| R1 Activity | 0.0 | Meropenem has no EUCAST breakpoint for S. aureus → classified UNKNOWN → R1=0 |
| R2–R5 | 0.0 | R1=0 collapses stewardship and guideline scores |

**Root cause:** Meropenem is not a drug used for S. aureus in any guideline. EUCAST does not publish a breakpoint for meropenem vs. S. aureus. The EUCAST parser returns `UNKNOWN`, which maps to R1=0. This is the most important failure: the "default broad-spectrum drug" assumption breaks down when the organism is outside that drug's spectrum.

**Insight for the random policy (SUBOPT, 0.575):** The random policy picks vancomycin — which does cover MRSA — but this patient is MSSA. Vancomycin *works* (R1=1.0) but IDSA guidelines explicitly prefer cefazolin over vancomycin for MSSA bacteremia because cefazolin achieves superior β-lactam mediated killing. R2 = 0.0 for vancomycin on MSSA (guideline alternative, not first-line), and R3 is penalized because cefazolin is narrower. The trained model must learn to distinguish MSSA (→ cefazolin) from MRSA (→ vancomycin).

---

### A6 — MDR Enterococcus + Dialysis (CrCl 8)

**Policy:** Random (seed=42) | **Score:** 0.10 | **Verdict:** FAIL

| Component | Score | Reason |
|-----------|-------|--------|
| R0 Allergy | 1.0 | No allergy |
| R1 Activity | 0.0 | Vancomycin MIC=32 → EUCAST Resistant for VRE → R1=0 |
| R2–R5 | 0.0 | No activity = no credit |

**Root cause:** The random policy (seed=42) picks vancomycin, which is the standard treatment for Enterococcus — but this case is specifically *vancomycin-resistant* Enterococcus (VRE). MIC=32 is 64x above the EUCAST susceptible breakpoint. R1=0 cascades immediately. The model cannot recover from selecting an inactive drug.

**Insight:** This case tests whether the agent knows to look up resistance *first* before defaulting to the "standard" drug for an organism class. The correct flow is: `interpret_resistance(vancomycin)` → sees RESISTANT → queries guidelines for alternative → daptomycin post-HD. An agent that skips resistance lookup will fail VRE cases every time.

---

### A9 — ESBL E. coli Bacteremia (Carbapenem Stewardship)

**Policy:** Random (seed=42) | **Score:** 0.053 | **Verdict:** FAIL

| Component | Score | Reason |
|-----------|-------|--------|
| R0 Allergy | 1.0 | No allergy |
| R1 Activity | 0.0 | Random picks ceftriaxone, which ESBL degrades → EUCAST R → R1=0 |
| R2–R5 | 0.0 | Drug is inactive |

**Root cause:** ESBL-producing E. coli is resistant to all third-generation cephalosporins (ceftriaxone, cefotaxime) by enzymatic degradation. The random policy selects ceftriaxone — a drug that works perfectly against susceptible E. coli — but ESBL hydrolyzes the beta-lactam ring, rendering it inactive. R1=0.

**Insight:** This is the hallmark ESBL failure. Even the broad-empiric policy (meropenem) passes this case (SUBOPT, 0.82) because meropenem is stable to ESBL. But the correct answer is ertapenem (narrowest carbapenem active against ESBL), not meropenem. The trained agent needs to: (1) interpret resistance to ceftriaxone, see RESISTANT, (2) consult guideline for ESBL bacteremia, and (3) select ertapenem as the narrowest carbapenem.

---

## 5 Success Cases

### A2 — CRE K. pneumoniae Bacteremia

**Policy:** EUCAST-only + Oracle | **Score:** 1.00 | **Verdict:** PASS

**Why it passes:** The EUCAST-only policy correctly looks up all MICs, identifies that meropenem is resistant (MIC=16) and ceftazidime-avibactam is susceptible (MIC=0.5), and selects ceftazidime-avibactam as the narrowest susceptible drug. R1=1.0, R3=1.0 (narrowest), R2=1.0 (IDSA first-line for KPC-CRE). Perfect score.

**Insight:** When the antibiogram is unambiguous and the IDSA first-line drug is also the narrowest active drug, the environment rewards maximally. No tension between R2 and R3. The model just needs to check the resistance pattern and follow the guideline.

---

### A4 — MRSA Pneumonia

**Policy:** Random (seed=42), EUCAST-only, Oracle | **Score:** 1.00 | **Verdict:** PASS (all three)

**Why it passes (including random!):** Vancomycin is the only drug in the antibiogram that covers MRSA pneumonia. The random policy happens to select vancomycin. EUCAST-only also finds vancomycin (narrowest susceptible drug). Oracle confirms vancomycin is IDSA first-line for MRSA pneumonia. R1=R2=R3=R4=1.0.

**Insight:** Cases where only one drug in the antibiogram is active AND it matches IDSA guidelines are "forced wins" — any policy that doesn't trigger the allergy gate will pass. This is expected: the environment design puts harder discrimination into multi-drug cases (A3, A8, A9) where both narrow and broad drugs are active.

---

### A5 — CRE Bacteremia + Moderate-Severe Renal Impairment (CrCl 25)

**Policy:** Random (seed=42), EUCAST-only, Oracle | **Score:** 1.00 | **Verdict:** PASS

**Why it passes:** Ceftazidime-avibactam is the only active drug for CRE in the antibiogram. The renal dose at CrCl 25 (CrCl_10_30 tier) is `0.94g IV q12h` — different from the normal dose of `2.5g IV q8h`. All passing policies correctly identify the renal-adjusted dose. R4=1.0.

**Insight:** The environment scores renal dose correctness by exact tier match from `drug_properties.json`. This case tests that `assess_patient_factors()` is called and the output is used to adjust the dose. An agent that prescribes 2.5g q8h (normal dose) for a CrCl-25 patient would score R4=0.5.

---

### A7 — XDR P. aeruginosa Pneumonia (Last-Line Agent)

**Policy:** EUCAST-only, Oracle | **Score:** 1.00 | **Verdict:** PASS

**Why EUCAST-only passes:** The antibiogram shows all standard agents resistant except cefiderocol. EUCAST-only correctly identifies cefiderocol as the only susceptible drug. R1=1.0. IDSA also lists cefiderocol as first-line for XDR Pseudomonas. R2=1.0.

**Why Random fails (SUBOPT, 0.76):** Random picks colistin, which has activity (R1=1.0) but colistin is listed as an *alternative* in IDSA guidelines (R2=0.5) and has a broader toxicity profile than cefiderocol (R3 penalty). The model must prefer the IDSA-recommended last-line agent.

**Insight:** Last-line agent cases (cefiderocol, colistin) test whether the model correctly ranks drugs by IDSA priority when multiple agents technically "work" in the antibiogram. R2 is the discriminating component.

---

### A10 — MDR E. coli CRE Intra-Abdominal Infection

**Policy:** EUCAST-only, Oracle | **Score:** 1.00 | **Verdict:** PASS

**Why it passes:** CRE E. coli in an intra-abdominal infection. Ceftazidime-avibactam is susceptible (MIC=0.5) and is IDSA first-line for CRE intra-abdominal infection. Dose is 2.5g IV q8h (normal renal function). All components align: R0=1, R1=1, R2=1, R3=1, R4=1.

**Why Broad-Empiric and Random fail:** Meropenem (MIC=16) is resistant — R1=0. Cefepime (random pick) is also resistant for CRE — R1=0. Both fail at the first checkpoint.

---

## Summary Pattern

| Failure Mode | Root cause | Cases affected | Fix |
|---|---|---|------|
| Allergy not checked | R0 fires — skipped `assess_patient_factors()` | A1 | Must call `assess_patient_factors()` before commit |
| Default drug inactive | R1=0 for the prescribed drug | A2, A6, A8, A10 | Must call `interpret_resistance(drug)` for any candidate |
| Guideline not consulted | R2=0 — drug works but isn't IDSA preferred | A3, A8 | Must call `check_guideline(syndrome)` |
| Stewardship violation | R3 low — drug works but too broad | A3, A9 | Guideline lookup reveals narrower option |
| Drug-organism mismatch | No EUCAST breakpoint → UNKNOWN → R1=0 | A8 (meropenem vs S. aureus) | Must call `interpret_resistance()` and check the result |

**The conclusion:** A policy that always calls all 3 tool types before committing — regardless of prior knowledge — will handle all of these failure modes. The trained model's JEPA world model is specifically designed to prioritize the investigation step with the highest predicted information gain, biasing the agent toward exactly this systematic approach.
