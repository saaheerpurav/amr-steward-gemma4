# Clinical Validation Matrix — AMR-Steward

> **Purpose:** Map every clinical guideline encoded in the environment to the specific reward component that enforces it, and to the test case(s) that verify it. This matrix proves the environment is a medically grounded validation tool, not a heuristic scoring function.

---

## How to Reproduce Any Row

All test cases are deterministic:
```bash
python eval_adversarial.py --seed 42        # verifies adversarial cases A1–A10
python eval_published_cases.py              # verifies published literature cases P1–P3
pytest test_env.py test_jepa_integration.py # verifies env + JEPA unit tests
```

---

## Matrix

| # | Clinical Guideline / Standard | Source | Env Requirement | Reward Component | Test Case(s) | Verified |
|---|-------------------------------|--------|-----------------|------------------|--------------|----------|
| 1 | CRE bacteremia → ceftazidime-avibactam first-line | IDSA 2023, Tamma PD *Clin Infect Dis* 2023 | R2=1.0 only for ceftazidime-avibactam; alternatives score R2=0.5 | R2 (guideline concordance) | A2, A5, A10, P1 | ✅ |
| 2 | MSSA bacteremia → cefazolin (non-inferior to nafcillin, better tolerability) | IDSA; Maraolo AE *Open Forum Infect Dis* 2018 | R2=1.0 for cefazolin; vancomycin scores R2=0.0 for MSSA | R2 | A8, P2 | ✅ |
| 3 | MRSA pneumonia → vancomycin first-line | IDSA CAP/HAP guidelines 2016 | R2=1.0 for vancomycin only | R2 | A4 | ✅ |
| 4 | VRE bacteremia → high-dose daptomycin | IDSA; Britt NS *Clin Infect Dis* 2015 | R2=1.0 for daptomycin; linezolid=R2=0.5 | R2 | A6, P3 | ✅ |
| 5 | Susceptible E. coli UTI → ceftriaxone (not carbapenem) | IDSA UTI guidelines | R2=0.0 for meropenem in UTI even if R1=1.0; R2=1.0 for ceftriaxone | R2 | A3 | ✅ |
| 6 | ESBL bacteremia → ertapenem (narrowest carbapenem) | IDSA; reserve broad carbapenems | R2=1.0 for ertapenem; meropenem scores R2=0.5 (alternative) | R2, R3 | A9 | ✅ |
| 7 | XDR Pseudomonas → cefiderocol first-line | IDSA; WHO critical priority pathogen list | R2=1.0 for cefiderocol; colistin scores R2=0.5 | R2 | A7 | ✅ |
| 8 | EUCAST: Meropenem MIC ≥8 mg/L → Resistant | EUCAST Clinical Breakpoints v16.0, 2026 | R1=0.0 when EUCAST classifies MIC as R | R1 (microbiological activity) | A2 (CRE, MIC=8), A10 (CRE, MIC=16) | ✅ |
| 9 | EUCAST: Vancomycin MIC ≥32 mg/L → Resistant (VRE) | EUCAST v16.0 | R1=0.0; drug is inactive | R1 | A6 (VRE, MIC=32) | ✅ |
| 10 | EUCAST: Cefazolin breakpoint for S. aureus MSSA | EUCAST v16.0 | R1=1.0 for cefazolin at MSSA MIC=0.5 | R1 | A8 | ✅ |
| 11 | Beta-lactam allergy → do not prescribe penicillins or carbapenems | Standard allergy practice; cross-reactivity guidance | R0=0.0, total=0.0 — hard gate; no partial credit | R0 (allergy safety) | A1 (meropenem + penicillin allergy) | ✅ |
| 12 | Stewardship: prefer narrowest active drug | IDSA stewardship principles; WHO action plan | R3 scores narrowness of spectrum relative to active alternatives | R3 (stewardship) | A3, A9 | ✅ |
| 13 | Stewardship: R3 only fires if drug is active (R1 ≥ threshold) | Anti-gaming principle | R3=0.0 if R1 < threshold — prescribing inactive narrow drug earns zero stewardship credit | R3 gated on R1 | Anti-hacking unit test in `test_env.py` | ✅ |
| 14 | Renal: CrCl <10 (dialysis) → post-HD dosing for daptomycin | Pharmacokinetic principles; standard dialysis dosing tables | R4=1.0 only if prescribed dose matches CrCl_under_10 tier in `drug_properties.json` | R4 (dose correctness) | A6 (CrCl=8, daptomycin post-HD) | ✅ |
| 15 | Renal: CrCl 10–30 → dose reduction for ceftazidime-avibactam | FDA label; IDSA CRE guidance | R4=1.0 for 0.94g IV q12h; R4=0.5 for standard dose | R4 | A5 (CrCl=25) | ✅ |
| 16 | Renal: CrCl 30–50 → renal-adjusted dose for renally-cleared drugs | Standard pharmacokinetics | Documented in `data/drug_properties.json` CrCl_30_50 tier | R4 | Published Case 1 (CrCl=40) | ✅ |
| 17 | Systematic investigation before prescribing | IDSA/stewardship process standards | R5 rewards distinct tool type diversity; JEPA weights reward toward informative tool picks | R5, JEPA dense shaping | All adversarial cases (R5=0.0 for all non-trained baselines) | ✅ |
| 18 | Five WHO critical priority pathogens covered | WHO Global Priority Pathogens List 2017 | Environment generates K. pneumoniae, E. coli, P. aeruginosa, S. aureus, Enterococcus cases | Entire reward stack | A1–A10 collectively | ✅ |
| 19 | Published expert recommendation concordance | Peer-reviewed clinical literature | Oracle scores published recommendations: P1=1.000, P2=1.000, P3=0.939 | All components | P1, P2, P3 | ✅ |

---

## Component → Guideline Mapping

| Reward Component | Clinical Standard Encoded |
|-----------------|--------------------------|
| **R0** Allergy hard gate | Drug allergy avoidance; beta-lactam cross-reactivity |
| **R1** Microbiological activity | EUCAST Clinical Breakpoints v16.0 (2026) |
| **R2** Guideline concordance | IDSA Clinical Practice Guidelines 2022/2023 |
| **R3** Stewardship | IDSA antimicrobial stewardship principles; WHO AMR Action Plan |
| **R4** Dose correctness | Standard prescribing references; FDA drug labeling |
| **R5** Tool efficiency | Investigation process quality (systematic vs. random prescribing) |

---

## Test Coverage Summary

| Test Suite | Cases | What's validated |
|---|---|---|
| `eval_adversarial.py --seed 42` | A1–A10 | R0–R5 per case; pass/subopt/fail verdict |
| `eval_published_cases.py` | P1–P3 | Oracle concordance with peer-reviewed recommendations |
| `test_env.py` | 8 unit tests | R0 hard gate, R3 gated on R1, budget enforcement, allergy detection |
| `test_jepa_integration.py` | 13 tests | JEPA info-gain bounds, dense reward cap, EMA loading |

---

## Data Source Traceability

Every entry in this matrix is traceable to a file in the repository:

| Data Source | File | Entries covered |
|---|---|---|
| IDSA Guidelines | `data/idsa_guidelines.json` | R2 scores for all organism+syndrome combinations |
| EUCAST Breakpoints | `data/eucast.csv` | R1 scores via `data/eucast_parser.py` |
| Renal Dosing | `data/drug_properties.json` | R4 scores; allergy flags for R0 |
| Reward logic | `env/reward.py` | All R0–R5 pure functions |
| Adversarial cases | `eval_adversarial.py` | Rows A1–A10 in this matrix |
| Published cases | `eval_published_cases.py` | Rows P1–P3 in this matrix |
