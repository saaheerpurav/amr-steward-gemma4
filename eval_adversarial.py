"""
eval_adversarial.py
===================
10-case hand-crafted adversarial test suite for AMR-Steward.

Each case is engineered to break at least one baseline policy in a predictable
way. Together they prove the reward stack discriminates good from bad prescribing
on hard clinical scenarios not present in the training distribution.

Reward component coverage:
  R0 allergy gate       : A1
  R1 microbiologic act. : A1, A2, A4, A5, A6, A7, A8, A10
  R2 IDSA concordance   : A2, A4, A7, A8, A9
  R3 stewardship        : A3, A8, A9
  R4 renal dose tier    : A5 (CrCl 25), A6 (CrCl 8 / dialysis)
  R5 tool efficiency    : implicit -- baselines get 0.0, trained model must
                          beat them by *also* nailing R0-R4.

Usage:
    python eval_adversarial.py
    python eval_adversarial.py --seed 42 --out adversarial_results.json --md README_adversarial_table.md

Outputs:
    - Console: ASCII summary table
    - adversarial_results.json: full per-case per-policy breakdown
    - README_adversarial_table.md: copy-paste section for README
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from env.models import PatientCase
from env.reward import (
    _load_drug_properties,
    _load_idsa,
    _normalize_drug,
    compute_total_reward,
    compute_optimal_prescription,
)
from data.eucast_parser import classify_mic

# Import baseline policy functions from eval.py (safe: has __main__ guard)
from eval import (
    prescribe_broad_empiric,
    prescribe_random,
    prescribe_eucast_narrowest,
    _renal_dose,
)

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

_IDSA = _load_idsa()
_DRUG_PROPS = _load_drug_properties()
_EUCAST = type("_Eucast", (), {"classify_mic": staticmethod(classify_mic)})

BUDGET_BY_LEVEL: dict[int, int] = {1: 5, 2: 4, 3: 3}

# Quality-ratio threshold to call a prescription a "pass"
PASS_THRESHOLD = 0.85
# "SUBOPT" = partially correct but below PASS_THRESHOLD; "FAIL" = near zero
SUBOPT_THRESHOLD = 0.40

# ---------------------------------------------------------------------------
# The 10 adversarial cases
# ---------------------------------------------------------------------------
# Each entry has:
#   id              : short slug used in filenames and tables
#   title           : one-line clinical description
#   tests           : list of reward components under stress (for docs)
#   expected_drug   : IDSA-optimal drug (str, normalised lower-case)
#   expected_dose   : correct renal-tier dose string for PatientCase
#   level           : curriculum level (1|2|3) determines episode budget
#   patient_kwargs  : raw dict to build a PatientCase dataclass
#
# MIC values are set so each drug's EUCAST classification is unambiguous
# (well above S-cutoff for resistant drugs, well below S-cutoff for susceptible).

ADVERSARIAL_CASES: list[dict[str, Any]] = [
    # ------------------------------------------------------------------ A1
    {
        "id": "A1",
        "title": "VSE bacteremia + penicillin allergy",
        "tests": ["R0_allergy"],
        "expected_drug": "vancomycin",
        "expected_dose": "15-20mg/kg IV q8-12h",   # CrCl_above_50 tier
        "level": 1,
        "patient_kwargs": {
            "age": 54,
            "sex": "F",
            "infection_site": "bacteremia",
            "organism": "Enterococcus",
            "phenotype": "susceptible",
            "creatinine_clearance": 70.0,
            "allergies": ["penicillin"],
            # Antibiogram: ampicillin susceptible but blocked by penicillin allergy.
            # vancomycin is the non-allergic IDSA alternative.
            "antibiogram": {
                "ampicillin":  2.0,   # enterococcus S=4 → MIC 2 is S, but ALLERGY blocks it
                "vancomycin":  2.0,   # enterococcus S=4 → S
                "linezolid":   2.0,   # enterococcus S=4 → S (broader spectrum)
                "daptomycin":  2.0,   # enterococcus S=4 → S
            },
            "curriculum_level": 1,
        },
    },
    # ------------------------------------------------------------------ A2
    {
        "id": "A2",
        "title": "CRE K. pneumoniae bacteremia",
        "tests": ["R1_activity", "R2_guideline"],
        "expected_drug": "ceftazidime-avibactam",
        "expected_dose": "1.25g IV q8h",           # CrCl_30_50 tier (CrCl=50)
        "level": 2,
        "patient_kwargs": {
            "age": 67,
            "sex": "M",
            "infection_site": "bacteremia",
            "organism": "K. pneumoniae",
            "phenotype": "resistant",              # → IDSA key "K. pneumoniae (CRE)"
            "creatinine_clearance": 50.0,
            "allergies": [],
            # Meropenem is resistant (MIC=16 >> R-cutoff 8).
            # Ceftazidime-avibactam is susceptible (MIC=4 <= S-cutoff 8).
            "antibiogram": {
                "meropenem":           16.0,
                "ceftazidime-avibactam": 4.0,
                "meropenem-vaborbactam": 1.0,
                "colistin":              1.0,
            },
            "curriculum_level": 2,
        },
    },
    # ------------------------------------------------------------------ A3
    {
        "id": "A3",
        "title": "Susceptible E. coli UTI -- stewardship trap",
        "tests": ["R3_stewardship"],
        "expected_drug": "ceftriaxone",
        "expected_dose": "2g IV q24h",             # ceftriaxone has no renal adj → 0.5 for R4
        "level": 1,
        "patient_kwargs": {
            "age": 44,
            "sex": "F",
            "infection_site": "UTI",
            "organism": "E. coli",
            "phenotype": "susceptible",
            "creatinine_clearance": 80.0,
            "allergies": [],
            # All drugs susceptible; meropenem is much broader than needed.
            # Cefazolin and ceftriaxone both spectrum=2 but cefazolin listed first →
            # EUCAST-narrowest picks cefazolin (not IDSA-concordant for complicated UTI).
            "antibiogram": {
                "cefazolin":              1.0,   # e.coli cefazolin S=4 → S
                "ceftriaxone":            0.25,  # e.coli S=1 → S
                "meropenem":              0.25,  # e.coli S=2 → S (but broad overkill)
                "cefepime":               0.25,  # e.coli S=1 → S
                "piperacillin-tazobactam": 4.0,  # e.coli S=8 → S
            },
            "curriculum_level": 1,
        },
    },
    # ------------------------------------------------------------------ A4
    {
        "id": "A4",
        "title": "MRSA pneumonia",
        "tests": ["R1_activity", "R2_guideline"],
        "expected_drug": "vancomycin",
        "expected_dose": "15-20mg/kg IV q8-12h",   # CrCl_above_50 tier
        "level": 1,
        "patient_kwargs": {
            "age": 61,
            "sex": "M",
            "infection_site": "pneumonia",
            "organism": "S. aureus",
            "phenotype": "resistant",               # → IDSA key "S. aureus (MRSA)"
            "creatinine_clearance": 70.0,
            "allergies": [],
            # Meropenem has no EUCAST breakpoint for S. aureus → UNKNOWN → R1=0.
            # Vancomycin MIC=1 ≤ S-cutoff 2 → susceptible; first-line for MRSA pneumonia.
            "antibiogram": {
                "vancomycin": 1.0,   # s.aureus S=2 → S
                "linezolid":  2.0,   # s.aureus S=4 → S (alternative)
                "meropenem":  0.5,   # NO breakpoint for s.aureus → UNKNOWN → R1=0
            },
            "curriculum_level": 1,
        },
    },
    # ------------------------------------------------------------------ A5
    {
        "id": "A5",
        "title": "CRE bacteremia + moderate-severe renal impairment (CrCl 25)",
        "tests": ["R1_activity", "R4_dose"],
        "expected_drug": "ceftazidime-avibactam",
        "expected_dose": "0.94g IV q12h",           # CrCl_10_30 tier
        "level": 2,
        "patient_kwargs": {
            "age": 72,
            "sex": "M",
            "infection_site": "bacteremia",
            "organism": "K. pneumoniae",
            "phenotype": "resistant",
            "creatinine_clearance": 25.0,           # → CrCl_10_30 renal tier
            "allergies": [],
            # Same CRE pattern as A2, different CrCl → different dose tier.
            "antibiogram": {
                "meropenem":             16.0,
                "ceftazidime-avibactam":  4.0,
                "meropenem-vaborbactam":  1.0,
                "colistin":               1.0,
            },
            "curriculum_level": 2,
        },
    },
    # ------------------------------------------------------------------ A6
    {
        "id": "A6",
        "title": "MDR Enterococcus bacteremia + dialysis (CrCl 8)",
        "tests": ["R1_activity", "R2_guideline", "R4_dose"],
        "expected_drug": "daptomycin",
        "expected_dose": "8mg/kg IV post-HD",       # CrCl_under_10 tier (post-dialysis dosing)
        "level": 2,
        "patient_kwargs": {
            "age": 78,
            "sex": "F",
            "infection_site": "bacteremia",
            "organism": "Enterococcus",
            "phenotype": "MDR",                     # → IDSA key "Enterococcus (MDR)"
            "creatinine_clearance": 8.0,            # → CrCl_under_10 (ESRD on HD)
            "allergies": [],
            # Ampicillin and vancomycin resistant (VRE-MDR pattern).
            # Linezolid and daptomycin susceptible; daptomycin is IDSA first-line for MDR Enterococcus.
            # Daptomycin is narrower (spectrum=3 vs linezolid=4) so EUCAST-narrowest picks it too.
            "antibiogram": {
                "ampicillin":  16.0,  # enterococcus R=8 → MIC 16 is R
                "vancomycin":   8.0,  # enterococcus R=4 → MIC 8 is R
                "linezolid":    2.0,  # enterococcus S=4 → S
                "daptomycin":   2.0,  # enterococcus S=4 → S
            },
            "curriculum_level": 2,
        },
    },
    # ------------------------------------------------------------------ A7
    {
        "id": "A7",
        "title": "XDR P. aeruginosa pneumonia -- last-line agent",
        "tests": ["R1_activity", "R2_guideline"],
        "expected_drug": "cefiderocol",
        "expected_dose": "2g IV q8h (3h infusion)",  # CrCl_above_50 tier for cefiderocol
        "level": 2,
        "patient_kwargs": {
            "age": 65,
            "sex": "M",
            "infection_site": "pneumonia",
            "organism": "P. aeruginosa",
            "phenotype": "MDR",                     # → IDSA key "P. aeruginosa (MDR)"
            "creatinine_clearance": 60.0,
            "allergies": [],
            # Standard anti-pseudomonals all resistant; only cefiderocol and colistin susceptible.
            # Cefiderocol is IDSA first-line for XDR Pseudomonas.
            "antibiogram": {
                "meropenem":              16.0,  # p.aeruginosa S=2,R=8 → R
                "cefepime":               16.0,  # p.aeruginosa S=8,R=8 → R
                "piperacillin-tazobactam": 64.0, # p.aeruginosa S=16,R=16 → R
                "ceftazidime-avibactam":   4.0,  # p.aeruginosa S=8,R=8 → S (susceptible but alternative)
                "cefiderocol":             1.0,  # p.aeruginosa S=2,R=2 → S (first-line)
                "colistin":                1.0,  # p.aeruginosa S=2,R=2 → S (last resort)
            },
            "curriculum_level": 2,
        },
    },
    # ------------------------------------------------------------------ A8
    {
        "id": "A8",
        "title": "MSSA bacteremia -- stewardship: cefazolin vs vancomycin",
        "tests": ["R2_guideline", "R3_stewardship"],
        "expected_drug": "cefazolin",
        "expected_dose": "2g IV q8h",               # CrCl_above_50 tier
        "level": 1,
        "patient_kwargs": {
            "age": 58,
            "sex": "M",
            "infection_site": "bacteremia",
            "organism": "S. aureus",
            "phenotype": "susceptible",              # → IDSA key "S. aureus (MSSA)"
            "creatinine_clearance": 70.0,
            "allergies": [],
            # Vancomycin is susceptible and 'works' for MSSA, but cefazolin is IDSA first-line
            # and narrower spectrum. Oxacillin listed before cefazolin so EUCAST-narrowest
            # picks oxacillin (spectrum=2, same as cefazolin) -- demonstrating guideline
            # knowledge is required beyond just 'narrowest susceptible'.
            "antibiogram": {
                "vancomycin": 1.0,  # s.aureus S=2 → S (broader, not first-line for MSSA)
                "oxacillin":  1.0,  # s.aureus S=2 → S (alternative, not first-line)
                "cefazolin":  2.0,  # s.aureus S=4 → S (IDSA first-line for MSSA)
                "meropenem":  0.5,  # NO s.aureus breakpoint → UNKNOWN → R1=0
            },
            "curriculum_level": 1,
        },
    },
    # ------------------------------------------------------------------ A9
    {
        "id": "A9",
        "title": "ESBL E. coli bacteremia -- carbapenem stewardship",
        "tests": ["R2_guideline", "R3_stewardship"],
        "expected_drug": "ertapenem",
        "expected_dose": "1g IV q24h",              # CrCl_above_30 tier (CrCl=80)
        "level": 1,
        "patient_kwargs": {
            "age": 71,
            "sex": "F",
            "infection_site": "bacteremia",
            "organism": "E. coli",
            "phenotype": "resistant",               # → IDSA key "E. coli (ESBL)"
            "creatinine_clearance": 80.0,
            "allergies": [],
            # Cephalosporins resistant (ESBL hydrolysis); meropenem and ertapenem active.
            # Ertapenem (spectrum=4) is IDSA first-line and narrower than meropenem (spectrum=5).
            # Broad-empiric meropenem 'works' but gets R2=0.5 + R3 stewardship penalty.
            "antibiogram": {
                "ceftriaxone":  8.0,   # e.coli S=1,R=2 → MIC 8 is R (ESBL hydrolysis)
                "cefepime":     8.0,   # e.coli S=1,R=4 → MIC 8 is R
                "ertapenem":    0.25,  # e.coli S=0.5,R=1 → S (narrower carbapenem)
                "meropenem":    0.5,   # e.coli S=2,R=8 → S (broader carbapenem)
            },
            "curriculum_level": 1,
        },
    },
    # ------------------------------------------------------------------ A10
    {
        "id": "A10",
        "title": "MDR E. coli CRE intra-abdominal infection",
        "tests": ["R1_activity", "R2_guideline"],
        "expected_drug": "ceftazidime-avibactam",
        "expected_dose": "2.5g IV q8h",             # CrCl_above_50 tier (CrCl=60)
        "level": 2,
        "patient_kwargs": {
            "age": 63,
            "sex": "M",
            "infection_site": "intra-abdominal",
            "organism": "E. coli",
            "phenotype": "MDR",                     # → IDSA key "E. coli (MDR)"
            "creatinine_clearance": 60.0,
            "allergies": [],
            # CRE E. coli: meropenem and cefepime resistant; ceftazidime-avibactam first-line.
            # Also covers the 4th infection site (intra-abdominal) not in cases A1-A9.
            "antibiogram": {
                "meropenem":             16.0,  # e.coli S=2,R=8 → R
                "ceftazidime-avibactam":  4.0,  # e.coli S=8,R=8 → S
                "meropenem-vaborbactam":  1.0,  # e.coli S=2,R=8 → S
                "cefepime":               8.0,  # e.coli S=1,R=4 → R
            },
            "curriculum_level": 2,
        },
    },
]


# ---------------------------------------------------------------------------
# Pre-flight validation
# ---------------------------------------------------------------------------

def _preflight_validate() -> None:
    """Fail loud if any case references missing IDSA keys, unknown drugs, or
    antibiograms that are too sparse to score correctly."""
    from env.reward import _organism_to_idsa_key

    errors: list[str] = []

    for case in ADVERSARIAL_CASES:
        cid = case["id"]
        p = case["patient_kwargs"]

        # 1. IDSA key must exist
        idsa_key = _organism_to_idsa_key(p["organism"], p["phenotype"])
        site_data = _IDSA.get(p["infection_site"], {})
        if idsa_key not in site_data:
            errors.append(
                f"{cid}: IDSA key '{idsa_key}' not found under site '{p['infection_site']}'"
            )

        # 2. Expected drug must have an entry in drug_properties
        exp_drug = _normalize_drug(case["expected_drug"])
        if exp_drug not in _DRUG_PROPS:
            errors.append(
                f"{cid}: expected_drug '{exp_drug}' not found in drug_properties.json"
            )

        # 3. Expected drug must appear in antibiogram
        norm_antibiogram = {_normalize_drug(k): v for k, v in p["antibiogram"].items()}
        if exp_drug not in norm_antibiogram:
            errors.append(
                f"{cid}: expected_drug '{exp_drug}' not in antibiogram"
            )

        # 4. At least 3 drugs in antibiogram for meaningful EUCAST-narrowest scoring
        if len(p["antibiogram"]) < 3:
            errors.append(
                f"{cid}: antibiogram has only {len(p['antibiogram'])} drugs (need >= 3)"
            )

    if errors:
        print("PRE-FLIGHT VALIDATION FAILED:")
        for e in errors:
            print(f"  ERROR: {e}")
        sys.exit(1)

    print(f"[preflight] All {len(ADVERSARIAL_CASES)} cases validated OK.")


# ---------------------------------------------------------------------------
# Oracle prescription: IDSA first-line at correct renal-tier dose
# ---------------------------------------------------------------------------

def _oracle_prescription(case: dict[str, Any], patient: PatientCase) -> dict:
    """Build the IDSA-recommended prescription at the exact renal-tier dose
    from drug_properties.json so we can score the env's reward ceiling."""
    drug = case["expected_drug"]
    dose = _renal_dose(drug, patient.creatinine_clearance) or case["expected_dose"]
    idsa_data = _IDSA.get(patient.infection_site, {})
    from env.reward import _organism_to_idsa_key
    idsa_key = _organism_to_idsa_key(patient.organism, patient.phenotype)
    org_data = idsa_data.get(idsa_key, {})
    duration = org_data.get("duration", "14 days")
    return {
        "drug": drug,
        "dose": dose,
        "duration": duration,
        "justification": f"IDSA first-line for {idsa_key}",
    }


# ---------------------------------------------------------------------------
# Scoring helper
# ---------------------------------------------------------------------------

def _score(
    prescription: dict,
    patient: PatientCase,
    level: int,
) -> tuple[float, dict]:
    """Run compute_total_reward for a baseline (zero tool calls, full budget)."""
    budget_total = BUDGET_BY_LEVEL[level]
    total, bd = compute_total_reward(
        prescription=prescription,
        patient=patient,
        tool_call_history=[],      # baselines make no tool calls
        eucast=_EUCAST,
        idsa=_IDSA,
        drug_properties=_DRUG_PROPS,
        budget_remaining=budget_total,  # no budget spent
        budget_total=budget_total,
        tool_history=[],            # structured history also empty
    )
    return total, bd


# ---------------------------------------------------------------------------
# Per-case result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    case_id: str
    case_title: str
    policy: str
    drug: str
    dose: str
    quality_ratio: float
    total: float
    R0: float
    R1: float
    R2: float
    R3: float
    R4: float
    R5: float

    @property
    def verdict(self) -> str:
        if self.quality_ratio >= PASS_THRESHOLD:
            return "PASS"
        elif self.quality_ratio >= SUBOPT_THRESHOLD:
            return "SUBOPT"
        return "FAIL"


# ---------------------------------------------------------------------------
# Run a single case under all policies
# ---------------------------------------------------------------------------

def run_case(case: dict[str, Any], rng: random.Random) -> list[CaseResult]:
    """Score one adversarial case under all 4 'policies' (broad, random, eucast, oracle)."""
    patient = PatientCase(**case["patient_kwargs"])
    level = case["level"]

    results: list[CaseResult] = []

    # --- broad empiric (always meropenem) ---
    prx_be = prescribe_broad_empiric(patient)
    total_be, bd_be = _score(prx_be, patient, level)
    results.append(CaseResult(
        case_id=case["id"], case_title=case["title"], policy="broad_empiric",
        drug=prx_be["drug"], dose=prx_be["dose"],
        quality_ratio=bd_be["quality_ratio"], total=total_be,
        R0=bd_be["R0_allergy"], R1=bd_be["R1_activity"],
        R2=bd_be["R2_guideline"], R3=bd_be["R3_stewardship"],
        R4=bd_be["R4_dose"], R5=bd_be["R5_efficiency"],
    ))

    # --- random (use caller-supplied RNG for reproducibility) ---
    drugs = [_normalize_drug(d) for d in patient.antibiogram]
    rand_drug = rng.choice(drugs) if drugs else "meropenem"
    rand_dose = _renal_dose(rand_drug, patient.creatinine_clearance) or "standard dose"
    prx_rand = {"drug": rand_drug, "dose": rand_dose, "duration": "7 days", "justification": "random"}
    total_rand, bd_rand = _score(prx_rand, patient, level)
    results.append(CaseResult(
        case_id=case["id"], case_title=case["title"], policy="random",
        drug=rand_drug, dose=rand_dose,
        quality_ratio=bd_rand["quality_ratio"], total=total_rand,
        R0=bd_rand["R0_allergy"], R1=bd_rand["R1_activity"],
        R2=bd_rand["R2_guideline"], R3=bd_rand["R3_stewardship"],
        R4=bd_rand["R4_dose"], R5=bd_rand["R5_efficiency"],
    ))

    # --- EUCAST narrowest ---
    prx_eu = prescribe_eucast_narrowest(patient)
    total_eu, bd_eu = _score(prx_eu, patient, level)
    results.append(CaseResult(
        case_id=case["id"], case_title=case["title"], policy="eucast_narrowest",
        drug=prx_eu["drug"], dose=prx_eu["dose"],
        quality_ratio=bd_eu["quality_ratio"], total=total_eu,
        R0=bd_eu["R0_allergy"], R1=bd_eu["R1_activity"],
        R2=bd_eu["R2_guideline"], R3=bd_eu["R3_stewardship"],
        R4=bd_eu["R4_dose"], R5=bd_eu["R5_efficiency"],
    ))

    # --- oracle (IDSA first-line at correct renal-tier dose) ---
    prx_or = _oracle_prescription(case, patient)
    total_or, bd_or = _score(prx_or, patient, level)
    results.append(CaseResult(
        case_id=case["id"], case_title=case["title"], policy="oracle",
        drug=prx_or["drug"], dose=prx_or["dose"],
        quality_ratio=bd_or["quality_ratio"], total=total_or,
        R0=bd_or["R0_allergy"], R1=bd_or["R1_activity"],
        R2=bd_or["R2_guideline"], R3=bd_or["R3_stewardship"],
        R4=bd_or["R4_dose"], R5=bd_or["R5_efficiency"],
    ))

    return results


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

POLICY_LABELS = {
    "broad_empiric":    "Broad-Empiric",
    "random":           "Random(42)",
    "eucast_narrowest": "EUCAST-Only",
    "oracle":           "Oracle(IDSA)",
}


def format_console_table(all_results: list[CaseResult]) -> str:
    """ASCII table: rows = cases, columns = policies."""
    policies = ["broad_empiric", "random", "eucast_narrowest", "oracle"]
    col_w = 18
    id_w = 5
    title_w = 44

    header_row = f"{'ID':<{id_w}}  {'Scenario':<{title_w}}"
    for p in policies:
        header_row += f"  {POLICY_LABELS[p]:^{col_w}}"

    sep = "-" * (id_w + 2 + title_w + 2 + (col_w + 2) * len(policies))

    rows = [
        "",
        "=" * len(sep),
        "AMR-Steward  Adversarial Stress Test (10 hard cases)",
        "=" * len(sep),
        header_row,
        sep,
    ]

    # Group by case
    by_case: dict[str, dict[str, CaseResult]] = {}
    for r in all_results:
        by_case.setdefault(r.case_id, {})[r.policy] = r

    for case in ADVERSARIAL_CASES:
        cid = case["id"]
        row = f"{cid:<{id_w}}  {case['title'][:title_w]:<{title_w}}"
        for p in policies:
            res = by_case.get(cid, {}).get(p)
            if res is None:
                row += f"  {'N/A':^{col_w}}"
            else:
                cell = f"{res.verdict} {res.quality_ratio:.2f}"
                row += f"  {cell:^{col_w}}"
        rows.append(row)

    rows.append(sep)

    # Summary pass rates (PASS + SUBOPT >= threshold)
    rows.append("")
    rows.append("Pass rate (quality_ratio >= {:.2f}):".format(PASS_THRESHOLD))
    for p in policies:
        policy_results = [r for r in all_results if r.policy == p]
        n_pass = sum(1 for r in policy_results if r.quality_ratio >= PASS_THRESHOLD)
        rows.append(f"  {POLICY_LABELS[p]:<18}: {n_pass}/{len(policy_results)}")

    rows.append("")
    rows.append(
        "Note: All baselines make zero tool calls (R5=0.0). The trained model"
    )
    rows.append(
        "      must match or beat EUCAST-Only on R0-R4 *and* earn R5 from investigations."
    )
    rows.append("")

    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------

def format_markdown_table(all_results: list[CaseResult]) -> str:
    """README-ready markdown section for the adversarial stress test."""
    by_case: dict[str, dict[str, CaseResult]] = {}
    for r in all_results:
        by_case.setdefault(r.case_id, {})[r.policy] = r

    be_pass   = sum(1 for c in ADVERSARIAL_CASES
                    if by_case.get(c["id"], {}).get("broad_empiric") and
                    by_case[c["id"]]["broad_empiric"].quality_ratio >= PASS_THRESHOLD)
    rand_pass = sum(1 for c in ADVERSARIAL_CASES
                    if by_case.get(c["id"], {}).get("random") and
                    by_case[c["id"]]["random"].quality_ratio >= PASS_THRESHOLD)
    eu_pass   = sum(1 for c in ADVERSARIAL_CASES
                    if by_case.get(c["id"], {}).get("eucast_narrowest") and
                    by_case[c["id"]]["eucast_narrowest"].quality_ratio >= PASS_THRESHOLD)

    def _cell(res: CaseResult | None) -> str:
        if res is None:
            return "N/A"
        icon = {"PASS": "PASS", "SUBOPT": "SUBOPT", "FAIL": "FAIL"}[res.verdict]
        return f"{icon} ({res.quality_ratio:.2f})"

    lines = [
        "## Adversarial Stress Test (10 hand-crafted hard cases)",
        "",
        "These cases are not in any training set; each is engineered to break a specific",
        "baseline failure mode. MIC values are set to be unambiguous against EUCAST v16.0",
        "breakpoints. *Trained* column links to the live HuggingFace Space where you can",
        "inject any case and observe the model's prescription in real time.",
        "",
        "**Pass threshold**: quality\\_ratio >= {:.2f} (near-optimal IDSA-concordant prescription).".format(PASS_THRESHOLD),
        "",
        "**Note on R5**: Baselines make zero tool calls so R5=0. The trained model must beat",
        "baselines on *both* R0-R4 (correct prescription) *and* R5 (systematic investigation).",
        "",
        "| ID | Scenario | Best Drug | Broad-Empiric | Random (seed=42) | EUCAST-Only | Trained |",
        "|----|----------|-----------|---------------|-----------------|-------------|---------|",
    ]

    for case in ADVERSARIAL_CASES:
        cid = case["id"]
        title = case["title"]
        best_drug = f"`{case['expected_drug']}`"
        be_res   = by_case.get(cid, {}).get("broad_empiric")
        rand_res = by_case.get(cid, {}).get("random")
        eu_res   = by_case.get(cid, {}).get("eucast_narrowest")

        lines.append(
            f"| **{cid}** | {title} | {best_drug} | {_cell(be_res)} "
            f"| {_cell(rand_res)} | {_cell(eu_res)} "
            f"| [Live demo](https://huggingface.co/spaces/saaheerpurav/amr-steward) |"
        )

    lines.extend([
        "",
        f"> **Summary**: Broad-empiric {be_pass}/10 pass."
        f" Random(42) {rand_pass}/10 pass."
        f" EUCAST-only {eu_pass}/10 pass."
        f" Trained model: see live HuggingFace Space.",
        "",
        "> **Reproduce**: `python eval_adversarial.py --seed 42`"
        " — runs in under 10 seconds on CPU, no GPU required.",
        "",
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def build_json_output(
    all_results: list[CaseResult],
    seed: int,
) -> dict[str, Any]:
    """Full structured output for machine consumption."""
    by_case: dict[str, dict[str, dict]] = {}
    for r in all_results:
        by_case.setdefault(r.case_id, {})[r.policy] = {
            "drug": r.drug,
            "dose": r.dose,
            "verdict": r.verdict,
            "quality_ratio": round(r.quality_ratio, 4),
            "total": round(r.total, 4),
            "breakdown": {
                "R0_allergy": round(r.R0, 4),
                "R1_activity": round(r.R1, 4),
                "R2_guideline": round(r.R2, 4),
                "R3_stewardship": round(r.R3, 4),
                "R4_dose": round(r.R4, 4),
                "R5_efficiency": round(r.R5, 4),
            },
        }

    pass_rates: dict[str, Any] = {}
    for p in ["broad_empiric", "random", "eucast_narrowest", "oracle"]:
        policy_results = [r for r in all_results if r.policy == p]
        n_pass = sum(1 for r in policy_results if r.quality_ratio >= PASS_THRESHOLD)
        n_subopt = sum(1 for r in policy_results
                       if SUBOPT_THRESHOLD <= r.quality_ratio < PASS_THRESHOLD)
        n_fail = len(policy_results) - n_pass - n_subopt
        pass_rates[p] = {
            "pass": n_pass,
            "subopt": n_subopt,
            "fail": n_fail,
            "pass_rate": round(n_pass / max(len(policy_results), 1), 3),
        }

    return {
        "config": {
            "seed": seed,
            "pass_threshold": PASS_THRESHOLD,
            "subopt_threshold": SUBOPT_THRESHOLD,
            "n_cases": len(ADVERSARIAL_CASES),
        },
        "pass_rates": pass_rates,
        "cases": by_case,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run AMR-Steward adversarial stress test (10 hard cases)."
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for the 'random' baseline (default: 42)")
    parser.add_argument("--out", default="adversarial_results.json",
                        help="JSON output path")
    parser.add_argument("--md", default="README_adversarial_table.md",
                        help="Markdown output path")
    args = parser.parse_args()

    print("[adversarial] Validating case definitions ...")
    _preflight_validate()

    print(f"[adversarial] Running {len(ADVERSARIAL_CASES)} cases x 4 policies (seed={args.seed}) ...")
    rng = random.Random(args.seed)
    all_results: list[CaseResult] = []
    for case in ADVERSARIAL_CASES:
        results = run_case(case, rng)
        all_results.extend(results)
        # Quick per-case summary
        by_policy = {r.policy: r for r in results}
        or_qr  = by_policy["oracle"].quality_ratio
        be_qr  = by_policy["broad_empiric"].quality_ratio
        eu_qr  = by_policy["eucast_narrowest"].quality_ratio
        print(
            f"  {case['id']:<3} oracle={or_qr:.2f}  "
            f"broad_empiric={be_qr:.2f}  eucast_only={eu_qr:.2f}"
        )

    # Console table
    console_output = format_console_table(all_results)
    print(console_output)

    # JSON
    json_data = build_json_output(all_results, args.seed)
    json_path = Path(args.out)
    json_path.write_text(json.dumps(json_data, indent=2), encoding="utf-8")
    print(f"[adversarial] JSON saved to {json_path}")

    # Markdown
    md_text = format_markdown_table(all_results)
    md_path = Path(args.md)
    md_path.write_text(md_text, encoding="utf-8")
    print(f"[adversarial] Markdown saved to {md_path}")

    # Sanity-check assertions (fail loud if predictions are wrong)
    be_pass_rate = json_data["pass_rates"]["broad_empiric"]["pass_rate"]
    eu_pass_rate = json_data["pass_rates"]["eucast_narrowest"]["pass_rate"]
    assert be_pass_rate <= 0.30, (
        f"[SANITY FAIL] broad_empiric pass rate {be_pass_rate:.0%} > 30% — "
        f"cases are not adversarial enough."
    )
    assert eu_pass_rate >= 0.70, (
        f"[SANITY FAIL] eucast_narrowest pass rate {eu_pass_rate:.0%} < 70% — "
        f"cases are too hard even for the oracle-style baseline."
    )
    print(
        f"[adversarial] Sanity checks passed: "
        f"broad_empiric {be_pass_rate:.0%} pass, "
        f"eucast_narrowest {eu_pass_rate:.0%} pass."
    )


if __name__ == "__main__":
    main()
