# AMR-Steward — Reward Specification

**Document purpose**: Complete formal specification of the AMR-Steward reward system for LLM evaluation,
reproducibility, and future development. All formulas reflect the live codebase (`env/reward.py`).

---

## 1. Overview

The reward system is **fully deterministic and verifiable** — no LLM judge, no human rater.
Every component is a pure function of `PatientCase` + `prescription` + lookup tables.

### Total reward formula

```
total = 0.90 × quality_ratio + 0.10 × R5
```

where:

```
quality_ratio = min(1.0, process_score / opt_score)

process_score = 0.40 × R1 + 0.25 × R2 + 0.15 × R3 + 0.10 × R4

opt_score     = compute_optimal_prescription(patient)   ∈ (0.01, 0.90]
```

`quality_ratio` is patient-specific: `opt_score` is the maximum process_score achievable
for *this* patient given their antibiogram, renal function, and allergies. This means
`quality_ratio = 1.0` if and only if the agent found the IDSA-first-line drug at the
correct renal-adjusted dose, given what is achievable for that patient.

### Episode-level reward structure

| Phase | Signal | Range | Source |
|---|---|---|---|
| Each INVESTIGATE step | Dense shaping + JEPA consistency | ≥ 0, capped at 0.20 total | `_handle_investigate()` |
| Budget exhausted without COMMIT | Hard penalty | −0.1 | `_handle_investigate()` |
| COMMIT | `0.90 × quality_ratio + 0.10 × R5` | [0.0, 1.0] | `compute_total_reward()` |

---

## 2. R0 — Allergy Safety Gate

**File**: `reward.py → R0_allergy_safety()`

**Purpose**: Hard safety gate. Prescribing a drug to which the patient is allergic
returns an immediate total reward of **0.0**, regardless of all other components.

**Logic**:
```python
def R0_allergy_safety(prescription, patient, drug_properties) -> float:
    if not patient.allergies:
        return 1.0  # no allergies → safe
    
    drug = normalize(prescription["drug"])
    allergy_flags = drug_properties[drug]["allergy_flags"]
    patient_allergies = [a.lower() for a in patient.allergies]
    
    for flag in allergy_flags:
        if any(pa in flag.lower() for pa in patient_allergies):
            return 0.0  # conflict found → episode reward = 0.0
    
    return 1.0
```

**Behaviour when R0 = 0.0**:
```python
if r0 == 0.0:
    return 0.0, {R0: 0.0, R1: 0.0, R2: 0.0, R3: 0.0, R4: 0.0, R5: 0.0, total: 0.0}
```

R0 fires first; all other components are short-circuited.

**Allergy flag examples** (from `drug_properties.json`):
- `meropenem` → `["penicillin_cross_reactivity_low_risk"]`
- `ceftazidime-avibactam` → `["cephalosporin", "penicillin_cross_reactivity_low_risk"]`
- `vancomycin` → `[]`

**Range**: `{0.0, 1.0}`

---

## 3. R1 — Microbiological Activity

**File**: `reward.py → R1_microbiological_activity()`

**Purpose**: Does the prescribed drug actually kill the patient's bacteria, according to EUCAST?

**Logic**:
```python
def R1_microbiological_activity(prescription, patient, eucast) -> float:
    drug = normalize(prescription["drug"])
    mic  = patient.antibiogram.get(drug)
    
    if mic is None:
        return 0.0  # drug not in antibiogram → cannot confirm activity
    
    classification = eucast.classify_mic(patient.organism, drug, mic)
    return 1.0 if classification == "S" else 0.0
```

EUCAST classification returns `"S"` (Susceptible), `"I"` (Intermediate), or `"R"` (Resistant).
Only `"S"` earns R1 = 1.0. Intermediate ("I") returns 0.0 — clinically conservative.

**Data source**: `data/eucast.csv` (EUCAST v16.0 clinical breakpoints).

**Range**: `{0.0, 1.0}`

**Weight in process_score**: 0.40 (highest — must work microbiologically)

---

## 4. R2 — Guideline Concordance

**File**: `reward.py → R2_guideline_concordance()`

**Purpose**: Is the prescribed drug what IDSA recommends for this organism + infection site?

**Logic**:
```python
def R2_guideline_concordance(prescription, patient, idsa) -> float:
    drug        = normalize(prescription["drug"])
    syndrome    = patient.infection_site
    idsa_key    = _organism_to_idsa_key(patient.organism, patient.phenotype)
    
    org_data = idsa.get(syndrome, {}).get(idsa_key, {})
    
    if not org_data:
        return 0.0  # no guideline data for this combination
    
    first_line   = normalize(org_data["first_line"])
    alternatives = [normalize(a) for a in org_data.get("alternatives", [])]
    
    if drug == first_line:
        return 1.0
    elif drug in alternatives:
        return 0.5
    else:
        return 0.0
```

**`_organism_to_idsa_key()` mapping examples**:
- `("K. pneumoniae", "resistant")` → `"K. pneumoniae (CRE)"`
- `("S. aureus", "resistant")` → `"S. aureus (MRSA)"`
- `("E. coli", "susceptible")` → `"E. coli (susceptible)"`
- `("Enterococcus", "resistant")` → `"Enterococcus (VRE)"`

**Data source**: `data/idsa_guidelines.json`

**Range**: `{0.0, 0.5, 1.0}`

**Weight in process_score**: 0.25

---

## 5. R3 — Stewardship (Antibiotic Spectrum)

**File**: `reward.py → R3_stewardship()`

**Purpose**: Is this the narrowest-spectrum drug that still works for this patient?
Rewards selecting a targeted antibiotic over a broad-spectrum one when a narrow option exists.

**Conditional on R1**: Returns 0.0 immediately if `r1_score == 0.0`. You cannot earn
stewardship credit for a drug that doesn't even work against the bacteria.

**`SPECTRUM_SCORE` table** (hand-crafted, lower = narrower spectrum):

| Drug | Score |
|---|---|
| nitrofurantoin | 1 |
| fosfomycin | 1 |
| ampicillin | 2 |
| cefazolin | 2 |
| doxycycline | 2 |
| trimethoprim-sulfamethoxazole | 2 |
| ceftriaxone | 3 |
| cefepime | 3 |
| vancomycin | 3 |
| daptomycin | 3 |
| linezolid | 3 |
| azithromycin | 3 |
| levofloxacin | 3 |
| piperacillin-tazobactam | 4 |
| ertapenem | 4 |
| ceftazidime-avibactam | 4 |
| meropenem | 5 |
| imipenem | 5 |
| ceftolozane-tazobactam | 4 |
| colistin | 7 |
| polymyxin b | 7 |

Default (unlisted drug) = 5.

**Logic**:
```python
def R3_stewardship(prescription, patient, eucast, r1_score) -> float:
    if r1_score == 0.0:
        return 0.0
    
    drug             = normalize(prescription["drug"])
    prescribed_score = SPECTRUM_SCORE.get(drug, 5)
    
    # Find all drugs from antibiogram that are susceptible and non-allergenic
    susceptible_drugs = [d for d, mic in patient.antibiogram.items()
                         if eucast.classify_mic(patient.organism, d, mic) == "S"]
    valid_options = [d for d in susceptible_drugs
                     if not has_allergy_conflict(d, patient)]
    
    if not valid_options:
        return 1.0  # only option → full stewardship
    
    min_spectrum = min(SPECTRUM_SCORE.get(d, 5) for d in valid_options)
    
    if prescribed_score == min_spectrum:
        return 1.0
    elif prescribed_score == min_spectrum + 1:
        return 0.5
    else:
        return max(0.0, 1.0 - 0.3 × (prescribed_score - min_spectrum))
```

**Range**: `[0.0, 1.0]`

**Weight in process_score**: 0.15

---

## 6. R4 — Dose Correctness

**File**: `reward.py → R4_dose_correctness()`

**Purpose**: Is the prescribed dose correct for the patient's renal function (CrCl)?

**CrCl tier mapping**:
```python
if   crcl > 50:   tier = "CrCl_above_50"
elif crcl >= 30:  tier = "CrCl_30_50"
elif crcl >= 10:  tier = "CrCl_10_30"
else:             tier = "CrCl_under_10"
```

**Logic**:
```python
def R4_dose_correctness(prescription, patient, drug_properties) -> float:
    drug          = normalize(prescription["drug"])
    prescribed    = prescription["dose"].lower()
    expected      = drug_properties[drug]["renal_adjustments"][tier]
    
    if not expected:
        return 0.5  # no data → neutral
    
    # Exact string match
    if prescribed == expected.lower():
        return 1.0
    
    # Fuzzy numeric match (±10% → 1.0, ±30% → 0.5)
    prescribed_mg = extract_dose_mg(prescribed)
    expected_mg   = extract_dose_mg(expected)
    
    if prescribed_mg and expected_mg:
        ratio = prescribed_mg / expected_mg
        if 0.9 <= ratio <= 1.1:  return 1.0
        if 0.7 <= ratio <= 1.3:  return 0.5
    
    return 0.0
```

`extract_dose_mg()` parses strings like `"1.25g IV q8h"` → 1250.0 mg, `"500mg IV q12h"` → 500.0 mg.

**Return value for missing data**: 0.5 (neutral — not penalised for uncommon drugs with no renal data).

**Range**: `{0.0, 0.5, 1.0}`

**Weight in process_score**: 0.10

---

## 7. R5 — Tool Efficiency (Investigation Thoroughness)

**File**: `reward.py → R5_tool_efficiency()`

**Purpose**: Did the agent conduct a thorough investigation before committing?
Rewards calling multiple *distinct* tool types; dilutes score slightly for redundant calls.
Does **not** penalise spending budget (an agent that calls all 3 tools in a 5-step budget is not penalised for using 3 steps).

**Logic**:
```python
def R5_tool_efficiency(unique_tool_types, budget_spent, budget_remaining, budget_total) -> float:
    if budget_spent == 0:
        return 0.0
    
    expected_tools   = min(3, budget_total)        # "complete" = 3 distinct tools
    efficiency_ratio = unique_tool_types / max(1, budget_spent)  # penalises spam
    completeness     = min(1.0, unique_tool_types / expected_tools)
    
    return round(0.8 × completeness + 0.2 × efficiency_ratio, 4)
```

**Scoring examples**:

| Scenario | unique_types | budget_spent | R5 |
|---|---|---|---|
| No investigation (blind COMMIT) | 0 | 0 | 0.0 |
| 1 tool call | 1 | 1 | 0.467 |
| 2 tool calls, 2 different tools | 2 | 2 | 0.733 |
| 3 tool calls, 3 different tools | 3 | 3 | **1.000** |
| 3 tool calls, 2 different tools (1 repeat) | 2 | 3 | 0.680 |
| 3 tool calls, all same tool (spam) | 1 | 3 | 0.333 |

`count_unique_tool_types(tool_history)` counts distinct `tool` *names* (not tool+arg pairs):
two calls to `interpret_resistance` on different drugs count as **one** tool type.

**Range**: `[0.0, 1.0]`

**Weight in total**: 0.10

---

## 8. `compute_optimal_prescription(patient, eucast, idsa, drug_properties) → float`

**Purpose**: Computes the maximum achievable `process_score` for this specific patient.
Used as the denominator in `quality_ratio`.

**Algorithm**:
```python
def compute_optimal_prescription(patient, eucast, idsa, drug_properties) -> float:
    best = 0.01  # never zero — avoids division by zero
    
    for drug in patient.antibiogram:
        drug_norm = normalize(drug)
        dose      = get_renal_adjusted_dose(drug_norm, patient.creatinine_clearance, drug_properties)
        
        prx   = {"drug": drug_norm, "dose": dose, "duration": "", "justification": ""}
        r1    = R1_microbiological_activity(prx, patient, eucast)
        r2    = R2_guideline_concordance(prx, patient, idsa)
        r3    = R3_stewardship(prx, patient, eucast, r1)
        r4    = R4_dose_correctness(prx, patient, drug_properties)
        score = 0.40*r1 + 0.25*r2 + 0.15*r3 + 0.10*r4
        
        if score > best:
            best = score
    
    return best  # ∈ (0.01, 0.90]
```

`quality_ratio = 1.0` iff the agent's process_score equals opt_score — i.e., it found the
IDSA-first-line drug at the correct dose, given this patient's antibiogram.

---

## 9. Dense Shaping During Investigation

### 9.1 JEPA-Weighted Dense Bonus

Per INVESTIGATE step, for each novel `(tool_name, tool_arg)` pair:

```python
jepa_info_gain = world_model.predict_information_gain(s_before, jepa_key)  # ∈ [0.0, 1.0]
jepa_scale     = 0.5 + jepa_info_gain                                       # ∈ [0.5, 1.5]
raw_bonus      = _DENSE_NOVEL_TOOL × jepa_scale                             # base = 0.04
inc            = min(raw_bonus, _DENSE_CAP - dense_accum)                  # cap at 0.20
```

**Interpretation**:
- World model's top-predicted tool: `jepa_info_gain ≈ 0.12` → `jepa_scale ≈ 1.12` → `bonus ≈ 0.045`
- World model's lowest-predicted tool: `jepa_info_gain ≈ 0.0` → `jepa_scale = 0.5` → `bonus = 0.020`
- Repeated tool call (already in `called_tools`): raw bonus = 0.0

### 9.2 Latent State Consistency Bonus

After each INVESTIGATE step:

```python
s_after        = world_model.encode_known_state(tool_results, patient)
tgt_before     = target_encoder(s_before)               # [1, 128]
tgt_after      = target_encoder(s_after)                # [1, 128]
actual_delta   = ‖tgt_after − tgt_before‖₂ / √128     # normalised ∈ [0.0, 1.0]
bonus          = min(actual_delta × 0.02, _DENSE_CAP - dense_accum)
```

Rewards tool calls that caused the agent's known clinical state to shift significantly in
target-encoder space — i.e., that revealed genuinely new information.

### 9.3 Dense Accumulator Cap

Both bonuses accumulate into `state.dense_accum`, which is hard-capped at `_DENSE_CAP = 0.20`.
This ensures the terminal `quality_ratio` (weight 0.90) always dominates the dense shaping (weight ≤ 0.20).

**Per-step budget exhaustion penalty**: If budget reaches 0 with no COMMIT:
```python
return -0.1, True  # terminal, penalised
```

---

## 10. Anti-Gaming Mechanisms

| Mechanism | What it prevents |
|---|---|
| R0 hard gate (allergy → 0.0) | Prescribing drugs that could kill the patient regardless of other quality |
| R3 gated on R1 | Getting stewardship credit for a narrow drug that doesn't even work |
| `quality_ratio = process_score / opt_score` | Patient-specific ceiling prevents easy cases inflating the score |
| R5 diversity term | Agent cannot spam one tool repeatedly to farm dense bonus |
| `_DENSE_CAP = 0.20` | Dense bonuses cannot exceed 22% of max total; terminal reward dominates |
| `called_tools` dedup set | Repeated `(tool, arg)` pair earns zero dense bonus |
| Budget exhaustion penalty (−0.1) | Agent cannot avoid committing by doing nothing |

---

## 11. `compute_total_reward()` — Full Signature

```python
def compute_total_reward(
    prescription:     dict,            # {"drug", "dose", "duration", "justification"}
    patient:          PatientCase,
    tool_call_history: list[str],      # legacy free-text results (fallback only)
    eucast:           EUCASTParser,
    idsa:             dict | None,     # loads from idsa_guidelines.json if None
    drug_properties:  dict | None,     # loads from drug_properties.json if None
    budget_remaining: int | None,      # steps remaining at COMMIT time
    budget_total:     int,             # episode budget at reset (default 5)
    tool_history:     list[dict] | None  # preferred: [{tool, arg, ...}, ...]
) -> tuple[float, dict]:
    """
    Returns: (total_reward, breakdown_dict)
    breakdown_dict keys: R0_allergy, R1_activity, R2_guideline, R3_stewardship,
                         R4_dose, R5_efficiency, quality_ratio, total
    """
```

---

## 12. Reward Component Summary Table

| Component | Function | Weight | Range | Notes |
|---|---|---|---|---|
| R0 Allergy Safety | `R0_allergy_safety()` | Gate | {0, 1} | 0 → total=0 immediately |
| R1 Activity | `R1_microbiological_activity()` | 0.40 | {0, 1} | EUCAST S=1, I/R=0 |
| R2 Concordance | `R2_guideline_concordance()` | 0.25 | {0, 0.5, 1} | IDSA first=1, alt=0.5 |
| R3 Stewardship | `R3_stewardship()` | 0.15 | [0, 1] | Gated on R1; SPECTRUM_SCORE graded |
| R4 Dose | `R4_dose_correctness()` | 0.10 | {0, 0.5, 1} | CrCl-tier lookup; ±10% fuzzy |
| R5 Efficiency | `R5_tool_efficiency()` | 0.10 (of total) | [0, 1] | Thoroughness; no budget penalty |
| Dense shaping | JEPA-weighted per-step | up to +0.20 | [0, 0.20] | Accumulated during INVESTIGATE |
| Consistency bonus | Latent delta curiosity | part of dense cap | [0, 0.02] per step | Target-encoder space delta |
| Terminal | `quality_ratio` | 0.90 (of total) | [0, 1] | Patient-specific ceiling |
