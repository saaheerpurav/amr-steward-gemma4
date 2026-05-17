# AMR-Steward — Full System Architecture

**Document purpose**: Comprehensive technical reference for LLM evaluation, code review, and future development.
All class names, function signatures, and data shapes reflect the live codebase.

---

## 1. System Overview

AMR-Steward is a reinforcement learning environment in which Gemma 4
(gemma-4-e2b-it + LoRA, trained with GRPO) learns to prescribe the correct antibiotic for
drug-resistant bacterial infections. The agent reasons over patient data through a structured
tool-call loop and is rewarded via deterministic clinical lookup tables — no LLM-as-judge.

### Key properties

| Property | Value |
|---|---|
| RL Algorithm | GRPO (Group Relative Policy Optimisation) via TRL |
| Base LLM | Gemma 4 (gemma-4-e2b-it) + LoRA (r=16) |
| Environment framework | Gymnasium-style (openenv-core base classes) |
| Reward type | RLVR — pure-function lookup, fully deterministic |
| World model | I-JEPA-style self-supervised latent predictor |
| Deployment | FastAPI on HuggingFace Spaces (CPU) |
| Concurrent sessions | Yes (`SUPPORTS_CONCURRENT_SESSIONS = True`) |

---

## 2. Repository Layout

```
amr-steward/
├── app.py                    # FastAPI server — singleton env pattern
├── train.py                  # GRPO training script (multi-head rewards)
├── jepa_pretrain.py          # JEPA world model pretraining (offline)
├── jepa_weights.pt           # Pretrained JEPA weights (LFS)
├── requirements.txt
├── Dockerfile                # CPU-optimised HuggingFace Spaces image
│
├── env/                      # Core environment package
│   ├── __init__.py           # Public re-exports
│   ├── environment.py        # AMREnvironment — reset/step/state
│   ├── models.py             # AMRAction, AMRObservation, AMRState, PatientCase
│   ├── reward.py             # R0-R5 pure-function reward components
│   └── world_model.py        # AMRWorldModel — JEPA latent predictor
│
├── data/
│   ├── eucast.csv            # EUCAST v16.0 clinical MIC breakpoints
│   ├── eucast_parser.py      # EUCASTParser.classify_mic()
│   ├── idsa_guidelines.json  # IDSA 2022/2023 first-line + alternatives
│   ├── drug_properties.json  # Renal dosing tiers, allergy flags, spectrum
│   ├── patient_generator.py  # Synthetic PatientCase generator (curriculum)
│   ├── antibiogram_generator.py
│   └── patient_vignettes.md
│
├── docs/                     # Extended documentation (this file)
├── assets/                   # Static images (PNGs embedded in BLOG.md)
├── training/                 # HuggingFace training space Dockerfile + script
├── demo_web/                 # Minimal HTML frontend for live demo
│
├── eval.py                   # Baseline + adversarial evaluation runner
├── eval_published_cases.py   # Validates env against peer-reviewed case reports
├── eval_adversarial.py       # 10 hand-crafted adversarial cases
│
├── test_env.py               # Environment integration smoke tests (8 tests)
├── test_jepa_integration.py  # JEPA reward shaping unit tests (13 tests)
│
└── README.md / BLOG.md       # Primary documentation / submission writeup
```

---

## 3. Data Layer

### 3.1 EUCAST Breakpoints (`data/eucast.csv` + `data/eucast_parser.py`)

`EUCASTParser` is the single source of truth for microbiological activity.

```python
class EUCASTParser:
    def classify_mic(self, organism: str, drug: str, mic: float) -> str:
        # Returns "S" (Susceptible), "I" (Intermediate), or "R" (Resistant)
        # Looks up EUCAST v16.0 breakpoints by organism + drug pair
        # Returns "R" if no breakpoint exists (fail-safe)
```

The CSV has columns: `organism, drug, S_breakpoint, R_breakpoint` (MIC mg/L).
Organism names are normalised to match `PatientCase.organism` spellings.

### 3.2 IDSA Guidelines (`data/idsa_guidelines.json`)

Schema:
```json
{
  "<infection_site>": {
    "<organism_key>": {
      "first_line": "<drug_name>",
      "dose": "<dose_string>",
      "duration": "<duration_string>",
      "notes": "<clinical_notes>",
      "alternatives": ["<drug_name>", ...]
    }
  }
}
```

Infection sites: `"bacteremia"`, `"UTI"`, `"pneumonia"`, `"intra-abdominal"`

Organism keys follow the pattern: `"K. pneumoniae (CRE)"`, `"S. aureus (MRSA)"`, etc.
The reward function maps `PatientCase.organism + PatientCase.phenotype` to these keys
via `_organism_to_idsa_key()` in `reward.py`.

### 3.3 Drug Properties (`data/drug_properties.json`)

Schema per drug:
```json
{
  "<drug_name>": {
    "class": "<antibiotic_class>",
    "standard_dose": "<dose_string>",
    "renal_adjustments": {
      "CrCl_above_50": "<dose_string>",
      "CrCl_30_50": "<dose_string>",
      "CrCl_10_30": "<dose_string>",
      "CrCl_under_10": "<dose_string>"
    },
    "allergy_flags": ["<flag_string>", ...],
    "spectrum": ["<coverage_category>", ...],
    "notes": "<clinical_notes>"
  }
}
```

`allergy_flags` are substring-matched against `PatientCase.allergies` in R0.

---

## 4. Core Data Structures (`env/models.py`)

### 4.1 `PatientCase` (internal dataclass, not serialised over wire)

```python
@dataclass
class PatientCase:
    age: int
    sex: str                     # "M" | "F"
    infection_site: str          # "bacteremia" | "UTI" | "pneumonia" | "intra-abdominal"
    organism: str                # "K. pneumoniae" | "E. coli" | "P. aeruginosa" | ...
    creatinine_clearance: float  # mL/min — continuous renal function proxy
    allergies: list[str]         # e.g. ["penicillin", "cephalosporin"]
    antibiogram: dict[str, float]  # drug → MIC value in mg/L
    phenotype: str               # "susceptible" | "resistant" | "MDR"
    curriculum_level: int        # 1 | 2 | 3
```

### 4.2 `AMRAction` (Pydantic, OpenEnv `Action` subclass)

```python
class AMRAction(Action):
    action_type: str          # "INVESTIGATE" | "COMMIT"
    tool_name: str | None     # "interpret_resistance" | "check_guideline" | "assess_patient_factors"
    tool_arg: str | None      # Drug name (for interpret_resistance) | infection site (for check_guideline)
    prescription: dict | None # Only for COMMIT: {"drug", "dose", "duration", "justification"}
```

Validators enforce:
- `action_type` must be `"INVESTIGATE"` or `"COMMIT"`
- `tool_name` must be one of the three registered tools when `action_type == "INVESTIGATE"`

### 4.3 `AMRObservation` (Pydantic, OpenEnv `Observation` subclass)

```python
class AMRObservation(Observation):
    episode_id: str
    step_count: int
    budget_remaining: int
    done: bool
    reward: float | None          # None during INVESTIGATE, float after COMMIT
    patient_summary: str          # Human-readable patient description for LLM
    tool_results: list[str]       # Accumulated results from all tool calls
    jepa_rankings: str            # PREDICTED INFORMATION GAIN block (from world model)
```

`to_prompt_text()` formats all fields into the text string sent to the LLM.

### 4.4 `AMRState` (Pydantic, OpenEnv `State` subclass)

```python
class AMRState(State):
    episode_id: str
    step_count: int
    curriculum_level: int
    budget_remaining: int
    done: bool
    patient: dict                 # PatientCase.__dict__ snapshot
    tool_results: list[str]
    called_tools: list[str]       # ["tool_name:tool_arg", ...] — dedup set serialised
    dense_accum: float            # Running sum of dense bonuses (capped at 0.20)
    tool_history: list[dict]      # [{tool, arg, jepa_info_gain, actual_delta, consistency_bonus}, ...]
    last_reward_breakdown: dict | None  # R0-R5 breakdown from last COMMIT
```

---

## 5. Environment Lifecycle (`env/environment.py`)

### 5.1 Class: `AMREnvironment`

```python
class AMREnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    _DENSE_NOVEL_TOOL = 0.04   # base dense bonus per novel (tool, arg) pair
    _DENSE_CAP = 0.20          # hard cap on total dense bonus per episode
    _CONSISTENCY_SCALE = 0.02  # max curiosity bonus per step from JEPA delta
```

### 5.2 `reset(curriculum_level=1, episode_id=None, **kwargs) -> AMRObservation`

1. Generates a `PatientCase` via `generate_patient(curriculum_level)` (seeded from episode_id)
2. Initialises a fresh `AMRState` with `budget_remaining = 6 - curriculum_level`
   - Level 1 → 5 tool budget
   - Level 2 → 4 tool budget
   - Level 3 → 3 tool budget
3. Calls `AMRWorldModel.predict_information_gain()` for all tools → builds JEPA rankings string
4. Returns `AMRObservation` with patient summary + JEPA rankings pre-populated

### 5.3 `step(action: AMRAction) -> AMRObservation`

Raises `ValueError` if `state.done == True`.

**INVESTIGATE branch** → `_handle_investigate(action)`:
1. Pre-step: encode `s_before = world_model.encode_known_state(tool_results, patient)`
2. Get `jepa_info_gain = world_model.predict_information_gain(s_before, tool_key)`
3. Execute tool → append result to `state.tool_results`
4. Post-step: encode `s_after`, compute `actual_delta = ‖tgt(s_after) − tgt(s_before)‖₂ / √128`
5. Dense bonus = `base × jepa_scale` where `jepa_scale = 0.5 + jepa_info_gain` (capped at `_DENSE_CAP`)
6. Consistency bonus = `actual_delta × 0.02` (also capped within `_DENSE_CAP`)
7. Append `{tool, arg, jepa_info_gain, actual_delta, consistency_bonus}` to `tool_history`
8. Returns partial `AMRObservation` with `done=False`, `reward=step_reward`

**COMMIT branch** → `_handle_commit(action)`:
1. Calls `compute_total_reward(prescription, patient, ...)` from `reward.py`
2. Sets `state.done = True`, stores `last_reward_breakdown`
3. Returns terminal `AMRObservation` with `done=True`, `reward=total_reward`

### 5.4 State Machine

```
reset()
  │
  ▼
[INVESTIGATING]  ──INVESTIGATE──►  [INVESTIGATING]
  │                                    (budget -= 1)
  │ COMMIT or budget==0
  ▼
[DONE]  (episode ends, reward returned)
```

Budget exhaustion (no COMMIT) → `reward = -0.1`, `done = True`.

---

## 6. JEPA World Model (`env/world_model.py`)

### 6.1 Architecture

```python
class AMRWorldModel(nn.Module):
    context_encoder:  nn.Sequential  # 64 → 256 → 128 (ReLU)
    predictor:        nn.Sequential  # 144 → 256 → 128 (ReLU)  [128 repr + 16 tool one-hot]
    target_encoder:   nn.Sequential  # EMA copy of context_encoder (τ = 0.99)
```

Constants:
```python
STATE_DIM = 64    # Clinical state vector dimensionality
REPR_DIM  = 128   # Latent representation dimensionality
TOOL_DIM  = 16    # Tool one-hot dimensionality (len(AVAILABLE_TOOLS))
```

### 6.2 `AVAILABLE_TOOLS` (16 entries, defines tool one-hot index)

```python
AVAILABLE_TOOLS = [
    "interpret_resistance_meropenem",
    "interpret_resistance_ceftazidime-avibactam",
    "interpret_resistance_ceftriaxone",
    "interpret_resistance_piperacillin-tazobactam",
    "interpret_resistance_colistin",
    "interpret_resistance_vancomycin",
    "interpret_resistance_daptomycin",
    "interpret_resistance_linezolid",
    "interpret_resistance_cefazolin",
    "interpret_resistance_ampicillin",
    "interpret_resistance_ertapenem",
    "interpret_resistance_cefepime",
    "check_guideline_bacteremia",
    "check_guideline_UTI",
    "check_guideline_pneumonia",
    "assess_patient_factors",
]
```

`_action_to_jepa_key(tool_name, tool_arg)` in `environment.py` maps `(tool_name, tool_arg)` → AVAILABLE_TOOLS key.

### 6.3 Training Objective (I-JEPA Pattern)

```python
# Forward pass
ctx_repr  = context_encoder(s_before)            # [B, 128]
tool_oh   = F.one_hot(tool_idx, TOOL_DIM).float() # [B, 16]
pred_repr = predictor(torch.cat([ctx_repr, tool_oh], dim=-1))  # [B, 128]
tgt_repr  = target_encoder(s_after)              # [B, 128]  ← stop-gradient

loss = F.mse_loss(pred_repr, tgt_repr)

# EMA update (after each backward)
for θ_t, θ_c in zip(target_encoder.parameters(), context_encoder.parameters()):
    θ_t.data = τ * θ_t.data + (1 - τ) * θ_c.data   # τ = 0.99
```

Pre-trained on 500 seeded synthetic episodes. Script: `jepa_pretrain.py`. Weights: `jepa_weights.pt`.

### 6.4 `encode_known_state(tool_results, patient_features) -> torch.Tensor [64]`

Builds the 64-dim clinical state vector:

| Dims | Content |
|---|---|
| 0–4 | Organism one-hot: K. pneumoniae, E. coli, P. aeruginosa, S. aureus, Enterococcus |
| 5–7 | Phenotype one-hot: susceptible, resistant, MDR |
| 8–11 | Infection site one-hot: bacteremia, UTI, pneumonia, intra-abdominal |
| 12 | Normalised CrCl: `min(crcl / 120.0, 1.0)` |
| 13 | Penicillin allergy flag |
| 14 | Cephalosporin/sulfa allergy flag |
| 15–30 | Tool-called flags: one per AVAILABLE_TOOLS slot (set to 1.0 if result in tool_results) |
| 31–46 | EUCAST classification per tool: S=1.0, I=0.5, R=0.0 |
| 47–63 | Antibiogram presence flags: one per drug in `_ANTIBIOGRAM_DRUGS` |

### 6.5 `predict_information_gain(known_state, tool_name) -> float`

```python
def predict_information_gain(self, known_state: Tensor, tool_name: str) -> float:
    if tool_name not in TOOL_TO_IDX:
        return 0.0  # fail closed — unknown tool gets zero gain
    
    ctx_repr  = context_encoder(known_state.unsqueeze(0))     # [1, 128]
    pred_repr = predictor(concat(ctx_repr, tool_one_hot))      # [1, 128]
    anchor    = target_encoder(known_state.unsqueeze(0))       # [1, 128]
    
    gain = ‖pred_repr − anchor‖₂ / √REPR_DIM                  # normalised L2
    return float(min(gain, 1.0))
```

Both `pred_repr` and `anchor` are in target-encoder space — matching the SSL training geometry.

### 6.6 EMA Target Encoder Update

```python
def update_target_encoder(self, decay: float = 0.99) -> None:
    for θ_target, θ_context in zip(
        self.target_encoder.parameters(),
        self.context_encoder.parameters()
    ):
        θ_target.data = decay * θ_target.data + (1 - decay) * θ_context.data
```

Called after each JEPA pretraining step. Frozen at inference.

---

## 7. Three JEPA Integration Points in the Training Loop

| # | Mechanism | Code Location | Effect |
|---|---|---|---|
| 1 | **Observation prior** | `environment.py → reset()` | JEPA-ranked top-K tools appended to every observation text |
| 2 | **Reward shaping** | `environment.py → _handle_investigate()` | Dense bonus scaled by `0.5 + jepa_info_gain` (range 0.5×–1.5×) |
| 3 | **Latent consistency** | `environment.py → _handle_investigate()` | `actual_delta × 0.02` curiosity bonus after each tool call |

---

## 8. FastAPI Server (`app.py`)

### 8.1 Singleton Pattern

```python
# Module-level singleton — survives across requests
_env: AMREnvironment | None = None

def _get_env() -> AMREnvironment:
    global _env
    if _env is None:
        _env = AMREnvironment()
    return _env
```

openenv-core's `create_app()` creates a fresh instance per request, which breaks session state.
AMR-Steward owns its routes directly to maintain the singleton.

### 8.2 Registered Routes

| Method | Path | Description |
|---|---|---|
| GET | `/` | HTML landing page |
| POST | `/reset` | Reset episode, returns `AMRObservation` |
| POST | `/step` | Advance episode with `AMRAction`, returns `AMRObservation` |
| GET | `/state` | Returns current `AMRState` |
| GET | `/health` | `{"status": "ok"}` |
| GET | `/docs` | Swagger UI (FastAPI auto-generated) |

### 8.3 Request/Response Schemas

**POST `/reset`**
```json
{
  "curriculum_level": 1,     // 1 | 2 | 3 (optional, default 1)
  "episode_id": "custom-id"  // optional
}
```

**POST `/step`**
```json
{
  "action_type": "INVESTIGATE",
  "tool_name": "interpret_resistance",
  "tool_arg": "meropenem"
}
```
or
```json
{
  "action_type": "COMMIT",
  "prescription": {
    "drug": "ceftazidime-avibactam",
    "dose": "1.25g IV q8h",
    "duration": "14 days",
    "justification": "CRE K. pneumoniae, IDSA first-line, renal-adjusted"
  }
}
```

---

## 9. Multi-Head GRPO Training (`train.py`)

Three independent reward functions passed to `GRPOTrainer.reward_funcs`:

| Head | Signal | Timescale | Purpose |
|---|---|---|---|
| **Format head** | R6 output format | Fast (step ~50) | Teach concise, structured COMMIT |
| **Process head** | R5 + dense shaping | Mid (step ~100) | Shape investigation behaviour |
| **Terminal head** | quality_ratio (R1-R4) | Slow (step ~200+) | RLVR oracle — verifiable clinical quality |

### 9.1 Curriculum Stages

| Stage | Cases | Organisms | Renal | Budget | Result |
|---|---|---|---|---|---|
| 1 | 128 | Susceptible only | Normal | 5 | 0.55 → **0.84** |
| 2 | 64 | + ESBL, MRSA, VRE | Mild–moderate | 4 | 0.40 → **0.79** |
| 3 | 32 | + CRE, XDR, VISA | Severe + allergies | 3 | **0.71** stable |

---

## 10. Available Tools (3 tools, registered in `environment.py`)

### `interpret_resistance(drug: str) → str`
Looks up `patient.antibiogram[drug]` and calls `EUCASTParser.classify_mic()`.
Returns: `"<Drug> MIC = <val> mg/L → EUCAST classification: <S/I/R> (organism: <org>)"`

### `check_guideline(infection_site: str) → str`
Looks up `idsa_guidelines[infection_site][organism_key]`.
Returns first-line drug, dose, duration, alternatives, and clinical notes.

### `assess_patient_factors() → str`
Summarises `PatientCase`: CrCl and renal impairment tier, allergies, age/sex.
Returns human-readable string (no JSON, directly LLM-readable).

---

## 11. Validation Suites

### 11.1 Published Clinical Cases (`eval_published_cases.py`)
3 cases from peer-reviewed literature. Encodes the expert-published prescription into the reward stack.

| Case | Expert Rx | Quality |
|---|---|---|
| CRE bacteremia, post-transplant | Ceftazidime-avibactam 1.25g IV q8h | **1.000** |
| MSSA bacteremia | Cefazolin 2g IV q8h | **1.000** |
| VRE on hemodialysis | Daptomycin 8mg/kg post-HD | **0.939** |

### 11.2 Adversarial Stress Test (`eval_adversarial.py`)
10 hand-crafted cases, each designed to break a specific naive policy.

### 11.3 JEPA Integration Tests (`test_jepa_integration.py`)
13 tests covering: key mapping, tool_history fields, gain/delta bounds, cap enforcement,
JEPA-scaled bonus, repeated-tool handling, consistency bonus bounds, graceful unknown-tool handling,
full episode accumulation, world model loading, step-after-done guard, and correct-Rx reward floor.
