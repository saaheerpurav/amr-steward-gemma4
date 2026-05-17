"""
env/environment.py — AMR-Steward RL environment.

Lifecycle:
  reset(seed, episode_id, curriculum_level=1) -> AMRObservation
  step(action)                                -> AMRObservation  (reward inside obs.reward)
  state                                       -> AMRState        (property, has episode_id + step_count)
"""

from __future__ import annotations

import json
import logging
import os
import random
import uuid
from pathlib import Path
from typing import Any, Optional

from typing import Generic, TypeVar as _TV

_A = _TV("_A"); _O = _TV("_O"); _S = _TV("_S")

class Environment(Generic[_A, _O, _S]):
    """Minimal RL environment base class."""
    def __init__(self): pass

from .models import AMRAction, AMRObservation, AMRState, PatientCase
from .world_model import AMRWorldModel, AVAILABLE_TOOLS, WEIGHTS_PATH, enrich_observation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BUDGET_BY_LEVEL: dict[int, int] = {1: 5, 2: 4, 3: 3}

_HERE = Path(__file__).parent.parent
DATA_DIR = _HERE / "data"


# ---------------------------------------------------------------------------
# Lazy data loaders (loaded once, shared across episodes)
# ---------------------------------------------------------------------------

def _load_json(filename: str) -> dict:
    path = DATA_DIR / filename
    if not path.exists():
        logger.warning("Data file not found: %s — using empty dict", path)
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


_idsa: dict | None = None
_drug_properties: dict | None = None
_eucast: Any | None = None
_world_model: AMRWorldModel | None = None


def _get_idsa() -> dict:
    global _idsa
    if _idsa is None:
        raw = _load_json("idsa_guidelines.json")
        _idsa = {k: v for k, v in raw.items() if not k.startswith("_")}
    return _idsa


def _get_drug_properties() -> dict:
    global _drug_properties
    if _drug_properties is None:
        raw = _load_json("drug_properties.json")
        _drug_properties = {k: v for k, v in raw.items() if not k.startswith("_")}
    return _drug_properties


def _get_eucast():
    global _eucast
    if _eucast is None:
        import sys
        sys.path.insert(0, str(_HERE))
        from data.eucast_parser import classify_mic, is_susceptible
        _eucast = type("EucastParser", (), {
            "classify_mic": staticmethod(classify_mic),
            "is_susceptible": staticmethod(is_susceptible),
        })
    return _eucast


def _get_world_model() -> AMRWorldModel:
    global _world_model
    if _world_model is None:
        if WEIGHTS_PATH.exists():
            _world_model = AMRWorldModel.load_from_weights(WEIGHTS_PATH)
            logger.info("JEPA world model loaded from %s", WEIGHTS_PATH)
        else:
            _world_model = AMRWorldModel()
            _world_model.eval()
            logger.warning("jepa_weights.pt not found — world model is randomly initialised. Run jepa_pretrain.py.")
    return _world_model


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

def interpret_resistance(drug: str, patient: PatientCase, eucast) -> str:
    mic = patient.antibiogram.get(drug)
    if mic is None:
        return (
            f"No MIC data available for {drug} in this antibiogram. "
            f"Available drugs: {', '.join(patient.antibiogram.keys()) or 'none'}."
        )
    classification = eucast.classify_mic(patient.organism, drug, mic)
    labels = {"S": "Susceptible", "I": "Intermediate", "R": "Resistant",
              "UNKNOWN": "UNKNOWN (no breakpoint)"}
    label = labels.get(classification, classification)
    return (
        f"{drug.capitalize()} MIC = {mic} mg/L -> EUCAST classification: {label} "
        f"(organism: {patient.organism})."
    )


def check_guideline(syndrome: str, patient: PatientCase, idsa: dict) -> str:
    syndrome_data = idsa.get(syndrome)
    if syndrome_data is None:
        available = ", ".join(idsa.keys())
        return f"No IDSA data found for syndrome '{syndrome}'. Available: {available}."

    organism = patient.organism
    phenotype = patient.phenotype
    # Specific keys first — matches _organism_to_idsa_key resolution in reward.py
    candidate_keys = [
        f"{organism} (MSSA)" if organism == "S. aureus" and phenotype == "susceptible" else None,
        f"{organism} (MRSA)" if organism == "S. aureus" and phenotype in ("resistant", "MDR") else None,
        f"{organism} (ESBL)" if phenotype == "resistant" else None,
        f"{organism} (CRE)"  if phenotype in ("resistant", "MDR") else None,
        f"{organism} (VSE)"  if organism == "Enterococcus" and phenotype == "susceptible" else None,
        f"{organism} (VRE)"  if organism == "Enterococcus" and phenotype in ("resistant", "MDR") else None,
        f"{organism} (susceptible)" if phenotype == "susceptible" else None,
        f"{organism} ({phenotype})",
        organism,
    ]
    candidate_keys = [k for k in candidate_keys if k is not None]

    rec = None
    matched_key = None
    for key in candidate_keys:
        if key in syndrome_data:
            rec = syndrome_data[key]
            matched_key = key
            break

    if rec is None:
        available_keys = ", ".join(syndrome_data.keys())
        return (
            f"No specific IDSA recommendation for {organism} ({phenotype}) + {syndrome}. "
            f"Available entries: {available_keys}."
        )

    alts = ", ".join(rec.get("alternatives", [])) or "none listed"
    return (
        f"IDSA recommendation for {matched_key} + {syndrome}:\n"
        f"  First-line: {rec['first_line']} - {rec.get('dose', 'dose not specified')} "
        f"for {rec.get('duration', 'duration not specified')}.\n"
        f"  Alternatives: {alts}.\n"
        f"  Notes: {rec.get('notes', 'none')}."
    )


def assess_patient_factors(patient: PatientCase, drug_properties: dict) -> str:
    crcl = patient.creatinine_clearance
    allergies = patient.allergies

    if crcl >= 50:
        renal_tier, renal_label = "CrCl_above_50", "normal / mild impairment"
    elif crcl >= 30:
        renal_tier, renal_label = "CrCl_30_50", "moderate impairment"
    elif crcl >= 10:
        renal_tier, renal_label = "CrCl_10_30", "severe impairment"
    else:
        renal_tier, renal_label = "CrCl_under_10", "kidney failure / dialysis-range"

    lines = [
        f"Renal function: CrCl {crcl} mL/min ({renal_label}).",
        f"Allergies reported: {', '.join(allergies) if allergies else 'none'}.",
        "Renal dosing alerts for drugs in this antibiogram:",
    ]
    rows: list[str] = []
    for drug in patient.antibiogram:
        props = drug_properties.get(drug)
        if props is None:
            continue
        adj = props.get("renal_adjustments", {})
        dose_at_tier = (
            adj.get(renal_tier)
            or adj.get("CrCl_above_50")
            or next(iter(adj.values()), "not specified")
        )
        flags: list[str] = props.get("allergy_flags", [])
        conflicts = [
            flag for flag in flags
            if any(a.lower() in flag.lower() for a in allergies)
        ]
        line = f"  - {drug}: {dose_at_tier}"
        if conflicts:
            line += f"   ALLERGY FLAG: {', '.join(conflicts)}"
        rows.append(line)

    lines.extend(rows or ["  (No dosing data found for antibiogram drugs in drug_properties.json)"])
    if allergies:
        lines.append(
            f"\nNote: Patient has documented allergy to {', '.join(allergies)}. "
            "Verify allergy history and cross-reactivity risk before prescribing."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JEPA helper
# ---------------------------------------------------------------------------

def _action_to_jepa_key(tool_name: str | None, tool_arg: str | None) -> str:
    """Map (tool_name, tool_arg) to an AVAILABLE_TOOLS compound key."""
    if not tool_name:
        return ""
    if tool_name == "interpret_resistance" and tool_arg:
        return f"interpret_resistance_{tool_arg.lower()}"
    if tool_name == "check_guideline" and tool_arg:
        return f"check_guideline_{tool_arg}"
    return tool_name  # "assess_patient_factors" maps directly


# ---------------------------------------------------------------------------
# Environment class — OpenEnv compliant
# ---------------------------------------------------------------------------

class AMREnvironment(Environment[AMRAction, AMRObservation, AMRState]):
    """
    OpenEnv-compatible RL environment for antibiotic prescribing decisions.

    Concurrent sessions are supported by the OpenEnv HTTP server: each
    session gets its own AMREnvironment instance, so per-episode mutable
    state on `self` is safe.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    _DENSE_NOVEL_TOOL = 0.04
    _DENSE_CAP = 0.20
    _CONSISTENCY_SCALE = 0.02   # max curiosity bonus per step from latent state delta

    def __init__(self) -> None:
        super().__init__()
        self.current_patient: PatientCase | None = None
        self._dense_accum: float = 0.0
        self._called_tools: set[str] = set()
        self._state: AMRState = AMRState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            curriculum_level=1,
            budget_remaining=BUDGET_BY_LEVEL[1],
            done=False,
        )

    # ------------------------------------------------------------------
    # Public OpenEnv API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        curriculum_level: int = 1,
        patient: Optional[PatientCase] = None,
        **kwargs: Any,
    ) -> AMRObservation:
        """Start a fresh episode with a new (or provided) patient.

        Args:
            patient: Optional pre-built PatientCase. When provided the random
                     patient sampler is skipped — used by training rollout to
                     replay completions against the exact same patient that was
                     used to build the prompt.
        """
        if curriculum_level not in BUDGET_BY_LEVEL:
            raise ValueError(f"curriculum_level must be 1, 2, or 3. Got: {curriculum_level}")

        if seed is not None:
            random.seed(seed)

        if patient is not None:
            self.current_patient = patient
        else:
            self.current_patient = self._sample_patient(curriculum_level)

        self._dense_accum = 0.0
        self._called_tools = set()
        self._state = AMRState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            curriculum_level=curriculum_level,
            budget_remaining=BUDGET_BY_LEVEL[curriculum_level],
            done=False,
            patient=self.current_patient.__dict__.copy(),
            tool_results=[],
            called_tools=[],
            dense_accum=0.0,
            tool_history=[],
            last_reward_breakdown=None,
        )

        logger.info(
            "Episode reset | episode_id=%s | level=%d | patient=%d%s %s | "
            "organism=%s | phenotype=%s | budget=%d",
            self._state.episode_id,
            curriculum_level,
            self.current_patient.age,
            self.current_patient.sex,
            self.current_patient.infection_site,
            self.current_patient.organism,
            self.current_patient.phenotype,
            self._state.budget_remaining,
        )
        return self._build_observation(reward=None)

    def step(
        self,
        action: AMRAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> AMRObservation:
        """Apply an action. Reward is returned via observation.reward."""
        # openenv-core may serialize env.state and inject it into a fresh instance
        # between requests. Restore Python-only attributes from the state dict.
        if self.current_patient is None and self._state.patient is not None:
            self.current_patient = PatientCase(**self._state.patient)
            self._dense_accum = self._state.dense_accum
            self._called_tools = set(self._state.called_tools)

        if self._state.done:
            raise ValueError("Episode is already done. Call reset() first.")
        if self.current_patient is None:
            raise ValueError("No active episode. Call reset() first.")

        self._state.step_count += 1

        if action.action_type == "INVESTIGATE":
            reward, done = self._handle_investigate(action)
        elif action.action_type == "COMMIT":
            reward, done = self._handle_commit(action)
        else:
            raise ValueError(
                f"Unknown action_type '{action.action_type}'. "
                "Must be 'INVESTIGATE' or 'COMMIT'."
            )

        self._state.done = done
        obs = self._build_observation(reward=reward)
        logger.info(
            "Step %d | episode_id=%s | action=%s | reward=%.4f | done=%s | budget=%d",
            self._state.step_count,
            self._state.episode_id,
            action.action_type,
            reward,
            done,
            self._state.budget_remaining,
        )
        return obs

    @property
    def state(self) -> AMRState:
        """Return current episode state — required by OpenEnv base class."""
        return self._state

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _handle_investigate(self, action: AMRAction) -> tuple[float, bool]:
        # --- JEPA: encode state BEFORE tool call ---
        jepa_info_gain = 0.0
        actual_delta = 0.0
        s_before = None
        try:
            import torch
            from .world_model import REPR_DIM
            wm = _get_world_model()
            patient_features = self.current_patient.__dict__
            s_before = wm.encode_known_state(
                list(self._state.tool_results), patient_features
            )
            jepa_key = _action_to_jepa_key(action.tool_name, action.tool_arg)
            jepa_info_gain = wm.predict_information_gain(s_before, jepa_key)
        except Exception as exc:
            logger.debug("JEPA pre-step failed: %s", exc)

        # --- Execute tool ---
        result = self._execute_tool(action.tool_name, action.tool_arg)
        self._state.tool_results.append(result)
        self._state.budget_remaining -= 1

        # --- JEPA: compute actual state delta AFTER tool call ---
        if s_before is not None:
            try:
                import torch
                from .world_model import REPR_DIM
                wm = _get_world_model()
                s_after = wm.encode_known_state(
                    list(self._state.tool_results), patient_features
                )
                with torch.no_grad():
                    tgt_before = wm.target_encoder(s_before.unsqueeze(0))
                    tgt_after = wm.target_encoder(s_after.unsqueeze(0))
                actual_delta = float(min(
                    1.0,
                    torch.norm(tgt_after - tgt_before, dim=-1).item() / (REPR_DIM ** 0.5)
                ))
            except Exception as exc:
                logger.debug("JEPA post-step delta failed: %s", exc)

        # --- Dense shaping: JEPA-scaled bonus for novel tool calls ---
        # Scale base bonus by JEPA score: [0, 1] -> [0.5x, 1.5x] multiplier
        jepa_scale = 0.5 + jepa_info_gain
        tool_key = f"{action.tool_name}:{action.tool_arg or ''}"
        if tool_key not in self._called_tools:
            raw = self._DENSE_NOVEL_TOOL * jepa_scale
            inc = min(raw, self._DENSE_CAP - self._dense_accum)
            self._dense_accum += inc
            step_reward = inc
        else:
            step_reward = 0.0
        self._called_tools.add(tool_key)

        # --- Latent state consistency bonus ---
        consistency_bonus = actual_delta * self._CONSISTENCY_SCALE
        # Ensure total dense stays within cap
        consistency_bonus = min(consistency_bonus, self._DENSE_CAP - self._dense_accum)
        self._dense_accum += consistency_bonus
        step_reward += consistency_bonus

        # Structured tool log — single source of truth for R5 unique_tool_types
        self._state.tool_history.append({
            "tool": action.tool_name or "unknown",
            "arg": action.tool_arg or "",
            "jepa_info_gain": round(jepa_info_gain, 4),
            "actual_delta": round(actual_delta, 4),
            "consistency_bonus": round(consistency_bonus, 4),
        })

        # Persist into state so it survives openenv-core serialization
        self._state.called_tools = list(self._called_tools)
        self._state.dense_accum = self._dense_accum

        logger.info(
            "Tool call | tool=%s | arg=%s | jepa_gain=%.3f | actual_delta=%.3f "
            "| dense_reward=%.4f | result=%s",
            action.tool_name, action.tool_arg,
            jepa_info_gain, actual_delta, step_reward,
            result[:120].replace("\n", " "),
        )

        if self._state.budget_remaining <= 0:
            logger.warning("Budget exhausted without COMMIT — penalising episode.")
            return -0.1, True
        return step_reward, False

    def _handle_commit(self, action: AMRAction) -> tuple[float, bool]:
        if not action.prescription:
            logger.warning("COMMIT missing prescription — returning 0 reward.")
            return 0.0, True

        try:
            from env.reward import compute_total_reward
            total, breakdown = compute_total_reward(
                prescription=action.prescription,
                patient=self.current_patient,
                tool_call_history=list(self._state.tool_results),
                eucast=_get_eucast(),
                idsa=_get_idsa(),
                drug_properties=_get_drug_properties(),
                budget_remaining=self._state.budget_remaining,
                budget_total=BUDGET_BY_LEVEL[self._state.curriculum_level],
                tool_history=list(self._state.tool_history),
            )
            self._state.last_reward_breakdown = breakdown
            logger.info(
                "COMMIT | drug=%s | total=%.4f | breakdown=%s",
                action.prescription.get("drug", "?"), total, breakdown,
            )
            return total, True
        except Exception as exc:
            logger.error("Reward computation error: %s", exc)
            self._state.last_reward_breakdown = {"error": str(exc)}
            return 0.0, True

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def _execute_tool(self, tool_name: str | None, tool_arg: str | None) -> str:
        if not tool_name:
            return "Error: tool_name is required for INVESTIGATE actions."

        patient = self.current_patient
        eucast = _get_eucast()
        idsa = _get_idsa()
        drug_properties = _get_drug_properties()

        if tool_name == "interpret_resistance":
            if not tool_arg:
                return "Error: tool_arg (drug name) is required for interpret_resistance."
            return interpret_resistance(drug=tool_arg, patient=patient, eucast=eucast)

        if tool_name == "check_guideline":
            syndrome = tool_arg or patient.infection_site
            return check_guideline(syndrome=syndrome, patient=patient, idsa=idsa)

        if tool_name == "assess_patient_factors":
            return assess_patient_factors(patient=patient, drug_properties=drug_properties)

        return (
            f"Unknown tool '{tool_name}'. Available: "
            "interpret_resistance | check_guideline | assess_patient_factors."
        )

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_observation(self, reward: float | None) -> AMRObservation:
        p = self.current_patient
        from data.patient_generator import patient_to_text  # type: ignore[import]
        try:
            patient_text = patient_to_text(p)
        except Exception:
            allergy_str = ", ".join(p.allergies) if p.allergies else "None reported"
            patient_text = (
                f"Patient: {p.age}-year-old {p.sex}.\n"
                f"Infection site: {p.infection_site}.\n"
                f"Culture result: {p.organism} isolated.\n"
                f"Renal function: CrCl {p.creatinine_clearance} mL/min.\n"
                f"Allergies: {allergy_str}.\n"
                f"Available antibiogram data: {list(p.antibiogram.keys())}.\n"
            )

        world_model_rankings = ""
        if not self._state.done and p.antibiogram:
            try:
                wm = _get_world_model()
                abx_keys = {k.lower() for k in p.antibiogram}
                relevant = [
                    t for t in AVAILABLE_TOOLS
                    if not t.startswith("interpret_resistance_")
                    or t[len("interpret_resistance_"):] in abx_keys
                ]
                world_model_rankings = enrich_observation(
                    "", wm, list(self._state.tool_results), p.__dict__, relevant
                ).strip()
            except Exception as exc:
                logger.debug("World model enrichment skipped: %s", exc)

        metadata: dict[str, Any] = {
            "episode_id": self._state.episode_id,
            "step_count": self._state.step_count,
            "curriculum_level": self._state.curriculum_level,
        }
        if self._state.done and self._state.last_reward_breakdown is not None:
            metadata["reward_breakdown"] = self._state.last_reward_breakdown

        return AMRObservation(
            done=self._state.done,
            reward=reward,
            metadata=metadata,
            patient_text=patient_text,
            tool_results=list(self._state.tool_results),
            budget_remaining=self._state.budget_remaining,
            world_model_rankings=world_model_rankings,
        )

    # ------------------------------------------------------------------
    # Patient sampling
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_patient(curriculum_level: int) -> PatientCase:
        try:
            import sys
            repo_root = str(Path(__file__).parent.parent)
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)
            from data.patient_generator import generate_patient  # type: ignore[import]
            return generate_patient(curriculum_level)
        except Exception as exc:
            logger.warning(
                "patient_generator unavailable (%s) — using demo patient (67F CRE bacteremia).", exc
            )
            return PatientCase(
                age=67, sex="F",
                infection_site="bacteremia",
                organism="K. pneumoniae",
                creatinine_clearance=35.0,
                allergies=[],
                antibiogram={"meropenem": 8.0, "ceftazidime-avibactam": 1.0, "colistin": 1.0},
                phenotype="resistant",
                curriculum_level=curriculum_level,
            )
