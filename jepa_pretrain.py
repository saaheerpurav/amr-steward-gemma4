"""
JEPA World Model Pre-training for AMR-Steward.

Generates synthetic clinical episodes, simulates tool calls, and trains the
JEPA predictor to predict how each tool call will change the known state.
Saves jepa_weights.pt to the repo root — the environment auto-loads it.

Run: python jepa_pretrain.py
No GPU required. Takes ~60-90 seconds.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from data.eucast_parser import classify_mic
from data.patient_generator import generate_patient
from env.world_model import (
    AVAILABLE_TOOLS,
    NUM_TOOLS,
    TOOL_TO_IDX,
    WEIGHTS_PATH,
    AMRWorldModel,
)

# ---------------------------------------------------------------------------
# Load data files (no openenv-core needed)
# ---------------------------------------------------------------------------

_DATA = Path(__file__).parent / "data"


def _load_json(name: str) -> dict:
    p = _DATA / name
    with open(p, encoding="utf-8") as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if not k.startswith("_")}


_IDSA = _load_json("idsa_guidelines.json")
_DRUG_PROPS = _load_json("drug_properties.json")


# ---------------------------------------------------------------------------
# Inline tool simulators (avoids openenv-core import in env/environment.py)
# ---------------------------------------------------------------------------

def _sim_interpret_resistance(drug: str, patient) -> str:
    mic = patient.antibiogram.get(drug)
    if mic is None:
        return f"No MIC data available for {drug} in this antibiogram."
    cls = classify_mic(patient.organism, drug, mic)
    labels = {"S": "Susceptible", "I": "Intermediate", "R": "Resistant"}
    label = labels.get(cls, "UNKNOWN (no breakpoint)")
    return (
        f"{drug.capitalize()} MIC = {mic} mg/L -> EUCAST classification: {label} "
        f"(organism: {patient.organism})."
    )


def _sim_check_guideline(syndrome: str, patient) -> str:
    syndrome_data = _IDSA.get(syndrome, {})
    pheno = patient.phenotype
    org = patient.organism
    candidates = [
        org,
        f"{org} ({pheno})",
        f"{org} (susceptible)" if pheno == "susceptible" else None,
        f"{org} (MSSA)" if org == "S. aureus" and pheno == "susceptible" else None,
        f"{org} (MRSA)" if org == "S. aureus" and pheno in ("resistant", "MDR") else None,
        f"{org} (ESBL)" if pheno == "resistant" else None,
        f"{org} (CRE)" if pheno in ("resistant", "MDR") else None,
        f"{org} (VSE)" if org == "Enterococcus" and pheno == "susceptible" else None,
        f"{org} (VRE)" if org == "Enterococcus" and pheno in ("resistant", "MDR") else None,
    ]
    for key in (k for k in candidates if k):
        if key in syndrome_data:
            rec = syndrome_data[key]
            alts = ", ".join(rec.get("alternatives", [])) or "none listed"
            return (
                f"IDSA recommendation for {key} + {syndrome}: "
                f"First-line: {rec['first_line']} - {rec.get('dose', '')} "
                f"for {rec.get('duration', '')}. "
                f"Alternatives: {alts}. Notes: {rec.get('notes', 'none')}."
            )
    return f"No specific IDSA recommendation for {org} ({pheno}) + {syndrome}."


def _sim_assess_patient(patient) -> str:
    crcl = patient.creatinine_clearance
    allergies = patient.allergies
    if crcl >= 50:
        renal_label = "normal / mild impairment"
        tier = "CrCl_above_50"
    elif crcl >= 30:
        renal_label = "moderate impairment"
        tier = "CrCl_30_50"
    elif crcl >= 10:
        renal_label = "severe impairment"
        tier = "CrCl_10_30"
    else:
        renal_label = "kidney failure / dialysis-range"
        tier = "CrCl_under_10"

    lines = [
        f"Renal function: CrCl {crcl} mL/min ({renal_label}).",
        f"Allergies reported: {', '.join(allergies) if allergies else 'none'}.",
        "Renal dosing alerts for drugs in this antibiogram:",
    ]
    for drug in patient.antibiogram:
        props = _DRUG_PROPS.get(drug, {})
        adj = props.get("renal_adjustments", {})
        dose = adj.get(tier) or adj.get("CrCl_above_50") or "not specified"
        lines.append(f"  - {drug}: {dose}")
    return "\n".join(lines)


def _call_tool(tool_key: str, patient) -> str:
    """Simulate a tool call and return the result text."""
    if tool_key.startswith("interpret_resistance_"):
        drug = tool_key[len("interpret_resistance_"):]
        return _sim_interpret_resistance(drug, patient)
    if tool_key.startswith("check_guideline_"):
        syndrome = tool_key[len("check_guideline_"):]
        return _sim_check_guideline(syndrome, patient)
    if tool_key == "assess_patient_factors":
        return _sim_assess_patient(patient)
    return ""


# ---------------------------------------------------------------------------
# Training data generation
# ---------------------------------------------------------------------------

def generate_triples(
    world_model: AMRWorldModel,
    n_episodes: int = 500,
) -> list[tuple[torch.Tensor, int, torch.Tensor]]:
    """Return (s_before, tool_idx, s_after) triples from synthetic episodes."""
    triples: list[tuple[torch.Tensor, int, torch.Tensor]] = []

    for ep in range(n_episodes):
        level = (ep % 3) + 1
        patient = generate_patient(level)
        pf = patient.__dict__
        abx_keys = {k.lower() for k in patient.antibiogram}

        # Determine which tools are relevant for this patient
        relevant = [
            t for t in AVAILABLE_TOOLS
            if not t.startswith("interpret_resistance_")
            or t[len("interpret_resistance_"):] in abx_keys
        ]

        # State before any tool calls
        s0 = world_model.encode_known_state([], pf)

        # Single-tool triples: s0 → call one tool → s1
        for tool_key in relevant:
            result = _call_tool(tool_key, patient)
            if not result:
                continue
            s1 = world_model.encode_known_state([result], pf)
            triples.append((s0.clone(), TOOL_TO_IDX[tool_key], s1.clone()))

        # Two-tool triples: s1 → call a second tool → s2
        if len(relevant) >= 2:
            t1_key = random.choice(relevant)
            r1 = _call_tool(t1_key, patient)
            s1 = world_model.encode_known_state([r1], pf)
            remaining = [t for t in relevant if t != t1_key]
            for t2_key in random.sample(remaining, min(2, len(remaining))):
                r2 = _call_tool(t2_key, patient)
                s2 = world_model.encode_known_state([r1, r2], pf)
                triples.append((s1.clone(), TOOL_TO_IDX[t2_key], s2.clone()))

    return triples


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    world_model: AMRWorldModel,
    triples: list[tuple[torch.Tensor, int, torch.Tensor]],
    n_epochs: int = 25,
    lr: float = 3e-4,
    batch_size: int = 64,
) -> None:
    optimizer = torch.optim.Adam(
        list(world_model.context_encoder.parameters())
        + list(world_model.predictor.parameters()),
        lr=lr,
    )

    s_before = torch.stack([t[0] for t in triples])
    tool_idxs = torch.tensor([t[1] for t in triples])
    s_after = torch.stack([t[2] for t in triples])
    n = len(triples)

    world_model.train()
    for epoch in range(n_epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        steps = 0

        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            sb = s_before[idx]
            ti = tool_idxs[idx]
            sa = s_after[idx]

            optimizer.zero_grad()

            ctx = world_model.context_encoder(sb)
            onehot = F.one_hot(ti, num_classes=NUM_TOOLS).float()
            pred_next = world_model.predictor(torch.cat([ctx, onehot], dim=-1))

            with torch.no_grad():
                tgt = world_model.target_encoder(sa)

            loss = F.mse_loss(pred_next, tgt)
            loss.backward()
            optimizer.step()
            world_model.update_target_encoder()

            epoch_loss += loss.item() * len(idx)
            steps += len(idx)

        if (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch + 1:>3}/{n_epochs}  loss={epoch_loss / steps:.6f}")

    world_model.eval()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("AMR-Steward JEPA Pre-training")
    print(f"Output: {WEIGHTS_PATH}\n")

    # Fixed seed → reproducible triple count and weights
    random.seed(42)
    torch.manual_seed(42)

    model = AMRWorldModel()

    print("Generating synthetic rollouts...")
    triples = generate_triples(model, n_episodes=500)
    print(f"Collected {len(triples)} (s_before, tool, s_after) triples from 500 seeded episodes.\n")

    print("Training JEPA predictor...")
    train(model, triples, n_epochs=25)

    model.save_weights(WEIGHTS_PATH)
    print(f"\nWeights saved to {WEIGHTS_PATH}")
    print("The environment will auto-load these on next startup.")
