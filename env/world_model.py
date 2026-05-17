
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from pathlib import Path

WEIGHTS_PATH = Path(__file__).parent.parent / "jepa_weights.pt"

AVAILABLE_TOOLS = [
    "interpret_resistance_meropenem",
    "interpret_resistance_ceftazidime-avibactam",
    "interpret_resistance_colistin",
    "interpret_resistance_vancomycin",
    "interpret_resistance_ceftriaxone",
    "interpret_resistance_piperacillin-tazobactam",
    "check_guideline_bacteremia",
    "check_guideline_UTI",
    "check_guideline_pneumonia",
    "check_guideline_intra-abdominal",
    "assess_patient_factors",
    "interpret_resistance_tigecycline",
    "interpret_resistance_cefepime",
    "interpret_resistance_ertapenem",
    "interpret_resistance_linezolid",
    "interpret_resistance_daptomycin",
]
NUM_TOOLS = len(AVAILABLE_TOOLS)
TOOL_TO_IDX = {t: i for i, t in enumerate(AVAILABLE_TOOLS)}

STATE_DIM = 64
REPR_DIM = 128


_ANTIBIOGRAM_DRUGS = [
    "meropenem", "ceftriaxone", "piperacillin-tazobactam", "ertapenem",
    "ceftazidime-avibactam", "colistin", "cefepime", "vancomycin",
    "daptomycin", "linezolid", "oxacillin", "cefazolin",
    "ampicillin", "tigecycline", "trimethoprim-sulfamethoxazole",
    "nitrofurantoin", "meropenem-vaborbactam", "ciprofloxacin",
    "ceftolozane-tazobactam", "cefiderocol", "nafcillin",
    "imipenem", "ampicillin-sulbactam", "fosfomycin", "azithromycin",
]


class AMRWorldModel(nn.Module):
    """JEPA-inspired world model.
    Predicts information gain of running each diagnostic test
    given what the agent already knows."""

    def __init__(self, state_dim: int = STATE_DIM, repr_dim: int = REPR_DIM, ema_decay: float = 0.99):
        super().__init__()
        self.ema_decay = ema_decay

        self.context_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, repr_dim),
        )

        self.target_encoder = deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.predictor = nn.Sequential(
            nn.Linear(repr_dim + NUM_TOOLS, 256),
            nn.ReLU(),
            nn.Linear(256, repr_dim),
        )

    @torch.no_grad()
    def update_target_encoder(self):
        """EMA update of target encoder."""
        for ctx_p, tgt_p in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            tgt_p.data = self.ema_decay * tgt_p.data + (1 - self.ema_decay) * ctx_p.data

    def forward(self, known_state: torch.Tensor, tool_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (predicted_repr, target_repr) for JEPA loss computation."""
        ctx_repr = self.context_encoder(known_state)
        tool_onehot = F.one_hot(tool_idx, num_classes=NUM_TOOLS).float()
        pred_repr = self.predictor(torch.cat([ctx_repr, tool_onehot], dim=-1))
        with torch.no_grad():
            tgt_repr = self.target_encoder(known_state)
        return pred_repr, tgt_repr

    def predict_information_gain(self, known_state: torch.Tensor, tool_name: str) -> float:
        """Information gain in target-encoder space.

        The predictor was trained to map (context_encoder(s_before), tool) →
        target_encoder(s_after). At inference, information gain is the L2 distance
        between that prediction and target_encoder(s_before), so both operands live
        in the same target-encoder embedding space (paper-faithful state delta).

        Returns 0.0 for unknown tool names rather than silently mapping to index 0.
        """
        if tool_name not in TOOL_TO_IDX:
            return 0.0  
        tool_idx = torch.tensor(TOOL_TO_IDX[tool_name])
        with torch.no_grad():
            ctx_repr = self.context_encoder(known_state.unsqueeze(0))
            tgt_anchor = self.target_encoder(known_state.unsqueeze(0))
            tool_onehot = F.one_hot(tool_idx.unsqueeze(0), num_classes=NUM_TOOLS).float()
            pred_next = self.predictor(torch.cat([ctx_repr, tool_onehot], dim=-1))
            gain = torch.norm(pred_next - tgt_anchor, dim=-1).item() / (REPR_DIM ** 0.5)
        return float(max(0.0, min(1.0, gain)))

    def save_weights(self, path: Path = WEIGHTS_PATH) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load_from_weights(cls, path: Path = WEIGHTS_PATH) -> "AMRWorldModel":
        model = cls()
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.eval()
        return model

    def get_test_rankings(self, known_state: torch.Tensor, available_tools: list[str]) -> list[tuple[str, float]]:
        """Returns tools sorted by predicted information gain (highest first)."""
        scores = [(tool, self.predict_information_gain(known_state, tool)) for tool in available_tools]
        return sorted(scores, key=lambda x: x[1], reverse=True)

    def encode_known_state(self, tool_results: list[str], patient_features: dict) -> torch.Tensor:
        """Convert tool results + patient features into a 64-dim state vector.

        Layout (64 dims):
          [0:5]   organism one-hot
          [5:8]   phenotype one-hot
          [8:12]  infection site one-hot
          [12]    CrCl / 120 (normalized)
          [13]    penicillin allergy flag
          [14]    cephalosporin / sulfonamide allergy flag
          [15:31] tool-already-called flags (16 tools)
          [31:47] drug classification from tool results (16 slots, one per tool;
                  1.0=S, 0.5=I, 0.0=R, -0.5=in-antibiogram-untested, 0.0=n/a)
          [47:64] antibiogram presence flags (17 of the 25 tracked drugs, 64-47=17)
        """
        vec = torch.zeros(STATE_DIM)

      
        organisms = ["K. pneumoniae", "E. coli", "P. aeruginosa", "S. aureus", "Enterococcus"]
        org = patient_features.get("organism", "")
        for i, o in enumerate(organisms):
            if o.lower() in org.lower():
                vec[i] = 1.0
                break

        
        phenotypes = ["susceptible", "resistant", "MDR"]
        pheno = patient_features.get("phenotype", "").lower()
        for i, p in enumerate(phenotypes):
            if p.lower() == pheno:
                vec[5 + i] = 1.0
                break


        sites = ["bacteremia", "UTI", "pneumonia", "intra-abdominal"]
        site = patient_features.get("infection_site", "")
        for i, s in enumerate(sites):
            if s.lower() == site.lower():
                vec[8 + i] = 1.0
                break

      
        crcl = float(patient_features.get("creatinine_clearance", 60.0))
        vec[12] = min(crcl / 120.0, 1.0)


        allergies = [a.lower() for a in patient_features.get("allergies", [])]
        vec[13] = 1.0 if any("penicillin" in a for a in allergies) else 0.0
        vec[14] = 1.0 if any(a in ("cephalosporin", "sulfonamide", "sulfa") for a in allergies) else 0.0

     
        drug_class: dict[str, float] = {}  # drug_name -> 1.0/0.5/0.0
        for result in tool_results:
            rl = result.lower()
            
            drug_hit = None
            for drug in _ANTIBIOGRAM_DRUGS:
                if drug in rl and ("mic" in rl or "susceptib" in rl or "resistant" in rl):
                    drug_hit = drug
                    break
            if drug_hit:
                if "susceptible" in rl:
                    drug_class[drug_hit] = 1.0
                elif "intermediate" in rl:
                    drug_class[drug_hit] = 0.5
                elif "resistant" in rl:
                    drug_class[drug_hit] = 0.0

        results_joined = " ".join(tool_results).lower()

        antibiogram = {k.lower(): v for k, v in patient_features.get("antibiogram", {}).items()}

        for i, tool_key in enumerate(AVAILABLE_TOOLS[:16]):
            slot_called = 15 + i
            slot_class = 31 + i

            if tool_key.startswith("interpret_resistance_"):
                drug = tool_key[len("interpret_resistance_"):]
                if drug in drug_class:
                    vec[slot_called] = 1.0
                    vec[slot_class] = drug_class[drug]
                elif drug in antibiogram:
                    vec[slot_class] = -0.5  # present but not yet interpreted

            elif tool_key.startswith("check_guideline_"):
                syndrome = tool_key[len("check_guideline_"):]
                if "idsa recommendation" in results_joined or "first-line" in results_joined:
                    if syndrome.lower() in results_joined:
                        vec[slot_called] = 1.0

            elif tool_key == "assess_patient_factors":
                if "renal function" in results_joined or "crcl" in results_joined:
                    vec[slot_called] = 1.0

        
        for j, drug in enumerate(_ANTIBIOGRAM_DRUGS[:17]):
            if drug in antibiogram:
                vec[47 + j] = 1.0

        return vec


def enrich_observation(
    base_obs_text: str,
    world_model: AMRWorldModel,
    tool_results: list[str],
    patient_features: dict,
    available_tools: list[str],
) -> str:
    """Append JEPA information gain predictions to the observation text."""
    known_state = world_model.encode_known_state(tool_results, patient_features)
    rankings = world_model.get_test_rankings(known_state, available_tools)

    lines = ["", "PREDICTED INFORMATION GAIN (highest = most useful to run next):"]
    for tool, score in rankings[:5]:  # show top 5 only
        lines.append(f"  - {tool}: {score:.2f}")

    return base_obs_text + "\n".join(lines)
