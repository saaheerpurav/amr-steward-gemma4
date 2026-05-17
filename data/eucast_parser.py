# SAAHEER OWNS THIS FILE
# Parses EUCAST clinical breakpoints CSV and classifies MIC values.
# Breakpoints sourced from EUCAST v16.0 (2026); committed as data/eucast.csv.

import csv
import os

_breakpoints: dict = {}  # loaded once at import


def _load_breakpoints(csv_path: str = "data/eucast.csv"):
    """Load EUCAST breakpoints into memory. Call once at startup."""
    global _breakpoints
    if not os.path.exists(csv_path):
        print(f"WARNING: {csv_path} not found. Using stub breakpoints.")
        _breakpoints = _stub_breakpoints()
        return
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["organism"].strip().lower(), row["drug"].strip().lower())
            _breakpoints[key] = {
                "S_breakpoint": float(row["S"]) if row.get("S") else None,
                "R_breakpoint": float(row["R"]) if row.get("R") else None,
            }


def _stub_breakpoints() -> dict:
    """EUCAST 2024 clinical breakpoints (hardcoded). Download eucast.csv to override."""
    return {
        # K. pneumoniae
        ("k. pneumoniae", "meropenem"):              {"S_breakpoint": 2.0,  "R_breakpoint": 8.0},
        ("k. pneumoniae", "ceftriaxone"):            {"S_breakpoint": 1.0,  "R_breakpoint": 2.0},
        ("k. pneumoniae", "ertapenem"):              {"S_breakpoint": 0.5,  "R_breakpoint": 1.0},
        ("k. pneumoniae", "piperacillin-tazobactam"):{"S_breakpoint": 8.0,  "R_breakpoint": 16.0},
        ("k. pneumoniae", "ceftazidime-avibactam"):  {"S_breakpoint": 8.0,  "R_breakpoint": 8.0},
        ("k. pneumoniae", "meropenem-vaborbactam"):  {"S_breakpoint": 2.0,  "R_breakpoint": 8.0},
        ("k. pneumoniae", "colistin"):               {"S_breakpoint": 2.0,  "R_breakpoint": 2.0},
        ("k. pneumoniae", "tigecycline"):            {"S_breakpoint": 1.0,  "R_breakpoint": 2.0},
        ("k. pneumoniae", "cefepime"):               {"S_breakpoint": 1.0,  "R_breakpoint": 4.0},

        # E. coli
        ("e. coli", "ceftriaxone"):                  {"S_breakpoint": 1.0,  "R_breakpoint": 2.0},
        ("e. coli", "meropenem"):                    {"S_breakpoint": 2.0,  "R_breakpoint": 8.0},
        ("e. coli", "ertapenem"):                    {"S_breakpoint": 0.5,  "R_breakpoint": 1.0},
        ("e. coli", "piperacillin-tazobactam"):      {"S_breakpoint": 8.0,  "R_breakpoint": 16.0},
        ("e. coli", "ceftazidime-avibactam"):        {"S_breakpoint": 8.0,  "R_breakpoint": 8.0},
        ("e. coli", "trimethoprim-sulfamethoxazole"):{"S_breakpoint": 2.0,  "R_breakpoint": 4.0},
        ("e. coli", "nitrofurantoin"):               {"S_breakpoint": 32.0, "R_breakpoint": 64.0},
        ("e. coli", "cefepime"):                     {"S_breakpoint": 1.0,  "R_breakpoint": 4.0},

        # P. aeruginosa
        ("p. aeruginosa", "piperacillin-tazobactam"):{"S_breakpoint": 16.0, "R_breakpoint": 16.0},
        ("p. aeruginosa", "cefepime"):               {"S_breakpoint": 8.0,  "R_breakpoint": 8.0},
        ("p. aeruginosa", "meropenem"):              {"S_breakpoint": 2.0,  "R_breakpoint": 8.0},
        ("p. aeruginosa", "ceftazidime-avibactam"):  {"S_breakpoint": 8.0,  "R_breakpoint": 8.0},
        ("p. aeruginosa", "colistin"):               {"S_breakpoint": 2.0,  "R_breakpoint": 2.0},
        ("p. aeruginosa", "imipenem"):               {"S_breakpoint": 2.0,  "R_breakpoint": 8.0},

        # S. aureus
        ("s. aureus", "vancomycin"):                 {"S_breakpoint": 2.0,  "R_breakpoint": 2.0},
        ("s. aureus", "daptomycin"):                 {"S_breakpoint": 1.0,  "R_breakpoint": 1.0},
        ("s. aureus", "linezolid"):                  {"S_breakpoint": 4.0,  "R_breakpoint": 4.0},
        ("s. aureus", "oxacillin"):                  {"S_breakpoint": 2.0,  "R_breakpoint": 4.0},
        ("s. aureus", "cefazolin"):                  {"S_breakpoint": 4.0,  "R_breakpoint": 4.0},

        # Enterococcus
        ("enterococcus", "ampicillin"):              {"S_breakpoint": 4.0,  "R_breakpoint": 8.0},
        ("enterococcus", "vancomycin"):              {"S_breakpoint": 4.0,  "R_breakpoint": 4.0},
        ("enterococcus", "linezolid"):               {"S_breakpoint": 4.0,  "R_breakpoint": 4.0},
        ("enterococcus", "daptomycin"):              {"S_breakpoint": 4.0,  "R_breakpoint": 4.0},

        # Generic fallbacks for common drugs across organisms
        ("k. pneumoniae", "ampicillin"):             {"S_breakpoint": 8.0,  "R_breakpoint": 8.0},
        ("e. coli", "ampicillin"):                   {"S_breakpoint": 8.0,  "R_breakpoint": 8.0},
    }


def classify_mic(organism: str, drug: str, mic_value: float) -> str:
    """Classify a MIC value as Susceptible (S), Intermediate (I), or Resistant (R).
    Returns 'S', 'I', or 'R'. Returns 'UNKNOWN' if no breakpoint found."""
    if not _breakpoints:
        _load_breakpoints()

    key = (organism.lower().strip(), drug.lower().strip())
    bp = _breakpoints.get(key)

    if bp is None:
        return "UNKNOWN"

    s_bp = bp["S_breakpoint"]
    r_bp = bp["R_breakpoint"]

    if s_bp is not None and mic_value <= s_bp:
        return "S"
    elif r_bp is not None and mic_value >= r_bp:
        return "R"
    else:
        return "I"


def is_susceptible(organism: str, drug: str, mic_value: float) -> bool:
    return classify_mic(organism, drug, mic_value) == "S"


# Load on import
_load_breakpoints()
