# PALAK OWNS THIS FILE
# Generates synthetic patient cases for RL training episodes.

import random
from dataclasses import dataclass
from sys import path
path.insert(0, ".")
try:
    from env.models import PatientCase
except TypeError:
    @dataclass
    class PatientCase:
        age: int
        sex: str
        infection_site: str
        organism: str
        creatinine_clearance: float
        allergies: list[str]
        antibiogram: dict[str, float]
        phenotype: str
        curriculum_level: int


ORGANISMS_BY_LEVEL = {
    1: ["K. pneumoniae", "E. coli", "S. aureus"],
    2: ["K. pneumoniae", "E. coli", "P. aeruginosa", "S. aureus", "Enterococcus"],
    3: ["K. pneumoniae", "E. coli", "P. aeruginosa", "S. aureus", "Enterococcus"],
}
INFECTION_SITES = ["bacteremia", "UTI", "pneumonia", "intra-abdominal"]

# MIC value ranges per organism + phenotype. These are synthetic training ranges,
# not clinical breakpoint tables.
MIC_RANGES = {
    ("K. pneumoniae", "susceptible"): {
        "meropenem": (0.06, 1.0), "ceftriaxone": (0.03, 0.5),
        "ceftazidime-avibactam": (0.12, 2.0), "colistin": (0.5, 1.0),
    },
    ("K. pneumoniae", "resistant"): {
        "meropenem": (8.0, 32.0), "ceftriaxone": (32.0, 128.0),
        "ceftazidime-avibactam": (0.5, 2.0), "colistin": (0.5, 2.0),
    },
    ("K. pneumoniae", "MDR"): {
        "meropenem": (16.0, 64.0), "ceftriaxone": (64.0, 256.0),
        "ceftazidime-avibactam": (8.0, 32.0), "colistin": (2.0, 8.0),
    },
    ("E. coli", "susceptible"): {
        "ceftriaxone": (0.03, 0.25), "meropenem": (0.06, 0.5),
        "piperacillin-tazobactam": (1.0, 8.0),
    },
    ("E. coli", "resistant"): {
        "ceftriaxone": (16.0, 128.0), "meropenem": (0.06, 0.5),
        "ertapenem": (0.03, 0.25), "piperacillin-tazobactam": (16.0, 64.0),
        "trimethoprim-sulfamethoxazole": (4.0, 16.0),
    },
    ("E. coli", "MDR"): {
        "ceftriaxone": (64.0, 256.0), "meropenem": (8.0, 32.0),
        "ceftazidime-avibactam": (0.5, 4.0), "meropenem-vaborbactam": (0.5, 4.0),
        "cefiderocol": (0.25, 2.0),
    },
    ("P. aeruginosa", "susceptible"): {
        "cefepime": (0.5, 4.0), "piperacillin-tazobactam": (2.0, 8.0),
        "meropenem": (0.25, 1.0), "ciprofloxacin": (0.06, 0.5),
    },
    ("P. aeruginosa", "resistant"): {
        "cefepime": (16.0, 64.0), "piperacillin-tazobactam": (32.0, 128.0),
        "meropenem": (4.0, 16.0), "ceftazidime-avibactam": (1.0, 4.0),
        "ceftolozane-tazobactam": (0.5, 2.0),
    },
    ("P. aeruginosa", "MDR"): {
        "cefepime": (32.0, 128.0), "piperacillin-tazobactam": (64.0, 256.0),
        "meropenem": (16.0, 64.0), "ceftazidime-avibactam": (8.0, 32.0),
        "cefiderocol": (0.25, 2.0),
    },
    ("S. aureus", "susceptible"): {
        "cefazolin": (0.25, 1.0), "vancomycin": (0.5, 1.0),
        "daptomycin": (0.25, 0.5), "linezolid": (1.0, 2.0),
    },
    ("S. aureus", "resistant"): {  # MRSA
        "cefazolin": (16.0, 64.0), "vancomycin": (0.5, 2.0),
        "daptomycin": (0.5, 1.0), "linezolid": (1.0, 2.0),
    },
    ("S. aureus", "MDR"): {
        "cefazolin": (32.0, 128.0), "vancomycin": (4.0, 16.0),
        "daptomycin": (0.5, 2.0), "linezolid": (1.0, 2.0),
    },
    ("Enterococcus", "susceptible"): {
        "ampicillin": (0.5, 2.0), "vancomycin": (0.5, 2.0),
        "daptomycin": (0.5, 2.0), "linezolid": (1.0, 2.0),
    },
    ("Enterococcus", "resistant"): {
        "ampicillin": (16.0, 64.0), "vancomycin": (0.5, 2.0),
        "daptomycin": (0.5, 2.0), "linezolid": (1.0, 2.0),
    },
    ("Enterococcus", "MDR"): {
        "ampicillin": (32.0, 128.0), "vancomycin": (32.0, 128.0),
        "daptomycin": (0.5, 2.0), "linezolid": (1.0, 2.0),
    },
}

PHENOTYPE_BY_LEVEL = {
    1: ["susceptible"],
    2: ["susceptible", "resistant"],
    3: ["susceptible", "resistant", "MDR"],
}


def generate_patient(curriculum_level: int = 1) -> PatientCase:
    """Generate a random patient case appropriate for the curriculum level."""
    if curriculum_level not in PHENOTYPE_BY_LEVEL:
        raise ValueError("curriculum_level must be 1, 2, or 3")

    organism = random.choice(ORGANISMS_BY_LEVEL[curriculum_level])
    infection_site = _choose_infection_site(organism)
    phenotype = random.choice(PHENOTYPE_BY_LEVEL[curriculum_level])

    # Renal function by level
    if curriculum_level == 1:
        crcl = random.uniform(60, 120)   # normal
    elif curriculum_level == 2:
        crcl = random.uniform(30, 60)    # mild-moderate impairment
    else:
        crcl = random.uniform(10, 45)    # moderate-severe impairment

    # Allergies (rare in level 1, possible in level 2+)
    allergies = []
    if curriculum_level >= 2 and random.random() < 0.25:
        allergies = [random.choice(["penicillin", "cephalosporin", "sulfonamide"])]
    if curriculum_level == 3 and random.random() < 0.15:
        allergies.append(random.choice(["vancomycin", "fluoroquinolone"]))

    # Generate antibiogram
    antibiogram = _generate_antibiogram(organism, phenotype)

    return PatientCase(
        age=random.randint(35, 85),
        sex=random.choice(["M", "F"]),
        infection_site=infection_site,
        organism=organism,
        creatinine_clearance=round(crcl, 1),
        allergies=allergies,
        antibiogram=antibiogram,
        phenotype=phenotype,
        curriculum_level=curriculum_level,
    )


def _generate_antibiogram(organism: str, phenotype: str) -> dict[str, float]:
    """Generate realistic MIC values for the organism + phenotype combo."""
    ranges = MIC_RANGES.get((organism, phenotype), {})
    antibiogram = {}
    for drug, (low, high) in ranges.items():
        antibiogram[drug] = round(random.uniform(low, high), 3)
    return antibiogram


def _choose_infection_site(organism: str) -> str:
    """Bias sites toward realistic organism/syndrome combinations."""
    site_weights = {
        "K. pneumoniae": {"bacteremia": 3, "UTI": 3, "pneumonia": 2, "intra-abdominal": 2},
        "E. coli": {"bacteremia": 3, "UTI": 5, "pneumonia": 1, "intra-abdominal": 3},
        "P. aeruginosa": {"bacteremia": 2, "UTI": 2, "pneumonia": 5, "intra-abdominal": 1},
        "S. aureus": {"bacteremia": 5, "UTI": 1, "pneumonia": 3, "intra-abdominal": 1},
        "Enterococcus": {"bacteremia": 3, "UTI": 3, "pneumonia": 1, "intra-abdominal": 3},
    }[organism]
    sites = list(site_weights.keys())
    weights = list(site_weights.values())
    return random.choices(sites, weights=weights, k=1)[0]


def patient_to_text(patient: PatientCase) -> str:
    """Convert a PatientCase to the plain English observation text the LLM sees."""
    allergy_str = ", ".join(patient.allergies) if patient.allergies else "None reported"
    return (
        f"Patient: {patient.age}-year-old {patient.sex}.\n"
        f"Infection site: {patient.infection_site}.\n"
        f"Culture result: {patient.organism} isolated.\n"
        f"Renal function: CrCl {patient.creatinine_clearance} mL/min.\n"
        f"Allergies: {allergy_str}.\n"
        f"Available antibiogram data: {list(patient.antibiogram.keys())}.\n"
    )
