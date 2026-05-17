# Standalone antibiogram generator.
# Generates realistic synthetic MIC values conditioned on organism + resistance phenotype.
# Used independently by Bhatia's environment tools; MIC logic also embedded in patient_generator.py.

import random

# MIC value ranges per (organism, phenotype). Synthetic training ranges — not clinical breakpoint tables.
MIC_RANGES: dict[tuple[str, str], dict[str, tuple[float, float]]] = {
    ("K. pneumoniae", "susceptible"): {
        "meropenem":              (0.06, 1.0),
        "ceftriaxone":            (0.03, 0.5),
        "ceftazidime-avibactam":  (0.12, 2.0),
        "ertapenem":              (0.06, 0.5),
        "piperacillin-tazobactam":(1.0,  8.0),
        "colistin":               (0.5,  1.0),
    },
    ("K. pneumoniae", "resistant"): {
        "meropenem":              (8.0,  32.0),
        "ceftriaxone":            (32.0, 128.0),
        "ceftazidime-avibactam":  (0.5,  2.0),
        "meropenem-vaborbactam":  (0.5,  2.0),
        "colistin":               (0.5,  2.0),
    },
    ("K. pneumoniae", "MDR"): {
        "meropenem":              (16.0, 64.0),
        "ceftriaxone":            (64.0, 256.0),
        "ceftazidime-avibactam":  (8.0,  32.0),
        "cefiderocol":            (0.25, 2.0),
        "colistin":               (2.0,  8.0),
    },
    ("E. coli", "susceptible"): {
        "ceftriaxone":            (0.03, 0.25),
        "meropenem":              (0.06, 0.5),
        "ertapenem":              (0.03, 0.25),
        "piperacillin-tazobactam":(1.0,  8.0),
        "trimethoprim-sulfamethoxazole": (0.5, 2.0),
    },
    ("E. coli", "resistant"): {
        "ceftriaxone":            (16.0, 128.0),
        "meropenem":              (0.06, 0.5),
        "ertapenem":              (0.03, 0.25),
        "piperacillin-tazobactam":(16.0, 64.0),
        "trimethoprim-sulfamethoxazole": (4.0, 16.0),
    },
    ("E. coli", "MDR"): {
        "ceftriaxone":            (64.0, 256.0),
        "meropenem":              (8.0,  32.0),
        "ceftazidime-avibactam":  (0.5,  4.0),
        "meropenem-vaborbactam":  (0.5,  4.0),
        "cefiderocol":            (0.25, 2.0),
    },
    ("P. aeruginosa", "susceptible"): {
        "cefepime":               (0.5,  4.0),
        "piperacillin-tazobactam":(2.0,  8.0),
        "meropenem":              (0.25, 1.0),
        "ciprofloxacin":          (0.06, 0.5),
        "ceftazidime-avibactam":  (0.5,  2.0),
    },
    ("P. aeruginosa", "resistant"): {
        "cefepime":               (16.0, 64.0),
        "piperacillin-tazobactam":(32.0, 128.0),
        "meropenem":              (4.0,  16.0),
        "ceftazidime-avibactam":  (1.0,  4.0),
        "ceftolozane-tazobactam": (0.5,  2.0),
    },
    ("P. aeruginosa", "MDR"): {
        "cefepime":               (32.0, 128.0),
        "piperacillin-tazobactam":(64.0, 256.0),
        "meropenem":              (16.0, 64.0),
        "ceftazidime-avibactam":  (8.0,  32.0),
        "cefiderocol":            (0.25, 2.0),
        "colistin":               (1.0,  4.0),
    },
    ("S. aureus", "susceptible"): {
        "cefazolin":              (0.25, 1.0),
        "vancomycin":             (0.5,  1.0),
        "daptomycin":             (0.25, 0.5),
        "linezolid":              (1.0,  2.0),
    },
    ("S. aureus", "resistant"): {  # MRSA
        "cefazolin":              (16.0, 64.0),
        "vancomycin":             (0.5,  2.0),
        "daptomycin":             (0.5,  1.0),
        "linezolid":              (1.0,  2.0),
    },
    ("S. aureus", "MDR"): {  # VISA / hVISA
        "cefazolin":              (32.0, 128.0),
        "vancomycin":             (4.0,  16.0),
        "daptomycin":             (0.5,  2.0),
        "linezolid":              (1.0,  2.0),
    },
    ("Enterococcus", "susceptible"): {
        "ampicillin":             (0.5,  2.0),
        "vancomycin":             (0.5,  2.0),
        "daptomycin":             (0.5,  2.0),
        "linezolid":              (1.0,  2.0),
    },
    ("Enterococcus", "resistant"): {  # VRE
        "ampicillin":             (16.0, 64.0),
        "vancomycin":             (32.0, 128.0),
        "daptomycin":             (0.5,  2.0),
        "linezolid":              (1.0,  2.0),
    },
    ("Enterococcus", "MDR"): {  # Pan-resistant VRE
        "ampicillin":             (32.0, 128.0),
        "vancomycin":             (64.0, 256.0),
        "daptomycin":             (4.0,  16.0),
        "linezolid":              (4.0,  16.0),
    },
}


def generate_antibiogram(organism: str, phenotype: str) -> dict[str, float]:
    """Generate synthetic MIC values for a given organism and resistance phenotype.

    Args:
        organism:  One of "K. pneumoniae", "E. coli", "P. aeruginosa",
                   "S. aureus", "Enterococcus".
        phenotype: One of "susceptible", "resistant", "MDR".

    Returns:
        dict mapping drug name -> MIC value (float, mg/L).
        Returns empty dict if organism/phenotype combo is not in the table.
    """
    ranges = MIC_RANGES.get((organism, phenotype), {})
    return {
        drug: round(random.uniform(low, high), 3)
        for drug, (low, high) in ranges.items()
    }


def get_supported_organisms() -> list[str]:
    """Return list of organisms with MIC data."""
    return sorted({org for org, _ in MIC_RANGES})


def get_supported_phenotypes(organism: str) -> list[str]:
    """Return phenotypes available for a given organism."""
    return [pheno for org, pheno in MIC_RANGES if org == organism]


if __name__ == "__main__":
    # Quick smoke test
    test_cases = [
        ("K. pneumoniae", "susceptible"),
        ("K. pneumoniae", "resistant"),
        ("E. coli", "MDR"),
        ("P. aeruginosa", "MDR"),
        ("S. aureus", "resistant"),
        ("Enterococcus", "resistant"),
    ]
    for org, pheno in test_cases:
        abg = generate_antibiogram(org, pheno)
        print(f"\n{org} ({pheno}):")
        for drug, mic in abg.items():
            print(f"  {drug}: {mic} mg/L")
