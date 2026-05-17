"""
eval.py -- AMR-Steward baseline evaluation framework.

Benchmarks three deterministic no-investigation baselines against the RLVR
oracle and prints a comparison table with improvement headroom numbers.
Produces eval_results.json consumed by viz.py --eval.

Usage:
    python eval.py                          # 100 episodes per level
    python eval.py --n 200 --levels 1 2 3
    python eval.py --out my_eval.json
    python eval.py --checkpoint-dir checkpoints/amr-grpo

No GPU required. Runs in ~5-10 seconds for default settings.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from env.reward import (
    SPECTRUM_SCORE,
    _normalize_drug,
    _load_drug_properties,
    _load_idsa,
    compute_total_reward,
    compute_optimal_prescription,
)
from env.models import PatientCase
from data.patient_generator import generate_patient
from data.eucast_parser import classify_mic

# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

_IDSA = _load_idsa()
_DRUG_PROPS = _load_drug_properties()
_EUCAST = type("_Eucast", (), {"classify_mic": staticmethod(classify_mic)})

BUDGET_BY_LEVEL: dict[int, int] = {1: 5, 2: 4, 3: 3}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _renal_dose(drug_norm: str, crcl: float) -> str:
    adj = _DRUG_PROPS.get(drug_norm, {}).get("renal_adjustments", {})
    if crcl > 50:
        return adj.get("CrCl_above_50", "")
    elif crcl >= 30:
        return adj.get("CrCl_30_50", "")
    elif crcl >= 10:
        return adj.get("CrCl_10_30", "")
    return adj.get("CrCl_under_10", "")


def _allergy_ok(drug_norm: str, allergies: list[str]) -> bool:
    flags = _DRUG_PROPS.get(drug_norm, {}).get("allergy_flags", [])
    patient_allergies = [a.lower() for a in allergies]
    return not any(
        any(pa in flag.lower() for pa in patient_allergies)
        for flag in flags
    )


def _susceptible_drugs(patient: PatientCase) -> list[str]:
    out = []
    for drug, mic in patient.antibiogram.items():
        drug_norm = _normalize_drug(drug)
        if classify_mic(patient.organism, drug_norm, mic) == "S":
            if _allergy_ok(drug_norm, patient.allergies):
                out.append(drug_norm)
    return out


def _score(prescription: dict, patient: PatientCase, level: int) -> dict[str, float]:
    budget_total = BUDGET_BY_LEVEL[level]
    _, bd = compute_total_reward(
        prescription=prescription,
        patient=patient,
        tool_call_history=[],
        eucast=_EUCAST,
        idsa=_IDSA,
        drug_properties=_DRUG_PROPS,
        budget_remaining=budget_total,
        budget_total=budget_total,
    )
    return bd


# ---------------------------------------------------------------------------
# Baseline policies (no investigation, deterministic)
# ---------------------------------------------------------------------------

def prescribe_broad_empiric(patient: PatientCase) -> dict:
    """Always meropenem -- the lazy 'cover everything' default."""
    drug = "meropenem"
    return {
        "drug": drug,
        "dose": _renal_dose(drug, patient.creatinine_clearance) or "1g IV q8h",
        "duration": "14 days",
        "justification": "broad-spectrum empiric",
    }


def prescribe_random(patient: PatientCase) -> dict:
    """Random drug from the antibiogram -- simulates an untrained model."""
    drugs = [_normalize_drug(d) for d in patient.antibiogram]
    drug = random.choice(drugs) if drugs else "meropenem"
    return {
        "drug": drug,
        "dose": _renal_dose(drug, patient.creatinine_clearance) or "standard dose",
        "duration": "7 days",
        "justification": "random selection",
    }


def prescribe_eucast_narrowest(patient: PatientCase) -> dict:
    """Narrowest-spectrum EUCAST-susceptible drug -- antibiogram-guided,
    no IDSA guideline lookup.  Best a prescriber can do with lab results only."""
    susceptible = _susceptible_drugs(patient)
    if not susceptible:
        drug = _normalize_drug(list(patient.antibiogram.keys())[0])
    else:
        drug = min(susceptible, key=lambda d: SPECTRUM_SCORE.get(d, 5))
    return {
        "drug": drug,
        "dose": _renal_dose(drug, patient.creatinine_clearance) or "standard dose",
        "duration": "14 days",
        "justification": "narrowest EUCAST-susceptible drug (no guideline lookup)",
    }


POLICIES: dict[str, Any] = {
    "broad_empiric":    prescribe_broad_empiric,
    "random":           prescribe_random,
    "eucast_narrowest": prescribe_eucast_narrowest,
}

POLICY_LABELS: dict[str, str] = {
    "broad_empiric":    "Broad empiric (meropenem)",
    "random":           "Random (antibiogram)",
    "eucast_narrowest": "EUCAST-only (narrowest susceptible)",
}


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    level: int
    policy: str
    quality_ratio: float
    total: float
    R0_allergy: float
    R1_activity: float
    R2_guideline: float
    R3_stewardship: float
    R4_dose: float
    R5_efficiency: float
    opt_score: float


def evaluate_policy(
    policy_name: str,
    policy_fn: Any,
    level: int,
    n: int,
    seed: int | None,
) -> list[EpisodeResult]:
    if seed is not None:
        random.seed(seed + level * 1000)

    results: list[EpisodeResult] = []
    for _ in range(n):
        patient = generate_patient(level)
        prescription = policy_fn(patient)
        bd = _score(prescription, patient, level)
        opt_score = compute_optimal_prescription(patient, _EUCAST, _IDSA, _DRUG_PROPS)
        results.append(EpisodeResult(
            level=level,
            policy=policy_name,
            quality_ratio=float(bd.get("quality_ratio", 0.0)),
            total=float(bd.get("total", 0.0)),
            R0_allergy=float(bd.get("R0_allergy", 0.0)),
            R1_activity=float(bd.get("R1_activity", 0.0)),
            R2_guideline=float(bd.get("R2_guideline", 0.0)),
            R3_stewardship=float(bd.get("R3_stewardship", 0.0)),
            R4_dose=float(bd.get("R4_dose", 0.0)),
            R5_efficiency=float(bd.get("R5_efficiency", 0.0)),
            opt_score=opt_score,
        ))
    return results


def aggregate(results: list[EpisodeResult]) -> dict[str, float]:
    if not results:
        return {}

    def _mean(vals: list[float]) -> float:
        return round(statistics.mean(vals), 4)

    def _std(vals: list[float]) -> float:
        return round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0

    qr  = [r.quality_ratio for r in results]
    tot = [r.total for r in results]
    return {
        "n":                    len(results),
        "quality_ratio_mean":   _mean(qr),
        "quality_ratio_std":    _std(qr),
        "total_mean":           _mean(tot),
        "total_std":            _std(tot),
        "R1_activity_mean":     _mean([r.R1_activity for r in results]),
        "R2_guideline_mean":    _mean([r.R2_guideline for r in results]),
        "R3_stewardship_mean":  _mean([r.R3_stewardship for r in results]),
        "R4_dose_mean":         _mean([r.R4_dose for r in results]),
        "opt_score_mean":       _mean([r.opt_score for r in results]),
        "R0_allergy_fail_rate": _mean([1.0 - r.R0_allergy for r in results]),
    }


# ---------------------------------------------------------------------------
# Trained-model numbers from log_history.json
# ---------------------------------------------------------------------------

def _read_trained_model(checkpoint_dir: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    cp = Path(checkpoint_dir)
    if not cp.exists():
        return out

    stage_map = {"stage1": 1, "stage2": 2, "stage3": 3}
    for stage_dir in sorted(cp.glob("stage*")):
        hist_file = stage_dir / "log_history.json"
        if not hist_file.exists():
            continue
        try:
            history = json.loads(hist_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        entries = [e for e in history if isinstance(e.get("step"), (int, float))]
        if not entries:
            continue

        reward_keys = [k for k in entries[-1] if "reward" in k.lower() and "std" not in k.lower()]
        primary_key = next(
            (k for k in ["reward", "rewards/mean", "reward/mean"] if k in entries[-1]),
            reward_keys[0] if reward_keys else None,
        )

        level = stage_map.get(stage_dir.name)
        if level is None:
            continue

        final_reward = float(entries[-1].get(primary_key, 0.0)) if primary_key else 0.0
        peak_reward  = max(
            (float(e.get(primary_key, 0.0)) for e in entries if primary_key),
            default=0.0,
        )
        out[str(level)] = {
            "total_mean": round(final_reward, 4),
            "peak":       round(peak_reward, 4),
            "steps":      int(entries[-1].get("step", 0)),
            "reward_key": primary_key or "unknown",
        }
    return out


# ---------------------------------------------------------------------------
# Pretty-print table
# ---------------------------------------------------------------------------

def _print_table(
    agg: dict[str, dict[int, dict[str, float]]],
    trained: dict[str, dict[str, float]],
    levels: list[int],
) -> None:
    col = 20

    print()
    print("=" * 76)
    print("AMR-Steward  Baseline Evaluation -- Quality Ratio (RLVR Oracle Score)")
    print("=" * 76)
    header = f"{'Policy':<34}" + "".join(f"{'Level ' + str(l):<{col}}" for l in levels)
    print(header)
    print("-" * 76)

    for policy_name, label in POLICY_LABELS.items():
        row = f"{label:<34}"
        for lvl in levels:
            m  = agg.get(policy_name, {}).get(lvl, {})
            qr = m.get("quality_ratio_mean", float("nan"))
            sd = m.get("quality_ratio_std",  float("nan"))
            row += f"{qr:.3f} +/-{sd:.3f}     "
        print(row)

    # Oracle ceiling
    row = f"{'RLVR oracle ceiling':<34}"
    for lvl in levels:
        opt = agg.get("broad_empiric", {}).get(lvl, {}).get("opt_score_mean", float("nan"))
        row += f"{opt:.3f}            "
    print(row)

    # Trained model
    if trained:
        row = f"{'Trained Gemma 4 (GRPO)':<34}"
        for lvl in levels:
            info = trained.get(str(lvl))
            row += f"{info['total_mean']:.3f} (blended)   " if info else f"{'N/A':<{col}}"
        print(row)

    print("=" * 76)
    print("Note: Baselines make zero tool calls (R5=0). Trained model blends")
    print("      format + process + terminal reward heads.")
    print()

    # Component breakdown at level 1
    print("Component means at Level 1:")
    print(f"  {'Component':<22} {'broad_empiric':>14} {'random':>10} {'eucast_only':>13}")
    print("  " + "-" * 60)
    comps = [
        ("R1_activity_mean",    "R1 Micro activity"),
        ("R2_guideline_mean",   "R2 IDSA guideline"),
        ("R3_stewardship_mean", "R3 Stewardship"),
        ("R4_dose_mean",        "R4 Dose"),
    ]
    for key, clabel in comps:
        be = agg.get("broad_empiric",    {}).get(1, {}).get(key, float("nan"))
        rn = agg.get("random",           {}).get(1, {}).get(key, float("nan"))
        en = agg.get("eucast_narrowest", {}).get(1, {}).get(key, float("nan"))
        print(f"  {clabel:<22} {be:>14.3f} {rn:>10.3f} {en:>13.3f}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate AMR-Steward baseline policies against the RLVR oracle.",
    )
    parser.add_argument("--n", type=int, default=100,
                        help="Episodes per policy per level (default: 100)")
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3],
                        choices=[1, 2, 3])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="eval_results.json")
    parser.add_argument("--checkpoint-dir", default="checkpoints/amr-grpo")
    args = parser.parse_args()

    print(f"Evaluating {len(POLICIES)} policies x {len(args.levels)} levels x {args.n} episodes ...")

    agg: dict[str, dict[int, dict[str, float]]] = {}

    for policy_name, policy_fn in POLICIES.items():
        agg[policy_name] = {}
        label = POLICY_LABELS[policy_name]
        for level in args.levels:
            print(f"  [{label}] level {level} ...", end="", flush=True)
            results = evaluate_policy(policy_name, policy_fn, level, args.n, args.seed)
            agg[policy_name][level] = aggregate(results)
            qr = agg[policy_name][level]["quality_ratio_mean"]
            print(f" quality_ratio={qr:.3f}")

    trained = _read_trained_model(args.checkpoint_dir)
    if trained:
        print(f"\nTrained model (from {args.checkpoint_dir}):")
        for lvl, info in trained.items():
            print(f"  Level {lvl}: total={info['total_mean']:.3f} "
                  f"peak={info['peak']:.3f} steps={info['steps']}")
    else:
        print(f"\n[info] No checkpoint log_history found -- trained row omitted.")

    _print_table(agg, trained, args.levels)

    # Headline improvement numbers
    headlines: dict[str, Any] = {}
    for lvl in args.levels:
        rand_qr = agg.get("random", {}).get(lvl, {}).get("quality_ratio_mean", 0.0)
        opt     = agg.get("broad_empiric", {}).get(lvl, {}).get("opt_score_mean", 1.0)
        t_info  = trained.get(str(lvl))
        t_total = t_info["total_mean"] if t_info else None
        t_qr    = round(t_total / 0.9, 3) if t_total is not None else None
        headlines[f"level_{lvl}"] = {
            "random_quality_ratio":      round(rand_qr, 4),
            "oracle_ceiling":            round(opt, 4),
            "trained_total":             t_total,
            "trained_quality_ratio_est": t_qr,
            "improvement_over_random": (
                round((t_qr - rand_qr) / max(rand_qr, 0.001), 3)
                if t_qr is not None else None
            ),
        }

    if any(h.get("improvement_over_random") is not None for h in headlines.values()):
        print("Improvement over random baseline:")
        for lvl in args.levels:
            h = headlines.get(f"level_{lvl}", {})
            imp = h.get("improvement_over_random")
            if imp is not None:
                print(f"  Level {lvl}: trained ~{h['trained_quality_ratio_est']:.3f} "
                      f"vs random {h['random_quality_ratio']:.3f}  (+{imp:.1%})")

    output: dict[str, Any] = {
        "config": {
            "n_episodes_per_level": args.n,
            "levels": args.levels,
            "seed": args.seed,
        },
        "aggregated": {
            policy: {str(lvl): metrics for lvl, metrics in level_data.items()}
            for policy, level_data in agg.items()
        },
        "trained_model": trained,
        "headlines":     headlines,
        "policy_labels": POLICY_LABELS,
    }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nResults saved to {out_path}")

    try:
        import viz  # noqa: PLC0415
        viz.plot_eval_comparison(str(out_path))
        viz.plot_component_breakdown(str(out_path), level=1)
        print("Charts saved to assets/")
    except Exception as exc:
        print(f"[info] Chart generation skipped: {exc}")
        print("       Run 'python viz.py --eval eval_results.json' separately.")


if __name__ == "__main__":
    main()
