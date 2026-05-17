"""
viz.py — AMR-Steward reward curve visualisation.

Reads checkpoints/amr-grpo/stage*/log_history.json produced by train.py
and saves reward curve plots to the same checkpoint directory.

Usage:
    python viz.py                          # uses default checkpoint dir
    python viz.py --output-dir path/to/checkpoints/amr-grpo
    python viz.py --show                   # also opens an interactive window
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ── Matplotlib backend selection ───────────────────────────────────────────────
# Use non-interactive backend by default so this runs on headless servers.
import matplotlib
matplotlib.use("Agg")  # set before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Constants ──────────────────────────────────────────────────────────────────

# Keys we know about; shown in a fixed colour if present.
_REWARD_KEY_COLOURS = {
    "reward":           "#2196F3",   # blue  — TRL aggregated reward
    "reward/mean":      "#2196F3",
    "rewards/mean":     "#2196F3",
    "reward_1":         "#FF9800",   # orange — format head
    "reward_2":         "#4CAF50",   # green  — process head
    "reward_3":         "#9C27B0",   # purple — terminal head
    "quality_ratio":    "#F44336",   # red    — RLVR ratio (if logged)
}
_LOSS_COLOUR = "#607D8B"
_FALLBACK_COLOURS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_stages(output_dir: Path) -> list[tuple[str, list[dict]]]:
    """Return [(stage_name, log_history), ...] sorted by stage number."""
    stages: list[tuple[str, list[dict]]] = []
    for stage_dir in sorted(output_dir.glob("stage*")):
        hist_file = stage_dir / "log_history.json"
        if not hist_file.exists():
            continue
        try:
            with open(hist_file, encoding="utf-8") as fh:
                history = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  [warn] Could not read {hist_file}: {exc}")
            continue
        # Filter to entries that have a numeric step value
        history = [e for e in history if isinstance(e.get("step"), (int, float))]
        if history:
            stages.append((stage_dir.name, history))
    return stages


def _reward_keys(history: list[dict]) -> list[str]:
    """Return all numeric reward-related keys present in the history."""
    all_keys: set[str] = set()
    for entry in history:
        all_keys |= {
            k for k, v in entry.items()
            if "reward" in k.lower() and isinstance(v, (int, float))
            and "std" not in k.lower()
        }
    # Stable ordering: known keys first, then alphabetical
    known_order = list(_REWARD_KEY_COLOURS.keys())
    ordered = [k for k in known_order if k in all_keys]
    ordered += sorted(all_keys - set(known_order))
    return ordered


def _series(history: list[dict], key: str) -> tuple[list[float], list[float]]:
    """Return (steps, values) for *key*, skipping entries where key is absent."""
    steps, vals = [], []
    for e in history:
        if key in e and isinstance(e[key], (int, float)):
            steps.append(float(e["step"]))
            vals.append(float(e[key]))
    return steps, vals


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_stages(
    stages: list[tuple[str, list[dict]]],
    output_dir: Path,
    show: bool = False,
) -> None:
    """One figure per stage; each figure has reward curves + optional loss."""
    if not stages:
        print("No stage log histories found — nothing to plot.")
        return

    for stage_name, history in stages:
        rkeys = _reward_keys(history)
        has_loss = any("loss" in e for e in history)
        n_rows = 2 if (rkeys and has_loss) else 1

        fig = plt.figure(figsize=(10, 4 * n_rows))
        gs = gridspec.GridSpec(n_rows, 1, hspace=0.4)

        # ── Reward subplot ─────────────────────────────────────────────────────
        if rkeys:
            ax_r = fig.add_subplot(gs[0])
            for idx, key in enumerate(rkeys):
                steps, vals = _series(history, key)
                if not vals:
                    continue
                colour = _REWARD_KEY_COLOURS.get(key, _FALLBACK_COLOURS[idx % len(_FALLBACK_COLOURS)])
                label = key.replace("rewards/", "").replace("reward/", "")
                ax_r.plot(steps, vals, color=colour, linewidth=2, label=label)

            ax_r.set_title(f"Reward curves — {stage_name}", fontsize=13, fontweight="bold")
            ax_r.set_xlabel("Training step")
            ax_r.set_ylabel("Reward")
            ax_r.set_ylim(bottom=0)
            ax_r.legend(loc="lower right", fontsize=9)
            ax_r.grid(True, alpha=0.3)

        # ── Loss subplot ───────────────────────────────────────────────────────
        if has_loss and n_rows == 2:
            ax_l = fig.add_subplot(gs[1])
            steps_l, vals_l = _series(history, "loss")
            if vals_l:
                ax_l.plot(steps_l, vals_l, color=_LOSS_COLOUR, linewidth=2, label="loss")
            ax_l.set_title(f"Training loss — {stage_name}", fontsize=13, fontweight="bold")
            ax_l.set_xlabel("Training step")
            ax_l.set_ylabel("Loss")
            ax_l.legend(loc="upper right", fontsize=9)
            ax_l.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = output_dir / f"{stage_name}_curves.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out_path}")

        if show:
            matplotlib.use("TkAgg")
            plt.show()
        plt.close(fig)


def plot_combined(
    stages: list[tuple[str, list[dict]]],
    output_dir: Path,
) -> None:
    """Single figure comparing the primary reward across all stages."""
    if not stages:
        return

    # Pick the most informative single reward key per stage
    preferred = ["reward", "rewards/mean", "reward/mean"]

    fig, ax = plt.subplots(figsize=(10, 5))
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    plotted = False
    for i, (stage_name, history) in enumerate(stages):
        rkeys = _reward_keys(history)
        chosen = next((k for k in preferred if k in rkeys), rkeys[0] if rkeys else None)
        if chosen is None:
            continue
        steps, vals = _series(history, chosen)
        if not vals:
            continue
        ax.plot(steps, vals, color=colours[i % len(colours)],
                linewidth=2, label=f"{stage_name} ({chosen})")
        plotted = True

    if plotted:
        ax.set_title("AMR-Steward — Reward across curriculum stages", fontsize=14, fontweight="bold")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Reward")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = output_dir / "all_stages_reward.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out_path}")

    plt.close(fig)


# ── Eval comparison chart ──────────────────────────────────────────────────────

def plot_eval_comparison(
    eval_results_path: str = "eval_results.json",
    assets_dir: str = "assets",
) -> None:
    """Generate a grouped-bar chart comparing baseline policies vs the trained
    model, reading eval_results.json produced by eval.py.

    Saves assets/eval_comparison.png.
    """
    eval_path = Path(eval_results_path)
    if not eval_path.exists():
        print(f"[warn] {eval_path} not found. Run eval.py first.")
        return

    try:
        data = json.loads(eval_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[warn] Could not read {eval_path}: {exc}")
        return

    aggregated = data.get("aggregated", {})
    trained    = data.get("trained_model", {})
    labels     = data.get("policy_labels", {})
    config     = data.get("config", {})
    levels     = [int(l) for l in sorted(set(
        int(k) for v in aggregated.values() for k in v
    ))]

    if not levels:
        print("[warn] No level data found in eval_results.json.")
        return

    policies     = list(aggregated.keys())
    policy_names = [labels.get(p, p) for p in policies]
    n_policies   = len(policies)
    n_levels     = len(levels)

    # ── Quality-ratio grouped bar chart ───────────────────────────────────────
    fig, axes = plt.subplots(1, n_levels, figsize=(5 * n_levels, 6), sharey=True)
    if n_levels == 1:
        axes = [axes]

    bar_colours = ["#E53935", "#FB8C00", "#43A047", "#1E88E5", "#8E24AA"]

    for ax_idx, (ax, lvl) in enumerate(zip(axes, levels)):
        lvl_str = str(lvl)
        qr_vals  = []
        qr_errs  = []
        names    = []

        for pol_name, pol_label in zip(policies, policy_names):
            m = aggregated.get(pol_name, {}).get(lvl_str, {})
            qr_vals.append(m.get("quality_ratio_mean", 0.0))
            qr_errs.append(m.get("quality_ratio_std",  0.0))
            names.append(pol_label)

        # Oracle ceiling bar
        opt = aggregated.get(policies[0], {}).get(lvl_str, {}).get("opt_score_mean", 1.0)
        qr_vals.append(opt)
        qr_errs.append(0.0)
        names.append("RLVR oracle ceiling")

        # Trained model bar (estimated from total)
        trained_info = trained.get(lvl_str)
        if trained_info:
            est_qr = round(trained_info.get("total_mean", 0.0) / 0.9, 3)
            qr_vals.append(est_qr)
            qr_errs.append(0.0)
            names.append(f"Trained (est. from total)")

        x = range(len(names))
        colours_for_bars = bar_colours[:n_policies] + ["#546E7A"] + (["#1565C0"] if trained_info else [])

        bars = ax.bar(
            x, qr_vals,
            yerr=qr_errs,
            color=colours_for_bars,
            capsize=4,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.8,
        )

        # Value labels on bars
        for bar, val in zip(bars, qr_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{val:.3f}",
                ha="center", va="bottom",
                fontsize=8, fontweight="bold",
            )

        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
        ax.set_title(f"Curriculum Level {lvl}", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Quality Ratio (RLVR score)" if ax_idx == 0 else "")
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(1.0, color="#546E7A", linestyle="--", linewidth=1, alpha=0.5)

    n_episodes = config.get("n_episodes_per_level", "?")
    fig.suptitle(
        f"AMR-Steward — Quality Ratio: Baseline Policies vs Trained Model\n"
        f"(n={n_episodes} episodes per level, no investigation for baselines)",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    out_dir = Path(assets_dir)
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "eval_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close(fig)


# ── Component breakdown chart ──────────────────────────────────────────────────

def plot_component_breakdown(
    eval_results_path: str = "eval_results.json",
    assets_dir: str = "assets",
    level: int = 1,
) -> None:
    """Horizontal bar chart showing R1–R4 mean scores for each baseline policy
    at the given curriculum level — reveals *which* component is the bottleneck."""
    eval_path = Path(eval_results_path)
    if not eval_path.exists():
        return

    try:
        data = json.loads(eval_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return

    aggregated = data.get("aggregated", {})
    labels     = data.get("policy_labels", {})
    lvl_str    = str(level)

    policies     = list(aggregated.keys())
    policy_names = [labels.get(p, p) for p in policies]
    components   = ["R1_activity_mean", "R2_guideline_mean", "R3_stewardship_mean", "R4_dose_mean"]
    comp_labels  = ["R1 Microbiological", "R2 IDSA Guideline", "R3 Stewardship", "R4 Dose Correct"]

    n_policies = len(policies)
    y          = range(len(comp_labels))
    bar_h      = 0.8 / n_policies

    fig, ax = plt.subplots(figsize=(9, 5))
    colours = ["#E53935", "#FB8C00", "#43A047", "#1E88E5"]

    for i, (pol_name, pol_label) in enumerate(zip(policies, policy_names)):
        m      = aggregated.get(pol_name, {}).get(lvl_str, {})
        values = [m.get(c, 0.0) for c in components]
        offset = (i - n_policies / 2 + 0.5) * bar_h
        ax.barh(
            [yi + offset for yi in y],
            values,
            height=bar_h * 0.85,
            color=colours[i % len(colours)],
            alpha=0.85,
            label=pol_label,
        )

    ax.set_yticks(list(y))
    ax.set_yticklabels(comp_labels, fontsize=10)
    ax.set_xlabel("Mean Score (0–1)")
    ax.set_xlim(0, 1.15)
    ax.set_title(
        f"AMR-Steward — Reward Components by Baseline Policy (Level {level})",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    ax.axvline(1.0, color="#546E7A", linestyle="--", linewidth=1, alpha=0.5)
    plt.tight_layout()

    out_dir = Path(assets_dir)
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"component_breakdown_level{level}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close(fig)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot AMR-Steward training curves and baseline evaluation charts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python viz.py                              # training curves only\n"
            "  python viz.py --eval eval_results.json    # + baseline comparison\n"
            "  python viz.py --eval eval_results.json --breakdown-level 2\n"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="checkpoints/amr-grpo",
        help="Directory containing stage*/log_history.json (default: checkpoints/amr-grpo)",
    )
    parser.add_argument(
        "--eval",
        default=None,
        metavar="EVAL_JSON",
        help="Path to eval_results.json produced by eval.py",
    )
    parser.add_argument(
        "--breakdown-level",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Curriculum level for the component breakdown chart (default: 1)",
    )
    parser.add_argument(
        "--assets-dir",
        default="assets",
        help="Directory to save eval chart PNGs (default: assets/)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also open an interactive plot window (requires display).",
    )
    args = parser.parse_args()

    # ── Training curves (silently skipped if checkpoint dir absent) ───────────
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        stages = _load_stages(output_dir)
        if stages:
            print(f"Training curves: {len(stages)} stage(s) found — generating plots...")
            plot_stages(stages, output_dir, show=args.show)
            plot_combined(stages, output_dir)

    # ── Eval comparison chart ──────────────────────────────────────────────────
    if args.eval:
        print(f"\nGenerating eval comparison charts from: {args.eval}")
        plot_eval_comparison(args.eval, args.assets_dir)
        plot_component_breakdown(args.eval, args.assets_dir, level=args.breakdown_level)
    else:
        print("\n[tip] Run with --eval eval_results.json to also generate comparison charts.")

    print("Done.")


if __name__ == "__main__":
    main()
