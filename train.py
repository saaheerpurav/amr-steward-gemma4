"""GRPO training script for AMR-Steward.

Trains Gemma 4 with TRL GRPOTrainer + PEFT LoRA across a 3-stage curriculum.

Example:
    python train.py --stage 1 --samples 64
    python train.py  # all 3 stages with default sample counts
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset

sys.path.insert(0, str(Path(__file__).parent))

from env import AMREnvironment, AMRAction
from env.models import PatientCase
from env.reward import (
    parse_prescription_from_text,
    parse_tool_calls_from_text,
    count_unique_tool_types,
    R5_tool_efficiency,
    R6_format,
)


DEFAULT_MODEL_NAME = "google/gemma-4-e2b-it"
MAX_SEQ_LEN = 2048
LORA_R = 16
LORA_ALPHA = 32
DEFAULT_OUTPUT_DIR = "checkpoints/amr-grpo"

# Curriculum: (num_samples, level) tuples
CURRICULUM = [
    (512, 1),
    (512, 2),
    (256, 3),
]

SYSTEM_PROMPT = """You are an antimicrobial stewardship AI. Prescribe the narrowest effective antibiotic.

INVESTIGATE tools (optional, costs 1 budget each):
  INVESTIGATE: {"tool": "interpret_resistance", "arg": "<drug>"}
  INVESTIGATE: {"tool": "check_guideline", "arg": "<syndrome>"}
  INVESTIGATE: {"tool": "assess_patient_factors", "arg": ""}

When ready, output EXACTLY this one line and stop:
  COMMIT: {"drug": "<name>", "dose": "<dose>", "duration": "<days>", "justification": "<one sentence>"}

Do not add any text before or after the COMMIT line."""

GRPO_CONFIG_CLS = None
GRPO_TRAINER_CLS = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AMR-Steward with GRPO.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2, 3],
        help="Train only a single curriculum stage.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        help="Override sample count for the selected stage or for all stages.",
    )
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--max-completion-length", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.7)
    return parser.parse_args()


def setup_runtime() -> None:
    global GRPO_CONFIG_CLS, GRPO_TRAINER_CLS
    from trl import GRPOConfig, GRPOTrainer
    GRPO_CONFIG_CLS = GRPOConfig
    GRPO_TRAINER_CLS = GRPOTrainer


def _render_prompt(obs) -> str:
    parts = [f"SYSTEM:\n{SYSTEM_PROMPT}", f"USER:\n{obs.patient_text}"]
    if obs.tool_results:
        parts.append("INVESTIGATION RESULTS:\n" + "\n---\n".join(obs.tool_results))
    if obs.world_model_rankings:
        parts.append(obs.world_model_rankings)
    parts.append(f"INVESTIGATION BUDGET REMAINING: {obs.budget_remaining}")
    if obs.budget_remaining == 0:
        parts.append("YOU MUST COMMIT NOW.")
    parts.append("ASSISTANT:")
    return "\n\n".join(parts)


def build_dataset(level: int, num_samples: int) -> Dataset:
    """Build a prompt dataset with the exact patient case embedded as metadata."""
    env = AMREnvironment()
    rows: list[dict[str, Any]] = []
    for idx in range(num_samples):
        obs = env.reset(curriculum_level=level)
        patient = env.current_patient
        rows.append(
            {
                "prompt": _render_prompt(obs),
                "patient_json": json.dumps(asdict(patient)),
                "curriculum_level": level,
                "case_id": f"level{level}-case{idx}",
            }
        )
    return Dataset.from_list(rows)


def _completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        chunks: list[str] = []
        for item in completion:
            if isinstance(item, dict):
                content = item.get("content")
                if isinstance(content, str):
                    chunks.append(content)
            elif isinstance(item, str):
                chunks.append(item)
        return "\n".join(chunks)
    return str(completion)


def _patient_from_json(payload: str) -> PatientCase:
    return PatientCase(**json.loads(payload))


_BUDGET_BY_LEVEL = {1: 5, 2: 4, 3: 3}


def _parse_investigate_action(raw_line: str) -> tuple[str, str | None] | None:
    """Parse a single INVESTIGATE payload into (tool_name, tool_arg).

    Handles JSON format: '{"tool": "interpret_resistance", "arg": "meropenem"}'
    Returns None if the line cannot be parsed into a valid tool call.
    """
    raw_line = raw_line.strip()
    try:
        parsed = json.loads(raw_line)
        tool_name = parsed.get("tool")
        if not isinstance(tool_name, str) or not tool_name:
            return None
        tool_arg = parsed.get("arg") or None
        return tool_name, tool_arg
    except (json.JSONDecodeError, AttributeError):
        return None


def _score_completion_with_env(
    completion_text: str,
    patient_payload: str,
    level: int,
) -> float:
    """Replay the INVESTIGATE + COMMIT actions from a completion through a fresh
    AMREnvironment instance seeded with the exact same patient used to build the
    prompt.

    Returns cumulative episode reward:
        dense shaping (novel INVESTIGATE steps) + terminal COMMIT reward.

    This is the multi-turn training signal: the env's budget enforcement, dense
    shaping cap, allergy gate, and RLVR quality_ratio all fire exactly as they
    would during real deployment.
    """
    patient = _patient_from_json(patient_payload)
    env = AMREnvironment()
    env.reset(curriculum_level=int(level), patient=patient)

    cumulative = 0.0

    # ── Replay INVESTIGATE actions in textual order ────────────────────────────
    for raw_line in parse_tool_calls_from_text(completion_text):
        if env._state.done:
            break
        parsed = _parse_investigate_action(raw_line)
        if parsed is None:
            continue
        tool_name, tool_arg = parsed
        try:
            action = AMRAction(
                action_type="INVESTIGATE",
                tool_name=tool_name,
                tool_arg=tool_arg,
            )
            obs = env.step(action)
            if obs.reward is not None:
                cumulative += obs.reward
        except Exception:
            # Malformed / out-of-vocabulary tool name — skip silently
            continue

    # ── Execute COMMIT if the episode is still open ────────────────────────────
    if not env._state.done:
        prescription = parse_prescription_from_text(completion_text)
        if prescription:
            try:
                action = AMRAction(action_type="COMMIT", prescription=prescription)
                obs = env.step(action)
                if obs.reward is not None:
                    cumulative += obs.reward
            except Exception:
                pass
        # No COMMIT found → no terminal reward (natural penalty)

    return cumulative


# Canonical tool names — used by the plain-text fallback parser below
_KNOWN_TOOLS = ("interpret_resistance", "check_guideline", "assess_patient_factors")


def _parse_tool_calls_to_history(text: str) -> list[dict]:
    """Parse INVESTIGATE lines into structured [{tool, arg}, ...] entries.

    Primary path: strict JSON with "tool" and "arg" keys (matches system prompt).
    Fallback path: plain-text detection of known tool names — keeps Head 2 alive
    during early training when the model hasn't learned JSON format yet.
    Head 3 (env replay) stays strict-JSON-only; wrong format there correctly gives 0 reward.
    """
    history: list[dict] = []
    for raw in parse_tool_calls_from_text(text):
        raw_s = raw.strip()

        # Primary: strict JSON parse (system prompt teaches this exact format)
        try:
            payload = json.loads(raw_s)
            tool = payload.get("tool", "")
            if tool:
                history.append({"tool": tool, "arg": payload.get("arg", "")})
                continue
        except (json.JSONDecodeError, AttributeError, ValueError):
            pass

        # Fallback: plain-text — detect known tool names even without JSON.
        # Prevents Head 2 from returning 0 for every early-training completion.
        raw_lower = raw_s.lower()
        for tname in _KNOWN_TOOLS:
            if raw_lower.startswith(tname):
                # Remainder is the arg (strip leading punctuation/whitespace)
                remainder = raw_s[len(tname):].strip().lstrip(",:\"'").strip().strip("\"'")
                history.append({"tool": tname, "arg": remainder})
                break

    return history


def make_format_reward_fn():
    """Head 1: fast format signal — penalizes verbose completions."""
    def fn(prompts, completions, **kwargs) -> list[float]:
        return [float(R6_format(_completion_to_text(c))) * 0.05 for c in completions]
    return fn


def make_process_reward_fn():
    """Head 2: investigation quality — diverse tool use with budget remaining.

    Parses completions into the same structured [{tool, arg}, ...] format as
    AMRState.tool_history, then calls count_unique_tool_types — the identical
    function used by the env terminal in compute_total_reward. This eliminates
    the prior split-brain where Head 2 and the env terminal computed R5 from
    different data sources with different parsing logic.
    """
    def fn(prompts, completions, patient_json, curriculum_level=None, **kwargs) -> list[float]:
        rewards: list[float] = []
        levels = list(curriculum_level) if curriculum_level is not None else [1] * len(completions)
        for completion, level in zip(completions, levels):
            text = _completion_to_text(completion)
            history = _parse_tool_calls_to_history(text)
            budget_total = _BUDGET_BY_LEVEL.get(int(level), 5)
            unique_types = count_unique_tool_types(history)
            budget_spent = len(history)  # only valid structured tool calls count
            budget_remaining = max(0, budget_total - budget_spent)
            rewards.append(float(R5_tool_efficiency(unique_types, budget_spent, budget_remaining, budget_total)))
        return rewards
    return fn


def make_terminal_reward_fn():
    """Head 3: cumulative env reward — dense shaping + RLVR quality_ratio.

    Replays the full completion through a fresh AMREnvironment using the exact
    patient case that was baked into the prompt at dataset-build time.  This
    makes the reward signal multi-turn: investigation steps earn dense rewards,
    budget exhaustion is penalised, and the final COMMIT is scored via the
    RLVR optimality ratio.  The env's allergy safety gate also fires here.
    """
    def fn(
        prompts,
        completions,
        patient_json,
        curriculum_level=None,
        **kwargs,
    ) -> list[float]:
        levels = list(curriculum_level) if curriculum_level is not None else [1] * len(completions)
        rewards: list[float] = []
        for completion, patient_payload, level in zip(completions, patient_json, levels):
            text = _completion_to_text(completion)
            try:
                reward = _score_completion_with_env(text, patient_payload, int(level))
                rewards.append(float(reward))
            except Exception:
                rewards.append(0.0)
        return rewards
    return fn


def load_model(model_name: str):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if bf16_supported else torch.float16
    else:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return model, tokenizer


def make_stage_plan(args: argparse.Namespace) -> list[tuple[int, int]]:
    if args.stage is not None:
        samples = args.samples if args.samples is not None else CURRICULUM[args.stage - 1][0]
        return [(samples, args.stage)]

    if args.samples is not None:
        return [(args.samples, level) for _, level in CURRICULUM]

    return CURRICULUM


def train_stage(
    model,
    tokenizer,
    args: argparse.Namespace,
    level: int,
    num_samples: int,
    stage_index: int,
):
    print(f"\n=== Stage {stage_index}: level={level}, samples={num_samples} ===")
    dataset = build_dataset(level, num_samples)
    stage_output_dir = f"{args.output_dir}/stage{stage_index}"

    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    config = GRPO_CONFIG_CLS(
        output_dir=stage_output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        bf16=bf16_supported,
        fp16=torch.cuda.is_available() and not bf16_supported,
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        report_to="tensorboard",
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        temperature=args.temperature,
        log_completions=True,
        use_vllm=False,
    )

    trainer = GRPO_TRAINER_CLS(
        model=model,
        reward_funcs=[make_format_reward_fn(), make_process_reward_fn(), make_terminal_reward_fn()],
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()

    history_path = Path(stage_output_dir) / "log_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(trainer.state.log_history, indent=2), encoding="utf-8")
    print(f"Saved training log history to {history_path}")
    return trainer


def save_final_artifacts(trainer, tokenizer, args: argparse.Namespace):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


def main():
    args = parse_args()
    setup_runtime()
    print(f"Loading model: {args.model_name}")
    model, tokenizer = load_model(args.model_name)

    trainer = None
    for stage_index, (num_samples, level) in enumerate(make_stage_plan(args), start=1):
        trainer = train_stage(model, tokenizer, args, level, num_samples, stage_index)

    if trainer is None:
        raise RuntimeError("No training stages were scheduled.")

    print(f"\nSaving final artifacts to {args.output_dir}")
    save_final_artifacts(trainer, tokenizer, args)
    print("Done.")


if __name__ == "__main__":
    main()
