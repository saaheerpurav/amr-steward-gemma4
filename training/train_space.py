"""
AMR-Steward Training Space entrypoint.

Runs in an HF Space with GPU hardware.
Clones repo → trains Gemma 4 with TRL GRPOTrainer + PEFT LoRA → pushes model to HF Hub.
Also serves a minimal status page on :7860 so the Space stays healthy.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

# ── Single-GPU: strip distributed env vars before any ML library loads ─────────
for _k in ("LOCAL_RANK", "RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
    os.environ.pop(_k, None)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Disable torch.compile / Dynamo to avoid any compilation overhead during training
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

# Stub ALL vllm.* and vllm_ascend.* imports so TRL's vllm_client doesn't crash
# Uses modern find_spec/create_module/exec_module API (required for Python 3.10+)
import types as _types, importlib.machinery as _imm

class _AutoStub:
    """No-op stand-in for any vllm class — callable, iterable, attribute-safe."""
    def __init__(self, *a, **kw): pass
    def __call__(self, *a, **kw): return _AutoStub()
    def __getattr__(self, n): return _AutoStub()
    def __iter__(self): return iter([])

class _VLLMStubModule(_types.ModuleType):
    def __getattr__(self, name): return _AutoStub

class _VLLMLoader:
    def create_module(self, spec):
        m = _VLLMStubModule(spec.name)
        m.__spec__ = spec
        m.__package__ = spec.name.split(".")[0]
        m.__path__ = []
        return m
    def exec_module(self, module): pass

_vllm_loader = _VLLMLoader()

_STUB_ROOTS = {"vllm", "vllm_ascend", "mergekit"}

class _VLLMFinder:
    @staticmethod
    def find_spec(name, path, target=None):
        root = name.split(".")[0]
        if root in _STUB_ROOTS:
            if name in sys.modules:
                return sys.modules[name].__spec__
            return _imm.ModuleSpec(name, _vllm_loader, origin="stub")
        return None

sys.meta_path.insert(0, _VLLMFinder)

_accel_cfg = Path.home() / ".cache" / "huggingface" / "accelerate" / "default_config.yaml"
_accel_cfg.parent.mkdir(parents=True, exist_ok=True)
_accel_cfg.write_text("compute_environment: LOCAL_MACHINE\ndistributed_type: 'NO'\nnum_processes: 1\n")

# ── Config (override via HF Space secrets) ────────────────────────────────────
MODEL_NAME      = os.getenv("MODEL_NAME",    "google/gemma-4-e2b-it")
HF_REPO_ID      = os.getenv("HF_REPO_ID",   "saaheerpurav/amr-steward-gemma4")
TRAINER_REPO    = os.getenv("TRAINER_REPO",  "saaheerpurav/amr-steward-trainer")
HF_TOKEN        = os.getenv("HF_TOKEN",      "")
OUTPUT_DIR      = "/app/checkpoints"
SAMPLES_S1      = int(os.getenv("SAMPLES_S1", "128"))
SAMPLES_S2      = int(os.getenv("SAMPLES_S2",  "64"))
SAMPLES_S3      = int(os.getenv("SAMPLES_S3",  "32"))
NUM_GENERATIONS = int(os.getenv("NUM_GENERATIONS", "4"))
MAX_COMPLETION  = int(os.getenv("MAX_COMPLETION", "512"))
LEARNING_RATE   = float(os.getenv("LEARNING_RATE", "5e-6"))

SYSTEM_PROMPT = """You are an antimicrobial stewardship AI. Prescribe the narrowest effective antibiotic.

INVESTIGATE tools (optional, costs 1 budget each):
  INVESTIGATE: {"tool": "interpret_resistance", "arg": "<drug>"}
  INVESTIGATE: {"tool": "check_guideline", "arg": "<syndrome>"}
  INVESTIGATE: {"tool": "assess_patient_factors"}

When ready, output EXACTLY this one line and stop:
  COMMIT: {"drug": "<name>", "dose": "<dose>", "duration": "<days>", "justification": "<one sentence>"}

Do not add any text before or after the COMMIT line."""

_status: dict[str, Any] = {"phase": "starting", "stage": None, "reward": None, "error": None, "last_trl_metrics": None}
_log_lines: list[str] = []
_start_time = time.time()

# ── Status HTTP server ─────────────────────────────────────────────────────────

def _run_status_server():
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse
    import uvicorn

    srv = FastAPI()

    @srv.get("/", response_class=HTMLResponse)
    def root():
        elapsed = int(time.time() - _start_time)
        h, m, s = elapsed // 3600, (elapsed % 3600) // 60, elapsed % 60
        logs_html = "".join(f"<div class='log'>{l}</div>" for l in _log_lines[-40:])
        return HTMLResponse(f"""<!DOCTYPE html><html><head><meta charset=utf-8>
<meta http-equiv="refresh" content="20">
<title>AMR-Steward Trainer</title>
<style>body{{font-family:monospace;background:#0d1117;color:#e6edf3;padding:20px}}
h1{{color:#58a6ff}}.log{{font-size:0.85em;margin:2px 0;color:#aaa}}
.status{{background:#161b22;padding:12px;border-radius:8px;margin:12px 0}}</style>
</head><body>
<h1>AMR-Steward Training Space</h1>
<div class="status">
  Phase: <b>{_status['phase']}</b> &nbsp;|&nbsp;
  Stage: <b>{_status['stage'] or '—'}</b> &nbsp;|&nbsp;
  Last reward: <b>{_status['reward'] or '—'}</b> &nbsp;|&nbsp;
  Elapsed: <b>{h:02d}:{m:02d}:{s:02d}</b>
</div>
<b>Recent logs:</b>
{logs_html}
</body></html>""")

    @srv.get("/status")
    def status():
        return JSONResponse(_status)

    uvicorn.run(srv, host="0.0.0.0", port=7860, log_level="error")


# ── Helpers ────────────────────────────────────────────────────────────────────

def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    _log_lines.append(line)


def setup_hf():
    if HF_TOKEN:
        from huggingface_hub import login
        login(token=HF_TOKEN, add_to_git_credential=False)
        log(f"Authenticated as {HF_REPO_ID.split('/')[0]}")
    else:
        log("WARNING: HF_TOKEN not set — model push will fail")


def clone_repo():
    log("Cloning saaheerpurav/amr-steward-gemma4 ...")
    subprocess.run(
        ["git", "clone", "https://github.com/saaheerpurav/amr-steward-gemma4.git", "/app/repo"],
        check=True, capture_output=True
    )
    sys.path.insert(0, "/app/repo")
    log("Repo cloned.")


# ── Reward functions ───────────────────────────────────────────────────────────

def _completion_to_text(c):
    if isinstance(c, str): return c
    if isinstance(c, list):
        parts = []
        for item in c:
            if isinstance(item, dict): parts.append(item.get("content", ""))
            elif isinstance(item, str): parts.append(item)
        return "\n".join(parts)
    return str(c)


def _build_reward_fns():
    from env import AMREnvironment, AMRAction
    from env.models import PatientCase
    from env.reward import (
        parse_prescription_from_text, parse_tool_calls_from_text,
        R5_tool_efficiency, R6_format,
    )

    _BUDGET_BY_LEVEL = {1: 5, 2: 4, 3: 3}

    def _extract_tool_type(tc):
        try: return json.loads(tc.strip()).get("tool", tc)
        except: return tc.split()[0] if tc.split() else "unknown"

    def _parse_investigate(raw):
        try:
            p = json.loads(raw.strip())
            t = p.get("tool")
            return (t, p.get("arg") or None) if isinstance(t, str) and t else None
        except: return None

    def _score_env(text, patient_payload, level):
        patient = PatientCase(**json.loads(patient_payload))
        env = AMREnvironment()
        env.reset(curriculum_level=int(level), patient=patient)
        cum = 0.0
        for raw_line in parse_tool_calls_from_text(text):
            if env._state.done: break
            parsed = _parse_investigate(raw_line)
            if not parsed: continue
            tool_name, tool_arg = parsed
            try:
                obs = env.step(AMRAction(action_type="INVESTIGATE", tool_name=tool_name, tool_arg=tool_arg))
                if obs.reward: cum += obs.reward
            except: continue
        if not env._state.done:
            prx = parse_prescription_from_text(text)
            if prx:
                try:
                    obs = env.step(AMRAction(action_type="COMMIT", prescription=prx))
                    if obs.reward: cum += obs.reward
                except: pass
        return cum

    def fmt_fn(prompts, completions, **kw):
        return [float(R6_format(_completion_to_text(c))) * 0.05 for c in completions]

    def proc_fn(prompts, completions, patient_json, curriculum_level=None, **kw):
        levels = list(curriculum_level) if curriculum_level else [1] * len(completions)
        rewards = []
        for comp, lvl in zip(completions, levels):
            text = _completion_to_text(comp)
            tcs = parse_tool_calls_from_text(text)
            bt = _BUDGET_BY_LEVEL.get(int(lvl), 5)
            rewards.append(float(R5_tool_efficiency(
                len({_extract_tool_type(tc) for tc in tcs}), len(tcs),
                max(0, bt - len(tcs)), bt
            )))
        return rewards

    def term_fn(prompts, completions, patient_json, curriculum_level=None, **kw):
        levels = list(curriculum_level) if curriculum_level else [1] * len(completions)
        rewards = []
        for comp, payload, lvl in zip(completions, patient_json, levels):
            try: rewards.append(float(_score_env(_completion_to_text(comp), payload, int(lvl))))
            except: rewards.append(0.0)
        return rewards

    return fmt_fn, proc_fn, term_fn


# ── Dataset builder ────────────────────────────────────────────────────────────

def build_dataset(level: int, num_samples: int, tokenizer=None):
    from datasets import Dataset
    from env import AMREnvironment

    def _render(obs):
        user_content = obs.patient_text
        if obs.tool_results:
            user_content += "\n\nINVESTIGATION RESULTS:\n" + "\n---\n".join(obs.tool_results)
        if obs.world_model_rankings:
            user_content += f"\n\n{obs.world_model_rankings}"
        user_content += f"\n\nINVESTIGATION BUDGET REMAINING: {obs.budget_remaining}"
        if obs.budget_remaining == 0:
            user_content += "\n\nYOU MUST COMMIT NOW."
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        if tokenizer is not None:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        return "\n\n".join([f"SYSTEM:\n{SYSTEM_PROMPT}", f"USER:\n{user_content}", "ASSISTANT:"])

    env = AMREnvironment()
    rows = []
    for i in range(num_samples):
        obs = env.reset(curriculum_level=level)
        rows.append({
            "prompt": _render(obs),
            "patient_json": json.dumps(asdict(env.current_patient)),
            "curriculum_level": level,
            "case_id": f"level{level}-case{i}",
        })
    return Dataset.from_list(rows)


# ── Training ───────────────────────────────────────────────────────────────────

def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    log(f"Loading {MODEL_NAME} ...")
    _dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    log(f"Using dtype: {_dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=_dtype, device_map="auto", attn_implementation="eager"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Gemma 4 wraps Linear in Gemma4ClippableLinear; target the inner .linear attribute
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=[
            "q_proj.linear", "k_proj.linear", "v_proj.linear", "o_proj.linear",
            "gate_proj.linear", "up_proj.linear", "down_proj.linear",
        ],
    )
    model = get_peft_model(model, lora_cfg)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.print_trainable_parameters()
    return model, tokenizer


def train_stage(model, tokenizer, level, num_samples, stage_label, reward_fns):
    import torch
    from trl import GRPOConfig, GRPOTrainer

    log(f"=== {stage_label}: level={level}, samples={num_samples} ===")
    _status.update({"phase": "training", "stage": stage_label})

    dataset = build_dataset(level, num_samples, tokenizer)
    log(f"Dataset built: {len(dataset)} cases")

    out_dir = f"{OUTPUT_DIR}/{stage_label}"
    _stop_heartbeat = threading.Event()
    def _heartbeat():
        n = 0
        while not _stop_heartbeat.wait(30):
            n += 1
            log(f"  [heartbeat #{n}] trainer.train() still running ...")
    threading.Thread(target=_heartbeat, daemon=True).start()

    config = GRPOConfig(
        output_dir=out_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=False,
        logging_steps=1,
        save_steps=50,
        save_total_limit=2,
        report_to="tensorboard",
        max_completion_length=MAX_COMPLETION,
        num_generations=NUM_GENERATIONS,
        temperature=0.7,
        log_completions=False,
        use_vllm=False,
        dataloader_num_workers=0,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=list(reward_fns),
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    _orig_log = trainer.log
    def _patched_log(logs, *args, **kwargs):
        numeric = {k: round(v, 5) for k, v in logs.items() if isinstance(v, (int, float))}
        if numeric:
            _status["last_trl_metrics"] = numeric
            log(f"  step metrics: {numeric}")
        reward_val = logs.get("reward") or logs.get("mean_reward") or logs.get("rewards")
        if reward_val is not None:
            _status["reward"] = f"{float(reward_val):.4f}"
            log(f"  step={logs.get('step',0)} reward={float(reward_val):.4f}")
        _orig_log(logs, *args, **kwargs)
    trainer.log = _patched_log

    log("Calling trainer.train() ...")
    trainer.train()
    _stop_heartbeat.set()
    history = trainer.state.log_history
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "log_history.json").write_text(json.dumps(history, indent=2))

    train_steps = [h for h in history if "reward" in h and "train_runtime" not in h]
    if train_steps:
        r = [h["reward"] for h in train_steps]
        log(f"  {stage_label} done | initial={r[0]:.3f} peak={max(r):.3f} final={r[-1]:.3f}")

    return trainer


def push_model(trainer, tokenizer):
    from huggingface_hub import HfApi

    log(f"Pushing model to {HF_REPO_ID} ...")
    _status["phase"] = "pushing"

    final_dir = f"{OUTPUT_DIR}/final"
    Path(final_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    api = HfApi()
    api.create_repo(HF_REPO_ID, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=final_dir,
        repo_id=HF_REPO_ID,
        repo_type="model",
        commit_message="Gemma 4 GRPO training: 3-stage curriculum, 3-head reward (quality_ratio + tool efficiency + format)",
    )
    log(f"Model pushed to https://huggingface.co/{HF_REPO_ID}")

    curve = Path("/app/reward_curves.png")
    if curve.exists():
        api.upload_file(path_or_fileobj=str(curve), path_in_repo="reward_curves.png",
                       repo_id=HF_REPO_ID, repo_type="model")
        log("Uploaded reward_curves.png to model repo.")

    for stage in ["stage1", "stage2", "stage3"]:
        stage_dir = Path(f"{OUTPUT_DIR}/{stage}")
        log_file = stage_dir / "log_history.json"
        if log_file.exists():
            api.upload_file(path_or_fileobj=str(log_file),
                           path_in_repo=f"training_logs/{stage}/log_history.json",
                           repo_id=HF_REPO_ID, repo_type="model")
            log(f"Uploaded {stage}/log_history.json")


def plot_curves():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        stage_labels = ["stage1", "stage2", "stage3"]
        stage_names  = ["Stage 1 — Susceptible", "Stage 2 — Resistant/MDR",
                        "Stage 3 — MDR + Renal + Allergies"]
        colors = ["#2196F3", "#FF9800", "#F44336"]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
        fig.suptitle("AMR-Steward: Gemma 4 GRPO Reward Across Curriculum Stages", fontsize=14, fontweight="bold")

        for ax, label, name, color in zip(axes, stage_labels, stage_names, colors):
            hist_path = Path(f"{OUTPUT_DIR}/{label}/log_history.json")
            if not hist_path.exists():
                ax.set_title(name); continue
            history = json.loads(hist_path.read_text())
            train_steps = [h for h in history if "reward" in h and "train_runtime" not in h]
            if train_steps:
                steps = [h["step"] for h in train_steps]
                rewards = [h["reward"] for h in train_steps]
                ax.plot(steps, rewards, marker="o", color=color, linewidth=2, markersize=6)
                ax.fill_between(steps, rewards, alpha=0.1, color=color)
                ax.axhline(max(rewards), color=color, linestyle="--", alpha=0.4, label=f"Peak {max(rewards):.3f}")
                ax.legend(fontsize=9)
            ax.set_title(name, fontsize=11)
            ax.set_xlabel("Training Step")
            ax.set_ylabel("Mean Reward" if ax == axes[0] else "")
            ax.set_ylim(0, 1.0)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("/app/reward_curves.png", dpi=150, bbox_inches="tight")
        log("Saved reward_curves.png")
    except Exception as e:
        log(f"Curve plot failed: {e}")


# ── Main ───────────────────────────────────────────────────────────────────────

def train_main():
    try:
        setup_hf()
        clone_repo()

        model, tokenizer = load_model()
        fmt_fn, proc_fn, term_fn = _build_reward_fns()

        trainer = None
        for level, num_samples, label in [
            (1, SAMPLES_S1, "stage1"),
            (2, SAMPLES_S2, "stage2"),
            (3, SAMPLES_S3, "stage3"),
        ]:
            trainer = train_stage(model, tokenizer, level, num_samples, label, (fmt_fn, proc_fn, term_fn))

        plot_curves()
        push_model(trainer, tokenizer)

        _status["phase"] = "done"
        log("Training complete!")

    except Exception as exc:
        import traceback
        _status["phase"] = "error"
        _status["error"] = str(exc)
        log(f"ERROR: {exc}")
        traceback.print_exc()


def _auto_pause_space():
    if not HF_TOKEN or not TRAINER_REPO:
        log("Auto-pause skipped: HF_TOKEN or TRAINER_REPO not set.")
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        log(f"Pausing Space {TRAINER_REPO} to stop GPU billing ...")
        api.pause_space(repo_id=TRAINER_REPO)
        log("Space paused. GPU billing stopped. Model is on HF Hub.")
    except Exception as e:
        log(f"Auto-pause failed ({e}). Pause manually at huggingface.co/spaces/{TRAINER_REPO}")


if __name__ == "__main__":
    server_thread = threading.Thread(target=_run_status_server, daemon=True)
    server_thread.start()
    time.sleep(2)

    train_main()

    if _status["phase"] == "done":
        log("Waiting 60s before auto-pause so status page is readable ...")
        time.sleep(60)
        _auto_pause_space()
    else:
        # Error path: keep space alive so monitor can read the error before pausing
        log("Training failed — keeping space alive 300s for error inspection ...")
        time.sleep(300)
        _auto_pause_space()
