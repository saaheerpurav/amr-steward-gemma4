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

class _AutoStubMeta(type):
    """Metaclass so the _AutoStub CLASS itself is iterable/subscriptable/attribute-safe."""
    def __iter__(cls): return iter([])
    def __len__(cls): return 0
    def __getitem__(cls, item): return cls()
    def __contains__(cls, item): return False
    def __bool__(cls): return True
    def __getattr__(cls, name): return cls()  # handles SomeStubClass.attr

class _AutoStub(metaclass=_AutoStubMeta):
    """No-op stand-in for any optional dep class — callable, iterable, attribute-safe."""
    def __init__(self, *a, **kw): pass
    def __call__(self, *a, **kw): return _AutoStub()
    def __getattr__(self, n): return _AutoStub()
    def __iter__(self): return iter([])
    def __len__(self): return 0
    def __bool__(self): return True
    def __str__(self): return ""
    def __repr__(self): return "AutoStub()"

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

# Stub all TRL optional deps that aren't installed; covers vllm, mergekit, llm_blender, etc.
_STUB_ROOTS = {
    "vllm", "vllm_ascend",
    "mergekit",            # TRL callbacks/mergekit_utils
    "llm_blender",         # TRL judges.py PairRM
    "liger_kernel",        # TRL chunked GRPO loss
    "unsloth",             # TRL unsloth trainer
    "deepspeed",           # deep distributed (not used single-GPU)
    "ray",                 # TRL ray remote
    "diffusers",           # TRL diffusion policy
}

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
MAX_COMPLETION  = int(os.getenv("MAX_COMPLETION", "256"))
LEARNING_RATE   = float(os.getenv("LEARNING_RATE", "5e-6"))
SFT_SAMPLES_PER_LEVEL = int(os.getenv("SFT_SAMPLES_PER_LEVEL", "25"))

SYSTEM_PROMPT = """You are an antimicrobial stewardship AI. Complete the prescription JSON for the patient."""

# Appended to every prompt so the model MUST generate a drug name first (assistant prefill).
# This eliminates the cold-start problem: format is forced, GRPO trains on drug choice quality.
COMPLETION_PREFIX = 'COMMIT: {"drug": "'

# Context shown in the user message so the model understands the task
FEW_SHOT_EXAMPLE = """\
Complete the prescription for this patient. You will be asked to finish:
COMMIT: {"drug": "[choose best drug]", "dose": "[appropriate dose]", "duration": "[N days]", "justification": "[reason]"}
---
PATIENT:"""

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

    def _score_env_prose(text, patient_payload, level):
        """Prose fallback: scan free text for any drug name from the patient's antibiogram.
        Returns 0.25× the formal reward so COMMIT: format is still strongly preferred."""
        formal = _score_env(text, patient_payload, level)
        if formal > 0:
            return formal
        import re as _re2
        try:
            patient = PatientCase(**json.loads(patient_payload))
            text_lower = text.lower()
            best = 0.0
            for drug in patient.antibiogram:
                # Match drug name with flexible separators (ceftazidime-avibactam → ceftazidime avibactam)
                pattern = drug.lower().replace("-", r"[-\s]?")
                if _re2.search(pattern, text_lower):
                    try:
                        env2 = AMREnvironment()
                        env2.reset(curriculum_level=int(level), patient=patient)
                        obs = env2.step(AMRAction(
                            action_type="COMMIT",
                            prescription={"drug": drug, "dose": "standard", "duration": "14 days", "justification": ""}
                        ))
                        if obs.reward:
                            best = max(best, float(obs.reward))
                    except Exception:
                        pass
            return best * 0.25
        except Exception:
            return 0.0

    def fmt_fn(prompts, completions, **kw):
        import re as _re
        rewards = []
        for c in completions:
            # Prefill "COMMIT: {\"drug\": \"" is in the prompt, not the completion —
            # reconstruct the full text so COMMIT: is visible to the reward function.
            text = COMPLETION_PREFIX + _completion_to_text(c)
            lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
            has_commit = any(_re.search(r"COMMIT\s*:", l, _re.IGNORECASE) for l in lines)
            has_action = any(_re.search(r"(INVESTIGATE|COMMIT)\s*:", l, _re.IGNORECASE) for l in lines)
            if not has_commit:
                rewards.append(0.02 if has_action else 0.0)
            else:
                r6 = float(R6_format(text))
                rewards.append(0.05 + r6 * 0.10)
        log(f"  fmt_rewards={[round(r,4) for r in rewards]} sample={_completion_to_text(completions[0])[:80]!r}")
        return rewards

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
            try: rewards.append(float(_score_env_prose(COMPLETION_PREFIX + _completion_to_text(comp), payload, int(lvl))))
            except: rewards.append(0.0)
        log(f"  term_rewards={[round(r,4) for r in rewards]}")
        return rewards

    return fmt_fn, proc_fn, term_fn


# ── Dataset builder ────────────────────────────────────────────────────────────

def build_dataset(level: int, num_samples: int, tokenizer=None):
    from datasets import Dataset
    from env import AMREnvironment

    def _render(obs):
        user_content = FEW_SHOT_EXAMPLE + "\n" + obs.patient_text
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
            ) + COMPLETION_PREFIX
        return "\n\n".join([f"SYSTEM:\n{SYSTEM_PROMPT}", f"USER:\n{user_content}", "ASSISTANT:"]) + COMPLETION_PREFIX

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


# ── SFT warm-up ───────────────────────────────────────────────────────────────

def build_sft_dataset(num_per_level: int = 25, tokenizer=None):
    """Build (prompt+completion) demonstrations for SFT warm-up.

    Scores every drug via AMREnvironment to pick the correct answer, then
    formats it as INVESTIGATE/COMMIT lines so the model learns the output format.
    """
    from datasets import Dataset
    from env import AMREnvironment, AMRAction

    DOSE_TABLE = {
        "meropenem":               "1g IV q8h",
        "ceftazidime-avibactam":   "2.5g IV q8h",
        "colistin":                "150mg IV q12h",
        "meropenem-vaborbactam":   "4g IV q8h",
        "ceftriaxone":             "2g IV q24h",
        "ciprofloxacin":           "400mg IV q12h",
        "ampicillin-sulbactam":    "3g IV q6h",
        "piperacillin-tazobactam": "4.5g IV q6h",
    }

    def _best_drug(patient, level):
        """Ask the reward function which drug scores highest."""
        best_drug, best_r = None, -1.0
        for drug in patient.antibiogram:
            try:
                e2 = AMREnvironment()
                e2.reset(curriculum_level=level, patient=patient)
                obs = e2.step(AMRAction(
                    action_type="COMMIT",
                    prescription={"drug": drug, "dose": "standard",
                                  "duration": "14 days", "justification": "test"},
                ))
                r = obs.reward or 0.0
                if r > best_r:
                    best_r, best_drug = r, drug
            except Exception:
                pass
        return best_drug or list(patient.antibiogram.keys())[0]

    def _renal_adjust(drug, dose, crcl):
        if not crcl or crcl >= 50:
            return dose
        if crcl < 30:
            if drug == "ceftazidime-avibactam": return "0.94g IV q24h"
            if drug == "meropenem":             return "500mg IV q12h"
            if drug == "meropenem-vaborbactam": return "2g IV q8h"
        return dose

    DURATION = {"bacteremia": "14 days", "pneumonia": "7 days", "uti": "7 days"}

    env = AMREnvironment()
    rows = []
    for level in [1, 2, 3]:
        for _ in range(num_per_level):
            obs = env.reset(curriculum_level=level)
            patient = env.current_patient
            drug = _best_drug(patient, level)
            crcl = getattr(patient, "creatinine_clearance", None)
            dose = _renal_adjust(drug, DOSE_TABLE.get(drug, "standard dose IV q8h"), crcl)
            site = getattr(patient, "infection_site", "bacteremia")
            mic  = patient.antibiogram.get(drug, 1.0)
            dur  = DURATION.get(site, "14 days")

            # Completion is everything AFTER the COMPLETION_PREFIX that ends the prompt.
            # prompt ends: ...COMMIT: {"drug": "
            # completion:  ceftazidime-avibactam", "dose": "2.5g IV q8h", ...}
            completion = (
                f'{drug}", "dose": "{dose}", "duration": "{dur}", '
                f'"justification": "MIC {mic}, guideline-appropriate for {site}"}}'
            )

            user_content = FEW_SHOT_EXAMPLE + "\n" + obs.patient_text
            user_content += f"\n\nINVESTIGATION BUDGET REMAINING: {obs.budget_remaining}"
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ]
            if tokenizer is not None:
                prompt_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                ) + COMPLETION_PREFIX
                eos = tokenizer.eos_token or ""
            else:
                prompt_text = f"SYSTEM:\n{SYSTEM_PROMPT}\n\nUSER:\n{user_content}\n\nASSISTANT:{COMPLETION_PREFIX}"
                eos = "\n"

            rows.append({"text": prompt_text + completion + eos})

    log(f"SFT dataset: {len(rows)} demonstrations across levels 1-3")
    return Dataset.from_list(rows)


def train_sft(model, tokenizer):
    """SFT warm-up: teach the model to produce COMMIT: format before GRPO."""
    import torch
    from trl import SFTTrainer, SFTConfig

    log("=== SFT Warm-up: teaching COMMIT: format ===")
    _status.update({"phase": "training", "stage": "sft_warmup"})

    dataset = build_sft_dataset(num_per_level=SFT_SAMPLES_PER_LEVEL, tokenizer=tokenizer)
    out_dir  = f"{OUTPUT_DIR}/sft"

    config = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=False,
        logging_steps=5,
        save_steps=999999,
        save_total_limit=1,
        report_to="none",
        max_seq_length=768,
        dataset_text_field="text",
        packing=False,
        dataloader_num_workers=0,
    )

    # Completion-only masking using COMPLETION_PREFIX as the boundary.
    # The prompt ends with COMPLETION_PREFIX; loss is computed only on what follows.
    try:
        from trl import DataCollatorForCompletionOnlyLM
        resp_tpl_ids = tokenizer.encode(COMPLETION_PREFIX, add_special_tokens=False)
        collator = DataCollatorForCompletionOnlyLM(resp_tpl_ids, tokenizer=tokenizer)
        log(f"  SFT: completion-only masking, template={COMPLETION_PREFIX!r} ids={resp_tpl_ids}")
    except Exception as e:
        collator = None
        log(f"  SFT: DataCollatorForCompletionOnlyLM unavailable ({e}), using full-text loss")

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=collator,
    )

    _orig_log = trainer.log
    def _patched_log(logs, *a, **kw):
        numeric = {k: round(v, 5) for k, v in logs.items() if isinstance(v, (int, float))}
        if numeric:
            log(f"  SFT: {numeric}")
        _orig_log(logs, *a, **kw)
    trainer.log = _patched_log

    _stop_hb = threading.Event()
    def _hb():
        n = 0
        while not _stop_hb.wait(30):
            n += 1
            log(f"  [SFT heartbeat #{n}] still training ...")
    threading.Thread(target=_hb, daemon=True).start()

    log("SFT trainer.train() ...")
    trainer.train()
    _stop_hb.set()

    history = trainer.state.log_history
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "log_history.json").write_text(json.dumps(history, indent=2))

    sft_steps = [h for h in history if "loss" in h and "train_runtime" not in h]
    if sft_steps:
        losses = [h["loss"] for h in sft_steps]
        log(f"  SFT done | initial_loss={losses[0]:.3f} final_loss={losses[-1]:.3f}")
    log("SFT warm-up complete — model can now produce COMMIT: tokens.")


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
    # PEFT/TRL expect warnings_issued on the base model; Gemma4 doesn't have it
    for _m in [model, getattr(model, 'base_model', None), getattr(getattr(model, 'base_model', None), 'model', None)]:
        if _m is not None and not hasattr(_m, 'warnings_issued'):
            _m.warnings_issued = {}
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Prevent the "INVESTINVESTINVEST" token-repetition loop during GRPO generation
    if hasattr(model, "generation_config"):
        model.generation_config.repetition_penalty = 1.3
        log("Set repetition_penalty=1.3 on generation_config")

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
        report_to="none",
        max_completion_length=MAX_COMPLETION,
        num_generations=NUM_GENERATIONS,
        temperature=0.9,
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

    for stage in ["sft", "stage1", "stage2", "stage3"]:
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

        fig, axes = plt.subplots(1, 4, figsize=(20, 4))
        fig.suptitle("AMR-Steward: Gemma 4 Training — SFT Warm-up + GRPO Curriculum",
                     fontsize=13, fontweight="bold")

        # Panel 0: SFT loss
        ax = axes[0]
        sft_path = Path(f"{OUTPUT_DIR}/sft/log_history.json")
        if sft_path.exists():
            history = json.loads(sft_path.read_text())
            steps_data = [h for h in history if "loss" in h and "train_runtime" not in h]
            if steps_data:
                steps  = [h["step"] for h in steps_data]
                losses = [h["loss"] for h in steps_data]
                ax.plot(steps, losses, color="#9C27B0", linewidth=2)
                ax.fill_between(steps, losses, alpha=0.1, color="#9C27B0")
        ax.set_title("SFT Warm-up\n(Cross-entropy loss)", fontsize=10)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)

        # Panels 1-3: GRPO reward per stage
        stage_labels = ["stage1", "stage2", "stage3"]
        stage_names  = ["Stage 1\nSusceptible", "Stage 2\nResistant/MDR", "Stage 3\nMDR+Renal"]
        colors       = ["#2196F3", "#FF9800", "#F44336"]
        for ax, label, name, color in zip(axes[1:], stage_labels, stage_names, colors):
            hist_path = Path(f"{OUTPUT_DIR}/{label}/log_history.json")
            if hist_path.exists():
                history = json.loads(hist_path.read_text())
                train_steps = [h for h in history if "reward" in h and "train_runtime" not in h]
                if train_steps:
                    steps   = [h["step"] for h in train_steps]
                    rewards = [h["reward"] for h in train_steps]
                    ax.plot(steps, rewards, color=color, linewidth=2)
                    ax.fill_between(steps, rewards, alpha=0.1, color=color)
                    ax.axhline(max(rewards), color=color, linestyle="--", alpha=0.4,
                               label=f"Peak {max(rewards):.3f}")
                    ax.legend(fontsize=9)
            ax.set_title(f"GRPO {name}", fontsize=10)
            ax.set_xlabel("Step")
            ax.set_ylabel("Mean Reward")
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
        train_sft(model, tokenizer)
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
        log("Training failed — keeping space alive 900s for error inspection ...")
        time.sleep(900)
        _auto_pause_space()
