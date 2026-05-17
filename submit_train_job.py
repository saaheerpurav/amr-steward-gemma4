"""
submit_train_job.py — Create and start an AMR-Steward training run on HF hardware.

Usage:
    python submit_train_job.py --token hf_xxx
    python submit_train_job.py --token hf_xxx --hardware a10g-large --model google/gemma-4-e2b-it
    python submit_train_job.py --token hf_xxx --delete   # delete Space when done

Hardware options and hourly cost:
    t4-medium     ~$0.60/hr  (T4 16GB)   — gemma-4-e2b-it fits, ~90 min
    a10g-small    ~$1.05/hr  (A10G 24GB) — gemma-4-e2b-it fast, ~45 min
    a10g-large    ~$3.15/hr  (A10G 40GB) — recommended, ~60 min
    a100-large    ~$7.60/hr  (A100 80GB) — fastest, ~40 min
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

try:
    from huggingface_hub import HfApi, SpaceHardware
except ImportError:
    print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub>=0.23")
    sys.exit(1)

TRAINER_REPO = "saaheerpurav/amr-steward-trainer"
TRAINING_DIR = Path(__file__).parent / "training"

HARDWARE_ALIASES = {
    "t4":         "t4-medium",
    "t4-medium":  "t4-medium",
    "a10g":       "a10g-small",
    "a10g-small": "a10g-small",
    "a10g-large": "a10g-large",
    "a100":       "a100-large",
    "a100-large": "a100-large",
}


def parse_args():
    p = argparse.ArgumentParser(description="Submit AMR-Steward training job to HF.")
    p.add_argument("--token",    required=True, help="HuggingFace write token")
    p.add_argument("--hardware", default="a10g-large",
                   choices=list(HARDWARE_ALIASES.keys()),
                   help="GPU hardware tier (default: a10g-large ~$3/hr)")
    p.add_argument("--model",    default="google/gemma-4-e2b-it",
                   help="Base model (default: google/gemma-4-e2b-it)")
    p.add_argument("--repo",     default=TRAINER_REPO,
                   help="Space repo id to create (default: %(default)s)")
    p.add_argument("--samples",  nargs=3, type=int, default=[128, 64, 32],
                   metavar=("S1","S2","S3"),
                   help="Samples per curriculum stage (default: 128 64 32)")
    p.add_argument("--delete",   action="store_true",
                   help="Delete the training Space when it finishes")
    p.add_argument("--watch",    action="store_true",
                   help="Tail Space logs in terminal (blocks until done)")
    return p.parse_args()


def main():
    args = parse_args()
    hw = HARDWARE_ALIASES.get(args.hardware, args.hardware)
    s1, s2, s3 = args.samples

    api = HfApi(token=args.token)
    me = api.whoami()["name"]
    print(f"Logged in as: {me}")

    # Resolve repo namespace
    repo_id = args.repo
    if "/" not in repo_id:
        repo_id = f"{me}/{repo_id}"

    # 1. Create the Space (Docker SDK)
    print(f"\nCreating training Space: {repo_id}")
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
        private=True,   # private so it doesn't show up publicly
    )

    # 2. Upload training files
    print("Uploading training files ...")
    api.upload_file(
        path_or_fileobj=str(TRAINING_DIR / "Dockerfile"),
        path_in_repo="Dockerfile",
        repo_id=repo_id, repo_type="space",
        commit_message="Add training Dockerfile",
    )
    api.upload_file(
        path_or_fileobj=str(TRAINING_DIR / "train_space.py"),
        path_in_repo="train_space.py",
        repo_id=repo_id, repo_type="space",
        commit_message="Add training entrypoint",
    )

    # 3. Set Space secrets (token + config)
    print("Setting Space secrets ...")
    api.add_space_secret(repo_id=repo_id, key="HF_TOKEN",      value=args.token)
    api.add_space_secret(repo_id=repo_id, key="TRAINER_REPO",  value=repo_id)   # for auto-pause
    api.add_space_secret(repo_id=repo_id, key="MODEL_NAME",    value=args.model)
    api.add_space_secret(repo_id=repo_id, key="SAMPLES_S1",    value=str(s1))
    api.add_space_secret(repo_id=repo_id, key="SAMPLES_S2",    value=str(s2))
    api.add_space_secret(repo_id=repo_id, key="SAMPLES_S3",    value=str(s3))

    # 4. Upgrade hardware
    print(f"Requesting hardware: {hw} ...")
    api.request_space_hardware(repo_id=repo_id, hardware=hw)

    space_url = f"https://huggingface.co/spaces/{repo_id}"
    print(f"\nTraining Space live: {space_url}")
    print(f"Model will be pushed to: https://huggingface.co/saaheerpurav/amr-steward-gemma4")
    print(f"\nStatus page (once Space starts): https://{repo_id.replace('/', '-')}.hf.space")
    rate = {'t4-medium':0.60,'a10g-small':1.05,'a10g-large':3.15,'a100-large':7.60}.get(hw,4.0)
    print(f"\nEstimated cost: ~${rate * 1.5:.2f} (Space auto-pauses when training finishes)")

    if args.watch:
        print("\nWatching logs (Ctrl-C to stop watching — training continues):\n")
        _watch_logs(api, repo_id, args.delete)
    else:
        print("\nRun with --watch to stream logs, or check the Space URL above.")
        print("Run with --delete to delete the Space automatically when done.")


def _watch_logs(api: HfApi, repo_id: str, delete_when_done: bool):
    try:
        for event in api.get_space_runtime(repo_id=repo_id).__class__.__mro__:
            pass
    except Exception:
        pass

    done_keywords = ["Training complete!", "ERROR:", "Status server running"]
    print("Waiting for Space to build ...")

    seen = set()
    while True:
        try:
            # Poll Space runtime status
            runtime = api.get_space_runtime(repo_id=repo_id)
            stage = getattr(runtime, "stage", "unknown")

            if stage == "RUNNING":
                # Try to fetch logs
                try:
                    logs = list(api.get_space_logs(repo_id=repo_id))
                    for entry in logs:
                        key = (getattr(entry, "timestamp", ""), getattr(entry, "data", ""))
                        if key not in seen:
                            seen.add(key)
                            print(getattr(entry, "data", str(entry)))
                            if any(kw in str(entry) for kw in done_keywords):
                                print("\nTraining finished.")
                                if delete_when_done:
                                    print(f"Deleting Space {repo_id} ...")
                                    api.delete_repo(repo_id=repo_id, repo_type="space")
                                    print("Space deleted.")
                                return
                except Exception:
                    pass
            elif stage in ("BUILD_ERROR", "RUNTIME_ERROR"):
                print(f"Space error: {stage}")
                return
            else:
                print(f"  Space status: {stage} ...")

        except Exception as exc:
            print(f"  Polling error: {exc}")

        time.sleep(30)


if __name__ == "__main__":
    main()
