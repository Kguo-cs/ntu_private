#!/usr/bin/env python3
"""
Run Hydra+Lightning training and auto-resume from the latest checkpoint if it fails.

Usage:
  python run_with_resume.py -- python run.py action=fit
  # (any extra Hydra overrides can follow; we'll append ckpt_path=... only on retries)

Options:
  --max-retries N       Max number of retries after failures (default: 5)
  --sleep-seconds S     Seconds to sleep between retries (default: 15)
  --search-root PATH    Where to search for *.ckpt (default: current working dir)
  --prefer-last         Prefer files named 'last.ckpt' if present (default: True)
  --offline             Set WANDB_MODE=offline for the child process
  --no-offline          Do not force W&B offline
  --print-cmd           Echo the spawned command before each run
  --hydra-full-error    Set HYDRA_FULL_ERROR=1 for clearer stack traces
  --env KEY=VALUE ...   Extra environment variables to set for the child process
"""
from __future__ import annotations

import argparse
import os
import shlex
import signal
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple
import subprocess
import glob

def parse_args() -> Tuple[argparse.Namespace, List[str]]:
    if "--" in sys.argv:
        sep = sys.argv.index("--")
        wrapper_argv = sys.argv[1:sep]
        child_argv = sys.argv[sep + 1:]
    else:
        # Default child command if not provided
        wrapper_argv = sys.argv[1:]
        child_argv = ["python", "run.py", "action=fit"]

    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--sleep-seconds", type=int, default=15)
    ap.add_argument("--search-root", type=str, default=os.getcwd())
    ap.add_argument("--prefer-last", action="store_true", default=True)
    ap.add_argument("--offline", dest="offline", action="store_true", default=False)
    ap.add_argument("--no-offline", dest="offline", action="store_false")
    ap.add_argument("--print-cmd", action="store_true", default=False)
    ap.add_argument("--hydra-full-error", action="store_true", default=True)
    ap.add_argument("--env", nargs="*", default=[], help="KEY=VALUE pairs for child env")
    args = ap.parse_args(wrapper_argv)
    return args, child_argv

def already_has_ckpt_override(argv: List[str]) -> bool:
    for a in argv:
        # Accept both 'ckpt_path=...' and 'trainer.ckpt_path=...' in case user wires it differently
        if a.startswith("ckpt_path=") or a.startswith("trainer.ckpt_path="):
            return True
    return False

def pick_latest_ckpt(search_root: str, prefer_last: bool=True) -> Optional[Path]:
    root = Path(search_root)
    if not root.exists():
        return None

    # Fast path: prefer files called 'last.ckpt'
    if prefer_last:
        last_candidates = list(root.rglob("last.ckpt"))
        if last_candidates:
            # pick newest by mtime
            last_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return last_candidates[0]

    # Otherwise pick newest *.ckpt by mtime
    candidates = list(root.rglob("*.ckpt"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]

def run_once(cmd: List[str], extra_env: dict, print_cmd: bool=False) -> int:
    if print_cmd:
        print("[run_with_resume] Command:", " ".join(shlex.quote(c) for c in cmd), flush=True)
    try:
        proc = subprocess.run(cmd, env={**os.environ, **extra_env})
        return proc.returncode
    except KeyboardInterrupt:
        # Propagate Ctrl-C cleanly
        return 130
    except Exception as e:
        print(f"[run_with_resume] Subprocess failed to start: {e}", flush=True)
        return 127

def main():
    args, child_argv = parse_args()

    # Build base env for child process
    child_env = {}
    if args.hydra_full_error:
        child_env["HYDRA_FULL_ERROR"] = "1"
    if args.offline:
        # safer on flaky networks; you can later `wandb sync` the folder
        child_env["WANDB_MODE"] = "offline"
    # Any extra KEY=VALUE
    for kv in args.env:
        if "=" in kv:
            k, v = kv.split("=", 1)
            child_env[k] = v

    # Normalize command: allow starting with 'python' omitted
    if child_argv and child_argv[0].endswith(".py"):
        child_argv = ["python"] + child_argv

    # First attempt: run as given (don’t force a ckpt override on the very first try)
    attempt = 0
    cmd = list(child_argv)
    rc =1#run_once(cmd, child_env, print_cmd=args.print_cmd)

    while rc not in (0, 130) and attempt < args.max_retries:
        attempt += 1
        print(f"[run_with_resume] Run failed with return code {rc}. Attempt {attempt}/{args.max_retries}.", flush=True)

        ckpt = pick_latest_ckpt(args.search_root, prefer_last=args.prefer_last)
        if ckpt is None:
            print(f"[run_with_resume] No checkpoint found under: {args.search_root}. Will retry without resume.", flush=True)
            resume_cmd = list(child_argv)
        else:
            # Append Hydra override if not already present
            resume_cmd = list(child_argv)
            if not already_has_ckpt_override(resume_cmd):
                resume_cmd += [f"ckpt_path={str(ckpt)}"]
            print(f"[run_with_resume] Resuming from checkpoint: {ckpt}", flush=True)

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

        rc = run_once(resume_cmd, child_env, print_cmd=args.print_cmd)

    if rc == 0:
        print("[run_with_resume] Completed successfully.", flush=True)
        sys.exit(0)
    elif rc == 130:
        print("[run_with_resume] Interrupted by user (SIGINT).", flush=True)
        sys.exit(130)
    else:
        print(f"[run_with_resume] Exhausted retries. Last return code: {rc}", flush=True)
        sys.exit(rc)

if __name__ == "__main__":
    main()