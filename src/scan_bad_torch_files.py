import argparse
import os
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm


LOAD_CODE = r"""
import sys
import torch
path = sys.argv[1]
obj = torch.load(path, map_location="cpu", weights_only=False)
print("OK", path)
"""


def check_one(path: Path, timeout: int):
    cmd = [sys.executable, "-c", LOAD_CODE, str(path)]
    env = os.environ.copy()
    env["PYTHONFAULTHANDLER"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"

    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        env=env,
    )

    return p.returncode, p.stdout, p.stderr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="dataset root or processed dir")
    parser.add_argument("--suffix", default=".pt,.pth", help="comma-separated suffixes")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--out", default="bad_torch_files.txt")
    args = parser.parse_args()

    root = Path(args.root)
    suffixes = tuple(s.strip() for s in args.suffix.split(","))

    files = []
    for s in suffixes:
        files.extend(root.rglob(f"*{s}"))

    files = sorted(set(files))
    print(f"[INFO] root={root}")
    print(f"[INFO] num files={len(files)}")
    print(f"[INFO] out={args.out}")

    bad = []

    with open(args.out, "w") as fout:
        for path in tqdm(files):
            try:
                code, stdout, stderr = check_one(path, args.timeout)
            except subprocess.TimeoutExpired:
                msg = f"[TIMEOUT] {path}"
                print(msg)
                fout.write(msg + "\n")
                fout.flush()
                bad.append(path)
                continue

            if code != 0:
                msg = f"[BAD] returncode={code} path={path}"
                print(msg)
                fout.write(msg + "\n")
                fout.write("----- stdout -----\n")
                fout.write(stdout[-4000:] + "\n")
                fout.write("----- stderr -----\n")
                fout.write(stderr[-4000:] + "\n")
                fout.write("\n")
                fout.flush()
                bad.append(path)

    print(f"[DONE] bad files={len(bad)}")
    for p in bad[:50]:
        print(p)


if __name__ == "__main__":
    main()