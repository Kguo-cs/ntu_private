"""Read-only, credential-safe W&B snapshot for the 2026-10-13 cc analysis."""
import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
from urllib.parse import urlsplit

os.environ.setdefault("WANDB_CACHE_DIR", "/tmp/sim_wandb_cc_readonly_cache")
os.environ.setdefault("WANDB_SILENT", "true")
os.environ.setdefault("WANDB_CONSOLE", "off")
import wandb


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def clean(value):
    if isinstance(value, dict):
        return {str(k): ("[REDACTED]" if any(s in str(k).lower() for s in
                ("password", "api_key", "apikey", "access_token", "auth_token", "secret"))
                else clean(v)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n")


def inventory(root):
    api = wandb.Api(timeout=45)
    host = urlsplit(str(api.settings.get("base_url", ""))).hostname
    rows = []
    for run in api.runs("u3006612/bc", order="-created_at", per_page=200):
        rows.append({"id": run.id, "name": run.name, "state": run.state,
                     "created_at": run.created_at, "updated_at": getattr(run, "updated_at", None),
                     "config": clean(dict(run.config)), "summary": clean(dict(run.summary)),
                     "tags": list(run.tags), "group": run.group, "job_type": run.job_type,
                     "commit": run.commit, "url": run.url})
    value = {"fetched_at_utc": utc(), "project": "u3006612/bc", "api_host": host,
             "run_count": len(rows), "read_only": True, "runs": rows}
    write(root / "inventory.json", value)
    print(json.dumps({"run_count": len(rows), "api_host": host,
                      "fetched_at_utc": value["fetched_at_utc"]}))
    for run in rows[:30]:
        print(run["id"], run["name"], run["state"], run["commit"])


def histories(root, ids):
    records = {r["id"]: r for r in json.loads((root / "inventory.json").read_text())["runs"]}
    if not ids:
        raise ValueError("Explicit run IDs are required")
    def fetch(rid):
        record = records[rid]
        api = wandb.Api(timeout=45)
        run = api.run("u3006612/bc/" + rid)
        ceiling = int(record["summary"].get("_step", 0)) + 1
        rows = list(run.scan_history(page_size=1000, max_step=ceiling))
        payload = {"run_id": rid, "name": record["name"], "fetched_at_utc": utc(),
                   "method": "unsampled scan_history(keys=None)",
                   "snapshot_max_step_exclusive": ceiling, "row_count": len(rows), "rows": rows}
        write(root / "histories" / (rid + ".json"), payload)
        # Metadata is read separately; only non-secret execution fields are retained.
        meta = {"run_id": rid, "read_only": True, "fetched_at_utc": utc()}
        try:
            f = run.file("wandb-metadata.json")
            downloaded = f.download(root=str(root / "metadata" / rid), replace=False)
            raw_path = Path(downloaded.name)
            source = json.loads(raw_path.read_text())
            meta["selected"] = {k: source.get(k) for k in [
                "os", "python", "program", "args", "git", "cuda", "gpu", "gpu_count", "gpu_devices", "cpu_count", "cpu_count_logical"] if k in source}
            meta["remote_metadata_sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
            # Do not persist full remote metadata (it may contain environment information).
            raw_path.unlink()
        except Exception as exc:
            meta["metadata_error_type"] = type(exc).__name__
        write(root / "metadata" / rid / "selected.json", meta)
        return {"run_id": rid, "status": "downloaded", "rows": len(rows)}
    errors = []
    complete = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fetch, rid): rid for rid in ids}
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                complete.append(result)
                print(json.dumps(result), flush=True)
            except Exception as exc:
                result = {"run_id": futures[future], "error_type": type(exc).__name__}
                errors.append(result)
                print(json.dumps(result), flush=True)
    write(root / "history_fetch_status.json", {"requested": ids, "completed": complete,
          "errors": errors, "completed_at_utc": utc(), "read_only": True})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["inventory", "histories"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("ids", nargs="*")
    args = parser.parse_intermixed_args()
    try:
        inventory(args.output) if args.mode == "inventory" else histories(args.output, args.ids)
    except Exception as exc:
        write(args.output / (args.mode + "_access_error.json"), {
            "fetched_at_utc": utc(), "error_type": type(exc).__name__, "read_only": True})
        print("READ_ONLY_ACCESS_ERROR", type(exc).__name__)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
