"""Pre-save scene basenames once, without loading any scene content."""
import argparse
import json
import os
from pathlib import Path
import pickle
import tempfile


def write_file_list(raw_dir, output=None):
    raw_dir = Path(raw_dir).expanduser()
    output = Path(output).expanduser() if output else raw_dir.with_name(raw_dir.name + "_files.pkl")
    excluded = output.name if output.parent.absolute() == raw_dir.absolute() else None
    with os.scandir(raw_dir) as entries:
        names = sorted(entry.name for entry in entries
                       if entry.name != excluded and entry.name.endswith((".pkl", ".pt"))
                       and entry.is_file())
    if not names:
        raise ValueError(f"No .pkl/.pt scenes in {raw_dir}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            pickle.dump(dict(version=1, source_root=str(raw_dir.resolve()), files=names),
                        handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return dict(file_list=str(output.resolve()), num_files=len(names), bytes=output.stat().st_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True, help="Directory containing scene .pkl/.pt files")
    parser.add_argument("--output", help="Defaults to <raw-dir-parent>/<split>_files.pkl")
    args = parser.parse_args()
    print(json.dumps(write_file_list(args.raw_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
