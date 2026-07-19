"""Pre-tokenize Waymo HeteroData scenes and save compact token files.

The TokenProcessor runs once in the main process, normally on one GPU.
Each input scene is converted to:

{
    "tokenized_map":   {... CPU tensors ...},
    "tokenized_agent": {... CPU tensors ...},
}

Do not use Python multiprocessing with one shared CUDA TokenProcessor. CUDA
contexts and model objects are not safely shared by forked worker processes.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from tqdm import tqdm


torch.set_float32_matmul_precision("highest")


DEFAULT_PROJECT_PATHS = (
    "/home/users/ntu/lyuchen/scratch/keguo_projects/sim",
    "/home/ke/code/sim",
    "/home/users/ntu/ke.guo/scratch/sim",
    "/home/ke/code/catk",
    "/home/users/ntu/zhangshu/scratch/sim",
    "/home/users/ntu/shanhelo/scratch/keguo_projects/sim",
    "/mnt/d/code/sim",
    "/home/ke/keguo/sim",
    "/home/guoke/sim",
)


def add_existing_project_paths() -> None:
    """Add only existing project directories to sys.path."""
    for path in DEFAULT_PROJECT_PATHS:
        if os.path.isdir(path) and path not in sys.path:
            sys.path.append(path)


def move_to_cpu(value: Any) -> Any:
    """Recursively detach tensors and move them to CPU."""
    if torch.is_tensor(value):
        return value.detach().cpu()

    if isinstance(value, Mapping):
        return {
            key: move_to_cpu(item)
            for key, item in value.items()
        }

    if isinstance(value, tuple):
        return tuple(move_to_cpu(item) for item in value)

    if isinstance(value, list):
        return [move_to_cpu(item) for item in value]

    return value


def remove_optional_keys(
    mapping: MutableMapping[str, Any],
    *keys: str,
) -> None:
    """Remove intermediate fields when present."""
    for key in keys:
        mapping.pop(key, None)


def to_heterodata(value: Any) -> HeteroData:
    """Normalize a loaded object into HeteroData."""
    if isinstance(value, HeteroData):
        return value

    if isinstance(value, dict):
        return HeteroData(value)

    raise TypeError(
        "Each input file must contain a HeteroData object or compatible dict, "
        f"got {type(value).__name__}."
    )


def load_scene(path: Path) -> HeteroData:
    """Load one scene from disk on CPU."""
    try:
        loaded = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        # Compatibility with PyTorch versions without weights_only.
        loaded = torch.load(path, map_location="cpu")

    return to_heterodata(loaded)


def set_single_scene_batch(data: HeteroData, device: torch.device) -> None:
    """Ensure all agents belong to scene 0."""
    if "agent" not in data.node_types:
        raise KeyError("Input HeteroData does not contain an 'agent' node type.")
    if "type" not in data["agent"]:
        raise KeyError("Input data['agent'] does not contain 'type'.")

    num_agents = int(data["agent"]["type"].shape[0])
    data["agent"]["batch"] = torch.zeros(
        num_agents,
        dtype=torch.long,
        device=device,
    )
    data.num_graphs = 1


def validate_tokenized_output(
    tokenized_map: Mapping[str, Any],
    tokenized_agent: Mapping[str, Any],
) -> None:
    """Check fields needed by subsequent SMART training."""
    for name, mapping in (
        ("tokenized_map", tokenized_map),
        ("tokenized_agent", tokenized_agent),
    ):
        if "type" not in mapping:
            raise KeyError(f"{name} is missing required key 'type'.")

    if "batch" not in tokenized_agent:
        raise KeyError("tokenized_agent is missing required key 'batch'.")


def add_num_nodes(
    tokenized_map: MutableMapping[str, Any],
    tokenized_agent: MutableMapping[str, Any],
) -> None:
    """Store explicit node counts for later HeteroData reconstruction."""
    tokenized_map["num_nodes"] = int(tokenized_map["type"].shape[0])
    tokenized_agent["num_nodes"] = int(
        tokenized_agent["type"].shape[0]
    )


def tokenize_scene(
    input_path: Path,
    token_processor,
    device: torch.device,
    pred_init: bool,
) -> dict[str, Any]:
    """Tokenize one input scene and return CPU-only output."""
    data = load_scene(input_path)
    data = data.to(device)
    set_single_scene_batch(data, device)

    with torch.inference_mode():
        tokenized_map = token_processor.tokenize_map(data)
        tokenized_agent = token_processor.tokenize_agent(
            data,
            tokenized_map,
        )

        if pred_init:
            token_processor.get_init(tokenized_agent)

    # Convert after all tokenizer operations have finished. Moving map tensors
    # to CPU before tokenize_agent may cause device mismatches.
    tokenized_map = move_to_cpu(tokenized_map)
    tokenized_agent = move_to_cpu(tokenized_agent)

    remove_optional_keys(
        tokenized_map,
        "traj_pos_local",
    )
    remove_optional_keys(
        tokenized_agent,
        "num_graphs",
        "traj_pos_local",
        "token_agent_shape",
        "batch",
        "token_traj_all",
        "token_traj",
        "trajectory_token_veh",
        "trajectory_token_ped",
        "trajectory_token_cyc",
        "gt_z_raw",
        'id'
    )

    validate_tokenized_output(
        tokenized_map,
        tokenized_agent,
    )
    add_num_nodes(
        tokenized_map,
        tokenized_agent,
    )

    return {
        "tokenized_map": tokenized_map,
        "tokenized_agent": tokenized_agent,
    }


def atomic_torch_save(value: Any, output_path: Path) -> None:
    """Save through a temporary file to avoid partial/corrupt outputs."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    try:
        torch.save(value, temporary_path)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def output_name(input_path: Path) -> str:
    """Convert any accepted input extension to a .pt filename."""
    return f"{input_path.stem}.pt"


def find_input_files(
    directory: Path,
    extensions: tuple[str, ...],
) -> list[Path]:
    """Return sorted regular files with accepted extensions."""
    normalized = {
        extension.lower()
        if extension.startswith(".")
        else f".{extension.lower()}"
        for extension in extensions
    }

    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in normalized
    )


def build_token_processor(args, device: torch.device):
    """Create one TokenProcessor for the complete preprocessing run."""
    add_existing_project_paths()
    from src.smart.tokens.token_processor import TokenProcessor

    processor = TokenProcessor(
        map_token_file=args.map_token_file,
        agent_token_file=args.agent_token_file,
        map_token_sampling={
            "num_k": args.map_num_k,
            "temp": args.map_temperature,
        },
        agent_token_sampling={
            "num_k": args.agent_num_k,
            "temp": args.agent_temperature,
        },
    )
    processor = processor.to(device)
    processor.eval()
    return processor


def process_directory(args) -> None:
    input_directory = Path(args.input_dir).expanduser().resolve()
    output_directory = Path(args.output_dir).expanduser().resolve()

    if not input_directory.is_dir():
        raise FileNotFoundError(
            f"Input directory does not exist: {input_directory}"
        )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    input_files = find_input_files(
        input_directory,
        tuple(args.extensions),
    )
    if not input_files:
        raise FileNotFoundError(
            f"No matching files found in {input_directory}. "
            f"Accepted extensions: {args.extensions}"
        )

    output_directory.mkdir(parents=True, exist_ok=True)
    token_processor = build_token_processor(args, device)

    failed: list[tuple[Path, Exception]] = []
    processed = 0
    skipped = 0

    progress = tqdm(input_files, desc="Tokenizing scenes")
    for input_path in progress:
        output_path = output_directory / output_name(input_path)

        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            token_data = tokenize_scene(
                input_path=input_path,
                token_processor=token_processor,
                device=device,
                pred_init=args.pred_init,
            )
            atomic_torch_save(token_data, output_path)
            processed += 1
        except Exception as error:
            failed.append((input_path, error))
            if args.fail_fast:
                raise
        finally:
            # Avoid retaining a reference to the full scene between iterations.
            if device.type == "cuda" and args.empty_cuda_cache:
                torch.cuda.empty_cache()

        progress.set_postfix(
            processed=processed,
            skipped=skipped,
            failed=len(failed),
        )

    print(
        f"Finished: processed={processed}, skipped={skipped}, "
        f"failed={len(failed)}"
    )

    if failed:
        print("\nFailed files:")
        for path, error in failed:
            print(f"  {path.name}: {type(error).__name__}: {error}")
        raise RuntimeError(
            f"{len(failed)} input file(s) failed preprocessing."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pre-tokenize SMART/Waymo scene files."
    )

    parser.add_argument(
        "--input-dir",
        default="./waymo_data/full/validation_map2light",
    )
    parser.add_argument(
        "--output-dir",
        default="./waymo_data/full/validation_token",
    )
    parser.add_argument(
        "--map-token-file",
        default="map_traj_token5.pkl",
    )
    parser.add_argument(
        "--agent-token-file",
        default="agent_vocab_555_s2.pkl",
    )

    parser.add_argument("--map-num-k", type=int, default=1)
    parser.add_argument("--map-temperature", type=float, default=1.0)
    parser.add_argument("--agent-num-k", type=int, default=1)
    parser.add_argument("--agent-temperature", type=float, default=1.0)

    parser.add_argument(
        "--device",
        default="cuda",
        help="Examples: cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".pt", ".pth", ".pkl"],
    )
    parser.add_argument(
        "--pred-init",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
    )
    parser.add_argument(
        "--empty-cuda-cache",
        action="store_true",
    )
    return parser


if __name__ == "__main__":
    process_directory(build_parser().parse_args())