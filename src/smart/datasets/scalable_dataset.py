# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling <CAT-K> or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import pickle
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
from torch_geometric.data import Dataset

from src.utils import RankedLogger
from random import shuffle
import os
import torch
log = RankedLogger(__name__, rank_zero_only=True)
working_dir = os.getcwd()
import torch

num_gpus = torch.cuda.device_count()
print("Total number of GPUs available:", num_gpus)

from pathlib import Path
from src.smart.metrics.metadata_adapter import attach_sd_metric_metadata
import torch

class MultiDataset(Dataset):
    def __init__(
        self,
        raw_dir: str,
        transform: Callable,
        tfrecord_dir: Optional[str] = None,
        scenario_dreamer_preprocessed: bool = False,
        sample_list: Optional[str] = None,
        scenario_dreamer_latent_cache: Optional[str] = None,
        record_latent_source: bool = False,
        scenario_dreamer_non_partitioned_only: bool = False,
        scenario_dreamer_graph_type_index: Optional[str] = None,
        sample_list_check_exists: bool = True,
    ) -> None:
        root = Path(raw_dir)
        self.scenario_dreamer_preprocessed = scenario_dreamer_preprocessed
        self.latent_cache_dir = Path(scenario_dreamer_latent_cache) if scenario_dreamer_latent_cache else None
        self.record_latent_source = record_latent_source
        self.non_partitioned_only = scenario_dreamer_non_partitioned_only
        self.non_partitioned_selection = None
        if self.non_partitioned_only and not scenario_dreamer_preprocessed:
            raise ValueError("Full-lane filtering requires scenario_dreamer_preprocessed=true")
        # Legacy graph_type_index paths are accepted but no longer read or written.
        if (self.latent_cache_dir is not None or record_latent_source) and not scenario_dreamer_preprocessed:
            raise ValueError("Latent caching requires scenario_dreamer_preprocessed=true")
        if self.latent_cache_dir is not None:
            from src.smart.scenario_dreamer.latent_cache import read_manifest
            self.latent_cache_manifest = read_manifest(self.latent_cache_dir)
        if sample_list is not None:
            with Path(sample_list).open("rb") as handle:
                names = pickle.load(handle)["files"]
            if (not isinstance(names, (list, tuple)) or not names
                    or any(not isinstance(name, str) or name in (".", "..")
                           or os.path.basename(name) != name for name in names)
                    or len(set(names)) != len(names)):
                raise ValueError("sample_list must contain unique cache basenames")
            # Basenames make the pre-saved list portable with the raw data directory.
            # Training defers file existence checks to get(), avoiding per-scene stat calls.
            paths = [os.path.join(os.fspath(root), name) for name in names]
            if sample_list_check_exists:
                missing = [p for p in paths if not Path(p).is_file()]
                if missing:
                    raise FileNotFoundError(f"Missing {len(missing)} selected scenes in {root}: {missing[:3]}")
        else:
            paths = sorted(p for p in root.iterdir() if p.is_file() and p.suffix in (".pt", ".pkl"))
        if self.non_partitioned_only:
            from src.smart.scenario_dreamer.graph_type_index import select_non_partitioned
            paths, self.non_partitioned_selection = select_non_partitioned(
                paths, root, scenario_dreamer_graph_type_index)
            log.info(f"Scenario Dreamer full-lane training selection: {self.non_partitioned_selection}")
        self._raw_paths = [os.fspath(p) for p in paths]
        self._tfrecord_dir = Path(tfrecord_dir) if tfrecord_dir is not None else None
        self._num_samples = len(self._raw_paths)

        log.info("Length of {} dataset is ".format(raw_dir) + str(self._num_samples))
        super(MultiDataset, self).__init__(
            transform=transform, pre_transform=None, pre_filter=None
        )

    @property
    def raw_paths(self) -> List[str]:
        return self._raw_paths

    def len(self) -> int:
        return self._num_samples

    def get(self, idx: int):

        # DataLoader/DistributedSampler owns indexing; never duplicate rows per GPU.

        source_hash = None
        if self.record_latent_source or self.latent_cache_dir is not None:
            import hashlib
            import io
            raw = Path(self.raw_paths[idx]).read_bytes()
            source_hash = hashlib.sha256(raw).hexdigest()
            data = (pickle.loads(raw) if Path(self.raw_paths[idx]).suffix == ".pkl" else
                    torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False))
        elif '.pkl' in self.raw_paths[idx]:
            with open(self.raw_paths[idx], "rb") as handle:
                data = pickle.load(handle)
        else:
            data = torch.load(self.raw_paths[idx], map_location="cpu", weights_only=False)

        if self.scenario_dreamer_preprocessed:
            if self.non_partitioned_only:
                from src.smart.scenario_dreamer.graph_type_index import scene_graph_type
                if scene_graph_type(data, self.raw_paths[idx]) != 0:
                    raise ValueError(f"Full-lane training requires lg_type=0; filename disagrees with scene metadata: {self.raw_paths[idx]}")
            from src.smart.scenario_dreamer.preprocessed import adapt_preprocessed_scene
            result = adapt_preprocessed_scene(data, self.raw_paths[idx])
            if source_hash is not None:
                result["sd_source_sha256"] = source_hash
            if self.latent_cache_dir is not None:
                from src.smart.scenario_dreamer.latent_cache import load_record
                n, l = result["sd_agent"]["num_nodes"], result["sd_lane"]["num_nodes"]
                record = load_record(self.latent_cache_dir, Path(self.raw_paths[idx]).name,
                                     source_hash, self.latent_cache_manifest, n, l)
                order = torch.cat((torch.arange(1, n), torch.zeros(1, dtype=torch.long)))
                for stat in ("mu", "log_var"):
                    result["sd_agent"][f"posterior_{stat}"] = record[f"agent_{stat}"][order]
                    result["sd_lane"][f"posterior_{stat}"] = record[f"lane_{stat}"]
                result["sd_latent_cache_fingerprint"] = record["encoder_fingerprint"]
            return result

        # ============================================================
        # Scenario Dreamer metric metadata
        # ============================================================
        if (
                 "scenario_dreamer" in data
                and "scenario_dreamer_cache_file" in data
        ):
            # IMPORTANT:
            # 只有当模型实际以这个 raw Waymo timestep 作为 initial scene
            # 时，才能这样设置。
            actual_generation_raw_t = int(data["scene_timestep"])
            from src.smart.scenario_dreamer.data import attach_model_map
            attach_model_map(data, data["scenario_dreamer"])

            data = attach_sd_metric_metadata(
                data,
                data,
                generation_scene_timestep=actual_generation_raw_t,
            )

        if self._tfrecord_dir is not None:
            data["tfrecord_path"] = (
                self._tfrecord_dir / (data["scenario_id"] + ".tfrecords")
            ).as_posix()
        return data
