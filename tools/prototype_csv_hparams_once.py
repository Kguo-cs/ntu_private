"""Uninstalled CPU-tested CSV telemetry option for a future isolated process.

No monkey-patch runs on import. Metrics retain Lightning's Fabric writer exactly.
Hparams are serialized on first save and after explicit log_hyperparams updates,
not on every metrics flush. Direct nested hparams mutations require invalidation.
This is not a proven fix for any native crash and is not used by current queues.
"""
from __future__ import annotations
import os
from typing import Any,Dict
from lightning.fabric.loggers.csv_logs import _ExperimentWriter as FabricWriter
from lightning.fabric.loggers.logger import rank_zero_experiment
from lightning.pytorch.loggers import csv_logs as native


class HparamsOnceWriter(native.ExperimentWriter):
    def __init__(self,log_dir: str):
        super().__init__(log_dir)
        self._hparams_dirty=True
        self.hparams_serializations=0

    def log_hparams(self,params: Dict[str,Any]):
        super().log_hparams(params)
        self.invalidate_hparams()

    def invalidate_hparams(self):
        """Call after direct/nested changes that bypass log_hyperparams."""
        self._hparams_dirty=True

    def save(self):
        hparams_file=os.path.join(self.log_dir,self.NAME_HPARAMS_FILE)
        # Keep native filesystem behavior, including recovery if file disappears.
        if self._hparams_dirty or not self._fs.isfile(hparams_file):
            native.save_hparams_to_yaml(hparams_file,self.hparams)
            self._hparams_dirty=False
            self.hparams_serializations+=1
        # Avoid native.ExperimentWriter.save: it serializes YAML unconditionally.
        return FabricWriter.save(self)


class HparamsOnceCSVLogger(native.CSVLogger):
    @property
    @rank_zero_experiment
    def experiment(self):
        if self._experiment is not None:return self._experiment
        self._fs.makedirs(self.root_dir,exist_ok=True)
        self._experiment=HparamsOnceWriter(log_dir=self.log_dir)
        return self._experiment
