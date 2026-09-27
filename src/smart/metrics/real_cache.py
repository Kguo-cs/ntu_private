"""One-time reference preparation and a persistent, read-only SQLite store.

Only load trusted author-provided pickle files and locally built databases.
A completed database is self-contained: subsequent evaluation needs no raw
TFRecord, official repository, or original GT pickle directory (unless exporting
additional fields not retained here). The model still needs its own input data.
"""
from __future__ import annotations
import hashlib
import json
import os
import pickle
import platform
import sqlite3
import tempfile
import zlib
from collections import OrderedDict
from pathlib import Path

import numpy as np
import scipy
import networkx as nx
import torch

if __package__:
    from .official_backend import load_official_backend
    from . import metric_core
    from .metric_core import DistributionAccumulator, SPECS, as_numpy, validate_unified
else:
    from official_backend import load_official_backend
    import metric_core
    from metric_core import DistributionAccumulator, SPECS, as_numpy, validate_unified

SCHEMA_VERSION = 1
DEFAULT_CACHE_NAME = '.sd_real_metric_cache.sqlite'


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def backend_signature():
    backend = load_official_backend()
    return {
        'source_sha256': backend.source_sha256,
        'metric_core_sha256': hashlib.sha256(Path(metric_core.__file__).read_bytes()).hexdigest(),
        'specs': [list(s) for s in SPECS],
        'python': '.'.join(platform.python_version().split('.')[:2]),
        'numpy': np.__version__, 'scipy': scipy.__version__,
        'networkx': nx.__version__, 'torch': str(torch.__version__),
    }


def _load_eval(eval_set):
    path = Path(eval_set).expanduser().resolve()
    payload = path.read_bytes()
    obj = pickle.loads(payload)
    if not isinstance(obj, dict) or 'files' not in obj:
        raise ValueError('eval_set must contain a dict with key files')
    filenames = []
    for name in obj['files']:
        if not isinstance(name, (str, Path)):
            raise TypeError('eval_set files must be filenames')
        filenames.append(Path(str(name)).name)
    if not filenames or len(set(filenames)) != len(filenames):
        raise ValueError('eval_set must have nonempty, unique cache basenames')
    return filenames, hashlib.sha256(payload).hexdigest()


def accumulator_state(acc):
    return {'histograms': [h.tolist() for h in acc.histograms],
            'totals': acc.totals.tolist(), 'num_scenes': int(acc.num_scenes),
            'num_vehicles': int(acc.num_vehicles), 'num_colliding': int(acc.num_colliding)}


def add_statistics(acc, stats):
    """Add sufficient statistics, NOT normalized histogram probabilities."""
    histograms = stats['histograms']
    totals = np.asarray(stats['totals'], dtype=np.int64)
    if len(histograms) != len(SPECS) or totals.shape != (len(SPECS),):
        raise ValueError('Malformed cached histogram structure')
    for i, (dst, values) in enumerate(zip(acc.histograms, histograms)):
        src = np.asarray(values, dtype=np.int64)
        if src.shape != dst.shape or (src < 0).any() or int(src.sum()) != int(totals[i]):
            raise ValueError(f'Cached histogram {i} is inconsistent with sample counts')
        dst += src
    acc.totals += totals
    acc.num_scenes += int(stats['num_scenes'])
    acc.num_vehicles += int(stats['num_vehicles'])
    acc.num_colliding += int(stats['num_colliding'])


def _cache_scalar(value, name):
    a = as_numpy(value)
    if a.size != 1 or not np.isfinite(a).all() or int(a.item()) != a.item():
        raise ValueError(f'{name} must be an integer scalar')
    return int(a.item())


def _prepare_one(source_path, official, *, require_explicit_edges=True):
    payload = source_path.read_bytes()
    raw = pickle.loads(payload)
    # Tensor->NumPy preserves dtype; no world-coordinate float32 conversion.
    raw = {k: as_numpy(v) if torch.is_tensor(v) else v for k,v in raw.items()}
    if _cache_scalar(raw['lg_type'], 'lg_type') != 0:
        raise ValueError('Expected lg_type=0 for initial-scene reference')
    timestep = _cache_scalar(raw['scene_timestep'], 'scene_timestep')
    if require_explicit_edges and 'edge_index_lane_to_lane' not in raw:
        raise ValueError('Official GT cache is missing edge_index_lane_to_lane; '
                         'do not guess road_connection_types ordering. Use the original official cache.')
    unified = official.convert_data_to_unified_format(raw, dataset_name='waymo_gt')
    validate_unified(unified)
    # Prepare this geometry ONCE for both GT and future generated agents.
    unified['metric_lanes'] = official.resample_lanes(unified['lanes'], num_points=100)
    stats = DistributionAccumulator()
    stats.update(unified, official, collision=False)
    # Keep the exact original map representation for optional generated exports.
    keep_keys = ('lg_type','num_lanes','road_points','road_connection_types','edge_index_lane_to_lane')
    cache = {k: raw[k] for k in keep_keys if k in raw}
    cache['scene_timestep'] = timestep
    stat = source_path.stat()
    return {'cache': cache, 'gt': unified, 'statistics': accumulator_state(stats)}, {
        'size': len(payload), 'mtime_ns': stat.st_mtime_ns,
        'sha256': hashlib.sha256(payload).hexdigest(),
    }


def prepare_real_cache(cache_root, eval_set, output, *, expected_scenes=50_000,
                       overwrite=False, require_explicit_edges=True, progress=True):
    """Build a complete DB atomically; errors never silently skip GT entries."""
    root = Path(cache_root).expanduser().resolve()
    dest = Path(output).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    files, eval_hash = _load_eval(eval_set)
    if expected_scenes and len(files) != expected_scenes:
        raise ValueError(f'Expected {expected_scenes} references, got {len(files)}')
    if dest.exists() and not overwrite:
        raise FileExistsError(f'{dest} already exists. Use --overwrite only to intentionally rebuild.')
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, staging = tempfile.mkstemp(prefix=dest.name+'.building-', dir=dest.parent)
    os.close(fd)
    connection = None
    try:
        connection = sqlite3.connect(staging)
        connection.execute('PRAGMA journal_mode=DELETE')
        connection.execute('CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
        connection.execute('''CREATE TABLE scenes (
            name TEXT PRIMARY KEY, ordinal INTEGER UNIQUE NOT NULL,
            payload BLOB NOT NULL, payload_sha256 TEXT NOT NULL,
            source_size INTEGER NOT NULL, source_mtime_ns INTEGER NOT NULL,
            source_sha256 TEXT NOT NULL)''')
        official = load_official_backend()
        total = DistributionAccumulator()
        manifest_hash = hashlib.sha256()
        for i, name in enumerate(files):
            path = root / name
            try:
                entry, source = _prepare_one(path, official, require_explicit_edges=require_explicit_edges)
            except Exception as exc:
                raise RuntimeError(f'Reference preparation failed at {i}: {name}: {exc}') from exc
            add_statistics(total, entry['statistics'])
            blob = zlib.compress(pickle.dumps(entry, protocol=pickle.HIGHEST_PROTOCOL), level=1)
            connection.execute('INSERT INTO scenes VALUES (?,?,?,?,?,?,?)',
                (name, i, sqlite3.Binary(blob), hashlib.sha256(blob).hexdigest(),
                 source['size'], source['mtime_ns'], source['sha256']))
            manifest_hash.update(_json([name,source['sha256']]).encode())
            manifest_hash.update(b'\n')
            if (i+1) % 100 == 0:
                connection.commit()
            if progress and ((i+1) % 100 == 0 or i+1 == len(files)):
                print(f'Prepared real scenes: {i+1}/{len(files)}', flush=True)
        metadata = {
            'schema_version': SCHEMA_VERSION, 'completed': True,
            'backend_signature': backend_signature(), 'eval_set_sha256': eval_hash,
            'files': files, 'num_scenes': len(files), 'reference_statistics': accumulator_state(total),
            'source_cache_root': str(root), 'source_manifest_sha256': manifest_hash.hexdigest(),
            'require_explicit_edges': require_explicit_edges,
        }
        connection.executemany('INSERT INTO metadata VALUES (?,?)',
            [(k,_json(v)) for k,v in metadata.items()])
        connection.commit()
        connection.close(); connection = None
        if dest.exists() and not overwrite:
            raise FileExistsError(f'{dest} was created by another process; refusing to replace it')
        os.replace(staging, dest)
        return metadata
    finally:
        if connection is not None:
            connection.close()
        for leftover in (staging, staging+'-journal'):
            if os.path.exists(leftover):
                os.unlink(leftover)


class CachedReferenceStore:
    """Read prepared statistics/maps without recalculating any GT features."""
    def __init__(self, database, *, eval_set=None, expected_scenes=50_000, lru_size=32):
        self.database = Path(database).expanduser().resolve()
        if not self.database.is_file():
            raise FileNotFoundError(f'Real-statistics cache missing: {self.database}. '
                                    'Run precompute_real.py once before validation.')
        self.connection = sqlite3.connect(self.database.as_uri()+'?mode=ro', uri=True)
        metadata = {k: json.loads(v) for k,v in self.connection.execute('SELECT key,value FROM metadata')}
        if metadata.get('schema_version') != SCHEMA_VERSION or not metadata.get('completed'):
            self.close(); raise ValueError('Incomplete or incompatible real-statistics cache')
        self.metadata = metadata
        self.files = metadata['files']
        self.index = {name:i for i,name in enumerate(self.files)}
        if len(self.files) != len(self.index):
            self.close(); raise ValueError('Duplicate reference filenames in database')
        if expected_scenes and len(self.files) != expected_scenes:
            self.close(); raise ValueError(f'Expected {expected_scenes} references, got {len(self.files)}')
        if self.connection.execute('SELECT COUNT(*) FROM scenes').fetchone()[0] != len(self.files):
            self.close(); raise ValueError('Reference database is missing rows')
        if eval_set is not None:
            files, checksum = _load_eval(eval_set)
            if files != self.files or checksum != metadata['eval_set_sha256']:
                self.close(); raise ValueError('eval_set does not match this real-statistics cache')
        self.eval_set = Path(eval_set) if eval_set is not None else None
        self.expected_scenes = int(expected_scenes)
        self.official = load_official_backend()
        self.cache = OrderedDict()
        self.lru_size = max(0, int(lru_size))
        self.full_real = DistributionAccumulator()
        add_statistics(self.full_real, metadata['reference_statistics'])

    def close(self):
        connection = getattr(self, 'connection', None)
        if connection is not None:
            connection.close()
            self.connection = None

    def __del__(self):
        self.close()

    def entry(self, filename):
        name = Path(str(filename)).name
        if name not in self.index:
            raise ValueError(f'{name} is not in the prepared evaluation list')
        if name in self.cache:
            self.cache.move_to_end(name)
            return self.cache[name]
        row = self.connection.execute('SELECT payload,payload_sha256 FROM scenes WHERE name=?', (name,)).fetchone()
        if row is None or hashlib.sha256(row[0]).hexdigest() != row[1]:
            raise ValueError(f'Missing or corrupt prepared scene: {name}')
        entry = pickle.loads(zlib.decompress(row[0]))
        if self.lru_size:
            self.cache[name] = entry
            if len(self.cache) > self.lru_size:
                self.cache.popitem(last=False)
        return entry

    def get(self, filename):
        entry = self.entry(filename)
        return entry['cache'], entry['gt']

    def add_real(self, accumulator, filename):
        add_statistics(accumulator, self.entry(filename)['statistics'])

    def verify_sources(self, source_root=None, *, mode='sha256'):
        """Optional audit, not run every epoch. Detect replaced original GT files."""
        if mode not in ('stat','sha256'):
            raise ValueError('mode must be stat or sha256')
        root = Path(source_root or self.metadata['source_cache_root']).expanduser().resolve()
        for name, size, mtime, checksum in self.connection.execute(
                'SELECT name,source_size,source_mtime_ns,source_sha256 FROM scenes ORDER BY ordinal'):
            path = root/name
            if not path.is_file():
                raise FileNotFoundError(path)
            stat = path.stat()
            if mode == 'stat':
                valid = stat.st_size == size and stat.st_mtime_ns == mtime
            else:
                valid = stat.st_size == size and hashlib.sha256(path.read_bytes()).hexdigest() == checksum
            if not valid:
                raise ValueError(f'Original reference file differs from prepared snapshot: {name}')
        return len(self.files)
