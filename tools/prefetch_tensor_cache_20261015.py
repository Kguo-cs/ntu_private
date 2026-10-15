"""Batch only NumPy/plain cache payloads across the existing CPU worker pipe."""
from pathlib import Path
import hashlib
import json
import pickle


def read_batch(paths):
    from tools.tensor_cache_numpy_worker_20261015 import read
    return [read(path) for path in paths]


class PendingCachedRead:
    def __init__(self, owner, path):
        self.owner, self.path = owner, path
    def get(self, timeout=None):
        return self.owner.get(self.path, timeout)


class PrefetchPool:
    def __init__(self, pool, manifest, policy, batch_files=64):
        self.pool, self.manifest, self.policy = pool, Path(manifest), Path(policy)
        self.batch_files = batch_files
        self.files, self.index, self.cache = None, None, {}
        self.sequence_digest = hashlib.sha256()
        self.records = dict(batch_files=batch_files, requested_files=0, batch_rpcs=0,
            worker_files_read=0, cache_hits=0, single_file_fallbacks=0,
            no_Torch_tensor_IPC=True, input_model_order_unchanged=True)
    def apply_async(self, function, args):
        if function.__name__ != 'read' or len(args) != 1:
            raise ValueError('Prefetch proxy accepts only the native TRAIN read(path)')
        return PendingCachedRead(self, str(args[0]))
    def get(self, path, timeout):
        name = Path(path).name
        if self.files is None and self.manifest.is_file():
            payload = self.manifest.read_bytes()
            self.files = pickle.loads(payload)['files']
            assert len(set(self.files)) == len(self.files)
            self.index = {Path(n).name:i for i,n in enumerate(self.files)}
            assert len(self.index) == len(self.files)
            self.records['train_manifest_sha256'] = hashlib.sha256(payload).hexdigest()
        if path in self.cache:
            self.records['cache_hits'] += 1
            value = self.cache.pop(path)
        else:
            if self.index is not None and name in self.index:
                offset = self.index[name]
                paths = [str(Path(path).parent/Path(n).name)
                         for n in self.files[offset:offset+self.batch_files]]
            else:
                paths = [path]
                self.records['single_file_fallbacks'] += 1
            values = self.pool.apply_async(read_batch,(paths,)).get(timeout=timeout)
            assert len(values) == len(paths)
            self.cache = dict(zip(paths,values))
            value = self.cache.pop(path)
            self.records['batch_rpcs'] += 1
            self.records['worker_files_read'] += len(paths)
        self.records['requested_files'] += 1
        self.sequence_digest.update((path+'\n').encode())
        if self.records['requested_files']==1 or self.records['requested_files']%4096==0:
            self.save()
        return value
    def save(self):
        self.policy.parent.mkdir(parents=True,exist_ok=True)
        self.policy.write_text(json.dumps(dict(self.records,
            requested_path_sequence_sha256=self.sequence_digest.hexdigest()),indent=2)+'\n')
    def terminate(self):
        self.save()
        self.pool.terminate()
    def join(self):
        self.pool.join()

