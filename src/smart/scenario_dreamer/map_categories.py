"""Waymo Nocturne labels with raw split provenance shared by both initializers.

The default policy uses training metadata only for training records and both
validation whitelists for validation records. Testing records are outside the
Nocturne whitelist. The released splitless cache rule remains explicitly
available for reproducing the original cache writer, including key collisions.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickle
import re
import warnings

SPLIT_POLICY = 'split_aware_nocturne_whitelist'
NATIVE_POLICY = 'native_vae_train_plus_val_whitelist'
KEY_PATTERN = re.compile(r'tfrecord-\d+-of-\d+_\d+')
SCENE_PATTERN = re.compile(r'(?P<split>training|validation|testing)\.(?P<key>tfrecord-\d+-of-\d+_\d+)_\d+_\d+\.pkl')
SOURCE_RAW_SPLITS = {
    'nocturne_train_filenames.pkl': 'training',
    'nocturne_val_filenames.pkl': 'validation',
    # These are heldout original validation scenarios moved into the test cache.
    'nocturne_test_filenames.pkl': 'validation',
}


@dataclass(frozen=True)
class MapCategoryIndex:
    policy: str
    training_keys: frozenset[str] = frozenset()
    validation_keys: frozenset[str] = frozenset()
    native_keys: frozenset[str] = frozenset()


def _scene_parts(filename):
    match = SCENE_PATTERN.fullmatch(Path(str(filename)).name)
    if match is None:
        raise ValueError(f'Cannot derive native Waymo map category from cache filename: {filename}')
    return match.group('split'), match.group('key')


def scene_category_key(filename):
    """Return the shard/record key; classification also preserves the raw split."""
    return _scene_parts(filename)[1]


def classify_category(filename, category_index):
    """Map a cache name to its label under the selected explicit policy."""
    if not isinstance(category_index, MapCategoryIndex):
        raise TypeError('Map categories require a loaded MapCategoryIndex')
    split, key = _scene_parts(filename)
    if category_index.policy == NATIVE_POLICY:
        return int(key in category_index.native_keys)
    if category_index.policy != SPLIT_POLICY:
        raise ValueError(f'Unknown map category policy: {category_index.policy}')
    if split == 'training':
        return int(key in category_index.training_keys)
    if split == 'validation':
        return int(key in category_index.validation_keys)
    return 0


def _validated_keys(values, name):
    if (not isinstance(values, (list, tuple))
            or any(not isinstance(key, str) or KEY_PATTERN.fullmatch(key) is None for key in values)
            or len(set(values)) != len(values)):
        raise ValueError(f'Map category index must contain unique TFRecord record keys: {name}')
    return frozenset(values)


def load_category_keys(path):
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError('Map category index must be a JSON object')
    policy = data.get('policy')
    if policy == NATIVE_POLICY:
        keys = _validated_keys(data.get('compatible_keys'), 'native')
        warnings.warn('Loading legacy splitless train+val map categories; this reproduces '
                      'the released cache rule, not the corrected split-aware labels. '
                      'Re-import metadata to use split-aware labels.', RuntimeWarning, stacklevel=2)
        return MapCategoryIndex(policy, native_keys=keys)
    if policy != SPLIT_POLICY:
        raise ValueError(f'Unknown map category index policy: {policy}')
    if type(data.get('schema_version')) is not int or data['schema_version'] != 2:
        raise ValueError('Split-aware map category index requires schema_version=2')
    if data.get('source_raw_splits') != SOURCE_RAW_SPLITS:
        raise ValueError('Split-aware category provenance must identify train and both validation whitelists')
    hashes = data.get('source_sha256')
    if (not isinstance(hashes, dict) or set(hashes) != set(SOURCE_RAW_SPLITS)
            or any(not isinstance(value, str) or re.fullmatch(r'[0-9a-f]{64}', value) is None
                   for value in hashes.values())):
        raise ValueError('Split-aware category provenance requires SHA256 hashes for all three whitelists')
    by_split = data.get('compatible_keys_by_raw_split')
    if not isinstance(by_split, dict) or set(by_split) != {'training', 'validation', 'testing'}:
        raise ValueError('Split-aware category index requires training/validation/testing key lists')
    train = _validated_keys(by_split['training'], 'training')
    val = _validated_keys(by_split['validation'], 'validation')
    test = _validated_keys(by_split['testing'], 'testing')
    if test:
        raise ValueError('Original testing records must not be included in the Nocturne whitelist')
    return MapCategoryIndex(policy, training_keys=train, validation_keys=val)


def import_category_keys(metadata_root, output, *, policy=SPLIT_POLICY):
    """Import portable labels; native splitless behavior requires explicit policy."""
    if policy not in (SPLIT_POLICY, NATIVE_POLICY):
        raise ValueError(f'Unknown map category policy: {policy}')
    root, output = Path(metadata_root), Path(output)
    keys, sources = {}, {}
    filenames = SOURCE_RAW_SPLITS if policy == SPLIT_POLICY else tuple(SOURCE_RAW_SPLITS)[:2]
    for name in filenames:
        path = root / name
        with path.open('rb') as handle:
            values = pickle.load(handle)
        keys[name] = _validated_keys(values, name)
        sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if policy == SPLIT_POLICY:
        payload = {'schema_version': 2, 'policy': policy,
                   'source_sha256': sources, 'source_raw_splits': SOURCE_RAW_SPLITS,
                   'compatible_keys_by_raw_split': {
                       'training': sorted(keys['nocturne_train_filenames.pkl']),
                       'validation': sorted(keys['nocturne_val_filenames.pkl'] | keys['nocturne_test_filenames.pkl']),
                       'testing': [],
                   }}
    else:
        payload = {'policy': policy, 'source_sha256': sources,
                   'compatible_keys': sorted(set().union(*keys.values()))}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload, separators=(',', ':')) + '\n')
    temporary.replace(output)
    return output


def apply_category_index(agent, category_keys, *, model_name):
    """Fill missing scene labels and preserve explicit input metadata precedence."""
    if category_keys is None:
        return agent
    from .preprocessed import read_vectorworld_map_metadata
    ids, valid, sources = read_vectorworld_map_metadata(agent, int(agent['num_graphs']))
    if bool(valid.all()):
        return agent
    ids, valid = ids.clone(), valid.clone()
    names = agent.get('scenario_dreamer_cache_file')
    if names is None:
        raise ValueError(f'{model_name} map category index needs scenario_dreamer_cache_file')
    names = [names] if isinstance(names, str) else list(names)
    if len(names) != len(ids):
        raise ValueError(f'{model_name} scene filenames must align with map category labels')
    source = ('split_aware_nocturne_filename_index' if category_keys.policy == SPLIT_POLICY
              else 'native_nocturne_filename_index')
    for i in range(len(ids)):
        if not bool(valid[i]):
            ids[i] = classify_category(names[i], category_keys)
            valid[i] = True
            sources[i] = source
    return dict(agent, vectorworld_map_id=ids, vectorworld_map_valid_mask=valid,
                vectorworld_map_source=sources)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata-root', required=True, type=Path,
                        help='Directory containing official nocturne_train/val/test_filenames.pkl')
    parser.add_argument('--policy', choices=(SPLIT_POLICY, NATIVE_POLICY), default=SPLIT_POLICY,
                        help='Use legacy native policy only for explicit cache-source reproduction')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[3] /
                        'src/waymo_data/scenario_dreamer/metadata/nocturne_compatible_keys.json')
    args = parser.parse_args()
    print(import_category_keys(args.metadata_root, args.output, policy=args.policy))


if __name__ == '__main__':
    main()
