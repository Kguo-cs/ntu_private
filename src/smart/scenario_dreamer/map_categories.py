"""Shared Waymo map categories used by Scenario Dreamer and VectorWorld."""
import hashlib
import json
from pathlib import Path
import pickle
import re

KEY_PATTERN = re.compile(r'tfrecord-\d+-of-\d+_\d+')
SCENE_PATTERN = re.compile(r'(?:training|validation|testing)\.(tfrecord-\d+-of-\d+_\d+)_\d+_\d+\.pkl')


def scene_category_key(filename):
    match = SCENE_PATTERN.fullmatch(Path(str(filename)).name)
    if match is None:
        raise ValueError(f'Cannot derive native Waymo map category from cache filename: {filename}')
    return match.group(1)


def load_category_keys(path):
    data = json.loads(Path(path).read_text())
    if data.get('policy') != 'native_vae_train_plus_val_whitelist':
        raise ValueError('Map category index must use the released VAE train+val whitelist policy')
    keys = data.get('compatible_keys')
    if (not isinstance(keys, list) or not keys
            or any(not isinstance(key, str) or KEY_PATTERN.fullmatch(key) is None for key in keys)
            or len(set(keys)) != len(keys)):
        raise ValueError('Map category index must contain unique native TFRecord record keys')
    return frozenset(keys)


def import_category_keys(metadata_root, output):
    """Use the train+val lists loaded by both released autoencoders."""
    root, output = Path(metadata_root), Path(output)
    keys, sources = set(), {}
    for name in ('nocturne_train_filenames.pkl', 'nocturne_val_filenames.pkl'):
        path = root / name
        with path.open('rb') as handle:
            values = pickle.load(handle)
        if not isinstance(values, (list, tuple)) or any(
                not isinstance(value, str) or KEY_PATTERN.fullmatch(value) is None for value in values):
            raise ValueError(f'Invalid native map category whitelist: {name}')
        keys.update(values)
        sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = {'policy': 'native_vae_train_plus_val_whitelist',
               'source_sha256': sources, 'compatible_keys': sorted(keys)}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload, separators=(',', ':')) + '\n')
    temporary.replace(output)
    return output


def apply_category_index(agent, category_keys, *, model_name):
    """Fill missing scene labels without modifying input metadata or filenames."""
    if category_keys is None:
        return agent
    from .preprocessed import read_vectorworld_map_metadata
    ids, valid, sources = read_vectorworld_map_metadata(agent, int(agent["num_graphs"]))
    if bool(valid.all()):
        return agent
    ids, valid = ids.clone(), valid.clone()
    names = agent.get("scenario_dreamer_cache_file")
    if names is None:
        raise ValueError(f"{model_name} map category index needs scenario_dreamer_cache_file")
    names = [names] if isinstance(names, str) else list(names)
    if len(names) != len(ids):
        raise ValueError(f"{model_name} scene filenames must align with map category labels")
    for i in range(len(ids)):
        if not bool(valid[i]):
            ids[i] = int(scene_category_key(names[i]) in category_keys)
            valid[i] = True
            sources[i] = "native_nocturne_filename_index"
    return dict(agent, vectorworld_map_id=ids, vectorworld_map_valid_mask=valid,
                vectorworld_map_source=sources)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", required=True, type=Path,
                        help="Directory containing official nocturne_train/val_filenames.pkl")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[3] /
                        "src/waymo_data/scenario_dreamer/metadata/nocturne_compatible_keys.json")
    args = parser.parse_args()
    print(import_category_keys(args.metadata_root, args.output))


if __name__ == "__main__":
    main()
