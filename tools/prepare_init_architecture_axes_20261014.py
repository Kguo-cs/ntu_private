"""Stage architecture options without changing an in-flight native experiment."""
from pathlib import Path
import ast
import difflib
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'output/reports/init_diffusion_expanded_20261014/map_architecture'

def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new, 1)

def main():
    files = {}
    name = 'src/smart/diffusion/denoiser.py'
    text = (ROOT/name).read_text()
    text = replace_once(text, '        map2agent_max_num_neighbors: int = 30,\n    ) -> None:',
        '        map2agent_max_num_neighbors: int = 30,\n'
        '        num_freq_bands: Optional[int] = None,\n'
        '        head_dim: Optional[int] = None,\n'
        '        map_feature_dim: int = 128,\n    ) -> None:')
    text = replace_once(text, '        num_freq_bands = hidden_dim//2\n',
        '        num_freq_bands = hidden_dim // 2 if num_freq_bands is None else num_freq_bands\n')
    text = replace_once(text, '        head_dim = hidden_dim//num_heads\n',
        '        if not isinstance(num_heads, Integral) or isinstance(num_heads, bool) or num_heads < 1:\n'
        '            raise ValueError("num_heads must be a positive integer")\n'
        '        if head_dim is None and hidden_dim % num_heads:\n'
        '            raise ValueError("hidden_dim must be divisible by num_heads when head_dim is inferred")\n'
        '        head_dim = hidden_dim // num_heads if head_dim is None else head_dim\n'
        '        for name, value in (("num_freq_bands", num_freq_bands), ("head_dim", head_dim), ("map_feature_dim", map_feature_dim)):\n'
        '            if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:\n'
        '                raise ValueError(f"{name} must be a positive integer")\n'
        '        self.num_freq_bands = int(num_freq_bands)\n'
        '        self.head_dim = int(head_dim)\n'
        '        self.map_feature_dim = int(map_feature_dim)\n')
    text = replace_once(text, '        self.lane_embed = MLPLayer(128, hidden_dim, hidden_dim)',
        '        self.lane_embed = MLPLayer(map_feature_dim, hidden_dim, hidden_dim)')
    files[name] = text

    name = 'src/smart/diffusion/initial_diffusion.py'
    text = (ROOT/name).read_text()
    text = replace_once(text, '        collision_guidance: Optional[Mapping[str, Any]] = None,\n    ) -> None:',
        '        collision_guidance: Optional[Mapping[str, Any]] = None,\n'
        '        denoiser_num_heads: Optional[int] = None,\n'
        '        denoiser_num_freq_bands: Optional[int] = None,\n'
        '        denoiser_head_dim: Optional[int] = None,\n    ) -> None:')
    text = replace_once(text, '        if map_condition_types not in ("road_edges", "lanes_and_edges"):',
        '        if map_condition_types not in ("road_edges", "lanes_and_edges", "lanes_only", "all"):')
    text = replace_once(text, '            raise ValueError("map_condition_types must be road_edges or lanes_and_edges")',
        '            raise ValueError("map_condition_types must be road_edges, lanes_and_edges, lanes_only or all")')
    text = replace_once(text, '        if denoiser_hidden_dim % args.num_heads:\n',
        '        args.num_heads = args.num_heads if denoiser_num_heads is None else denoiser_num_heads\n'
        '        for name, value in (("denoiser_num_heads", args.num_heads),\n'
        '                            ("denoiser_num_freq_bands", denoiser_num_freq_bands),\n'
        '                            ("denoiser_head_dim", denoiser_head_dim)):\n'
        '            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):\n'
        '                raise ValueError(f"{name} must be a positive integer")\n'
        '        args.denoiser_num_freq_bands = denoiser_num_freq_bands\n'
        '        args.denoiser_head_dim = denoiser_head_dim\n'
        '        args.map_feature_dim = hidden_dim\n'
        '        if denoiser_head_dim is None and denoiser_hidden_dim % args.num_heads:\n')
    text = replace_once(text, '            "denoiser_hidden_dim": denoiser_hidden_dim, "denoiser_num_layers": denoiser_num_layers,\n',
        '            "denoiser_hidden_dim": denoiser_hidden_dim, "denoiser_num_layers": denoiser_num_layers,\n'
        '            "denoiser_num_heads": self.G1.model.num_heads,\n'
        '            "denoiser_num_freq_bands": self.G1.model.num_freq_bands,\n'
        '            "denoiser_head_dim": self.G1.model.head_dim,\n'
        '            "map_feature_dim": self.G1.model.map_feature_dim,\n')
    text = replace_once(text, '        saved_options = dict(legacy_options, **state.get("denoiser_options", {}))\n',
        '        saved_options = dict(legacy_options, **state.get("denoiser_options", {}))\n'
        '        saved_hidden = saved_options["denoiser_hidden_dim"]\n'
        '        for key, value in (("denoiser_num_heads", 8), ("denoiser_num_freq_bands", saved_hidden // 2),\n'
        '                           ("denoiser_head_dim", saved_hidden // 8), ("map_feature_dim", 128)):\n'
        '            saved_options.setdefault(key, value)\n')
    files[name] = text

    name = 'src/smart/diffusion/scale_flow.py'
    text = (ROOT/name).read_text()
    text = replace_once(text, '            num_heads=args.num_heads,\n            dropout=args.dropout,\n',
        '            num_heads=args.num_heads,\n'
        '            num_freq_bands=getattr(args, "denoiser_num_freq_bands", None),\n'
        '            head_dim=getattr(args, "denoiser_head_dim", None),\n'
        '            map_feature_dim=getattr(args, "map_feature_dim", 128),\n'
        '            dropout=args.dropout,\n')
    files[name] = text

    name = 'src/smart/modules/map_decoder.py'
    text = (ROOT/name).read_text()
    text = replace_once(text, '    if output_types == "lanes_and_edges":\n',
        '    if output_types == "lanes_only":\n'
        '        return (map_type == 0) | (map_type == 1) | (map_type == 2)\n'
        '    if output_types == "all":\n'
        '        return torch.ones_like(map_type, dtype=torch.bool)\n'
        '    if output_types == "lanes_and_edges":\n')
    text = replace_once(text, '    raise ValueError("init_map_output_types must be road_edges or lanes_and_edges")',
        '    raise ValueError("init_map_output_types must be road_edges, lanes_and_edges, lanes_only or all")')
    text = replace_once(text, '        elif square_crop or output_types == "lanes_and_edges":',
        '        elif square_crop or output_types != "road_edges":')
    files[name] = text

    name = 'configs/model/smart.yaml'
    text = (ROOT/name).read_text()
    text = replace_once(text, '      denoiser_hidden_dim: 256\n',
        '      denoiser_hidden_dim: 256\n'
        '      denoiser_num_heads: null # null preserves the legacy eight heads\n'
        '      denoiser_num_freq_bands: null # null infers hidden_dim // 2\n'
        '      denoiser_head_dim: null # null infers hidden_dim // num_heads\n')
    files[name] = text
    name = 'tests/test_init_diffusion_map_condition.py'
    text = (ROOT/name).read_text()
    text = replace_once(text, '            initial_map_output_mask(torch.arange(10), "all")',
        '            initial_map_output_mask(torch.arange(10), "unsupported")')
    text = replace_once(text, '    def test_invalid_output_policy_fails_early(self):',
        '    def test_new_output_policies_return_the_requested_queries(self):\n'
        '        types = torch.arange(10)\n'
        '        self.assertEqual(torch.where(initial_map_output_mask(types, "lanes_only"))[0].tolist(), [0, 1, 2])\n'
        '        self.assertTrue(initial_map_output_mask(types, "all").all())\n\n'
        '    def test_invalid_output_policy_fails_early(self):')
    files[name] = text
    name = 'tests/test_init_diffusion_sep_map.py'
    text = (ROOT/name).read_text()
    text = replace_once(text, '        self.model.lane_embed = nn.Linear(8, 4, bias=False)\n',
        '        self.model.lane_embed = nn.Linear(8, 4, bias=False)\n'
        '        settings = args[0]\n'
        '        self.component_loss = None\n'
        '        self.feature_type = False\n'
        '        self.model.attribute_source = None\n'
        '        self.model.num_heads = settings.num_heads\n'
        '        self.model.num_freq_bands = settings.hidden_dim // 2\n'
        '        self.model.head_dim = settings.hidden_dim // settings.num_heads\n'
        '        self.model.map_feature_dim = getattr(settings, "map_feature_dim", 128)\n')
    files[name] = text
    patches = []
    manifest = {}
    for name, text in files.items():
        if name.endswith('.py'):
            ast.parse(text)
        target = ART/'prepared_sources'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        manifest[name] = dict(original_sha256=hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),
                              prepared_sha256=hashlib.sha256(text.encode()).hexdigest())
        patches.extend(difflib.unified_diff((ROOT/name).read_text().splitlines(True), text.splitlines(True), fromfile='a/'+name, tofile='b/'+name))
    (ART/'prepared.patch').write_text(''.join(patches))
    (ART/'prepared_sources.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('STAGED_ONLY', list(files))

if __name__ == '__main__':
    main()
