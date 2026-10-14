"""CPU real-data parity and architecture checks for staged, unapplied sources."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import argparse
import copy
import hashlib
import importlib
import importlib.util
import json
import pickle
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ART = ROOT/'output/reports/init_diffusion_expanded_20261014/map_architecture'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def overlay():
    for name in ('src.smart.modules.map_decoder', 'src.smart.diffusion.denoiser',
                 'src.smart.diffusion.scale_flow', 'src.smart.diffusion.initial_diffusion'):
        parent, child = name.rsplit('.', 1)
        package = importlib.import_module(parent)
        path = ART/'prepared_sources'/(name.replace('.', '/')+'.py')
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        setattr(package, child, module)
        spec.loader.exec_module(module)

def main(mode):
    if mode == 'prepared':
        overlay()
    import torch
    from tools.runtime_cached_torch_loader_sweep import install_cached_loader
    install_cached_loader(torch)
    import hydra
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict
    from torch_geometric.loader import DataLoader
    from src.smart.datasets import MultiDataset
    from src.smart.datamodules.target_builder import WaymoTargetBuilderTrain
    from src.smart.modules.map_decoder import initial_map_output_mask
    torch.set_num_threads(2)
    OmegaConf.register_new_resolver('sim_root', lambda: str(ROOT), replace=True)
    with initialize_config_dir(config_dir=str(ROOT/'configs'), version_base=None):
        cfg = compose(config_name='run', overrides=['experiment=init_diffusion_657_n20_road_k8_full','optimizer=default'])
    with open_dict(cfg):
        cfg.paths.root_dir=str(ROOT/'src')
        cfg.paths.output_dir=str(ART/'cpu_preflight')
        cfg.paths.log_dir=str(ART/'cpu_preflight')
    with (ROOT/'logs/component_loss_scratch/calibration/calibration_files.pkl').open('rb') as f:
        files = pickle.load(f)['files'][:2]
    manifest = ART/'cpu_train2.pkl'
    with manifest.open('wb') as f:
        pickle.dump({'files': files}, f)
    dataset = MultiDataset(cfg.data.train_raw_dir, WaymoTargetBuilderTrain(32),
                           sample_list=str(manifest), sample_list_check_exists=False)
    batch = next(iter(DataLoader(dataset, batch_size=2, num_workers=0, shuffle=False)))
    def instantiate(c):
        torch.manual_seed(4817)
        return hydra.utils.instantiate(c.model, _recursive_=False).cpu().train()
    def digest(module):
        h=hashlib.sha256()
        for name,t in sorted(module.state_dict().items()):
            if torch.is_tensor(t):
                h.update(name.encode());h.update(t.detach().contiguous().numpy().tobytes())
        return h.hexdigest()
    def run(model):
        torch.manual_seed(26817)
        maps,agents=model.token_processor(copy.deepcopy(batch))
        losses=model.encoder(maps,agents)['initial_logit']
        loss=losses[0]+losses[1]
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        return torch.stack([x.detach() for x in losses]),torch.get_rng_state().clone()
    baseline=instantiate(cfg)
    initial_hash=digest(baseline.encoder.init_decoder.G1)
    losses,rng=run(baseline)
    result=dict(mode=mode,baseline_generator_tensor_sha256=initial_hash,baseline_losses=losses.tolist(),
                initial_map_checkpoint_sha256=baseline.initial_map_checkpoint_metadata['file_sha256'],
                train_manifest_sha256=sha(manifest),script_sha256=sha(__file__))
    if mode=='legacy':
        torch.save(dict(losses=losses,rng=rng,decoder=baseline.encoder.init_decoder.state_dict()),ART/'legacy_baseline.pt')
        (ART/'legacy_baseline.json').write_text(json.dumps(result,indent=2)+'\n')
        print('LEGACY_REFERENCE_SAVED',flush=True)
        return
    old=torch.load(ART/'legacy_baseline.pt',map_location='cpu',weights_only=False)
    original=json.loads((ART/'legacy_baseline.json').read_text())
    assert initial_hash==original['baseline_generator_tensor_sha256']
    assert torch.equal(losses,old['losses'])
    assert torch.equal(rng,old['rng'])
    baseline.encoder.init_decoder.load_state_dict(old['decoder'],strict=True)
    assert not baseline.encoder.map_encoder.training
    assert all(p.grad is None for p in baseline.encoder.map_encoder.parameters())
    result['legacy_tensor_output_rng_and_checkpoint_parity']=True
    tests=[('freq32',{'denoiser_num_freq_bands':32},{}),
           ('heads4dim32',{'denoiser_num_heads':4,'denoiser_head_dim':32},{}),
           ('lanes_only',{'map_condition_types':'lanes_only'},{}),
           ('all_map_types',{'map_condition_types':'all'},{}),
           ('train_map64_layers2',{'denoiser_hidden_dim':192,'denoiser_num_heads':6,
                                   'denoiser_head_dim':24,'denoiser_num_freq_bands':16},
                                  {'hidden_dim':64,'num_freq_bands':16,'num_heads':4,'head_dim':16,'num_map_layers':2,'sep_map':True})]
    checked=[]
    for name,options,outer in tests:
        candidate=copy.deepcopy(cfg)
        for key,value in options.items():
            OmegaConf.update(candidate,'model.model_config.decoder.init_diffusion.'+key,value,force_add=True)
        for key,value in outer.items():
            OmegaConf.update(candidate,'model.model_config.decoder.'+key,value,force_add=True)
        if outer:
            candidate.model.model_config.initial_map_checkpoint=None
        model=instantiate(candidate)
        values,_=run(model)
        decoder=model.encoder.init_decoder
        denoiser=decoder.G1.model
        assert denoiser.num_heads==options.get('denoiser_num_heads',8)
        assert denoiser.num_freq_bands==options.get('denoiser_num_freq_bands',denoiser.hidden_dim//2)
        assert denoiser.head_dim==options.get('denoiser_head_dim',denoiser.hidden_dim//denoiser.num_heads)
        assert denoiser.map_feature_dim==outer.get('hidden_dim',128)
        if outer:
            assert model.encoder.init_map_encoder.training
            assert any(p.grad is not None for p in model.encoder.init_map_encoder.parameters())
        else:
            assert all(p.grad is None for p in model.encoder.map_encoder.parameters())
        checked.append(dict(name=name,options=options,outer=outer,losses=values.tolist(),
                            generator_parameters=sum(p.numel() for p in decoder.G1.parameters()),
                            map_feature_dim=denoiser.map_feature_dim))
        print('CHECKED',name,flush=True)
        del model
    types=torch.arange(10)
    assert torch.where(initial_map_output_mask(types,'lanes_only'))[0].tolist()==[0,1,2]
    assert initial_map_output_mask(types,'all').all()
    result.update(status='PASS',variants=checked,prepared_sources=json.loads((ART/'prepared_sources.json').read_text()),
                  scope='CPU real TRAIN2 forward/backward, default tensor/outputs/RNG exact equality, old InitDiffusion checkpoint load, new mask semantics, and trainable changed-dimension map gradients. Sources remain staged, not applied to production.')
    (ART/'prepared_preflight.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PREPARED_ARCHITECTURE_PREFLIGHT_PASS',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['legacy','prepared']);args=parser.parse_args()
    main(args.mode)
