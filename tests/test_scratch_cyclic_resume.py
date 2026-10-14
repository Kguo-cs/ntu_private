"""Real-Lightning regression for multi-epoch continuation of native sweeps."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import math
import json
import random
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
import lightning.pytorch as L
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.loader import DataLoader
from tools.scratch_component_loss_sweep import install_cyclic_resume_loader, ScratchDiagnostics


class TinyData(L.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.num_workers=0;self.shuffle=False;self.train_batch_size=3;self.pin_memory=False
        self.train_dataset=torch.arange(7,dtype=torch.float32).reshape(-1,1)
    def train_dataloader(self):
        return DataLoader(self.train_dataset,batch_size=3,shuffle=False,drop_last=False)


class TinyModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.linear=torch.nn.Linear(1,1)
        self.register_buffer('ema',self.linear.weight.detach().clone())
        self.register_buffer('ema_updates',torch.tensor(0,dtype=torch.long))
        self.trace=[]
        self.scheduler_trace=[]
    def training_step(self,batch,batch_idx):
        draw=torch.rand(())+random.random()+float(np.random.random())
        self.trace.append(dict(step=self.global_step,epoch=self.current_epoch,
            ids=batch[:,0].tolist(),draw=float(draw),lr=self.optimizers().param_groups[0]['lr']))
        return (self.linear(batch)-(batch*.1+draw)).square().mean()
    def lr_scheduler_step(self,scheduler,metric):
        progress=self.trainer.fit_loop.epoch_loop.batch_progress
        self.scheduler_trace.append((self.global_step,self.current_epoch,progress.current.ready,
                                     self.trainer.num_training_batches,progress.is_last_batch))
        super().lr_scheduler_step(scheduler,metric)
    def configure_optimizers(self):
        optimizer=torch.optim.AdamW(self.linear.parameters(),lr=.01)
        def factor(unused):
            return .05+.95*.5*(1+math.cos(math.pi*min((self.current_epoch+1)/128,1.)))
        return [optimizer],[LambdaLR(optimizer,factor)]


class Checkpoints(L.Callback):
    def __init__(self,folder,split,resume_rng=None):
        self.folder=folder;self.split=split;self.resume_rng=resume_rng;self.metadata={}
    def on_train_start(self,trainer,model):
        ScratchDiagnostics.on_train_start(self,trainer,model)
    def on_save_checkpoint(self,trainer,model,checkpoint):
        ScratchDiagnostics.on_save_checkpoint(self,trainer,model,checkpoint)
    def on_train_batch_end(self,trainer,model,outputs,batch,batch_idx):
        model.ema.mul_(.9).add_(model.linear.weight.detach(),alpha=.1)
        model.ema_updates.add_(1)
        if trainer.global_step in (self.split,11):
            trainer.save_checkpoint(self.folder/f'step{trainer.global_step}.ckpt')


def run(folder,steps,split,source=None):
    L.seed_everything(251017,workers=True)
    model=TinyModel();dm=TinyData();rng=None
    if source:
        ckpt=torch.load(source,map_location='cpu',weights_only=False)
        rng=ckpt['scratch_sweep_rng']
        install_cyclic_resume_loader(dm,ckpt['global_step'],7,3,folder/"loader_events.json")
    callback=Checkpoints(folder,split,rng)
    trainer=L.Trainer(accelerator='cpu',devices=1,max_epochs=128,max_steps=steps,
        logger=False,enable_checkpointing=False,enable_model_summary=False,
        enable_progress_bar=False,limit_val_batches=0,num_sanity_val_steps=0,
        default_root_dir=folder,callbacks=[callback],
        reload_dataloaders_every_n_epochs=1 if source else 0)
    trainer.fit(model,datamodule=dm,ckpt_path=str(source) if source else None)
    return model


class CyclicResumeTests(unittest.TestCase):
    def test_resume_matches_uninterrupted_across_multiple_epochs(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for split in (1,2,4,5):
                with self.subTest(split=split):
                    folders=[root/f'{split}_{name}' for name in ('full','part','resume')]
                    for folder in folders:folder.mkdir()
                    full=run(folders[0],11,split)
                    part=run(folders[1],split,split)
                    resumed=run(folders[2],11,split,folders[1]/f'step{split}.ckpt')
                    self.assertEqual(full.trace,part.trace+resumed.trace)
                    events=json.loads((folders[2]/'loader_events.json').read_text())
                    self.assertEqual([e['epoch'] for e in events],list(range(split//3,4)))
                    self.assertEqual(events[0]['offset'],(split%3)*3)
                    self.assertTrue(all(e['offset']==0 and e['dataset_scenes']==7
                                        and e['batches']==3 for e in events[1:]))
                    for key,value in full.state_dict().items():
                        torch.testing.assert_close(value,resumed.state_dict()[key],rtol=0,atol=0)
                    a=torch.load(folders[0]/'step11.ckpt',map_location='cpu',weights_only=False)
                    b=torch.load(folders[2]/'step11.ckpt',map_location='cpu',weights_only=False)
                    self.assertEqual(a['lr_schedulers'],b['lr_schedulers'],
                                     (full.scheduler_trace,part.scheduler_trace,resumed.scheduler_trace))
                    self.assertEqual(a['epoch'],b['epoch'])
                    self.assertEqual(a['global_step'],b['global_step'])
                    for key,values in a['optimizer_states'][0]['state'].items():
                        for name,value in values.items():
                            torch.testing.assert_close(value,b['optimizer_states'][0]['state'][key][name],rtol=0,atol=0)
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':unittest.main()
