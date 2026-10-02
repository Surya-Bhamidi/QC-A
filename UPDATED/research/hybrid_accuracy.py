"""Matched convolutional-token fidelity/dot accuracy experiment."""
import argparse
from dataclasses import dataclass
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from .config import Config
from .data import load_splits
from .metrics import probabilities,classification_metrics,fit_temperature,bootstrap_seed_mean
from .models import Block
from .runtime import seed_all,json_write,torch_save_atomic,snapshot

MANIFEST=Path('configs/hybrid_accuracy.json')
ROOT=Path('runs/hybrid_accuracy')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class Architecture:
    model: str
    seed: int
    image_size: int=28
    patch_size: int=7
    embed_dim: int=48
    heads: int=3
    qk_dim: int=4
    depth: int=2
    mlp_dim: int=96
    dropout: float=.1
    beta: float=2.
    token_grid: int=4


class ConvTokenClassifier(nn.Module):
    def __init__(self,c,classes=2):
        super().__init__()
        self.c=c
        self.stem=nn.Sequential(
            nn.Conv2d(1,16,3,padding=1),nn.GroupNorm(4,16),nn.GELU(),
            nn.Conv2d(16,32,3,stride=2,padding=1),nn.GroupNorm(8,32),nn.GELU(),
            nn.Conv2d(32,c.embed_dim,3,stride=2,padding=1),nn.GroupNorm(8,c.embed_dim),nn.GELU(),
            nn.AdaptiveAvgPool2d((c.token_grid,c.token_grid)))
        tokens=c.token_grid**2+1
        self.cls=nn.Parameter(torch.zeros(1,1,c.embed_dim))
        self.position=nn.Parameter(torch.randn(1,tokens,c.embed_dim)*.02)
        self.drop=nn.Dropout(c.dropout)
        self.blocks=nn.ModuleList([Block(c) for _ in range(c.depth)])
        self.norm=nn.LayerNorm(c.embed_dim)
        self.head=nn.Linear(c.embed_dim,classes)

    def forward(self,x):
        x=self.stem(x).flatten(2).transpose(1,2)
        x=self.drop(torch.cat([self.cls.expand(len(x),-1,-1),x],1)+self.position)
        for block in self.blocks:
            x=block(x)
        return self.head(self.norm(x)[:,0])


@torch.no_grad()
def predict(model,dataset,batch_size):
    model.eval(); logits=[]; labels=[]
    for x,y in DataLoader(dataset,batch_size=batch_size,shuffle=False):
        logits.append(model(x).numpy());labels.append(y.numpy())
    return np.concatenate(logits),np.concatenate(labels)


def run_one(spec,seed,kernel,test=False):
    arch=Architecture(model=kernel,seed=seed,**spec['architecture'])
    tag=f'{kernel}-s{seed}-'+hashlib.sha256(json.dumps({'arch':arch.__dict__,'train':spec['training']},sort_keys=True).encode()).hexdigest()[:10]
    folder=ROOT/tag
    final=folder/('test_metrics.json' if test else 'validation_metrics.json')
    if final.exists():
        return folder,json.loads(final.read_text())
    folder.mkdir(parents=True,exist_ok=True)
    seed_all(seed,spec['training']['threads'])
    data_config=Config(dataset=spec['dataset'],model=kernel,seed=seed)
    datasets,_,classes,meta=load_splits(data_config)
    model=ConvTokenClassifier(arch,classes)
    generator=torch.Generator().manual_seed(seed+510000)
    loader=DataLoader(datasets['train'],batch_size=spec['training']['batch_size'],shuffle=True,generator=generator)
    counts=torch.bincount(datasets['train'].tensors[1],minlength=classes).float()
    weights=counts.sum()/(classes*counts.clamp_min(1)) if spec['training']['class_weight'] else None
    criterion=nn.CrossEntropyLoss(weight=weights)
    optimizer=torch.optim.AdamW(model.parameters(),lr=spec['training']['lr'],weight_decay=spec['training']['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=spec['training']['epochs'],eta_min=1e-6)
    best=float('inf'); elapsed=0.
    json_write(folder/'config.json',{'architecture':arch.__dict__,'training':spec['training']})
    json_write(folder/'data.json',meta)
    snapshot(folder)
    for epoch in range(1,spec['training']['epochs']+1):
        start=time.perf_counter();model.train()
        for x,y in loader:
            optimizer.zero_grad(set_to_none=True);loss=criterion(model(x),y);loss.backward();optimizer.step()
        val_z,val_y=predict(model,datasets['val'],spec['training']['batch_size'])
        val_nll=float(nn.functional.cross_entropy(torch.from_numpy(val_z),torch.from_numpy(val_y)))
        scheduler.step();elapsed+=time.perf_counter()-start
        if val_nll<best:
            best=val_nll;torch_save_atomic({'model':model.state_dict(),'epoch':epoch,'val_nll':best},folder/'best.pt')
    state=torch.load(folder/'best.pt',weights_only=True);model.load_state_dict(state['model'])
    val_z,val_y=predict(model,datasets['val'],spec['training']['batch_size'])
    temperature=fit_temperature(val_z,val_y)
    val_p=probabilities(val_z,temperature)
    np.savez_compressed(folder/'validation_predictions.npz',logits=val_z,labels=val_y,probabilities=val_p,sample_ids=np.arange(len(val_y)))
    vm=classification_metrics(val_y,val_p);vm.update(best_epoch=state['epoch'],temperature=temperature,training_seconds=elapsed)
    json_write(folder/'validation_metrics.json',vm)
    if not test:
        return folder,vm
    test_z,test_y=predict(model,datasets['test'],spec['training']['batch_size']);test_p=probabilities(test_z,temperature)
    np.savez_compressed(folder/'test_predictions.npz',logits=test_z,labels=test_y,probabilities=test_p,sample_ids=np.arange(len(test_y)))
    tm=classification_metrics(test_y,test_p);tm.update(best_epoch=state['epoch'],temperature=temperature,training_seconds=elapsed,
        execution_mode='classical_convolutional_encoder_with_analytical_'+kernel+'_attention',shots=0,circuit_calls=0)
    json_write(folder/'test_metrics.json',tm)
    return folder,tm


def summarize(rows,phase):
    result=[]
    for kernel in ['fidelity','dot']:
        part=[r for r in rows if r['kernel']==kernel]
        record={'kernel':kernel,'seeds':len(part)}
        for metric in ['accuracy','auroc','balanced_accuracy']:
            s=bootstrap_seed_mean([r[metric] for r in part])
            record[metric]=s
        result.append(record)
    json_write(ROOT/(phase+'_summary.json'),{'phase':phase,'rows':rows,'summary':result})
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['pilot','freeze','run']);args=parser.parse_args()
    spec=json.loads(MANIFEST.read_text());frozen=MANIFEST.with_suffix('.frozen.json')
    source={'research/hybrid_accuracy.py':digest(__file__),'research/models.py':digest('research/models.py'),
            'research/data.py':digest('research/data.py'),'research/metrics.py':digest('research/metrics.py')}
    if args.action=='freeze':
        if frozen.exists():raise FileExistsError(frozen)
        if not (ROOT/'pilot_summary.json').exists():raise ValueError('Run validation pilot first')
        json_write(frozen,{'manifest_sha256':digest(MANIFEST),'source_sha256':source,
            'utc':datetime.now(timezone.utc).isoformat(),'test_disclosure':'Official test split was inspected in earlier experiments'})
        return
    if args.action=='run':
        lock=json.loads(frozen.read_text());assert lock['manifest_sha256']==digest(MANIFEST) and lock['source_sha256']==source
        seeds=spec['final_seeds'];phase='test';test=True
    else:
        seeds=spec['pilot_seeds'];phase='pilot';test=False
    rows=[]
    for kernel in spec['kernels']:
        for seed in seeds:
            folder,m=run_one(spec,seed,kernel,test)
            rows.append({'kernel':kernel,'seed':seed,'run':str(folder),**{k:m[k] for k in ['accuracy','auroc','balanced_accuracy']}})
            print(phase,kernel,seed,'accuracy',m['accuracy'],'auroc',m['auroc'],flush=True)
    print(json.dumps(summarize(rows,phase),indent=2))


if __name__=='__main__':
    main()
