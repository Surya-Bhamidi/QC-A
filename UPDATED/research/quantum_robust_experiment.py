"""Validation-selected, frozen quantum-preserving accuracy experiment."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from .config import Config
from .data import load_splits
from .train import predict
from .metrics import classification_metrics,probabilities,fit_temperature
from .runtime import seed_all,json_write,event,snapshot,torch_save_atomic
from .quantum_robust import QuantumRobustViT,SAMStep,augment

ROOT=Path('runs/quantum_robust_accuracy')
MANIFEST=Path('configs/quantum_robust_accuracy.json')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hashes():
    return {p:digest(p) for p in ['research/quantum_robust.py','research/quantum_robust_experiment.py',
        'research/models.py','research/config.py','research/data.py','research/train.py',
        'research/metrics.py','research/runtime.py']}


def model_config(spec,seed,kernel):
    return Config(**(spec['architecture']|{'dataset':spec['dataset'],'seed':seed,
                    'model':'dot' if kernel=='dot' else 'fidelity'}))


def run_folder(spec,recipe,seed,kernel):
    tag=hashlib.sha256(json.dumps({'architecture':spec['architecture'],'training':spec['training'],
            'recipe':recipe,'seed':seed,'kernel':kernel},sort_keys=True).encode()).hexdigest()[:10]
    return ROOT/f"{recipe['name']}-{kernel}-s{seed}-{tag}"


def balanced_nll(logits,labels):
    loss=torch.nn.functional.cross_entropy(torch.from_numpy(logits),torch.from_numpy(labels),reduction='none')
    return float(torch.stack([loss[torch.from_numpy(labels)==c].mean() for c in np.unique(labels)]).mean())


def run_one(spec,recipe,seed,kernel,test=False):
    folder=run_folder(spec,recipe,seed,kernel)
    final=folder/('test_metrics.json' if test else 'validation_metrics.json')
    if final.exists():
        return json.loads(final.read_text())
    folder.mkdir(parents=True,exist_ok=True)
    train_cfg=spec['training']
    seed_all(seed,train_cfg['threads'])
    c=model_config(spec,seed,kernel)
    ds,_,classes,meta=load_splits(c)
    model=QuantumRobustViT(c,kernel)
    ema=deepcopy(model)
    ema.eval()
    optimizer=torch.optim.AdamW(model.parameters(),lr=train_cfg['lr'],weight_decay=train_cfg['weight_decay'])
    def lr_lambda(epoch):
        warmup=train_cfg['warmup_epochs']
        return (epoch+1)/warmup if epoch<warmup else .01+.99*.5*(1+math.cos(math.pi*(epoch-warmup)/(train_cfg['epochs']-warmup)))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda)
    sam=SAMStep(optimizer,recipe['sam_rho']) if recipe['sam_rho'] else None
    generator=torch.Generator().manual_seed(seed+612000)
    loader=DataLoader(ds['train'],batch_size=train_cfg['batch_size'],shuffle=True,generator=generator)
    counts=torch.bincount(ds['train'].tensors[1],minlength=classes).float()
    weight=counts.sum()/(classes*counts.clamp_min(1)) if train_cfg['class_weight'] else None
    criterion=torch.nn.CrossEntropyLoss(weight=weight)
    start_epoch,best,elapsed=0,float('inf'),0.
    if not (folder/'config.json').exists():
        json_write(folder/'config.json',{'architecture':c.dict(),'kernel':kernel,'recipe':recipe,'training':train_cfg})
        json_write(folder/'data.json',meta)
        json_write(folder/'resources.json',{'parameters':sum(p.numel() for p in model.parameters()),
            'qubits':0 if kernel=='dot' else 4,'quantum_training_backend':'batched_differentiable_gate_statevector_simulation' if kernel!='dot' else 'classical_dot',
            'physical_hardware':False,'pennylane_qnode_calls_during_training':0})
        snapshot(folder/'initial_source')
    if (folder/'last.pt').exists():
        state=torch.load(folder/'last.pt',weights_only=False)
        model.load_state_dict(state['model']);ema.load_state_dict(state['ema'])
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        start_epoch,best,elapsed=state['epoch'],state['best'],state['elapsed']
        torch.set_rng_state(state['rng_torch']);np.random.set_state(state['rng_numpy'])
        random.setstate(state['rng_python']);generator.set_state(state['rng_loader'])
        event(folder,kind='resume',epoch=start_epoch)
    for epoch in range(start_epoch+1,train_cfg['epochs']+1):
        start=time.perf_counter();model.train();train_loss=0.
        for x,y in loader:
            x=augment(x,train_cfg['augmentation'])
            lam=float(np.random.beta(train_cfg['mixup_alpha'],train_cfg['mixup_alpha']))
            order=torch.randperm(len(x))
            mixed=lam*x+(1-lam)*x[order]
            def closure():
                logits=model(mixed)
                return lam*criterion(logits,y)+(1-lam)*criterion(logits,y[order])
            if sam is not None:
                loss=sam.step(closure)
            else:
                optimizer.zero_grad(set_to_none=True);loss=closure();loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
            train_loss+=float(loss.detach())*len(y)
            with torch.no_grad():
                for target,source in zip(ema.parameters(),model.parameters()):
                    target.lerp_(source,1-train_cfg['ema_decay'])
        val_z,val_y,_=predict(ema,ds['val'],train_cfg['batch_size'])
        score=balanced_nll(val_z,val_y)
        elapsed+=time.perf_counter()-start
        if score<best:
            best=score
            torch_save_atomic({'model':ema.state_dict(),'epoch':epoch,'val_balanced_nll':score},folder/'best.pt')
        scheduler.step()
        torch_save_atomic({'model':model.state_dict(),'ema':ema.state_dict(),'optimizer':optimizer.state_dict(),
            'scheduler':scheduler.state_dict(),'epoch':epoch,'best':best,'elapsed':elapsed,
            'rng_torch':torch.get_rng_state(),'rng_numpy':np.random.get_state(),'rng_python':random.getstate(),
            'rng_loader':generator.get_state()},folder/'last.pt')
        event(folder,kind='epoch',epoch=epoch,train_loss=train_loss/len(ds['train']),
              validation_balanced_nll=score,validation_accuracy=float(np.mean(val_z.argmax(1)==val_y)),seconds=elapsed)
        if epoch%6==0:
            print(recipe['name'],kernel,seed,'epoch',epoch,'balanced val NLL',round(score,4),'seconds',round(elapsed,1),flush=True)
    best_state=torch.load(folder/'best.pt',weights_only=True)
    model.load_state_dict(best_state['model'])
    val_z,val_y,_=predict(model,ds['val'],train_cfg['batch_size'])
    temperature=fit_temperature(val_z,val_y)
    for split,stem in ([('val','validation'),('test','test')] if test else [('val','validation')]):
        z,y,seconds=predict(model,ds[split],train_cfg['batch_size'])
        p=probabilities(z,temperature)
        np.savez_compressed(folder/(stem+'_predictions.npz'),logits=z,labels=y,probabilities=p,sample_ids=np.arange(len(y)))
        metrics=classification_metrics(y,p)
        metrics.update(seed=seed,kernel=kernel,recipe=recipe['name'],temperature=temperature,
            balanced_nll=balanced_nll(z,y),best_epoch=best_state['epoch'],training_seconds=elapsed,
            inference_seconds=seconds,physical_hardware=False,
            execution_mode='differentiable_quantum_gate_statevector_simulation' if kernel!='dot' else 'classical_dot',
            shots=0,pennylane_qnode_calls=0,parameters=sum(p.numel() for p in model.parameters()))
        json_write(folder/(stem+'_metrics.json'),metrics)
    print('Completed',recipe['name'],kernel,seed,'test' if test else 'validation',flush=True)
    return metrics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['pilot','freeze','run'])
    a=parser.parse_args()
    spec=json.loads(MANIFEST.read_text())
    frozen=MANIFEST.with_suffix('.frozen.json')
    if a.action=='freeze':
        if frozen.exists():
            raise FileExistsError(frozen)
        pilot=json.loads((ROOT/'pilot_summary.json').read_text())['rows']
        choices={recipe['name']:float(np.mean([r['balanced_nll'] for r in pilot if r['recipe']==recipe['name']])) for recipe in spec['recipes']}
        selected=min(choices,key=choices.get)
        json_write(frozen,{'utc':datetime.now(timezone.utc).isoformat(),'manifest_sha256':digest(MANIFEST),
            'source_sha256':hashes(),'pilot_recipe_scores':choices,'selected_recipe':selected,
            'pilot_summary_sha256':digest(ROOT/'pilot_summary.json'),'disclosure':spec['test_disclosure']})
        snapshot(ROOT/'frozen_source')
        print('Frozen recipe',selected,choices,flush=True)
        return
    if a.action=='run':
        lock=json.loads(frozen.read_text())
        assert lock['manifest_sha256']==digest(MANIFEST) and lock['source_sha256']==hashes()
        recipes=[r for r in spec['recipes'] if r['name']==lock['selected_recipe']]
        seeds=spec['final_seeds'];test=True;phase='test'
    else:
        recipes=spec['recipes'];seeds=spec['pilot_seeds'];test=False;phase='pilot'
    rows=[]
    for recipe in recipes:
        for kernel in spec['kernels']:
            for seed in seeds:
                metrics=run_one(spec,recipe,seed,kernel,test)
                rows.append(metrics)
                json_write(ROOT/(phase+'_progress.json'),{'rows':rows})
    json_write(ROOT/(phase+'_summary.json'),{'complete':True,'conditions':len(rows),'rows':rows})
    print('Completed',phase,len(rows),'conditions',flush=True)


if __name__=='__main__':
    main()
