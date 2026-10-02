"""Fixed, matched pretrained-feature quantum accuracy follow-up."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from .quantum_transfer import (MANIFEST,CACHE,QuantumTransferViT,head_config,
                               load_features,feature_hashes)
from .quantum_robust_experiment import digest,balanced_nll,hashes as robust_hashes
from .metrics import probabilities,classification_metrics,fit_temperature
from .runtime import seed_all,json_write,event,snapshot,torch_save_atomic
from .train import predict

ROOT=Path('runs/pretrained_quantum_accuracy')


def source_hashes():
    return robust_hashes()|{p:digest(p) for p in ['research/quantum_transfer.py',
                                               'research/quantum_transfer_experiment.py']}


def verify_lock(spec):
    lock=json.loads(MANIFEST.with_suffix('.frozen.json').read_text())
    meta=json.loads((CACHE/'metadata.json').read_text())
    assert lock['manifest_sha256']==digest(MANIFEST) and lock['source_sha256']==source_hashes()
    assert lock['feature_metadata_sha256']==digest(CACHE/'metadata.json')
    assert lock['feature_files_sha256']==meta['files_sha256']==feature_hashes()
    assert lock['weights_sha256']==digest(spec['backbone']['weight_file'])
    return lock


def save_predictions(folder,stem,model,features,labels,temperature,details):
    z,y,seconds=predict(model,TensorDataset(features,labels),64)
    p=probabilities(z,temperature)
    np.savez_compressed(folder/(stem+'_predictions.npz'),logits=z,probabilities=p,
                        labels=y,sample_ids=np.arange(len(y)))
    metrics=classification_metrics(y,p)
    metrics.update(details,temperature=temperature,balanced_nll=balanced_nll(z,y),
                   inference_seconds_head_only=seconds)
    json_write(folder/(stem+'_metrics.json'),metrics)
    return metrics


def run_one(spec,banks,kernel,seed):
    folder=ROOT/f'{kernel}-s{seed}'
    if (folder/'test_metrics.json').exists():
        return json.loads((folder/'test_metrics.json').read_text())
    folder.mkdir(parents=True,exist_ok=True)
    tr=spec['training'];seed_all(seed,tr['threads'])
    model=QuantumTransferViT(head_config(spec,seed,kernel),kernel)
    ema=deepcopy(model).eval()
    opt=torch.optim.AdamW(model.parameters(),lr=tr['lr'],weight_decay=tr['weight_decay'])
    def schedule(epoch):
        w=tr['warmup_epochs']
        return (epoch+1)/w if epoch<w else .01+.99*.5*(1+math.cos(math.pi*(epoch-w)/(tr['epochs']-w)))
    sched=torch.optim.lr_scheduler.LambdaLR(opt,schedule)
    bank=banks['train'];n=len(bank['labels'])
    generator=torch.Generator().manual_seed(seed+731000)
    loader=DataLoader(TensorDataset(torch.arange(n),bank['labels']),batch_size=tr['batch_size'],
                      shuffle=True,generator=generator)
    counts=torch.bincount(bank['labels'],minlength=2).float()
    weight=counts.sum()/(2*counts) if tr['class_weight'] else None
    loss_fn=torch.nn.CrossEntropyLoss(weight=weight)
    start_epoch,best,elapsed=0,float('inf'),0.
    if not (folder/'config.json').exists():
        json_write(folder/'config.json',{'seed':seed,'kernel':kernel,'specification':spec})
        snapshot(folder/'initial_source')
    if (folder/'last.pt').exists():
        state=torch.load(folder/'last.pt',weights_only=False)
        model.load_state_dict(state['model']);ema.load_state_dict(state['ema'])
        opt.load_state_dict(state['optimizer']);sched.load_state_dict(state['scheduler'])
        start_epoch,best,elapsed=state['epoch'],state['best'],state['elapsed']
        torch.set_rng_state(state['rng_torch']);np.random.set_state(state['rng_numpy'])
        random.setstate(state['rng_python']);generator.set_state(state['rng_loader'])
        event(folder,kind='resume',epoch=start_epoch)
    for epoch in range(start_epoch+1,tr['epochs']+1):
        start=time.perf_counter();model.train();total=0.
        for indices,y in loader:
            views=torch.randint(bank['features'].shape[0],(len(indices),))
            x=bank['features'][views,indices]
            lam=float(np.random.beta(tr['mixup_alpha'],tr['mixup_alpha']))
            order=torch.randperm(len(x));x=lam*x+(1-lam)*x[order]
            opt.zero_grad(set_to_none=True)
            z=model(x);loss=lam*loss_fn(z,y)+(1-lam)*loss_fn(z,y[order])
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            total+=float(loss.detach())*len(y)
            with torch.no_grad():
                for target,source in zip(ema.parameters(),model.parameters()):
                    target.lerp_(source,1-tr['ema_decay'])
        vz,vy,_=predict(ema,TensorDataset(banks['val']['features'][0],banks['val']['labels']),64)
        score=balanced_nll(vz,vy);elapsed+=time.perf_counter()-start
        if score<best:
            best=score
            torch_save_atomic({'model':ema.state_dict(),'epoch':epoch,'validation_balanced_nll':best},folder/'best.pt')
        sched.step()
        torch_save_atomic({'model':model.state_dict(),'ema':ema.state_dict(),'optimizer':opt.state_dict(),
            'scheduler':sched.state_dict(),'epoch':epoch,'best':best,'elapsed':elapsed,
            'rng_torch':torch.get_rng_state(),'rng_numpy':np.random.get_state(),
            'rng_python':random.getstate(),'rng_loader':generator.get_state()},folder/'last.pt')
        event(folder,kind='epoch',epoch=epoch,train_loss=total/n,validation_balanced_nll=score,seconds=elapsed)
        if epoch%4==0:
            print(kernel,seed,'epoch',epoch,'validation balanced NLL',round(score,4),'seconds',round(elapsed,1),flush=True)
    state=torch.load(folder/'best.pt',weights_only=True);model.load_state_dict(state['model'])
    vz,vy,_=predict(model,TensorDataset(banks['val']['features'][0],banks['val']['labels']),64)
    temperature=fit_temperature(vz,vy)
    meta=json.loads((CACHE/'metadata.json').read_text())
    details={'kernel':kernel,'seed':seed,'best_epoch':state['epoch'],'training_seconds':elapsed,
        'trainable_parameters':sum(p.numel() for p in model.parameters()),
        'frozen_encoder_parameters':meta['frozen_encoder_parameters'],'external_pretraining':'ImageNet-1K',
        'physical_hardware':False,'shots':0,'pennylane_qnode_calls':0,
        'head_backend':'quantum_gate_statevector_simulation' if kernel!='dot' else 'classical_dot'}
    result=None
    for split,stem in [('val','validation'),('test','test')]:
        result=save_predictions(folder,stem,model,banks[split]['features'][0],banks[split]['labels'],temperature,details)
    print('Completed pretrained',kernel,seed,'test accuracy',round(result['accuracy'],5),flush=True)
    return result


@threadpool_limits.wrap(limits=4)
def linear_control(spec,banks):
    folder=ROOT/'linear';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'test_metrics.json').exists():
        return json.loads((folder/'test_metrics.json').read_text())
    x=banks['train']['features'].mean(2).reshape(-1,512).numpy()
    y=banks['train']['labels'].repeat(banks['train']['features'].shape[0]).numpy()
    vx=banks['val']['features'][0].mean(1).numpy();vy=banks['val']['labels'].numpy()
    choices=[];models=[]
    for c in spec['linear_C_candidates']:
        model=LogisticRegression(C=c,class_weight='balanced',max_iter=2000,solver='lbfgs',tol=1e-6)
        model.fit(x,y);s=model.decision_function(vx)
        z=np.stack([-.5*s,.5*s],1).astype(np.float32)
        choices.append({'C':c,'validation_balanced_nll':balanced_nll(z,vy),'iterations':int(model.n_iter_[0])})
        models.append(model)
    chosen=int(np.argmin([r['validation_balanced_nll'] for r in choices]));model=models[chosen]
    s=model.decision_function(vx);vz=np.stack([-.5*s,.5*s],1).astype(np.float32)
    temperature=fit_temperature(vz,vy)
    np.savez_compressed(folder/'coefficients.npz',coef=model.coef_,intercept=model.intercept_)
    json_write(folder/'selection.json',{'candidates':choices,'selected':choices[chosen],
               'criterion':'validation_class_balanced_NLL','test_labels_used_for_selection':False})
    for split,stem in [('val','validation'),('test','test')]:
        s=model.decision_function(banks[split]['features'][0].mean(1).numpy())
        z=np.stack([-.5*s,.5*s],1).astype(np.float32)
        y=banks[split]['labels'].numpy();p=probabilities(z,temperature)
        np.savez_compressed(folder/(stem+'_predictions.npz'),logits=z,probabilities=p,labels=y,sample_ids=np.arange(len(y)))
        metrics=classification_metrics(y,p)
        metrics.update(kernel='linear',C=choices[chosen]['C'],temperature=temperature,
                       parameters=513,external_pretraining='ImageNet-1K',physical_hardware=False)
        json_write(folder/(stem+'_metrics.json'),metrics)
    print('Pretrained linear control test accuracy',metrics['accuracy'],flush=True)
    return metrics


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','run']);a=parser.parse_args()
    spec=json.loads(MANIFEST.read_text());frozen=MANIFEST.with_suffix('.frozen.json')
    if a.action=='freeze':
        if frozen.exists():
            raise FileExistsError(frozen)
        meta=json.loads((CACHE/'metadata.json').read_text())
        assert meta['manifest_sha256']==digest(MANIFEST) and meta['files_sha256']==feature_hashes()
        json_write(frozen,{'utc':datetime.now(timezone.utc).isoformat(),'manifest_sha256':digest(MANIFEST),
            'source_sha256':source_hashes(),'feature_metadata_sha256':digest(CACHE/'metadata.json'),
            'feature_files_sha256':feature_hashes(),'weights_sha256':digest(spec['backbone']['weight_file']),
            'disclosure':spec['disclosure']})
        snapshot(ROOT/'frozen_source');print('Frozen pretrained quantum experiment',flush=True)
        return
    verify_lock(spec);seed_all(0,spec['training']['threads']);banks=load_features();rows=[]
    for kernel in spec['kernels']:
        for seed in spec['seeds']:
            rows.append(run_one(spec,banks,kernel,seed));json_write(ROOT/'progress.json',{'rows':rows})
    linear=linear_control(spec,banks)
    json_write(ROOT/'summary.json',{'complete':True,'neural_conditions':len(rows),'rows':rows,'linear':linear})


if __name__=='__main__':
    main()
