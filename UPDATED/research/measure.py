"""Complete-split finite-shot emulation with calibration and auditable allocations."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from .config import Config
from .data import load_splits
from .metrics import probabilities, classification_metrics, fit_temperature
from .models import MatchedViT
from .runtime import json_write, event, seed_all, snapshot
from .shots import ShotConfig, ShotEstimator


@torch.no_grad()
def measured_predict(model,dataset,batch_size,cfg,seed,out,split,save_allocations=True):
    model.eval()
    estimator=ShotEstimator(cfg,seed)
    model.set_estimator(estimator,capture=True)
    logits,labels,per_example_shots=[],[],[]
    errors=[]
    start=time.perf_counter()
    folder=out/'allocations'/split
    folder.mkdir(parents=True,exist_ok=True)
    for batch,(x,y) in enumerate(DataLoader(dataset,batch_size=batch_size,shuffle=False)):
        z=model(x)
        logits.append(z.numpy())
        labels.append(y.numpy())
        records={}
        used=np.zeros(len(y),dtype=np.int64)
        batch_errors=np.zeros((len(y),4))
        for layer,block in enumerate(model.blocks):
            r=block.attn.last
            audit=r['audit']
            used+=audit['shots'].sum(axis=(1,2,3))
            ideal=(model.config.beta*r['ideal']).softmax(-1)
            estimated=r['attention']
            value=r['v']
            # Layer-local conditional comparison uses the same current noisy-input Q/K/V.
            output_error=((estimated@value-ideal@value).square().sum(-1)).mean((1,2)).numpy()
            fidelity_mse=(r['estimated']-r['ideal']).square().mean((1,2,3)).numpy()
            attention_mse=(estimated-ideal).square().mean((1,2,3)).numpy()
            kl=(ideal*(ideal.clamp_min(1e-12).log()-estimated.clamp_min(1e-12).log())).sum(-1).mean((1,2)).numpy()
            batch_errors+=np.stack([fidelity_mse,attention_mse,output_error,kl],1)/len(model.blocks)
            if save_allocations:
                records[f'l{layer}_shots']=audit['shots'].astype(np.int32)
                records[f'l{layer}_successes']=audit['successes'].astype(np.int32)
                records[f'l{layer}_rounds']=np.stack(audit['decisions']).astype(np.int32)
        if save_allocations:
            np.savez_compressed(folder/f'batch-{batch:04d}.npz',**records)
        per_example_shots.append(used)
        errors.append(batch_errors)
    elapsed=time.perf_counter()-start
    model.set_estimator(None)
    z,y=np.concatenate(logits),np.concatenate(labels)
    assert len(y)==len(dataset)
    detail={'logits':z,'labels':y,'sample_ids':np.arange(len(y)), 'shots_per_example':np.concatenate(per_example_shots),
            'conditional_errors':np.concatenate(errors)}
    counters={'total_shots':estimator.total_shots,'circuit_calls':estimator.circuit_calls,
              'modeled_circuit_executions':estimator.modeled_circuit_executions,'seconds':elapsed}
    return detail,counters


def run(run_dir,cfg,root='runs/measurements',save_allocations=True):
    run_dir=Path(run_dir)
    c=Config.read(run_dir/'config.json')
    if c.model!='fidelity':
        raise ValueError('Only fidelity checkpoints support measurement substitution')
    seed_all(c.seed,c.threads)
    out=Path(root)/c.experiment_id/f'{cfg.policy}-s{cfg.shots}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'test_metrics.json').exists():
        if json.loads((out/'measurement.json').read_text())['allocation']!=asdict(cfg):
            raise ValueError('Existing measurement has a different allocation configuration')
        return out
    snapshot(out)
    json_write(out/'measurement.json',{'training_run':str(run_dir),'allocation':asdict(cfg),
               'seed':c.seed+700000,'execution_mode':'classical_probability_sampling_emulator',
               'calibration':'separate validation predictions for this measurement mode',
               'error_columns':['fidelity_mse','attention_mse','output_squared_l2','attention_kl'],
               'error_reference':'conditional layer-local exact attention; not end-to-end ideal-model error'})
    ds,ch,nclass,_=load_splits(c)
    model=MatchedViT(c,ch,nclass)
    model.load_state_dict(torch.load(run_dir/'best.pt',weights_only=True)['model'])
    val,vc=measured_predict(model,ds['val'],c.batch_size,cfg,c.seed+700000,out,'val',save_allocations)
    temperature=fit_temperature(val['logits'],val['labels'])
    np.savez_compressed(out/'validation_predictions.npz',**val,probabilities=probabilities(val['logits'],temperature))
    test,tc=measured_predict(model,ds['test'],c.batch_size,cfg,c.seed+800000,out,'test',save_allocations)
    p=probabilities(test['logits'],temperature)
    np.savez_compressed(out/'test_predictions.npz',**test,probabilities=p)
    m=classification_metrics(test['labels'],p)
    m.update({'execution_mode':'classical_probability_sampling_emulator','temperature':temperature,
              'validation_resources':vc,'test_resources':tc,'shots':tc['total_shots'],'circuit_calls':tc['circuit_calls'],
              'conditional_errors_mean':test['conditional_errors'].mean(0).tolist(),
              'accuracy_per_million_modeled_shots':m['accuracy']/(tc['total_shots']/1e6),
              'auroc_per_million_modeled_shots':m['auroc']/(tc['total_shots']/1e6) if m['auroc'] is not None else None})
    json_write(out/'test_metrics.json',m)
    event(out,kind='complete',test_n=len(test['labels']),**tc)
    print(c.seed,cfg.policy,cfg.shots,'accuracy',round(m['accuracy'],4),'seconds',round(tc['seconds'],2),flush=True)
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--runs',default='runs/central')
    p.add_argument('--output',default='runs/measurements')
    p.add_argument('--manifest',default='configs/central.json')
    p.add_argument('--budgets',nargs='+',type=int)
    p.add_argument('--policies',nargs='+')
    p.add_argument('--summary-only-allocations',action='store_true')
    a=p.parse_args()
    manifest=json.loads(Path(a.manifest).read_text())['measurement']
    for path in sorted(Path(a.runs).glob('*fidelity*/config.json')):
        for budget in a.budgets or manifest['budgets']:
            for policy in a.policies or manifest['policies']:
                run(path.parent,ShotConfig(shots=budget,policy=policy,pilot=manifest['pilot'],rounds=manifest['rounds']),
                    a.output,not a.summary_only_allocations)


if __name__=='__main__':
    main()
