"""Frozen follow-up: noisy final-CLS attention, selective mitigation and shots."""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from .bias_aware import RiskConfig,POLICIES,run_policy
from .config import Config
from .data import load_splits
from .models import MatchedViT,angles
from .noisy_overlap import destructive_means,reference_angles,NoisyParitySampler
from .noise_probe import calibration_profile
from .circuits import Noise
from .metrics import classification_metrics,probabilities
from .runtime import seed_all,json_write,snapshot
from .shots import softmax
from .train import train

ROOT=Path('runs/bias_aware')
MANIFEST=Path('configs/bias_aware.json')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def computational_hashes():
    return {p:sha(p) for p in ['research/bias_aware.py','research/noisy_overlap.py',
                              'research/bias_experiment.py','research/models.py',
                              'research/shots.py','research/circuits.py','research/noise_probe.py',
                              'research/train.py','research/data.py','research/config.py',
                              'research/runtime.py','research/metrics.py','configs/central.json']}


def noise_settings():
    calibrated,_=calibration_profile()
    return {'ideal':Noise(),'historical':calibrated,
            'stress':Noise(preparation=.003,one_qubit=.005,two_qubit=.03,
                           amplitude=.003,phase=.003,readout=.04)}


def checkpoint(seed,allow_train=False):
    base=json.loads(Path('configs/central.json').read_text())['base']
    c=Config(**(base|{'seed':seed,'dataset':'pneumoniamnist','model':'fidelity'}))
    existing=Path('runs/central')/c.experiment_id
    folder=existing if (existing/'best.pt').exists() else ROOT/'checkpoints'/c.experiment_id
    if not (folder/'best.pt').exists():
        if not allow_train:
            raise FileNotFoundError(folder)
        # Train/validation only: follow-up test inference remains in the frozen runner.
        train(c,ROOT/'checkpoints',validation_only=True)
    model=MatchedViT(c)
    model.load_state_dict(torch.load(folder/'best.pt',weights_only=True)['model'])
    model.eval()
    return c,model,folder


@torch.no_grad()
def classify(model,bank,estimates):
    b,h,n=bank['ideal'].shape
    att=softmax(model.config.beta*estimates.reshape(b,h,n))
    values=np.einsum('bhn,bhnd->bhd',att,bank['values']).reshape(b,-1)
    final=model.blocks[-1]
    x=torch.from_numpy(bank['residual'])+final.attn.out(torch.as_tensor(values,dtype=torch.float32))
    x=x+final.mlp(final.norm2(x))
    logits=model.head(model.norm(x)).numpy()
    return logits,att


@torch.no_grad()
def feature_bank(c,model,folder,split,limit=None):
    name=f'seed-{c.seed}-{split}'+(f'-n{limit}' if limit else '')
    path=ROOT/'banks'/(name+'.npz')
    if path.exists():
        with np.load(path) as stored:
            return {k:stored[k] for k in stored.files}
    ds,_,_,meta=load_splits(c)
    x,y=ds[split].tensors
    if limit:
        x,y=x[:limit],y[:limit]
    output={k:[] for k in ['q','k','values','ideal','residual','clean_logits']}
    residual=[]
    hook=model.blocks[-1].register_forward_pre_hook(lambda module,args:residual.append(args[0][:,0].detach()))
    model.set_estimator(capture=True)
    for start in range(0,len(y),64):
        z=model(x[start:start+64])
        last=model.blocks[-1].attn.last
        output['q'].append(angles(last['q'][:,:,0:1]).numpy())
        output['k'].append(angles(last['k']).numpy())
        output['values'].append(last['v'].numpy())
        output['ideal'].append(last['ideal'][:,:,0].numpy())
        output['residual'].append(residual.pop().numpy())
        output['clean_logits'].append(z.numpy())
    hook.remove()
    model.set_estimator(capture=False)
    bank={k:np.concatenate(v) for k,v in output.items()}
    bank.update(labels=y.numpy(),sample_ids=np.arange(len(y)))
    reconstructed,_=classify(model,bank,bank['ideal'])
    np.testing.assert_allclose(reconstructed,bank['clean_logits'],atol=2e-6,rtol=1e-6)
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**bank)
    json_write(path.with_suffix('.json'),{'training_run':str(folder),'checkpoint_sha256':sha(folder/'best.pt'),
        'data_sha256':meta['sha256'],'split':split,'n':len(y),'model_seed':c.seed,
        'scope':'Only final-block CLS-to-token scores replaced; earlier layers analytical',
        'clean_reconstruction_max_absolute_error':float(np.max(abs(reconstructed-bank['clean_logits']))),
        'code_sha256':computational_hashes()})
    return bank


def noisy_bank(c,bank,label,noise,split):
    path=ROOT/'banks'/f'seed-{c.seed}-{split}-n{len(bank["labels"])}-{label}-noisy.npz'
    if path.exists():
        with np.load(path) as z:
            return z['means'],z['reference_means']
    start=time.perf_counter()
    parts=[destructive_means(bank['q'][i:i+64],bank['k'][i:i+64],noise) for i in range(0,len(bank['labels']),64)]
    means=np.concatenate(parts)
    q,k,_=reference_angles(c.qk_dim)
    references=destructive_means(q,k,noise)
    np.savez_compressed(path,means=means,reference_means=references)
    json_write(path.with_suffix('.json'),{'noise':asdict(noise),'simulator_seconds':time.perf_counter()-start,
        'mode':'factorized_noisy_density_matrix_circuit_simulation','physical_hardware':False,
        'target_circuit_settings_simulated':int(means.size),'reference_settings_simulated':4,
        'two_qubit_factor_evaluations':int((means.size+4)*c.qk_dim),'pennylane_qnode_calls':0,
        'logical_qubits':2*c.qk_dim,'limitations':'Product features, local independent noise, all-to-all destructive SWAP only'})
    return means,references


def condition(c,model,bank,means,references,label,policy,budget,repeat,cfg,phase):
    path=ROOT/phase/f'seed-{c.seed}'/f'{label}-s{budget}-{policy}-r{repeat}'
    metric_path=path/'metrics.json'
    if metric_path.exists():
        return json.loads(metric_path.read_text())
    path.mkdir(parents=True,exist_ok=True)
    b,h,n=bank['ideal'].shape
    key=f'{phase}-{c.seed}-{label}-{budget}-{repeat}'
    rng_seed=int(hashlib.sha256(key.encode()).hexdigest()[:8],16)
    sampler=NoisyParitySampler(means.reshape(b*h,n),np.random.default_rng(rng_seed))
    refs=NoisyParitySampler(np.broadcast_to(references,(b,4)),np.random.default_rng(rng_seed+1))
    risk_cfg=RiskConfig(shots=budget,**cfg)
    start=time.perf_counter()
    f,audit=run_policy(sampler,refs,bank['values'].reshape(b*h,n,-1),c.beta,policy,risk_cfg,heads=h)
    logits,attention=classify(model,bank,f)
    elapsed=time.perf_counter()-start
    ideal_attention=softmax(c.beta*bank['ideal'])
    output=np.einsum('bhn,bhnd->bhd',attention,bank['values'])
    ideal_output=np.einsum('bhn,bhnd->bhd',ideal_attention,bank['values'])
    error=((output-ideal_output)**2).sum(-1).mean(-1)
    kl=(ideal_attention*(np.log(np.maximum(ideal_attention,1e-15))-np.log(np.maximum(attention,1e-15)))).sum(-1).mean(-1)
    p=probabilities(logits)
    metrics=classification_metrics(bank['labels'],p)
    metrics.update({'seed':c.seed,'noise':label,'policy':policy,'budget':budget,'repeat':repeat,'phase':phase,
        'output_mse':float(error.mean()),'attention_kl':float(kl.mean()),
        'fidelity_mse':float(np.mean((f.reshape(b,h,n)-bank['ideal'])**2)),
        'clean_disagreement':float(np.mean(logits.argmax(1)!=bank['clean_logits'].argmax(1))),
        'logit_mse':float(np.mean((logits-bank['clean_logits'])**2)),
        'allocation_and_classification_seconds':elapsed,'total_shots':sampler.shots+refs.shots,
        'calibration_shots':refs.shots,'measurement_jobs':sampler.measurement_jobs+refs.measurement_jobs,
        'physical_hardware':False,'pennylane_qnode_calls':0,'mode':sampler.mode,
        'probability_calibration':'Uncalibrated logits; AUROC/accuracy do not require temperature fitting',
        'scope':'Noisy final-block CLS attention only; earlier backbone analytical',
        'config':asdict(risk_cfg),'rng_seed':rng_seed})
    if 'mitigation_weights' in audit:
        metrics['mean_mitigation_weight']=float(audit['mitigation_weights'].mean())
        metrics['calibration_failure_fraction']=float(audit['calibration_failed'].mean())
    assert metrics['total_shots']==b*h*n*budget
    np.savez_compressed(path/'predictions.npz',logits=logits,probabilities=p,labels=bank['labels'],
        sample_ids=bank['sample_ids'],conditional_output_error=error,attention_kl=kl,
        estimated_fidelity=f.reshape(b,h,n),**audit)
    json_write(metric_path,metrics)
    return metrics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['pilot','freeze','run'])
    a=parser.parse_args()
    seed_all(902,2)
    spec=json.loads(MANIFEST.read_text())
    freeze=MANIFEST.with_suffix('.frozen.json')
    if a.action=='freeze':
        if freeze.exists():
            raise FileExistsError('Already frozen; do not overwrite')
        if not (ROOT/'pilot_summary.json').exists():
            raise ValueError('Complete validation pilot first')
        json_write(freeze,{'manifest_sha256':sha(MANIFEST),'utc':datetime.now(timezone.utc).isoformat(),
            'gate_sha256':sha('BIAS_AWARE_NOVELTY_GATE.md'),'code_sha256':computational_hashes(),
            'disclosure':'Follow-up on previously inspected dataset; new method settings selected without new test outcomes'})
        snapshot(ROOT/'frozen_source')
        print('Frozen',freeze)
        return
    if a.action=='run':
        frozen=json.loads(freeze.read_text())
        assert frozen['manifest_sha256']==sha(MANIFEST)
        assert frozen['code_sha256']==computational_hashes(),'Computational source changed after freeze'
        seeds,budgets,repeats=spec['seeds'],spec['budgets'],spec['measurement_repeats']
        split,limit,phase='test',None,'test'
        noise_labels=spec['noises']
    else:
        seeds,budgets,repeats=[0],[64,256],[0,1,2]
        split,limit,phase='val',64,'pilot'
        noise_labels=['ideal','historical','stress']
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        snapshot(ROOT/('pilot_source_'+stamp))
    rows=[]
    noises=noise_settings()
    for seed in seeds:
        c,model,folder=checkpoint(seed,allow_train=a.action=='run')
        bank=feature_bank(c,model,folder,split,limit)
        for label in noise_labels:
            means,references=noisy_bank(c,bank,label,noises[label],split)
            for budget in budgets:
                for repeat in repeats:
                    for policy in spec['policies']:
                        row=condition(c,model,bank,means,references,label,policy,budget,repeat,
                                      spec['risk_config'],phase)
                        rows.append(row)
                print(phase,'seed',seed,label,'budget',budget,'conditions',len(rows),flush=True)
    json_write(ROOT/(phase+'_summary.json'),{'conditions':len(rows),'complete':True,
        'primary_unit':'training seed after averaging shot repeats','metrics':[{
            k:r[k] for k in ['seed','noise','policy','budget','repeat','accuracy','auroc','output_mse',
                            'allocation_and_classification_seconds','total_shots','calibration_shots']} for r in rows]})
    print('Completed',phase,len(rows),'conditions',flush=True)


if __name__=='__main__':
    main()
