"""Verify complete image/feature/circuit predictions and report matched outcomes."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import TensorDataset
from .quantum_transfer import (MANIFEST,CACHE,head_config,QuantumTransferViT,
    make_encoder,image_features,training_stats,load_features)
from .quantum_transfer_experiment import ROOT,verify_lock,source_hashes
from .quantum_robust import PennyLaneOverlap
from .quantum_robust_experiment import digest
from .data import load_splits
from .train import predict
from .runtime import seed_all,json_write
from .metrics import probabilities,classification_metrics,bootstrap_seed_mean,paired_predictions,holm
from .analyze import csv_write,paired_seeds,figure_save

OUT=Path('artifacts/pretrained_quantum_accuracy')
REPORT_FILE=Path('PRETRAINED_QUANTUM_ACCURACY_RESULTS.md')


def check_saved(folder,stem,labels):
    metrics=json.loads((folder/(stem+'_metrics.json')).read_text())
    with np.load(folder/(stem+'_predictions.npz')) as z:
        bank={k:z[k].copy() for k in z.files}
    np.testing.assert_array_equal(bank['labels'],labels)
    np.testing.assert_array_equal(bank['sample_ids'],np.arange(len(labels)))
    np.testing.assert_allclose(bank['probabilities'],probabilities(bank['logits'],metrics['temperature']),atol=1e-7)
    measured=classification_metrics(labels,bank['probabilities'])
    for key in ['accuracy','balanced_accuracy','auroc','nll','sensitivity','specificity','ece','brier','f1']:
        assert abs(measured[key]-metrics[key])<1e-12,(folder,key)
    return bank,metrics


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--verify-circuits',action='store_true');a=parser.parse_args()
    spec=json.loads(MANIFEST.read_text());lock=verify_lock(spec)
    meta=json.loads((CACHE/'metadata.json').read_text())
    assert meta['feature_source_sha256']==digest('research/quantum_transfer.py')
    seed_all(993,spec['backbone']['threads'])
    ds,_,_,data=load_splits(head_config(spec,0,'entangled'))
    assert data['sha256']==meta['dataset_sha256']
    raw={split:torch.load(CACHE/(split+'.pt'),weights_only=True) for split in ['train','val','test']}
    for split in raw:
        torch.testing.assert_close(raw[split]['labels'],ds[split].tensors[1],atol=0,rtol=0)
        torch.testing.assert_close(raw[split]['sample_ids'],torch.arange(len(ds[split])),atol=0,rtol=0)
    stats=torch.load(CACHE/'normalization.pt',weights_only=True)
    mean,std=training_stats(raw['train']['features'])
    torch.testing.assert_close(stats['mean'],mean,atol=0,rtol=0)
    torch.testing.assert_close(stats['std'],std,atol=0,rtol=0)
    encoder,transform=make_encoder(spec);fresh={};encoder_check={}
    for split in ['val','test']:
        start=time.perf_counter()
        features=image_features(encoder,transform,ds[split].tensors[0],spec['backbone']['feature_batch_size'])
        torch.testing.assert_close(features,raw[split]['features'][0],atol=1e-5,rtol=1e-5)
        encoder_check[split]={'images':len(features),'max_absolute_feature_difference':float((features-raw[split]['features'][0]).abs().max()),
            'seconds':time.perf_counter()-start,'all_original_pixels_reencoded':True}
        fresh[split]=(features-mean)/std
    del raw
    banks=load_features();seed_all(994,spec['training']['threads']);OUT.mkdir(parents=True,exist_ok=True)
    rows=[];predictions={};checks=[];circuit_predictions={}
    for kernel in spec['kernels']:
        predictions[kernel]=[];circuit_predictions[kernel]=[]
        for seed in spec['seeds']:
            folder=ROOT/f'{kernel}-s{seed}'
            state=torch.load(folder/'best.pt',weights_only=True)
            model=QuantumTransferViT(head_config(spec,seed,kernel),kernel);model.load_state_dict(state['model'])
            for split,stem in [('val','validation'),('test','test')]:
                bank,metrics=check_saved(folder,stem,banks[split]['labels'].numpy())
                z,y,_=predict(model,TensorDataset(fresh[split],banks[split]['labels']),64)
                np.testing.assert_allclose(z,bank['logits'],atol=1e-5,rtol=1e-5)
                if split=='test':
                    predictions[kernel].append(bank['probabilities'])
            rows.append({k:metrics[k] for k in ['kernel','seed','accuracy','auroc','balanced_accuracy',
                'sensitivity','specificity','best_epoch','trainable_parameters','training_seconds']})
            record={'kernel':kernel,'seed':seed,'full_raw_image_inference_verified':True,
                    'checkpoint_sha256':digest(folder/'best.pt'),'predictions_sha256':digest(folder/'test_predictions.npz')}
            if a.verify_circuits and kernel!='dot':
                backend=PennyLaneOverlap(entangled=kernel=='entangled');model.set_circuit_backend(backend)
                start=time.perf_counter()
                z,y,_=predict(model,TensorDataset(fresh['test'],banks['test']['labels']),16)
                np.testing.assert_allclose(z,bank['logits'],atol=1e-5,rtol=1e-5)
                np.testing.assert_array_equal(z.argmax(1),bank['logits'].argmax(1))
                p=probabilities(z,metrics['temperature']);qm=classification_metrics(y,p)
                circuit_predictions[kernel].append(p)
                np.savez_compressed(OUT/f'{kernel}-circuit-s{seed}.npz',logits=z,probabilities=p,labels=y,sample_ids=np.arange(len(y)))
                record['quantum']={'qnode_calls':backend.qnode_calls,'circuit_settings':backend.circuit_settings,
                    'qubits':4,'shots':0,'physical_hardware':False,'seconds':time.perf_counter()-start,
                    'accuracy':qm['accuracy'],'max_absolute_logit_difference':float(np.max(abs(z-bank['logits'])))}
                print('Verified full quantum classifier',kernel,seed,record['quantum'],flush=True)
            checks.append(record)
    assert len({r['trainable_parameters'] for r in rows})==1
    for split,stem in [('val','validation'),('test','test')]:
        bank,linear=check_saved(ROOT/'linear',stem,banks[split]['labels'].numpy())
        with np.load(ROOT/'linear/coefficients.npz') as coefficients:
            s=(fresh[split].mean(1).numpy()@coefficients['coef'].T+coefficients['intercept']).ravel()
        logits=np.stack([-.5*s,.5*s],1).astype(np.float32)
        np.testing.assert_allclose(logits,bank['logits'],atol=1e-5,rtol=1e-5)
    linear_p=bank['probabilities'];labels=banks['test']['labels'].numpy()
    summary=[];ensembles={};quantum_ensembles={}
    for kernel in spec['kernels']:
        part=[r for r in rows if r['kernel']==kernel]
        entry={'kernel':kernel,'seeds':len(part)}
        for metric in ['accuracy','auroc','balanced_accuracy','sensitivity','specificity']:
            entry[metric]=bootstrap_seed_mean([r[metric] for r in part])
        ensembles[kernel]=np.mean(predictions[kernel],axis=0)
        entry['ensemble']=classification_metrics(labels,ensembles[kernel]);summary.append(entry)
        np.savez_compressed(OUT/f'{kernel}-ensemble.npz',probabilities=ensembles[kernel],labels=labels,sample_ids=np.arange(len(labels)))
        if circuit_predictions[kernel]:
            qp=np.mean(circuit_predictions[kernel],axis=0)
            np.testing.assert_array_equal(qp.argmax(1),ensembles[kernel].argmax(1))
            quantum_ensembles[kernel]=classification_metrics(labels,qp)
    contrasts=[]
    for control in ['product','dot']:
        for metric in ['accuracy','auroc']:
            difference=[next(r[metric] for r in rows if r['kernel']=='entangled' and r['seed']==seed)
                        -next(r[metric] for r in rows if r['kernel']==control and r['seed']==seed) for seed in spec['seeds']]
            contrasts.append({'control':control,'metric':metric,**paired_seeds(difference)})
    for r,p in zip(contrasts,holm([r['sign_flip_exact_p'] for r in contrasts])):
        r['holm_p']=p
    results={'summary':summary,'linear_control':linear,'paired_seed_contrasts':contrasts,
        'ensemble_example_comparisons':{name:paired_predictions(labels,ensembles['entangled'],p)
            for name,p in [('product',ensembles['product']),('dot',ensembles['dot']),('linear',linear_p)]},
        'quantum_circuit_ensembles':quantum_ensembles,'pretraining':'ImageNet-1K','exploratory':True,
        'any_quantum_ensemble_over_90':any(r['ensemble']['accuracy']>.9 for r in summary if r['kernel']!='dot')}
    json_write(OUT/'summary.json',results);csv_write(OUT/'per_seed.csv',rows)
    json_write(OUT/'verification.json',{'source_sha256':source_hashes(),'analysis_sha256':digest(__file__),
        'manifest_sha256':digest(MANIFEST),'data_sha256':data['sha256'],'weights_sha256':lock['weights_sha256'],
        'feature_cache_hashes_verified':True,'training_only_normalization_verified':True,
        'encoder':encoder_check,'checkpoints':checks,'all_full_splits_verified':True,
        'full_quantum_test_verified':a.verify_circuits,'physical_hardware':False})
    report(spec,meta,summary,linear,contrasts,checks,quantum_ensembles)
    print(json.dumps({'accuracy':{r['kernel']:{'single_mean':r['accuracy']['mean'],
        'ensemble':r['ensemble']['accuracy']} for r in summary},'linear':linear['accuracy'],
        'quantum_ensemble_accuracies':{k:v['accuracy'] for k,v in quantum_ensembles.items()}},indent=2))


def report(spec,meta,summary,linear,contrasts,checks,quantum_ensembles):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6.5,4))
    ax.errorbar(range(3),[100*r['accuracy']['mean'] for r in summary],
        yerr=[100*r['accuracy']['sd'] for r in summary],fmt='o',capsize=4,label='Single-model mean +/- seed SD')
    ax.scatter(range(3),[100*r['ensemble']['accuracy'] for r in summary],marker='s',label='Five-model ensemble')
    ax.axhline(90,linestyle='--',color='gray',label='90% target')
    ax.axhline(100*linear['accuracy'],linestyle=':',color='purple',label='Linear control')
    ax.set_xticks(range(3),[r['kernel'] for r in summary]);ax.set_ylabel('Test accuracy (%)')
    ax.legend(loc='lower center',bbox_to_anchor=(.5,1.02),ncol=2,fontsize=8,frameon=False)
    figure_save(fig,OUT,'accuracy')
    resolution=spec.get('native_image_size',28)
    lines=['# Pretrained image encoder with quantum fidelity attention','',
        f"Frozen ImageNet-1K ResNet-18 features feed four-qubit entangled or product-fidelity attention. The input uses the official {resolution}-pixel PneumoniaMNIST images with the weights' resize-to-256/center-crop-to-224 preprocessing. These results are exploratory because the official test cohort was examined earlier.",'',
        '| Kernel | Single-model accuracy, mean +/- SD | Mean AUROC +/- SD | Five-model ensemble accuracy |',
        '|---|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['kernel']} | {100*r['accuracy']['mean']:.2f}% +/- {100*r['accuracy']['sd']:.2f}% | {r['auroc']['mean']:.5f} +/- {r['auroc']['sd']:.5f} | {100*r['ensemble']['accuracy']:.2f}% |")
    lines.append(f"| Classical linear control | {100*linear['accuracy']:.2f}% (one deterministic classifier) | {linear['auroc']:.5f} | Not an ensemble |")
    met=[r for r in summary if r['kernel']!='dot' and r['ensemble']['accuracy']>.9]
    lines+=['',f"The >90% quantum-ensemble target is {'achieved' if met else 'not achieved'} on all 624 official test images. All fifteen neural checkpoints and the linear control are retained. Ensembles average five validation-calibrated probability arrays with equal weights; they are one prediction set, not five repetitions.",'',
        '## Quantum algorithms and evidence','',
        '- Entangled attention prepares four RY-encoded qubits, applies three CNOTs, then RZ(theta) and RY(theta/2) re-uploading per wire. Product attention prepares four independent RY-encoded qubits. Every attention overlap is a quantum-state fidelity.',
        '- Training executes differentiable gates on four-qubit statevectors. Independent inference obtains the zero-state probability of the PennyLane compute-uncompute circuit U(k)^dagger U(q). Neither execution mode uses a physical QPU.',
        '- The encoder, feature projection, value aggregation, classifier and optimizer are classical. The quantum circuits supply query/key attention similarities. Gradients through the quantum path and agreement with the PennyLane classifier are tested.',
        f"- Frozen encoder: {meta['frozen_encoder_parameters']:,} parameters. All three neural heads have {json.loads((ROOT/'entangled-s0/test_metrics.json').read_text())['trainable_parameters']:,} trainable parameters. The linear control fits 513 coefficients on the same frozen features."]
    if quantum_ensembles:
        qc=[r['quantum'] for r in checks if 'quantum' in r]
        lines.append(f"- Complete-test verification: {sum(r['qnode_calls'] for r in qc):,} batched QNode calls and {sum(r['circuit_settings'] for r in qc):,} four-qubit circuit settings across ten quantum classifiers. Shots: zero, since these are ideal probabilities. Every saved classifier prediction is reproduced from original pixels, and circuit decisions match the gate-statevector decisions.")
        for kernel,m in quantum_ensembles.items():
            correct=sum(m['confusion_matrix'][i][i] for i in range(2))
            lines.append(f"- Independently executed {kernel} quantum ensemble: {100*m['accuracy']:.2f}% ({correct}/624 correct), AUROC {m['auroc']:.5f}.")
    lines+=['','## Training, selection and attribution','',
        f"- Four spatial ResNet tokens, a learned projection to 32 dimensions, two attention blocks and two heads. All neural kernels share the same {spec['training']['epochs']}-epoch AdamW recipe, class-weighted cross-entropy, feature MixUp(alpha=0.1), training-only flip views, warmup/cosine learning rate and EMA(decay=0.95).",
        '- Feature normalization fits training views only. The frozen encoder stays in evaluation mode. All 4,708 training examples, 524 validation examples and 624 test examples are retained; the test set supplies scoring labels only.',
        '- Best checkpoints minimize class-balanced validation NLL. Temperatures fit validation labels only. The linear control selects C from 0.01/0.1/1/10 by class-balanced validation NLL; every candidate score is preserved.',
        '- The previous 88.94% reference was a small model ensemble with target-prior adjustment. The previous robust product ensemble reached 89.74%. Neither used ImageNet pretraining. The new experiment adds external pretraining and uses ordinary argmax without target adaptation; its comparison to the older models changes the model/data regime.',
        '- Transfer learning, entangling attention and re-uploading have prior art. An accuracy increase is not proof of algorithmic novelty, quantum speedup, clinical validity or publication readiness. A quantum-specific accuracy benefit must be assessed against the equally treated dot and linear controls.',
        '- The product ablation removes both the CNOTs and the second encoding stage. This contrast does not isolate entanglement alone. No finite-shot or noisy classifier accuracy is claimed here.']
    if resolution==28:
        lines.append('- Upsampling the 28-pixel source to 224 does not recover higher-resolution source-image information.')
    else:
        lines.append('- Native 224-pixel images are a separate resolution regime from the old 28-pixel benchmark. The official checksum, split labels and same-index image correspondence are audited; results must be labeled by resolution.')
    lines+=['','## Paired seed comparisons','',
        '| Entangled minus control | Metric | Mean difference | Seed-bootstrap 95% interval | Holm p |',
        '|---|---|---:|---|---:|']
    for r in contrasts:
        scale=100 if r['metric']=='accuracy' else 1;lo,hi=r['effect']['ci95_seed_bootstrap']
        lines.append(f"| {r['control']} | {r['metric']} | {scale*r['effect']['mean']:+.5f} | [{scale*lo:+.5f}, {scale*hi:+.5f}] | {r['holm_p']:.5f} |")
    lines+=['','Accuracy effects are percentage points. Intervals cover five-seed variability on this fixed cohort. The minimum nonzero two-sided exact sign-flip p with five seeds is 0.0625. Four seed comparisons use Holm correction. Ensemble paired-example McNemar/DeLong tests are exploratory and retained separately.','',
        '## Reproduction','',
        '[Prior-art gate](PRETRAINED_QUANTUM_NOVELTY_GATE.md). The manifest, weights, feature caches, checkpoints and saved predictions have retained hashes. `verification.json` independently checks full image features, training-only normalization, every checkpoint, metrics and circuit inference. The original compiled paper and slides describe the earlier study.','',
        f"Evidence: `{OUT.as_posix()}/summary.json`, `per_seed.csv`, `verification.json` and `accuracy.pdf`.",'',
        '```powershell']
    if resolution==28:
        lines+=['python -m research.quantum_transfer_experiment run',
                'python -m research.quantum_transfer_analyze --verify-circuits']
    else:
        lines+=['python -m research.quantum_native_accuracy run',
                'python -m research.quantum_native_accuracy analyze']
    lines+=['python tests/run_suite.py','```','',
        'Reuse the existing frozen manifest; do not refreeze or tune settings from these test outcomes.']
    REPORT_FILE.write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
