"""Complete matched analysis and full-test compute-uncompute verification."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .analyze import paired_seeds,csv_write,figure_save
from .data import load_splits
from .metrics import classification_metrics,probabilities,bootstrap_seed_mean,paired_predictions,holm
from .runtime import seed_all,json_write
from .train import predict
from .quantum_robust import QuantumRobustViT,PennyLaneOverlap
from .quantum_robust_experiment import ROOT,MANIFEST,digest,hashes,model_config,run_folder

OUT=Path('artifacts/quantum_robust_accuracy')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--verify-quantum',action='store_true')
    a=parser.parse_args()
    spec=json.loads(MANIFEST.read_text())
    lock=json.loads(MANIFEST.with_suffix('.frozen.json').read_text())
    assert lock['source_sha256']==hashes() and lock['manifest_sha256']==digest(MANIFEST)
    assert lock['pilot_summary_sha256']==digest(ROOT/'pilot_summary.json')
    pilot=json.loads((ROOT/'pilot_summary.json').read_text())['rows']
    recipe=next(r for r in spec['recipes'] if r['name']==lock['selected_recipe'])
    seed_all(777,spec['training']['threads'])
    ds,_,_,meta=load_splits(model_config(spec,0,'entangled'))
    rows=[];banks={};verification=[]
    OUT.mkdir(parents=True,exist_ok=True)
    for kernel in spec['kernels']:
        banks[kernel]=[]
        for seed in spec['final_seeds']:
            folder=run_folder(spec,recipe,seed,kernel)
            assert json.loads((folder/'data.json').read_text())['sha256']==meta['sha256']
            m=json.loads((folder/'test_metrics.json').read_text())
            model=QuantumRobustViT(model_config(spec,seed,kernel),kernel)
            state=torch.load(folder/'best.pt',weights_only=True);model.load_state_dict(state['model']);model.eval()
            for split,stem in [('val','validation'),('test','test')]:
                saved=json.loads((folder/(stem+'_metrics.json')).read_text())
                with np.load(folder/(stem+'_predictions.npz')) as z:
                    np.testing.assert_array_equal(z['sample_ids'],np.arange(len(ds[split])))
                    np.testing.assert_array_equal(z['labels'],ds[split].tensors[1].numpy())
                    np.testing.assert_allclose(probabilities(z['logits'],saved['temperature']),z['probabilities'],atol=1e-7)
                    measured=classification_metrics(z['labels'],z['probabilities'])
                    for metric in ['accuracy','auroc','balanced_accuracy','nll']:
                        assert abs(saved[metric]-measured[metric])<1e-12,(kernel,seed,metric)
                    logits,labels,_=predict(model,ds[split],64)
                    np.testing.assert_allclose(logits,z['logits'],atol=3e-6,rtol=3e-6)
                    if split=='test':
                        banks[kernel].append({'logits':z['logits'],'probabilities':z['probabilities'],
                                              'labels':z['labels'],'sample_ids':z['sample_ids']})
            rows.append({k:m[k] for k in ['kernel','seed','recipe','accuracy','auroc','balanced_accuracy',
                                        'sensitivity','specificity','best_epoch','parameters','training_seconds']})
            check={'kernel':kernel,'seed':seed,'checkpoint_sha256':digest(folder/'best.pt'),
                   'test_predictions_sha256':digest(folder/'test_predictions.npz'),'all_metrics_and_inference_verified':True}
            if a.verify_quantum and kernel!='dot':
                backend=PennyLaneOverlap(entangled=kernel=='entangled',chunk_size=4096)
                model.set_circuit_backend(backend)
                start=time.perf_counter()
                qz,qy,_=predict(model,ds['test'],16)
                qp=probabilities(qz,m['temperature'])
                max_gap=float(np.max(abs(qz-banks[kernel][-1]['logits'])))
                np.testing.assert_allclose(qz,banks[kernel][-1]['logits'],atol=1e-5,rtol=1e-5)
                qm=classification_metrics(qy,qp)
                np.savez_compressed(OUT/f'{kernel}-quantum-circuit-seed-{seed}.npz',logits=qz,probabilities=qp,
                                    labels=qy,sample_ids=np.arange(len(qy)))
                check['quantum_verification']={'actual_pennylane_qnode_calls':backend.qnode_calls,
                    'batched_compute_uncompute_circuit_settings':backend.circuit_settings,'logical_qubits':4,
                    'shots':0,'physical_hardware':False,'backend':'ideal_PennyLane_default_qubit_compute_uncompute',
                    'max_absolute_logit_difference':max_gap,'seconds':time.perf_counter()-start,
                    'accuracy':qm['accuracy'],'auroc':qm['auroc']}
                print('Full test quantum circuit verification',kernel,'seed',seed,check['quantum_verification'],flush=True)
            verification.append(check)
    assert len({r['parameters'] for r in rows})==1
    summary=[];ensembles={}
    labels=ds['test'].tensors[1].numpy()
    for kernel in spec['kernels']:
        part=[r for r in rows if r['kernel']==kernel]
        record={'kernel':kernel,'seeds':len(part)}
        for metric in ['accuracy','auroc','balanced_accuracy','sensitivity','specificity']:
            stats=bootstrap_seed_mean([r[metric] for r in part]);record[metric]=stats
        ep=np.mean([r['probabilities'] for r in banks[kernel]],axis=0)
        ensembles[kernel]=ep
        record['ensemble']=classification_metrics(labels,ep)
        np.savez_compressed(OUT/f'{kernel}-ensemble.npz',probabilities=ep,labels=labels,sample_ids=np.arange(len(labels)))
        summary.append(record)
    contrasts=[]
    for baseline in ['product','dot']:
        for metric in ['accuracy','auroc']:
            differences=[next(r[metric] for r in rows if r['kernel']=='entangled' and r['seed']==s)
                        -next(r[metric] for r in rows if r['kernel']==baseline and r['seed']==s)
                        for s in spec['final_seeds']]
            contrasts.append({'baseline':baseline,'metric':metric,**paired_seeds(differences)})
    for r,p in zip(contrasts,holm([r['sign_flip_exact_p'] for r in contrasts])):
        r['holm_p_four_contrasts']=p
    ensemble_comparisons={kernel:paired_predictions(labels,ensembles['entangled'],ensembles[kernel]) for kernel in ['product','dot']}
    quantum_ensembles={}
    if a.verify_quantum:
        for kernel in ['entangled','product']:
            qp=[]
            for seed in spec['final_seeds']:
                with np.load(OUT/f'{kernel}-quantum-circuit-seed-{seed}.npz') as z:
                    qp.append(z['probabilities'])
            quantum_ensembles[kernel]=classification_metrics(labels,np.mean(qp,axis=0))
    json_write(OUT/'summary.json',{'selected_recipe':recipe,'summary':summary,'paired_seed_contrasts':contrasts,
        'ensemble_paired_example_tests':ensemble_comparisons,'quantum_circuit_ensembles':quantum_ensembles,
        'goal_over_90_ensemble':next(r['ensemble']['accuracy'] for r in summary if r['kernel']=='entangled')>.9,
        'any_quantum_ensemble_over_90':any(r['ensemble']['accuracy']>.9 for r in summary if r['kernel']!='dot'),
        'exploratory':True,'physical_hardware':False})
    json_write(OUT/'verification.json',{'manifest_sha256':digest(MANIFEST),'source_sha256':hashes(),
        'analysis_sha256':digest(__file__),'data_sha256':meta['sha256'],'checks':verification,
        'all_full_splits_verified':True,'full_pennylane_quantum_test_verification':a.verify_quantum})
    csv_write(OUT/'per_seed.csv',rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6,4))
    ax.errorbar(range(3),[100*r['accuracy']['mean'] for r in summary],yerr=[100*r['accuracy']['sd'] for r in summary],
                fmt='o',capsize=4,label='Single-model mean +/- seed SD')
    ax.scatter(range(3),[100*r['ensemble']['accuracy'] for r in summary],marker='s',label='Five-model ensemble')
    ax.axhline(90,color='gray',linestyle='--',label='90% target')
    ax.set_xticks(range(3),[r['kernel'] for r in summary]);ax.set_ylabel('Test accuracy (%)')
    ax.legend(fontsize=8);figure_save(fig,OUT,'accuracy')
    lines=['# Quantum circuit attention: robust accuracy experiment','',
        'This experiment retains a quantum feature circuit and compares it with product-state fidelity and matched dot attention. It uses four-qubit gate simulation for differentiable training and an independent PennyLane compute-uncompute backend for quantum inference verification. All results are exploratory because the official test set was examined in earlier experiments.','',
        f"The validation pilot selected the shared **{recipe['name']}** recipe. Every kernel receives the same training settings and five seeds. The previous 88.94% result used an ensemble plus target-prior adjustment; this experiment uses ordinary argmax and an equal-weight ensemble, without target adaptation.",'',
        '| Kernel | Single-model accuracy, mean +/- seed SD | AUROC, mean +/- seed SD | Five-model ensemble accuracy |',
        '|---|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['kernel']} | {100*r['accuracy']['mean']:.2f}% +/- {100*r['accuracy']['sd']:.2f}% | {r['auroc']['mean']:.5f} +/- {r['auroc']['sd']:.5f} | {100*r['ensemble']['accuracy']:.2f}% |")
    ent=next(r for r in summary if r['kernel']=='entangled')
    goal=ent['ensemble']['accuracy']>.9
    lines+=['',f"The entangled ensemble {'exceeds' if goal else 'does not exceed'} 90% on all 624 test images: {100*ent['ensemble']['accuracy']:.2f}%. The single-model mean is {100*ent['accuracy']['mean']:.2f}%. Ensemble accuracy is one prediction set costing five model inferences, not five independent repetitions.",'',
        '## Quantum algorithms and execution','',
        '- Four RY encoding gates, three CNOT entanglers, then data-dependent RZ and RY(theta/2) re-uploading per wire. Fidelity is the zero-state probability after compute-uncompute U(k)^dagger U(q). The product ablation uses four RY gates without entanglement.',
        '- Gate-based statevector simulation trains query/key projections by backpropagation. A classical computer simulates the quantum algorithm. No physical QPU or quantum speedup is implied.',
        f"- Parameter counts are matched: {rows[0]['parameters']:,} per kernel. The local patch encoder, classical value aggregation and classification head are shared."]
    if quantum_ensembles:
        qs=[r['quantum_verification'] for r in verification if 'quantum_verification' in r]
        lines.append(f"- Complete-test PennyLane verification for both quantum maps: {sum(r['actual_pennylane_qnode_calls'] for r in qs):,} batched QNode calls, {sum(r['batched_compute_uncompute_circuit_settings'] for r in qs):,} four-qubit overlap settings, zero measurement shots (ideal probabilities). All saved statevector and PennyLane logits agree within numerical tolerance.")
        for kernel,qm in quantum_ensembles.items():
            lines.append(f"- Independently executed {kernel} quantum-circuit ensemble accuracy: {100*qm['accuracy']:.2f}%.")
    else:
        lines.append('- Complete-test PennyLane verification has not yet been run by this analysis invocation.')
    lines+=['','## Validation recipe selection','',
            '| Recipe | Entangled balanced NLL | Product balanced NLL | Dot balanced NLL | Shared mean |',
            '|---|---:|---:|---:|---:|']
    for candidate in spec['recipes']:
        scores=[np.mean([r['balanced_nll'] for r in pilot if r['recipe']==candidate['name'] and r['kernel']==kernel])
                for kernel in spec['kernels']]
        assert abs(np.mean(scores)-lock['pilot_recipe_scores'][candidate['name']])<1e-12
        lines.append(f"| {candidate['name']} | {scores[0]:.5f} | {scores[1]:.5f} | {scores[2]:.5f} | {np.mean(scores):.5f} |")
    lines+=['','These are validation losses averaged over the two pilot seeds. Lower is better. All twelve pilot conditions are retained in `runs/quantum_robust_accuracy/pilot_summary.json`; the shared mean selects the recipe. The additional planned AdamW seeds were trained on training/validation data in parallel while the pilot finished and did not enter recipe selection.']
    lines+=['','## Changes and selection','',
        '- Training-only per-image rotation up to 10 degrees, scale and translation, border padding, horizontal flip, brightness/contrast jitter and MixUp(alpha=0.2).',
        '- Twenty-four training epochs, AdamW with weight decay 0.001, warmup and cosine learning rate, and EMA(decay=0.98). The SAM candidate uses radius 0.05 and two gradient passes.',
        '- The best EMA checkpoint minimizes class-balanced validation NLL. One shared optimizer recipe was selected by validation scores across two pilot seeds and all three kernels, then frozen before this experiment\'s test inference.',
        '- SAM, augmentation, EMA and data re-uploading are established methods. Higher accuracy alone is not proof of algorithmic novelty. A fidelity-specific contribution requires outperforming the equally treated controls.',
        '', '## Paired comparisons','',
        '| Entangled minus control | Metric | Mean difference | Seed-bootstrap 95% interval | Holm p |',
        '|---|---|---:|---|---:|']
    for r in contrasts:
        effect=r['effect'];scale=100 if r['metric']=='accuracy' else 1;lo,hi=effect['ci95_seed_bootstrap']
        lines.append(f"| {r['baseline']} | {r['metric']} | {scale*effect['mean']:+.5f} | [{scale*lo:+.5f}, {scale*hi:+.5f}] | {r['holm_p_four_contrasts']:.5f} |")
    lines+=['','Accuracy differences are percentage points. Bootstrap intervals cover model-seed variability on this fixed cohort. With five seeds, the smallest nonzero two-sided exact sign-flip p is 0.0625. Four seed comparisons use Holm correction. Ensemble paired-example tests are reported separately and are exploratory.','',
        '## Evidence and reproduction','',
        '[Design and prior art](QUANTUM_ACCURACY_PLAN.md), [frozen manifest](configs/quantum_robust_accuracy.frozen.json), [per-seed results](artifacts/quantum_robust_accuracy/per_seed.csv), [statistics](artifacts/quantum_robust_accuracy/summary.json), [verification](artifacts/quantum_robust_accuracy/verification.json), [figure](artifacts/quantum_robust_accuracy/accuracy.pdf).','',
        '```powershell','python -m research.quantum_robust_experiment run',
        'python -m research.quantum_robust_analyze --verify-quantum','python tests/run_suite.py','```','',
        'Do not refreeze or change the selected settings based on these test outcomes. The original paper and slide PDFs document the earlier study. This file records the later quantum-preserving accuracy experiment.']
    Path('QUANTUM_ROBUST_ACCURACY_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'summary':summary,'quantum_ensembles':quantum_ensembles,'entangled_goal_90_met':goal},indent=2))


if __name__=='__main__':
    main()
