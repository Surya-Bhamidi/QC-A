"""Verify and report the frozen hybrid accuracy experiment from retained evidence."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np
import torch
from .analyze import paired_seeds,csv_write
from .config import Config
from .data import load_splits
from .hybrid_accuracy import Architecture,ConvTokenClassifier,predict,MANIFEST,ROOT,digest
from .metrics import classification_metrics,probabilities,bootstrap_seed_mean,holm
from .runtime import json_write,seed_all

OUT=Path('artifacts/hybrid_accuracy')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--verify-inference',action='store_true')
    args=parser.parse_args()
    spec=json.loads(MANIFEST.read_text())
    frozen=json.loads(MANIFEST.with_suffix('.frozen.json').read_text())
    assert frozen['manifest_sha256']==digest(MANIFEST)
    for file,expected in frozen['source_sha256'].items():
        assert digest(file)==expected,(file,'Frozen source changed')
    seed_all(811,spec['training']['threads'])
    datasets,_,classes,data_meta=load_splits(Config(dataset=spec['dataset']))
    rows=[]
    hashes={}
    validation_mean={}
    for kernel in spec['kernels']:
        for seed in spec['final_seeds']:
            arch=Architecture(model=kernel,seed=seed,**spec['architecture'])
            matches=[]
            for path in ROOT.glob(f'{kernel}-s{seed}-*/config.json'):
                config=json.loads(path.read_text())
                if config=={'architecture':arch.__dict__,'training':spec['training']}:
                    matches.append(path.parent)
            assert len(matches)==1,(kernel,seed,matches)
            folder=matches[0]
            assert json.loads((folder/'data.json').read_text())['sha256']==data_meta['sha256']
            model=ConvTokenClassifier(arch,classes)
            state=torch.load(folder/'best.pt',weights_only=True)
            model.load_state_dict(state['model']);model.eval()
            row={'kernel':kernel,'seed':seed,'run':str(folder),
                 'parameters':sum(p.numel() for p in model.parameters()),'best_epoch':state['epoch']}
            for split,stem in [('val','validation'),('test','test')]:
                path=folder/(stem+'_predictions.npz')
                saved=json.loads((folder/(stem+'_metrics.json')).read_text())
                with np.load(path) as z:
                    np.testing.assert_array_equal(z['sample_ids'],np.arange(len(datasets[split])))
                    np.testing.assert_array_equal(z['labels'],datasets[split].tensors[1].numpy())
                    np.testing.assert_allclose(probabilities(z['logits'],saved['temperature']),z['probabilities'],atol=1e-7)
                    recalculated=classification_metrics(z['labels'],z['probabilities'])
                    for metric in ['accuracy','auroc','balanced_accuracy','nll','sensitivity','specificity']:
                        assert abs(recalculated[metric]-saved[metric])<1e-12,(folder,metric)
                        row[stem+'_'+metric]=saved[metric]
                    if args.verify_inference:
                        logits,labels=predict(model,datasets[split],spec['training']['batch_size'])
                        np.testing.assert_array_equal(labels,z['labels'])
                        np.testing.assert_allclose(logits,z['logits'],atol=2e-6,rtol=1e-6)
                hashes[str(path)]=digest(path)
                hashes[str(folder/(stem+'_metrics.json'))]=digest(folder/(stem+'_metrics.json'))
            hashes[str(folder/'best.pt')]=digest(folder/'best.pt')
            rows.append(row)
    assert len({r['parameters'] for r in rows})==1,'Unmatched fidelity/dot parameters'
    summary={}
    for kernel in spec['kernels']:
        part=[r for r in rows if r['kernel']==kernel]
        summary[kernel]={metric:bootstrap_seed_mean([r[metric] for r in part])
                        for metric in ['test_accuracy','test_auroc','test_balanced_accuracy','validation_accuracy','validation_auroc']}
    contrasts=[]
    for metric in ['test_accuracy','test_auroc']:
        differences=[next(r[metric] for r in rows if r['kernel']=='fidelity' and r['seed']==s)
                    -next(r[metric] for r in rows if r['kernel']=='dot' and r['seed']==s)
                    for s in spec['final_seeds']]
        contrasts.append({'metric':metric,'contrast':'fidelity minus matched dot',**paired_seeds(differences)})
    for row,p in zip(contrasts,holm([r['sign_flip_exact_p'] for r in contrasts])):
        row['holm_p_two_metrics']=p
    OUT.mkdir(parents=True,exist_ok=True)
    csv_write(OUT/'per_seed.csv',rows)
    json_write(OUT/'summary.json',{'summary':summary,'paired_contrasts':contrasts,'seeds':5,
        'test_images_per_run':len(datasets['test']),'parameters_per_model':rows[0]['parameters'],
        'test_previously_inspected':True,'threshold':'.5; ordinary class argmax',
        'goal_above_90_percent_met':summary['fidelity']['test_accuracy']['mean']>.9})
    json_write(OUT/'verification.json',{'utc':datetime.now(timezone.utc).isoformat(),'runs':len(rows),
        'all_saved_prediction_metrics_and_sample_ids_verified':True,
        'full_checkpoint_inference_reproduced':args.verify_inference,
        'manifest_sha256':digest(MANIFEST),'data_sha256':data_meta['sha256'],
        'analysis_sha256':digest(__file__),'evidence_sha256':hashes})
    baseline=[]
    for s in spec['final_seeds']:
        paths=list(Path('runs/central').glob(f'pneumoniamnist-fidelity-s{s}-*/test_metrics.json'))
        assert len(paths)==1
        baseline.append(json.loads(paths[0].read_text())['accuracy'])
    original=bootstrap_seed_mean(baseline)
    text=['# Hybrid accuracy experiment: completed results','',
        'This frozen exploratory comparison tests a convolutional image encoder followed by fidelity or matched dot attention. Every model uses the same encoder, dimensions, training recipe and complete official splits. The target was greater than 90% test accuracy. The official test split was previously inspected, so this is not an independent confirmatory experiment.','',
        '## Result','',
        '| Architecture | Test accuracy, mean +/- seed SD | Test AUROC, mean +/- seed SD | Validation accuracy, mean +/- seed SD |',
        '|---|---:|---:|---:|']
    for kernel in spec['kernels']:
        r=summary[kernel]
        text.append(f"| Convolutional encoder + {kernel} attention | {100*r['test_accuracy']['mean']:.2f}% +/- {100*r['test_accuracy']['sd']:.2f}% | {r['test_auroc']['mean']:.5f} +/- {r['test_auroc']['sd']:.5f} | {100*r['validation_accuracy']['mean']:.2f}% +/- {100*r['validation_accuracy']['sd']:.2f}% |")
    f=summary['fidelity']['test_accuracy']
    text+=['',f"The fidelity hybrid's five-seed mean test accuracy is {100*f['mean']:.2f}%, with seed-bootstrap 95% interval [{100*f['ci95_seed_bootstrap'][0]:.2f}%, {100*f['ci95_seed_bootstrap'][1]:.2f}%]. The original smaller fidelity model scored {100*original['mean']:.2f}% +/- {100*original['sd']:.2f}%. This architecture change does not demonstrate the requested accuracy leap.",'']
    for r in contrasts:
        effect=r['effect'];scale=100 if r['metric']=='test_accuracy' else 1
        lo,hi=effect['ci95_seed_bootstrap']
        unit=' percentage points' if scale==100 else ''
        text.append(f"Fidelity minus matched dot, {r['metric']}: {scale*effect['mean']:+.5f}{unit}; seed-bootstrap 95% interval [{scale*lo:+.5f}, {scale*hi:+.5f}]; exact paired sign-flip p={r['sign_flip_exact_p']:.5f}, Holm p across two metrics={r['holm_p_two_metrics']:.5f}.")
        text.append('')
    text+=['The large validation-to-test gap is observed in both architectures. Label proportions also differ: 74.2% class 1 in training/validation and 62.5% in test. These observations do not by themselves establish the cause of the gap or prove that all future methods must fail. Greater than 90% remains a research target, not an achieved result.','',
        '## Protocol and verification','',
        f"- Five seeds per kernel, ten complete 624-image test predictions. Both kernels have {rows[0]['parameters']:,} trainable parameters.",
        '- Three convolutional layers (16/32/48 channels), GroupNorm/GELU, 4x4 pooled tokens, CLS token, two attention blocks, three heads and four query/key features per head.',
        '- AdamW, 30 epochs, learning rate 0.0006, cosine schedule, class-weighted training cross entropy; checkpoint selection by unweighted validation NLL. Temperature fits only validation labels and does not change argmax accuracy.',
        '- The validation pilot used seeds 0-2. The design was frozen before hybrid test inference. No hyperparameters or decision thresholds changed after hybrid test outcomes.',
        '- Inference uses analytical product fidelity or dot attention; quantum circuit calls and measurement shots are zero. No quantum advantage or methodological novelty is claimed.',
        '- The interrupted run was continued using the existing frozen source. This runner skips completed test conditions but retrains an incomplete condition deterministically; it does not restore partial optimizer state.',
        f"- Saved logits/probabilities, labels, sample IDs, metrics, parameter matching and frozen hashes are checked. Full inference from all ten best checkpoints {'was reproduced successfully' if args.verify_inference else 'has not been repeated by this analysis run'}.",
        '', '## Reproduction','', '```powershell','python -m research.hybrid_accuracy run',
        'python -m research.hybrid_analyze --verify-inference','python tests/run_suite.py','```','',
        '[Frozen design](configs/hybrid_accuracy.frozen.json), [per-seed results](artifacts/hybrid_accuracy/per_seed.csv), [paired statistics](artifacts/hybrid_accuracy/summary.json), [evidence audit](artifacts/hybrid_accuracy/verification.json).','',
        'The original paper and presentation retain their earlier experimental scope. This report records the later accuracy experiment.']
    Path('HYBRID_ACCURACY_RESULTS.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps({'summary':summary,'paired_contrasts':contrasts,'inference_verified':args.verify_inference},indent=2))


if __name__=='__main__':
    main()
