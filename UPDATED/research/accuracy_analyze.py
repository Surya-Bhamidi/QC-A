"""Summarize the frozen exploratory accuracy follow-up from saved predictions."""
import json
from pathlib import Path
import numpy as np
from .metrics import bootstrap_seed_mean
from .analyze import paired_seeds
from .runtime import json_write


def main():
    root=Path('runs/accuracy_followup')
    manifest=Path('configs/accuracy_followup.json')
    spec=json.loads(manifest.read_text())
    rows=[]
    for model in spec['models']:
        for seed in spec['seeds']:
            paths=list(root.glob(f'pneumoniamnist-{model}-s{seed}-*/test_metrics.json'))
            if len(paths)!=1:
                raise ValueError(f'Expected one saved run for {model} seed {seed}; got {paths}')
            m=json.loads(paths[0].read_text())
            rows.append({'model':model,'seed':seed,'run':str(paths[0].parent),
                         **{k:m[k] for k in ['accuracy','balanced_accuracy','auroc','nll','best_epoch','training_seconds']}})
    summary=[]
    for model in spec['models']:
        part=[r for r in rows if r['model']==model]
        record={'model':model,'seeds':len(part)}
        for metric in ['accuracy','balanced_accuracy','auroc','nll']:
            s=bootstrap_seed_mean([r[metric] for r in part])
            record.update({metric+'_mean':s['mean'],metric+'_sd':s['sd'],metric+'_ci95':s['ci95_seed_bootstrap']})
        summary.append(record)
    f=np.array([next(r['accuracy'] for r in rows if r['model']=='fidelity' and r['seed']==s) for s in spec['seeds']])
    d=np.array([next(r['accuracy'] for r in rows if r['model']=='dot' and r['seed']==s) for s in spec['seeds']])
    fa=np.array([next(r['auroc'] for r in rows if r['model']=='fidelity' and r['seed']==s) for s in spec['seeds']])
    da=np.array([next(r['auroc'] for r in rows if r['model']=='dot' and r['seed']==s) for s in spec['seeds']])
    contrasts={'accuracy_fidelity_minus_dot':paired_seeds(f-d),'auroc_fidelity_minus_dot':paired_seeds(fa-da)}
    out=Path('artifacts/accuracy_followup')
    out.mkdir(parents=True,exist_ok=True)
    json_write(out/'summary.json',{'manifest':str(manifest),'frozen_manifest':json.loads(manifest.with_suffix('.frozen.json').read_text()),
        'per_seed':rows,'summary':summary,'contrasts':contrasts,
        'disclosure':'Exploratory test follow-up: original test set was previously inspected. Selection used validation only.'})
    fs,ds=summary[0],summary[1]
    ci=contrasts['accuracy_fidelity_minus_dot']['effect']['ci95_seed_bootstrap']
    report=f'''# Frozen accuracy follow-up\n\nThis is an exploratory five-seed follow-up. The 40-epoch configuration was selected from a three-seed validation-only pilot; no test score selected it. The official test set had already been inspected in the earlier study, so these are not fresh confirmatory results.\n\n| Model | Test accuracy | Test AUROC |\n|---|---:|---:|\n| Fidelity | {100*fs['accuracy_mean']:.2f}% +/- {100*fs['accuracy_sd']:.2f}% | {fs['auroc_mean']:.5f} +/- {fs['auroc_sd']:.5f} |\n| Matched dot | {100*ds['accuracy_mean']:.2f}% +/- {100*ds['accuracy_sd']:.2f}% | {ds['auroc_mean']:.5f} +/- {ds['auroc_sd']:.5f} |\n\nFidelity minus dot accuracy is {100*contrasts['accuracy_fidelity_minus_dot']['effect']['mean']:+.2f} percentage points (seed-bootstrap 95% interval [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}]); exact paired sign-flip p={contrasts['accuracy_fidelity_minus_dot']['sign_flip_exact_p']:.4f}. This does not establish a fidelity accuracy advantage.\n\nThe validation pilot had favored fidelity slightly, but that small edge did not generalize cleanly to this matched five-seed test comparison. The result should be retained as evidence that longer training alone does not solve the kernel-comparison problem.\n\nRaw runs are in `runs/accuracy_followup/`; machine-readable output is [summary.json](artifacts/accuracy_followup/summary.json).\n'''
    Path('ACCURACY_FOLLOWUP_RESULTS.md').write_text(report,encoding='utf-8')
    print(report)


if __name__=='__main__':
    main()
