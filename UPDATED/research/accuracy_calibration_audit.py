"""Exploratory calibration/ensemble audit; adaptation APIs take no target labels.

No test labels, known test class fraction or test-optimal cutoff enter planning.
This audit was designed after earlier test outcomes and is not confirmatory.
"""
from pathlib import Path
import json
from datetime import datetime,timezone
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from .analyze import csv_write
from .metrics import classification_metrics
from .runtime import json_write


def platt(source_probabilities,source_labels,target_probabilities):
    def score(p):
        p=np.clip(p[:,1],1e-7,1-1e-7)
        return np.log(p)-np.log1p(-p)
    x=score(source_probabilities)
    def loss(ab):
        z=ab[0]*x+ab[1]
        return float(np.mean(np.logaddexp(0,z)-source_labels*z))
    fit=minimize(loss,[1.,0.],method='L-BFGS-B',bounds=[(.05,20),(-10,10)],
                 options={'ftol':1e-12,'gtol':1e-8,'maxiter':1000})
    if not fit.success:
        raise RuntimeError(fit.message)
    p=expit(fit.x[0]*score(target_probabilities)+fit.x[1])
    return np.column_stack([1-p,p]),{'slope':float(fit.x[0]),'intercept':float(fit.x[1])}


def em_adjust(probabilities,source_prior,max_iterations=1000):
    prior=source_prior.copy()
    for iteration in range(max_iterations):
        adapted=probabilities*(prior/source_prior)[None,:]
        adapted/=adapted.sum(1,keepdims=True)
        next_prior=np.maximum(adapted.mean(0),1e-7);next_prior/=next_prior.sum()
        gap=float(np.max(abs(next_prior-prior)))
        prior=next_prior
        if gap<1e-10:
            break
    adapted=probabilities*(prior/source_prior)[None,:]
    adapted/=adapted.sum(1,keepdims=True)
    return adapted,{'target_prior_estimate':prior.tolist(),'iterations':iteration+1,
                    'converged':gap<1e-10}


def bbse_adjust(source_probabilities,source_labels,target_probabilities):
    source_prior=np.bincount(source_labels,minlength=2)/len(source_labels)
    hard=source_probabilities.argmax(1)
    confusion=np.array([[np.mean(hard[source_labels==c]==pred) for c in [0,1]] for pred in [0,1]])
    if np.linalg.cond(confusion)>1e6:
        raise ValueError('BBSE confusion matrix is too ill-conditioned')
    target_counts=np.bincount(target_probabilities.argmax(1),minlength=2)/len(target_probabilities)
    prior=np.linalg.solve(confusion,target_counts)
    prior=np.maximum(prior,1e-7);prior/=prior.sum()
    adapted=target_probabilities*(prior/source_prior)[None,:]
    adapted/=adapted.sum(1,keepdims=True)
    return adapted,{'target_prior_estimate':prior.tolist(),'confusion_condition':float(np.linalg.cond(confusion))}


def apply_methods(source_probabilities,source_labels,target_probabilities):
    prior=np.bincount(source_labels,minlength=2)/len(source_labels)
    result={'default':(target_probabilities.argmax(1),target_probabilities,{})}
    # Only validation labels choose this threshold; tie breaker was fixed in the earlier diagnostic.
    thresholds=np.unique(np.r_[0.,source_probabilities[:,1],1.])
    accuracy=np.array([np.mean((source_probabilities[:,1]>=t)==source_labels) for t in thresholds])
    winners=np.flatnonzero(accuracy==accuracy.max())
    threshold=float(thresholds[winners[np.argmin(abs(thresholds[winners]-.5))]])
    result['validation_threshold']=((target_probabilities[:,1]>=threshold).astype(int),target_probabilities,
                                    {'validation_threshold':threshold})
    for name,fun in [('em',lambda:em_adjust(target_probabilities,prior)),
                     ('bbse',lambda:bbse_adjust(source_probabilities,source_labels,target_probabilities)),
                     ('platt',lambda:platt(source_probabilities,source_labels,target_probabilities))]:
        p,metadata=fun();result[name]=(p.argmax(1),p,metadata)
    calibrated,fit=platt(source_probabilities,source_labels,target_probabilities)
    corrected,shift=em_adjust(calibrated,prior)
    result['platt_em']=(corrected.argmax(1),corrected,fit|shift)
    return result


def main():
    out=Path('artifacts/accuracy_calibration')
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for study,root in [('original',Path('runs/central')),('longer_training',Path('runs/accuracy_followup')),
                       ('hybrid',Path('runs/hybrid_accuracy'))]:
        for kernel in ['fidelity','dot']:
            banks=[]
            for seed in range(5):
                pattern=f'pneumoniamnist-{kernel}-s{seed}-*/validation_predictions.npz' if study!='hybrid' else f'{kernel}-s{seed}-*/validation_predictions.npz'
                paths=list(root.glob(pattern));assert len(paths)==1,(root,pattern,paths)
                folder=paths[0].parent
                with np.load(paths[0]) as v,np.load(folder/'test_predictions.npz') as t:
                    banks.append((v['probabilities'],v['labels'],t['probabilities'],t['labels']))
            for vp,vy,tp,ty in banks:
                np.testing.assert_array_equal(vy,banks[0][1]);np.testing.assert_array_equal(ty,banks[0][3])
            evaluations=[(str(seed),*bank) for seed,bank in enumerate(banks)]
            evaluations.append(('ensemble',np.mean([b[0] for b in banks],0),banks[0][1],
                                 np.mean([b[2] for b in banks],0),banks[0][3]))
            for seed,vp,vy,tp,ty in evaluations:
                # ty is never passed to apply_methods; labels enter only scoring below.
                for method,(pred,p,metadata) in apply_methods(vp,vy,tp).items():
                    m=classification_metrics(ty,p)
                    row={'study':study,'kernel':kernel,'seed':seed,'method':method,
                         'accuracy':float(np.mean(pred==ty)),'auroc':m['auroc'],'n':len(ty),
                         'decision_parameters':metadata}
                    rows.append(row)
                    path=out/f'{study}-{kernel}-{seed}-{method}.npz'
                    np.savez_compressed(path,probabilities=p,predictions=pred,labels=ty,sample_ids=np.arange(len(ty)))
    summary=[]
    for study in ['original','longer_training','hybrid']:
        for kernel in ['fidelity','dot']:
            for method in ['default','validation_threshold','em','bbse','platt','platt_em']:
                part=[r for r in rows if (r['study'],r['kernel'],r['method'])==(study,kernel,method)]
                individual=[r['accuracy'] for r in part if r['seed']!='ensemble']
                ensemble=next(r for r in part if r['seed']=='ensemble')
                summary.append({'study':study,'kernel':kernel,'method':method,
                    'single_model_accuracy_mean':float(np.mean(individual)),
                    'single_model_accuracy_sd':float(np.std(individual,ddof=1)),
                    'ensemble_accuracy':ensemble['accuracy'],'ensemble_auroc':ensemble['auroc']})
    csv_write(out/'summary.csv',summary)
    json_write(out/'audit.json',{'utc':datetime.now(timezone.utc).isoformat(),'rows':rows,'summary':summary,
        'disclosure':'Post hoc exploratory methods on previously inspected test data; all settings retained. Target labels enter scoring only. Transductive EM/BBSE methods use the entire unlabeled target cohort.',
        'oracle_diagnostic_excluded':'Earlier exploratory probes maximized thresholds on test labels and used known test class counts. These are not valid deployable methods and are excluded from achieved-accuracy claims.'})
    lines=['# Exploratory calibration and ensemble audit','',
        'These methods were evaluated after test outcomes were already available. They are exploratory, not independent confirmation. Calibration fits validation labels only. EM and BBSE use the entire unlabeled target batch and therefore require a transductive evaluation protocol. Target labels score predictions only. All methods receive the same five seeds for both kernels.','',
        '| Study | Kernel | Method | Individual accuracy, mean +/- SD | Five-model ensemble accuracy |',
        '|---|---|---|---:|---:|']
    for r in summary:
        lines.append(f"| {r['study']} | {r['kernel']} | {r['method']} | {100*r['single_model_accuracy_mean']:.2f}% +/- {100*r['single_model_accuracy_sd']:.2f}% | {100*r['ensemble_accuracy']:.2f}% |")
    lines+=['',
        'An ensemble is one predictor costing five forward passes; it is not five independent replications. Per-seed and ensemble results must not be interchanged.','',
        'The audit includes the original validation-threshold and averaging checks; EM adjustment; hard-prediction BBSE; positive-slope logistic (Platt) probability calibration; and Platt followed by EM. Label-shift adjustment assumes stable class-conditional distributions; an observed class-prevalence change alone does not verify that assumption. Positive-slope calibration and binary prior adjustment preserve per-model score ordering, so they do not create an AUROC improvement.','',
        'References: [BBSE, Lipton et al. (2018)](https://proceedings.mlr.press/v80/lipton18a.html); [EM prior adjustment, Saerens et al. (2002), DOI](https://doi.org/10.1162/089976602753284446). These are established methods and do not establish algorithmic novelty for this project.','',
        'Earlier diagnostics that chose a threshold using test labels, or forced predictions to match the known test class counts, are oracle calculations. Any greater-than-90% values from those calculations are excluded from valid accuracy claims.','',
        '[Saved decision parameters and complete results](artifacts/accuracy_calibration/audit.json), [summary CSV](artifacts/accuracy_calibration/summary.csv). Reproduce with `python -m research.accuracy_calibration_audit`.']
    Path('ACCURACY_CALIBRATION_AUDIT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
