"""Regenerate tables, uncertainty, paired tests and vector figures from saved predictions."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .metrics import bootstrap_seed_mean, paired_predictions, holm
from .runtime import json_write

METRICS=['accuracy','balanced_accuracy','auroc','sensitivity','specificity','precision','f1','npv','ece','brier']


def csv_write(path,rows):
    if not rows:
        return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def paired_seeds(differences):
    d=np.asarray(differences,dtype=float)
    if len(d)<2:
        return {'n':len(d),'difference':float(d.mean()),'p':None}
    signs=np.array(list(itertools.product([-1,1],repeat=len(d))))
    p=float((np.abs((signs*d).mean(1))>=abs(d.mean())-1e-15).mean())
    sd=d.std(ddof=1)
    return {'n':len(d), 'effect':bootstrap_seed_mean(d), 'paired_standardized_difference':float(d.mean()/sd) if sd>1e-15 else None,
            'sign_flip_exact_p':p,'wilcoxon_p':float(stats.wilcoxon(d,method='auto').pvalue) if np.any(d) else 1.,
            'normality_assumed':False}


def figure_save(fig,out,name):
    fig.tight_layout()
    fig.savefig(out/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(out/(name+'.svg'),bbox_inches='tight')
    fig.savefig(out/(name+'.png'),dpi=220,bbox_inches='tight')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--runs',nargs='+',default=['runs/central','runs/additional'])
    p.add_argument('--measurements',default='runs/measurements')
    p.add_argument('--output',default='artifacts')
    a=p.parse_args()
    out=Path(a.output)
    out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    rows,groups,lookup=[],{},{}
    provenance=[]
    for root in a.runs:
        for path in sorted(Path(root).glob('*/test_metrics.json')):
            c=json.loads((path.parent/'config.json').read_text())
            m=json.loads(path.read_text())
            resources=json.loads((path.parent/'resources.json').read_text())
            row={k:c[k] for k in ['dataset','model','seed']}
            row.update({k:m[k] for k in METRICS})
            row.update({'n':m['n'],'parameters':resources['parameters'],'dense_macs':resources['dense_macs_per_image'],
                        'training_seconds':m['training_seconds'],'inference_seconds':m['inference_seconds'],'run':str(path.parent)})
            rows.append(row)
            groups.setdefault((c['dataset'],c['model']),[]).append(row)
            lookup[(c['dataset'],c['model'],c['seed'])]=path.parent
            provenance.append(str(path))
    csv_write(out/'central_per_seed.csv',rows)
    aggregated=[]
    for (dataset,model),items in groups.items():
        assert len({r['seed'] for r in items})==len(items),'Duplicate seed in aggregation'
        row={'dataset':dataset,'model':model,'seeds':len(items),'test_n':items[0]['n'],'parameters':items[0]['parameters']}
        for metric in METRICS:
            summary=bootstrap_seed_mean([r[metric] for r in items])
            row.update({metric+'_mean':summary['mean'],metric+'_sd':summary['sd'],
                        metric+'_ci_low':summary['ci95_seed_bootstrap'][0],metric+'_ci_high':summary['ci95_seed_bootstrap'][1]})
        aggregated.append(row)
    csv_write(out/'central_summary.csv',aggregated)
    json_write(out/'central_summary.json',aggregated)
    tex=['\\begin{tabular}{llrrr}','\\toprule','Dataset & Kernel & Parameters & Accuracy & AUROC \\\\','\\midrule']
    for r in aggregated:
        name=r['model'].replace('_',r'\_')
        tex.append(f"{r['dataset'].replace('mnist','')} & {name} & {r['parameters']} & {r['accuracy_mean']:.3f} $\\pm$ {r['accuracy_sd'] or 0:.3f} & {r['auroc_mean']:.3f} $\\pm$ {r['auroc_sd'] or 0:.3f} \\\\")
    tex.extend(['\\bottomrule','\\end{tabular}'])
    (out/'central_table.tex').write_text('\n'.join(tex))
    comparisons=[]
    for dataset in sorted({r['dataset'] for r in rows}):
        for comparator in ['dot_full','dot','cosine','rbf']:
            paired=[]
            per_example=[]
            for seed in range(5):
                ra=lookup.get((dataset,'fidelity',seed))
                rb=lookup.get((dataset,comparator,seed))
                if ra is None or rb is None:
                    continue
                ma,mb=json.loads((ra/'test_metrics.json').read_text()),json.loads((rb/'test_metrics.json').read_text())
                paired.append(ma['auroc']-mb['auroc'])
                pa,pb=np.load(ra/'test_predictions.npz'),np.load(rb/'test_predictions.npz')
                np.testing.assert_array_equal(pa['sample_ids'],pb['sample_ids'])
                np.testing.assert_array_equal(pa['labels'],pb['labels'])
                per_example.append({'seed':seed,**paired_predictions(pa['labels'],pa['probabilities'],pb['probabilities'])})
            if paired:
                comparisons.append({'dataset':dataset,'contrast':'fidelity minus '+comparator,'metric':'auroc',
                                    **paired_seeds(paired),'per_seed_paired_example_tests':per_example})
    if comparisons:
        corrections=holm([r['sign_flip_exact_p'] for r in comparisons])
        for r,padj in zip(comparisons,corrections):
            r['holm_p_all_dataset_kernel_contrasts']=padj
    json_write(out/'paired_classification.json',comparisons)
    fig,axes=plt.subplots(1,max(1,len({r['dataset'] for r in rows})),figsize=(13,3.5),squeeze=False)
    for ax,dataset in zip(axes[0],sorted({r['dataset'] for r in rows})):
        part=[r for r in aggregated if r['dataset']==dataset]
        ax.errorbar(np.arange(len(part)),[r['auroc_mean'] for r in part],yerr=[r['auroc_sd'] or 0 for r in part],fmt='o',capsize=3)
        ax.set_xticks(np.arange(len(part)),[r['model'] for r in part],rotation=45,ha='right')
        ax.set_title(dataset)
        ax.set_ylabel('AUROC (mean ± seed SD)')
    figure_save(fig,out,'matched_baselines')
    measurements=[]
    measurement_lookup={}
    for path in sorted(Path(a.measurements).glob('*/*/test_metrics.json')):
        meta=json.loads((path.parent/'measurement.json').read_text())
        c=json.loads((Path(meta['training_run'])/'config.json').read_text())
        m=json.loads(path.read_text())
        row={'dataset':c['dataset'],'seed':c['seed'],'policy':meta['allocation']['policy'],'budget':meta['allocation']['shots'],
             'accuracy':m['accuracy'],'auroc':m['auroc'],'shots':m['shots'],'actual_circuit_calls':m['circuit_calls'],
             'inference_seconds':m['test_resources']['seconds'],'fidelity_mse':m['conditional_errors_mean'][0],
             'attention_mse':m['conditional_errors_mean'][1],'output_mse':m['conditional_errors_mean'][2],
             'attention_kl':m['conditional_errors_mean'][3], 'run':str(path.parent)}
        measurements.append(row)
        measurement_lookup[(c['seed'],row['policy'],row['budget'])]=row
        provenance.append(str(path))
    csv_write(out/'measurement_per_seed.csv',measurements)
    measurement_summary=[]
    for policy,budget in sorted({(r['policy'],r['budget']) for r in measurements}):
        part=[r for r in measurements if r['policy']==policy and r['budget']==budget]
        row={'policy':policy,'budget':budget,'seeds':len(part)}
        for metric in ['accuracy','auroc','fidelity_mse','attention_mse','output_mse','attention_kl','shots']:
            summary=bootstrap_seed_mean([r[metric] for r in part])
            row.update({metric+'_mean':summary['mean'],metric+'_sd':summary['sd']})
        measurement_summary.append(row)
    csv_write(out/'measurement_summary.csv',measurement_summary)
    macros=[]
    for kind,macro in [('fidelity','FidelityAUC'),('dot','DotAUC'),('rbf','RBFAUC')]:
        row=next((r for r in aggregated if r['dataset']=='pneumoniamnist' and r['model']==kind),None)
        if row:
            macros.append('\\newcommand{\\'+macro+'}{'+f"{row['auroc_mean']:.4f}"+'}')
    uniform=next((r for r in measurement_summary if r['policy']=='uniform' and r['budget']==64),None)
    combined=next((r for r in measurement_summary if r['policy']=='combined' and r['budget']==64),None)
    if uniform and combined:
        for macro,value in [('UniformMSE',uniform['output_mse_mean']),('CombinedMSE',combined['output_mse_mean']),
                            ('UniformAUC',uniform['auroc_mean']),('CombinedAUC',combined['auroc_mean'])]:
            macros.append('\\newcommand{\\'+macro+'}{'+f'{value:.5f}'+'}')
        macros.append('\\newcommand{\\MSEIncrease}{'+f"{100*(combined['output_mse_mean']/uniform['output_mse_mean']-1):.1f}"+'}')
    (out/'result_macros.tex').write_text('\n'.join(macros)+'\n')
    table=['\\begin{tabular}{lrrr}','\\toprule','Policy & AUROC & Output squared error & Test shots \\\\','\\midrule']
    for r in measurement_summary:
        if r['budget']==64:
            table.append(f"{r['policy']} & {r['auroc_mean']:.4f} $\\pm$ {r['auroc_sd']:.4f} & {r['output_mse_mean']:.5f} & {r['shots_mean']/1e6:.3f}M \\\\")
    table.extend(['\\bottomrule','\\end{tabular}'])
    (out/'measurement_table.tex').write_text('\n'.join(table))
    measurement_comparisons=[]
    for metric in ['auroc','output_mse']:
        diffs=[]
        for seed in range(5):
            ra=measurement_lookup.get((seed,'combined',64))
            rb=measurement_lookup.get((seed,'uniform',64))
            if ra and rb:
                diffs.append(ra[metric]-rb[metric])
        if diffs:
            measurement_comparisons.append({'contrast':'combined minus uniform at 64 mean shots','metric':metric,**paired_seeds(diffs)})
    json_write(out/'paired_measurement.json',measurement_comparisons)
    primary=[r for r in comparisons if r['dataset']=='pneumoniamnist' and r['contrast']=='fidelity minus dot']+measurement_comparisons
    if primary:
        for r,padj in zip(primary,holm([r['sign_flip_exact_p'] for r in primary])):
            r['holm_primary_family_p']=padj
        json_write(out/'primary_hypotheses.json',primary)
    if measurement_summary:
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for policy in ['uniform','variance','sensitivity','combined']:
            part=sorted([r for r in measurement_summary if r['policy']==policy],key=lambda r:r['budget'])
            if not part:
                continue
            for ax,metric in zip(axes,['auroc','output_mse']):
                ax.errorbar([r['budget'] for r in part],[r[metric+'_mean'] for r in part],
                            yerr=[r[metric+'_sd'] or 0 for r in part],marker='o',capsize=2,label=policy)
                ax.set_xscale('log',base=2)
                ax.set_xlabel('Mean modeled shots per pair')
        axes[0].set_ylabel('Test AUROC (mean ± seed SD)')
        axes[1].set_ylabel('Conditional attention-output squared error')
        axes[1].set_yscale('log')
        axes[0].legend(fontsize=8)
        for ax in axes:
            ax.set_xlim(13,1250)
            ticks=[16,32,64,128,256,512,1024]
            ax.set_xticks(ticks,[str(x) for x in ticks])
        figure_save(fig,out,'shot_tradeoff')
    # Reliability: one disclosed seed, calibrated and uncalibrated, not pooled pseudo-replicates.
    fig,ax=plt.subplots(figsize=(4.5,4))
    for kind in ['dot','fidelity']:
        path=lookup.get(('pneumoniamnist',kind,0))
        if path:
            m=json.loads((path/'test_metrics.json').read_text())
            reliability=m['reliability']
            ax.plot([r['confidence'] for r in reliability],[r['accuracy'] for r in reliability],'-o',label=kind)
    ax.plot([0,1],[0,1],'k--',lw=.8)
    ax.set(xlabel='Mean confidence',ylabel='Empirical accuracy',title='PneumoniaMNIST: seed 0, calibrated')
    ax.legend()
    figure_save(fig,out,'reliability')
    # Secondary ablations, preserving every declared contrast and its actual seed count.
    ablation_path=Path('runs/ablations/index.json')
    if ablation_path.exists():
        ablations={}
        for r in json.loads(ablation_path.read_text()):
            m=json.loads((Path(r['run'])/'test_metrics.json').read_text())
            ablations.setdefault(r['ablation'],[]).append(m['auroc'])
        ablation_rows=[{'ablation':k,**bootstrap_seed_mean(v)} for k,v in ablations.items()]
        csv_write(out/'ablation_summary.csv',ablation_rows)
        fig,axes=plt.subplots(1,2,figsize=(10,3.7))
        for model in ['dot','fidelity']:
            for ax,prefix,xlabel in zip(axes,['fraction-','qk-'],['Training fraction','Q/K features per head']):
                selected=sorted([r for r in ablation_rows if r['ablation'].startswith(prefix) and r['ablation'].endswith('-'+model)],
                                key=lambda r:float(r['ablation'].split('-')[1]))
                ax.errorbar([float(r['ablation'].split('-')[1]) for r in selected],[r['mean'] for r in selected],
                            yerr=[r['sd'] for r in selected],marker='o',capsize=2,label=model)
                ax.set(xlabel=xlabel,ylabel='Test AUROC (mean and seed SD)')
                ax.legend()
        axes[0].set_xscale('log')
        figure_save(fig,out,'ablations')
    probe_path=Path('runs/circuits/probes.jsonl')
    if probe_path.exists():
        probes=[json.loads(line) for line in probe_path.read_text().splitlines()]
        csv_write(out/'circuit_probes.csv',[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in probes])
        fig,ax=plt.subplots(figsize=(6,4))
        for method in ['swap','destructive','uncompute']:
            part=[r for r in probes if r['method']==method]
            scenarios=['ideal','readout_only','calibration_proxy','full_synthetic']
            ax.plot(scenarios,[np.mean([r['squared_error'] for r in part if r['scenario']==s]) for s in scenarios],'-o',label=method)
        ax.set_ylabel('Fidelity squared error (3 angle pairs)')
        ax.tick_params(axis='x',rotation=20)
        ax.legend()
        figure_save(fig,out,'circuit_noise')
        mitigation=[]
        for scenario in ['readout_only','calibration_proxy','full_synthetic']:
            part=[r for r in probes if r['method']=='swap' and r['scenario']==scenario]
            mitigation.append({'scenario':scenario,'pairs':len(part),
                'unmitigated_mse':float(np.mean([r['squared_error'] for r in part])),
                'known_readout_inverse_mse':float(np.mean([r['mitigated_squared_error'] for r in part])),
                'calibration_shot_overhead_included':False})
        csv_write(out/'mitigation_summary.csv',mitigation)
    strengths_path=Path('runs/noise_strengths/probes.jsonl')
    if strengths_path.exists():
        strengths=[json.loads(line) for line in strengths_path.read_text().splitlines()]
        csv_write(out/'noise_strengths_per_pair.csv',[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in strengths])
        channels=['preparation','one_qubit','two_qubit','amplitude','phase','bitflip','phaseflip','readout']
        fig,axes=plt.subplots(2,4,figsize=(12,6))
        for ax,channel in zip(axes.ravel(),channels):
            for method in ['swap','destructive','uncompute']:
                part=[r for r in strengths if r['channel']==channel and r['method']==method]
                x=sorted({r['strength'] for r in part})
                ax.plot(x,[np.mean([r['squared_error'] for r in part if r['strength']==v]) for v in x],'-o',label=method)
            ax.set(title=channel,xlabel='Channel probability',ylabel='Fidelity squared error')
            ax.set_xscale('log')
        axes[0,0].legend(fontsize=7)
        figure_save(fig,out,'noise_strengths')
    json_write(out/'provenance.json',{'source_metrics':provenance,'central_runs':len(rows),'measurement_runs':len(measurements),
               'ci_unit':'seed mean conditional on fixed test set','physical_hardware':False})
    print('Generated',len(rows),'training-run rows and',len(measurements),'measurement rows in',out)


if __name__=='__main__':
    main()
