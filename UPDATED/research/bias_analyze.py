"""Summarize the frozen bias-aware follow-up without selecting favorable runs."""
import itertools
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .analyze import csv_write,paired_seeds,figure_save
from .metrics import bootstrap_seed_mean,holm,classification_metrics,probabilities
from .runtime import json_write
from .bias_experiment import ROOT,MANIFEST,sha,computational_hashes

OUT=Path('artifacts/bias_aware')
METRICS=['output_mse','accuracy','auroc','attention_kl','fidelity_mse','clean_disagreement',
         'mean_mitigation_weight','calibration_failure_fraction']
LABELS={'raw_uniform':'Raw uniform','raw_variance':'Raw variance',
        'raw_sensitivity':'Raw sensitivity','raw_combined':'Raw combined',
        'calibrated_uniform':'Calibrated uniform','calibrated_pooled':'Pooled calibrated uniform',
        'calibrated_neyman':'Calibrated adaptive','risk_uniform':'Selective / uniform shots',
        'risk_joint':'Selective / joint shots','risk_pooled':'Selective / pooled calibration',
        'risk_diagonal':'Selective / diagonal risk','risk_no_cal_cov':'Selective / no calibration covariance'}


def main():
    spec=json.loads(MANIFEST.read_text())
    frozen=json.loads(MANIFEST.with_suffix('.frozen.json').read_text())
    assert frozen['manifest_sha256']==sha(MANIFEST)
    assert frozen['code_sha256']==computational_hashes()
    expected=list(itertools.product(spec['seeds'],spec['noises'],spec['budgets'],
                                    spec['policies'],spec['measurement_repeats']))
    rows=[]
    for seed,noise,budget,policy,repeat in expected:
        path=ROOT/'test'/f'seed-{seed}'/f'{noise}-s{budget}-{policy}-r{repeat}'/'metrics.json'
        r=json.loads(path.read_text())
        assert (r['seed'],r['noise'],r['budget'],r['policy'],r['repeat'])==(seed,noise,budget,policy,repeat)
        assert r['n']==624 and r['total_shots']==624*34*budget
        rows.append({k:r[k] for k in ['seed','noise','budget','policy','repeat','n','total_shots',
                     'calibration_shots','allocation_and_classification_seconds']+METRICS if k in r})
    OUT.mkdir(parents=True,exist_ok=True)
    csv_write(OUT/'conditions.csv',rows)
    # One observation per model seed: sampling repeats are not independent models.
    seed_rows=[]
    for seed,noise,budget,policy in itertools.product(spec['seeds'],spec['noises'],spec['budgets'],spec['policies']):
        part=[r for r in rows if (r['seed'],r['noise'],r['budget'],r['policy'])==(seed,noise,budget,policy)]
        row={'seed':seed,'noise':noise,'budget':budget,'policy':policy,'repeats':len(part)}
        for metric in METRICS:
            if metric in part[0]:
                row[metric]=float(np.mean([r[metric] for r in part]))
        seed_rows.append(row)
    csv_write(OUT/'per_seed.csv',seed_rows)
    aggregates=[]
    lookup={}
    for noise,budget,policy in itertools.product(spec['noises'],spec['budgets'],spec['policies']):
        part=[r for r in seed_rows if (r['noise'],r['budget'],r['policy'])==(noise,budget,policy)]
        row={'noise':noise,'budget':budget,'policy':policy,'seeds':len(part),'repeats_per_seed':5}
        for metric in METRICS:
            if metric in part[0]:
                stats=bootstrap_seed_mean([r[metric] for r in part])
                row.update({metric+'_mean':stats['mean'],metric+'_sd':stats['sd'],
                            metric+'_ci_low':stats['ci95_seed_bootstrap'][0],
                            metric+'_ci_high':stats['ci95_seed_bootstrap'][1]})
        aggregates.append(row)
        lookup[(noise,budget,policy)]=row
    csv_write(OUT/'summary.csv',aggregates)
    json_write(OUT/'summary.json',aggregates)
    seeds={(r['seed'],r['noise'],r['budget'],r['policy']):r for r in seed_rows}

    def contrast(noise,budget,a,b,metric,selected_seeds=None):
        selected_seeds=spec['seeds'] if selected_seeds is None else selected_seeds
        av=np.array([seeds[(s,noise,budget,a)][metric] for s in selected_seeds])
        bv=np.array([seeds[(s,noise,budget,b)][metric] for s in selected_seeds])
        return {'noise':noise,'budget':budget,'candidate':a,'baseline':b,'metric':metric,
                'candidate_mean':float(av.mean()),'baseline_mean':float(bv.mean()),
                'relative_change_percent':float(100*(av.mean()/bv.mean()-1)),
                'seed_differences':(av-bv).tolist(),**paired_seeds(av-bv)}

    primary=[]
    for name in spec['primary']['contrasts']:
        a,b=name.split(' minus ')
        primary.append(contrast(spec['primary']['noise'],spec['primary']['budget'],a,b,spec['primary']['metric']))
    for row,p in zip(primary,holm([r['sign_flip_exact_p'] for r in primary])):
        row['holm_p_primary_family']=p
    json_write(OUT/'primary_tests.json',primary)
    secondary=[]
    for noise,budget,policy,metric in itertools.product(spec['noises'],spec['budgets'],
            ['risk_joint','risk_pooled','risk_diagonal'],['output_mse','accuracy','auroc']):
        for baseline in ['raw_uniform','raw_combined','calibrated_uniform','calibrated_pooled','calibrated_neyman']:
            secondary.append(contrast(noise,budget,policy,baseline,metric))
    for row,p in zip(secondary,holm([r['sign_flip_exact_p'] for r in secondary])):
        row['holm_p_secondary_family']=p
    json_write(OUT/'secondary_tests.json',secondary)
    subgroups=[dict(seed_group=label,**contrast('historical',64,'risk_joint',baseline,'output_mse',ss))
               for label,ss in [('existing_seeds_0_4',list(range(5))),('new_seeds_5_9',list(range(5,10)))]
               for baseline in ['raw_combined','calibrated_pooled']]
    json_write(OUT/'seed_subgroups_descriptive.json',subgroups)
    clean=[]
    for s in spec['seeds']:
        with np.load(ROOT/'banks'/f'seed-{s}-test.npz') as z:
            m=classification_metrics(z['labels'],probabilities(z['clean_logits']))
            clean.append({'seed':s,'accuracy':m['accuracy'],'auroc':m['auroc']})
    clean_summary={metric:bootstrap_seed_mean([r[metric] for r in clean]) for metric in ['accuracy','auroc']}
    json_write(OUT/'analytical_reference.json',{'per_seed':clean,'summary':clean_summary})

    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'svg.fonttype':'none',
                        'axes.spines.top':False,'axes.spines.right':False})
    shown=['raw_uniform','raw_combined','calibrated_uniform','calibrated_pooled',
           'calibrated_neyman','risk_joint','risk_pooled','risk_diagonal']
    fig,axes=plt.subplots(2,2,figsize=(13,9),sharex=True)
    for ax,(noise,budget) in zip(axes.flat,itertools.product(spec['noises'],spec['budgets'])):
        values=[lookup[(noise,budget,p)] for p in shown]
        ax.errorbar(range(len(shown)),[r['output_mse_mean'] for r in values],
                    yerr=[r['output_mse_sd'] for r in values],fmt='o',capsize=3)
        ax.set_title(f'{noise}, {budget} nominal shots/pair')
        ax.set_ylabel('Attention output MSE (mean +/- model-seed SD)')
        ax.set_yscale('log')
        ax.set_xticks(range(len(shown)),[LABELS[p] for p in shown],rotation=40,ha='right')
    figure_save(fig,OUT,'output_error')
    primary_wins=[r for r in primary if r['effect']['mean']<0 and r['holm_p_primary_family']<.05]
    lines=['# Bias-aware follow-up: executed results','',
        'The novelty check preceded implementation. General joint mitigation and shot allocation already have close prior art; this is an attention-specific experimental specialization. These results do not establish algorithmic novelty, quantum advantage or publication readiness.','',
        '**Decision: the proposed joint method does not beat the strongest controls.** At the primary setting its output error is 55.1% higher than raw combined allocation and 22.0% higher than pooled calibrated uniform. Every one of the ten model seeds has the same direction for these contrasts (Holm p=0.00977). Accuracy is 86.99% versus 87.11% for raw combined. This supports a negative empirical finding, not a successful new-method claim.','',
        f'Completed {len(rows):,} conditions: 10 trained model seeds x 5 measurement repetitions x 2 noise models x 2 budgets x 12 policies, each on all 624 official PneumoniaMNIST test images. Repetitions are averaged within each seed before inference.','',
        '## Frozen primary comparison','',
        'Historical noise, 64 nominal shots per overlap. Lower conditional attention-output MSE is better. Changes compare the joint selective method against each baseline.','',
        '| Baseline | Baseline MSE | Joint selective MSE | Relative change | Paired difference 95% seed CI | Holm p |',
        '|---|---:|---:|---:|---|---:|']
    for r in primary:
        lo,hi=r['effect']['ci95_seed_bootstrap']
        lines.append(f"| {LABELS[r['baseline']]} | {r['baseline_mean']:.6f} | {r['candidate_mean']:.6f} | {r['relative_change_percent']:+.1f}% | [{lo:+.6f}, {hi:+.6f}] | {r['holm_p_primary_family']:.5f} |")
    lines+=['',f'The proposed joint method shows corrected significant improvement against {len(primary_wins)} of the five prespecified primary baselines. Superiority requires comparison with the strong controls, not just a weaker baseline.','',
        'Intervals are percentile bootstraps over ten model seeds. Exact two-sided sign-flip tests enumerate 1,024 sign patterns under the symmetric/exchangeable paired-null assumption; Holm correction covers the five frozen primary contrasts. These intervals condition on this fixed dataset and do not measure external-patient uncertainty.','',
        '## All methods and conditions','',
        'Accuracy is mean +/- SD across model seeds after averaging sampling repeats. AUROC and MSE are means. All secondary configurations are retained.','',
        '| Noise | Budget | Method | Output MSE | Accuracy (%) | AUROC |',
        '|---|---:|---|---:|---:|---:|']
    for r in aggregates:
        lines.append(f"| {r['noise']} | {r['budget']} | {LABELS[r['policy']]} | {r['output_mse_mean']:.6f} | {100*r['accuracy_mean']:.2f} +/- {100*r['accuracy_sd']:.2f} | {r['auroc_mean']:.5f} |")
    lines+=['',f"Analytical reference for these ten seeds: accuracy {100*clean_summary['accuracy']['mean']:.2f}% +/- {100*clean_summary['accuracy']['sd']:.2f}%; AUROC {clean_summary['auroc']['mean']:.5f}. This is a different seed count and measurement scope from the original five-seed, full-attention experiment.",'',
        '## Interpretation and publication judgment','',
        'Selective correction beats the weaker per-image fully calibrated baselines at the primary setting, but that improvement disappears against strong raw or pooled-calibration controls. At 256 shots under historical noise, joint selective correction slightly improves output MSE versus raw combined (0.005630 versus 0.005834), while pooled calibrated uniform reaches 0.004522. Under stress at 256 shots, selective correction improves over raw combined (0.015743 versus 0.028813), but pooled calibrated uniform is again better (0.007754). The pooled version of the proposed method also loses to pooled calibrated uniform in all four noise/budget settings.','',
        'The full covariance surrogate does not consistently beat its diagonal ablation; the diagonal version has lower output error in three of four settings. Correct local geometry alone therefore does not establish an effective finite-shot policy. No causal attribution to one approximation is proven: pilot uncertainty, affine-response mismatch, nonlinear propagation, calibration cost and finite optimization can all matter. These results do not justify tuning on the same test set until the proposed method wins.','',
        'A methodological novelty claim is unsupported because general decision-aware mitigation and joint coefficient/allocation design have close prior art, and this specialization fails against strong baselines. A negative-results or benchmarking paper could be developed, but acceptance is not established; it would need a distinct generalizable lesson, wider tasks/circuit families, independent confirmation and a venue-specific assessment. Adding more runs of this configuration alone would not establish novelty.','',
        '## Costs, execution and limitations','',
        '- Each image has 2 heads x 17 final-CLS overlaps. Total budget is 2,176 or 8,704 shots per image, including calibration, pilots and production; respectively 1,357,824 or 5,431,296 for 624 images per condition.',
        '- Raw policies spend the full budget on targets. Corrected policies spend 12.5% on four known references. Adaptive corrected policies also spend 272 pilot shots per image. Independent production outcomes exclude pilots. Calibrated uniform avoids this pilot cost.',
        '- Pooled controls share all paid reference outcomes across test images under the stationary-noise assumption. They use no labels, target fidelities or true noise parameters. Their online/latency requirements differ from per-image calibration.',
        '- Simulation applies gates and local noise channels to exact disjoint two-qubit density matrices, then samples the destructive-SWAP parity distribution. No full-register PennyLane calls or physical QPU runs are counted in the 2,400-condition sweep. Independent full-register checks are recorded separately.',
        '- Only final-block CLS attention is noisy. The earlier backbone is analytical. This is not end-to-end noisy quantum-transformer inference. Product feature states and independent local noise enable efficient classical simulation.',
        '- Historical noise uses a retained April 2021 device calibration proxy, not a current device. The second noise model is a synthetic stress mixture. Affine calibration is an approximation and can mismatch target circuits.',
        '- The primary endpoint is attention-output reconstruction, not clinical accuracy. Better reconstruction need not improve classification.',
        '- The official test set was examined in the earlier study. This disclosed frozen follow-up is not fresh external confirmation. No settings changed after its test freeze.','',
        '## Evidence and reproduction','',
        '- [Novelty gate and primary sources](BIAS_AWARE_NOVELTY_GATE.md)',
        '- [Mathematical method and assumptions](BIAS_AWARE_METHOD.md)',
        '- [Frozen manifest](configs/bias_aware.frozen.json)',
        '- [Per-condition results](artifacts/bias_aware/conditions.csv)',
        '- [Per-seed results](artifacts/bias_aware/per_seed.csv)',
        '- [Primary paired tests](artifacts/bias_aware/primary_tests.json)',
        '- [All secondary contrasts](artifacts/bias_aware/secondary_tests.json)',
        '- [Output-error figure](artifacts/bias_aware/output_error.pdf)',
        '- [Complete prediction and shot-ledger audit](verification/bias_aware/evidence_manifest.json)',
        '- [Independent full-register circuit checks](verification/bias_aware/full_register_checks.json)',
        '- Saved predictions, measured outcomes and allocation ledgers: runs/bias_aware/test.',
        '', '```powershell','python -m research.bias_experiment run','python -m research.bias_analyze',
        'python verification/check_bias_aware.py','python tests/run_suite.py','```','',
        'The run command validates frozen source/configuration hashes and resumes completed conditions. Do not refreeze or tune on these test outcomes.']
    Path('BIAS_AWARE_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    json_write(OUT/'analysis_provenance.json',{'conditions':len(rows),'manifest_sha256':sha(MANIFEST),
        'frozen_utc':frozen['utc'],'analysis_sha256':sha(__file__),'primary_significant_improvements':len(primary_wins),
        'total_sampled_shots_all_conditions':sum(r['total_shots'] for r in rows),
        'unit_of_inference':'10 model seeds after averaging 5 shot repetitions',
        'all_complete':True})
    print(json.dumps({'conditions':len(rows),'primary':primary,'clean':clean_summary},indent=2))


if __name__=='__main__':
    main()
