"""Validation-only accuracy exploration, deliberately unable to run test inference."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .config import Config
from .train import train
from .runtime import json_write
from .metrics import bootstrap_seed_mean


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',default='configs/accuracy_exploration.json')
    parser.add_argument('--output',default='runs/accuracy_exploration')
    args=parser.parse_args()
    manifest_path=Path(args.manifest)
    manifest=json.loads(manifest_path.read_text())
    out=Path(args.output)
    rows=[]
    for candidate in manifest['candidates']:
        for model in manifest['models']:
            for seed in manifest['seeds']:
                c=Config(**(manifest['base']|candidate['override']|{
                    'dataset':manifest['dataset'],'model':model,'seed':seed}))
                folder=train(c,out,validation_only=True)
                metric=json.loads((folder/'validation_metrics.json').read_text())
                calibration=json.loads((folder/'calibration.json').read_text())
                rows.append({'candidate':candidate['name'],'model':model,'seed':seed,
                    'validation_accuracy':metric['accuracy'],'validation_auroc':metric['auroc'],
                    'validation_nll':metric['nll'],'best_epoch':calibration['best_epoch'],
                    'run':str(folder),'config':c.dict()})
                json_write(out/'per_seed.json',rows)
                print(candidate['name'],model,seed,
                      f"validation AUROC={metric['auroc']:.5f}",flush=True)
    summary=[]
    for candidate in manifest['candidates']:
        for model in manifest['models']:
            part=[r for r in rows if r['candidate']==candidate['name'] and r['model']==model]
            record={'candidate':candidate['name'],'model':model,'seeds':len(part)}
            for metric in ['validation_accuracy','validation_auroc','validation_nll']:
                s=bootstrap_seed_mean([r[metric] for r in part])
                record.update({metric+'_mean':s['mean'],metric+'_sd':s['sd'],
                               metric+'_ci95':s['ci95_seed_bootstrap']})
            summary.append(record)
    json_write(out/'summary.json',{'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'validation_only':True,'selection_metric':manifest['selection_metric'],
        'rows':summary,'interpretation':manifest['interpretation']})
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
