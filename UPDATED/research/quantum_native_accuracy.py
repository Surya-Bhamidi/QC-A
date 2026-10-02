"""Reuse the frozen pretrained learner under a native-resolution namespace.

No earlier computational source is edited. This explicit adapter substitutes
only the dataset resolution, manifest and artifact directories. Its own source
hash is included in the new freeze, and all matched controls share the adapter.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .data import load_splits as original_load_splits
from . import quantum_transfer as features
from . import quantum_transfer_experiment as experiment
from . import quantum_transfer_analyze as analysis
from .quantum_robust_experiment import digest
from .runtime import json_write

MANIFEST=Path('configs/native_quantum_accuracy.json')
CACHE=Path('artifacts/native_quantum_features')
ROOT=Path('runs/native_quantum_accuracy')
OUT=Path('artifacts/native_quantum_accuracy')
BASE_SOURCE_HASHES=experiment.source_hashes


def source_hashes():
    return BASE_SOURCE_HASHES()|{'research/quantum_native_accuracy.py':digest(__file__)}


def native_splits(c,data_dir='data'):
    spec=json.loads(MANIFEST.read_text())
    path=Path(spec['dataset_root'])/(c.dataset+'.npz')
    assert hashlib.md5(path.read_bytes()).hexdigest()==spec['official_md5']
    assert c.fraction==1.,'This resolution study uses complete official splits'
    datasets,channels,classes,meta=original_load_splits(replace(c,image_size=224,patch_size=112),spec['dataset_root'])
    alignment={}
    with np.load(Path('data')/(c.dataset+'.npz')) as reference:
        for split,ds in datasets.items():
            np.testing.assert_array_equal(ds.tensors[1].numpy(),reference[split+'_labels'].reshape(-1))
            reduced=torch.nn.functional.interpolate(ds.tensors[0],(28,28),mode='area').flatten(1)
            old=torch.from_numpy(reference[split+'_images'].copy()).float().flatten(1)
            reduced=reduced-reduced.mean(1,keepdim=True);old=old-old.mean(1,keepdim=True)
            denom=reduced.norm(dim=1)*old.norm(dim=1)
            valid=denom>1e-8
            corr=((reduced*old).sum(1)/denom.clamp_min(1e-8))[valid]
            fraction=float((corr>.9).float().mean())
            assert fraction>.99,(split,'Unexpected image-order correspondence',fraction)
            alignment[split]={'cases':len(ds),'labels_identical':True,'same_index_images_compared':True,
                'nonconstant_images':len(corr),'pearson_min':float(corr.min()),
                'pearson_median':float(corr.median()),'fraction_correlation_above_point9':fraction}
    meta.update(native_image_size=224,official_md5=spec['official_md5'],
                reference_28_sha256=digest('data/pneumoniamnist.npz'),resolution_alignment=alignment)
    CACHE.mkdir(parents=True,exist_ok=True)
    json_write(CACHE/'resolution_alignment.json',meta)
    return datasets,channels,classes,meta


def configure():
    features.MANIFEST=MANIFEST;features.CACHE=CACHE;features.load_splits=native_splits
    experiment.MANIFEST=MANIFEST;experiment.CACHE=CACHE;experiment.ROOT=ROOT
    experiment.source_hashes=source_hashes
    analysis.MANIFEST=MANIFEST;analysis.CACHE=CACHE;analysis.ROOT=ROOT;analysis.OUT=OUT
    analysis.REPORT_FILE=Path('NATIVE_QUANTUM_ACCURACY_RESULTS.md')
    analysis.source_hashes=source_hashes;analysis.load_splits=native_splits


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','freeze','run','analyze'])
    parser.add_argument('--kernel',choices=['entangled','product','dot'])
    a=parser.parse_args();configure()
    if a.action=='prepare':
        features.prepare()
    elif a.action=='analyze':
        import sys
        sys.argv=[sys.argv[0],'--verify-circuits']
        analysis.main()
    elif a.action=='freeze':
        import sys
        sys.argv=[sys.argv[0],'freeze'];experiment.main()
    else:
        spec=json.loads(MANIFEST.read_text());experiment.verify_lock(spec)
        banks=features.load_features();rows=[]
        for kernel in ([a.kernel] if a.kernel else spec['kernels']):
            for seed in spec['seeds']:
                rows.append(experiment.run_one(spec,banks,kernel,seed))
                json_write(ROOT/f"progress-{a.kernel or 'all'}.json",{'rows':rows})
        linear=experiment.linear_control(spec,banks) if a.kernel in [None,'dot'] else None
        if a.kernel is None:
            json_write(ROOT/'summary.json',{'complete':True,'neural_conditions':len(rows),'rows':rows,'linear':linear})


if __name__=='__main__':
    main()
