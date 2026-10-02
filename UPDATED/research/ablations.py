"""Run explicit ablation entries from a frozen JSON manifest."""
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
from .config import Config
from .train import train
from .runtime import json_write


def main():
    p=argparse.ArgumentParser()
    p.add_argument('manifest')
    p.add_argument('--output',default='runs/ablations')
    p.add_argument('--freeze',action='store_true')
    a=p.parse_args()
    path=Path(a.manifest)
    manifest=json.loads(path.read_text())
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    frozen=path.with_suffix('.frozen.json')
    if a.freeze:
        if frozen.exists():
            raise FileExistsError(frozen)
        json_write(frozen,{'sha256':digest,'utc':datetime.now(timezone.utc).isoformat(),
                          'status':'fixed secondary analyses, no tuning based on their test outcomes'})
        return
    if json.loads(frozen.read_text())['sha256']!=digest:
        raise ValueError('Ablation manifest changed after freeze')
    rows=[]
    for entry in manifest['entries']:
        for seed in manifest['seeds']:
            c=Config(**(manifest['base']|entry['override']|{'seed':seed}))
            baseline=Path('runs/central')/c.experiment_id
            folder=baseline if (baseline/'test_metrics.json').exists() else train(c,a.output)
            rows.append({'ablation':entry['name'],'seed':seed,'run':str(folder),'config':c.dict()})
            json_write(Path(a.output)/'index.json',rows)


if __name__=='__main__':
    main()
