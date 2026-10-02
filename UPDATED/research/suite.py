"""Frozen manifest runner; resumable without overwriting completed runs."""
import argparse
import hashlib
import json
from pathlib import Path
from .config import Config
from .train import train
from .runtime import json_write


def main():
    p=argparse.ArgumentParser()
    p.add_argument('manifest')
    p.add_argument('--output',default='runs/central')
    p.add_argument('--validation-only',action='store_true')
    p.add_argument('--freeze',action='store_true')
    a=p.parse_args()
    path=Path(a.manifest)
    manifest=json.loads(path.read_text())
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    freeze_path=path.with_suffix('.frozen.json')
    if a.freeze:
        if freeze_path.exists():
            raise FileExistsError('Protocol is already frozen; create a new version to change it')
        from datetime import datetime, timezone
        json_write(freeze_path, {'sha256':digest,'utc':datetime.now(timezone.utc).isoformat(),
                                'selection':'validation-only pilot; no new test evaluation before this freeze'})
        print('Frozen',digest)
        return
    if not a.validation_only:
        if not freeze_path.exists() or json.loads(freeze_path.read_text())['sha256']!=digest:
            raise ValueError('Freeze this exact manifest before final test evaluation')
    for dataset in manifest['datasets']:
        for seed in manifest['seeds']:
            for model in manifest['models']:
                c=Config(**(manifest['base']|{'dataset':dataset,'model':model,'seed':seed}))
                train(c,a.output,validation_only=a.validation_only)


if __name__=='__main__':
    main()
