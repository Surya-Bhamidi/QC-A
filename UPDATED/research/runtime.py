from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import time
import zipfile
import numpy as np
import torch


def seed_all(seed, threads=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def json_write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')
    replace_retry(temp, path)


def replace_retry(source, destination):
    """Atomic replacement with bounded retries for transient Windows/OneDrive file locks."""
    for attempt in range(20):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.05*(attempt+1))


def torch_save_atomic(data, path):
    temp = Path(path).with_suffix('.tmp')
    torch.save(data, temp)
    replace_retry(temp, path)


def event(directory, **data):
    data['utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    with (Path(directory)/'events.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(data, allow_nan=False)+'\n')


def environment():
    def git(*args):
        try:
            return subprocess.check_output(['git',*args], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    packages = {p.metadata['Name']: p.version for p in importlib.metadata.distributions() if p.metadata['Name']}
    sources = {p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path('research').glob('*.py'))}
    return {'python': sys.version, 'platform': platform.platform(), 'device': 'cpu', 'processor': platform.processor(),
            'torch_threads': torch.get_num_threads(), 'cuda_available': torch.cuda.is_available(),
            'packages': packages, 'commit': git('rev-parse','HEAD'), 'working_tree': git('status','--short'),
            'source_sha256': sources}


def snapshot(directory):
    directory = Path(directory)
    json_write(directory/'environment.json', environment())
    with zipfile.ZipFile(directory/'source.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(Path('research').glob('*.py')):
            z.write(p, p.as_posix())


def save_checkpoint(path, model, optimizer, scheduler, generator, epoch, best, elapsed):
    data = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'epoch': epoch, 'best': best, 'elapsed': elapsed, 'rng_torch': torch.get_rng_state(),
            'rng_numpy': np.random.get_state(), 'rng_python': random.getstate(), 'rng_loader': generator.get_state()}
    torch_save_atomic(data, path)


def restore_checkpoint(path, model, optimizer, scheduler, generator):
    # Only locally produced checkpoints are accepted by the command-line workflow.
    state = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    torch.set_rng_state(state['rng_torch'])
    np.random.set_state(state['rng_numpy'])
    random.setstate(state['rng_python'])
    generator.set_state(state['rng_loader'])
    return state
