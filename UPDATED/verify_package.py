"""Verify byte identity of packaged research evidence without training."""
from pathlib import Path
import hashlib
import json

root=Path(__file__).resolve().parent
records=json.loads((root/'PACKAGE_CONTENTS.json').read_text(encoding='utf-8'))
failures=[]
for name,record in records.items():
    path=root/name
    if not path.is_file():
        failures.append(name+': missing (run fetch_assets.py for the raw dataset)')
    elif path.stat().st_size!=record['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:
        failures.append(name+': content mismatch')
if failures:
    raise SystemExit('\n'.join(failures))
print('Verified',len(records),'packaged files against their SHA256 manifest')
