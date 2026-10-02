"""Maintain the distribution inventory after documentation or evidence changes.

This does not alter or regenerate any experiment's frozen scientific hash lock.
"""
from pathlib import Path
import hashlib
import json

root=Path(__file__).resolve().parent
ignored={'.log','.aux','.toc','.out','.lof','.lot','.pyc'}
records={}
for path in sorted(root.rglob('*')):
    if not path.is_file() or '__pycache__' in path.parts or path.suffix in ignored or path.name=='PACKAGE_CONTENTS.json':
        continue
    records[path.relative_to(root).as_posix()]={'bytes':path.stat().st_size,
        'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
(root/'PACKAGE_CONTENTS.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
print('Inventory:',len(records),'files')
