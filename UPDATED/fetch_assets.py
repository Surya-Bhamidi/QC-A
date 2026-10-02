"""Restore public native data; validate the assets shipped with this package."""
from pathlib import Path
import hashlib
import os
import urllib.request

ROOT=Path(__file__).resolve().parent
os.chdir(ROOT)
from research.fetch_native_dataset import main

main()
weight=Path('pretrained_models/resnet18-f37072fd.pth')
expected='f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'
if not weight.exists():
    weight.parent.mkdir(exist_ok=True)
    temporary=weight.with_suffix('.download')
    urllib.request.urlretrieve('https://download.pytorch.org/models/resnet18-f37072fd.pth',temporary)
    assert hashlib.sha256(temporary.read_bytes()).hexdigest()==expected
    temporary.replace(weight)
assert hashlib.sha256(weight.read_bytes()).hexdigest()==expected
print('Public data and encoder assets verified')
