"""Preserve every verification log, including failures."""
from pathlib import Path
import datetime
import sys
import unittest

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
out=root/'verification'
out.mkdir(exist_ok=True)
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
with (out/f'tests-{stamp}.txt').open('w',encoding='utf-8') as log:
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.discover(str(root/'tests'),pattern='test_*.py'))
print('Tests:',result.testsRun,'failures:',len(result.failures),'errors:',len(result.errors),'log:',out/f'tests-{stamp}.txt')
raise SystemExit(not result.wasSuccessful())
