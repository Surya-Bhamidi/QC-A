"""Secondary isolated-channel strength sweep on saved synthetic angle pairs."""
from pathlib import Path
from dataclasses import asdict
import hashlib
import json
import numpy as np
from .circuits import Noise,run_circuit
from .runtime import json_write,snapshot


def main():
    out=Path('runs/noise_strengths')
    out.mkdir(parents=True,exist_ok=True)
    source=Path('runs/circuits/pair_angles.npy')
    spec={'pairs_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'shots':1024,
          'strengths':[.001,.01,.05],
          'channels':['preparation','one_qubit','two_qubit','amplitude','phase','bitflip','phaseflip','readout'],
          'methods':['swap','destructive','uncompute'],
          'scope':'Secondary synthetic-pair experiment; no clinical data or parameter selection',
          'seed_rule':'1800 + pair*100 + channel_index*10 + strength_index'}
    spec_path=out/'protocol.json'
    if spec_path.exists():
        assert json.loads(spec_path.read_text())==spec,'Protocol changed; use a new directory'
    else:
        json_write(spec_path,spec)
        snapshot(out)
    path=out/'probes.jsonl'
    rows=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    done={r['id'] for r in rows}
    for pair,(q,k) in enumerate(np.load(source)):
        ideal=float(np.prod(np.cos((q-k)/2)**2))
        for method in spec['methods']:
            for ci,channel in enumerate(spec['channels']):
                for si,strength in enumerate(spec['strengths']):
                    key=f'{pair}-{method}-{channel}-{strength}'
                    if key in done:
                        continue
                    r=run_circuit(q,k,method,shots=spec['shots'],seed=1800+pair*100+ci*10+si,
                                  noise=Noise(**{channel:strength}))
                    np.save(out/(key+'-outcomes.npy'),r.pop('values'))
                    r.update({'id':key,'pair':pair,'method':method,'channel':channel,'strength':strength,
                              'ideal_fidelity':ideal,'squared_error':(r['fidelity']-ideal)**2})
                    if method=='swap' and channel=='readout':
                        r['mitigated_fidelity']=r['fidelity']/(1-2*strength)
                        r['mitigated_squared_error']=(r['mitigated_fidelity']-ideal)**2
                    with path.open('a') as f:
                        f.write(json.dumps(r)+'\n')
                    rows.append(r)
            print('Completed pair',pair,method,'total',len(rows),flush=True)
    assert len(rows)==216
    json_write(out/'summary.json',{'circuits':len(rows),'shots':sum(r['shots'] for r in rows),
        'simulator_seconds':sum(r['seconds'] for r in rows),'physical_hardware':False,'scope':spec['scope']})


if __name__=='__main__':
    main()
