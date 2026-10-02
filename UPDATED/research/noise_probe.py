"""Small disclosed circuit-only experiment; never substitutes for full-test noisy inference."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
from .circuits import Noise, run_circuit
from .runtime import json_write, snapshot


def calibration_profile(path='calibrations/props_guadalupe.json'):
    path=Path(path)
    data=json.loads(path.read_text())
    readouts=[next(p['value'] for p in q if p['name']=='readout_error') for q in data['qubits']]
    gate_errors={}
    for name in ['sx','cx']:
        gate_errors[name]=float(np.median([next(p['value'] for p in g['parameters'] if p['name']=='gate_error')
                                         for g in data['gates'] if g['gate']==name]))
    profile={'source':'https://raw.githubusercontent.com/Qiskit/qiskit-ibm-runtime/main/qiskit_ibm_runtime/fake_provider/backends/guadalupe/props_guadalupe.json',
             'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'backend':data['backend_name'],
             'calibration_date':data['last_update_date'],'median_average_gate_infidelity':gate_errors,
             'median_symmetric_readout':float(np.median(readouts)),
             'scope':'Historical calibration-derived homogeneous proxy, not backend-faithful simulation. Native gate synthesis, per-qubit asymmetry, scheduling, idle errors and current device drift are not represented.'}
    # For a uniform non-identity Pauli channel on Hilbert dimension D, p=r*(D+1)/D.
    noise=Noise(one_qubit=1.5*gate_errors['sx'],two_qubit=1.25*gate_errors['cx'],readout=profile['median_symmetric_readout'])
    return noise,profile


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='runs/circuits')
    p.add_argument('--pairs',type=int,default=3)
    a=p.parse_args()
    out=Path(a.output)
    out.mkdir(parents=True,exist_ok=True)
    snapshot(out)
    calibrated,profile=calibration_profile()
    json_write(out/'calibration_profile.json',profile)
    rng=np.random.default_rng(710)
    pair_angles=rng.uniform(0,np.pi,(a.pairs,2,4))
    np.save(out/'pair_angles.npy',pair_angles)
    results=[]
    path=out/'probes.jsonl'
    done=set()
    if path.exists():
        results=[json.loads(line) for line in path.read_text().splitlines()]
        done={r['id'] for r in results}
    settings=[('ideal',Noise()),('readout_only',Noise(readout=.05)),
              ('calibration_proxy',calibrated),
              ('full_synthetic',Noise(preparation=.002,one_qubit=.001,two_qubit=.01,amplitude=.001,phase=.001,bitflip=.0002,phaseflip=.0002,readout=.02))]
    for pair,(q,k) in enumerate(pair_angles):
        for method in ['swap','destructive','uncompute']:
            ideal=run_circuit(q,k,method)
            for label,noise in settings:
                key=f'{pair}-{method}-{label}'
                if key in done:
                    continue
                r=run_circuit(q,k,method,shots=1024,seed=900+pair,noise=noise)
                raw=r.pop('values')
                np.save(out/f'{key}-outcomes.npy',raw)
                r.update({'id':key,'pair':pair,'method':method,'scenario':label,'ideal_fidelity':ideal['fidelity'],
                          'squared_error':(r['fidelity']-ideal['fidelity'])**2})
                # Measured effect of a known symmetric readout inversion. Does not correct gate noise.
                # Ancilla SWAP only: destructive parity has no simple common attenuation factor.
                if method=='swap' and noise.readout:
                    mitigation=r['fidelity']/(1-2*noise.readout)
                    r.update({'mitigated_fidelity':mitigation,'mitigated_squared_error':(mitigation-ideal['fidelity'])**2,
                              'mitigation':'known-channel symmetric readout inversion; calibration-shot overhead excluded'})
                results.append(r)
                with path.open('a') as f:
                    f.write(json.dumps(r)+'\n')
                print(key,'seconds',round(r['seconds'],3),flush=True)
    resources=[]
    for d in [2,4,6,8]:
        from .circuits import logical_operations,basis_operations,resource_counts
        for method in ['swap','destructive','uncompute']:
            ops=logical_operations(np.zeros(d),np.ones(d),method)
            resources.append({'register_qubits':d,'method':method,'logical':resource_counts(ops),
                              'all_to_all':resource_counts(basis_operations(ops)),
                              'synthetic_line':resource_counts(basis_operations(ops,True)),
                              'logical_qubits':d if method=='uncompute' else 2*d+int(method=='swap')})
    json_write(out/'resources.json',resources)
    entangled=[]
    for pair,(q,k) in enumerate(pair_angles):
        for entangle in [False,True]:
            r=run_circuit(q,k,'uncompute',shots=1024,seed=501+pair,entangle=entangle)
            r.pop('values')
            entangled.append({'pair':pair,'entangle':entangle,**r})
    json_write(out/'entangling_probes.json',entangled)
    json_write(out/'summary.json',{'pairs':a.pairs,'shots':sum(r['shots'] for r in results),
               'circuit_calls':len(results),'simulator_seconds':sum(r['seconds'] for r in results),
               'scope':'Synthetic angle pairs; no classification accuracy and no physical QPU execution',
               'additional_calls':{'ideal_reference':a.pairs*3,'entangling_finite_shot':len(entangled)},
               'additional_shots':1024*len(entangled)})


if __name__=='__main__':
    main()
