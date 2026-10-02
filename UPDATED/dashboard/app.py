"""Research-only dashboard backed by completed research checkpoints."""
from pathlib import Path
import io
import base64
import json
import sys
import time
import numpy as np
import torch
from PIL import Image
from flask import Flask, jsonify, request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.config import Config
from research.models import MatchedViT
from research.shots import ShotConfig,ShotEstimator
from research.metrics import probabilities
from research.circuits import IdealCircuitEstimator,CircuitSampler,Noise,run_circuit,logical_operations,basis_operations,resource_counts

app=Flask(__name__,static_folder='static')
torch.set_num_threads(2)


def runs():
    return {p.parent.name:p.parent for p in (ROOT/'runs'/'central').glob('*/test_metrics.json')}


@app.get('/')
def index():
    return app.send_static_file('research.html')


@app.get('/api/models')
def models():
    out=[]
    for name,path in runs().items():
        c=Config.read(path/'config.json')
        out.append({'id':name,'model':c.model,'seed':c.seed,'dataset':c.dataset})
    return jsonify({'models':sorted(out,key=lambda r:(r['seed'],r['model'])),'research_only':True})


@app.post('/api/predict')
def predict():
    try:
        data=request.get_json() or {}
        run_id=data.get('run')
        available=runs()
        if run_id not in available:
            return jsonify(error='Select a completed research checkpoint'),400
        folder=available[run_id]
        c=Config.read(folder/'config.json')
        mode=data.get('mode','analytical')
        if mode not in {'analytical','emulator','ideal_circuit','finite_circuit'}:
            return jsonify(error='Unknown execution mode'),400
        if mode!='analytical' and c.model!='fidelity':
            return jsonify(error='Circuit and shot evaluation apply to fidelity attention'),400
        sample=int(data.get('sample',0))
        with np.load(ROOT/'data'/f'{c.dataset}.npz') as raw:
            if not 0<=sample<len(raw['val_images']):
                return jsonify(error='Validation sample index out of range'),400
            array=raw['val_images'][sample]
            label=int(raw['val_labels'][sample].item())
        if array.ndim==2:
            array=array[...,None]
        x=torch.from_numpy(array.copy()).permute(2,0,1).unsqueeze(0).float()/127.5-1
        model=MatchedViT(c,array.shape[-1],2)
        model.load_state_dict(torch.load(folder/'best.pt',weights_only=True)['model'])
        model.eval()
        estimator=None
        shots=int(data.get('shots',64))
        policy=data.get('policy','uniform')
        circuit_method=data.get('circuit_method','swap')
        if circuit_method not in {'swap','destructive'}:
            return jsonify(error='Classifier circuit must be swap or destructive'),400
        if shots not in [16,32,64,128,256,512,1024]:
            return jsonify(error='Unsupported shot budget'),400
        if mode=='emulator':
            estimator=ShotEstimator(ShotConfig(shots=shots,policy=policy),seed=123)
        elif mode=='ideal_circuit':
            estimator=IdealCircuitEstimator(circuit_method)
        elif mode=='finite_circuit':
            estimator=ShotEstimator(ShotConfig(shots=shots,policy=policy),seed=123,
                                    sampler_factory=lambda q,k,rng:CircuitSampler(q,k,rng,method=circuit_method))
        model.set_estimator(estimator,capture=True)
        start=time.perf_counter()
        with torch.no_grad():
            logits=model(x).numpy()
        seconds=time.perf_counter()-start
        calibration=json.loads((folder/'calibration.json').read_text())
        calibration_label='validation-fitted analytical temperature'
        if mode in {'emulator','finite_circuit'}:
            measured=ROOT/'runs'/'measurements'/run_id/f'{policy}-s{shots}'/'test_metrics.json'
            if measured.exists():
                calibration['temperature']=json.loads(measured.read_text())['temperature']
                calibration_label='validation-fitted temperature for matching ideal signed-shot distribution'
            else:
                calibration_label+='; shot-mode calibration unavailable'
        p=probabilities(logits,calibration['temperature'])[0]
        last=model.blocks[-1].attn.last
        audit=last['audit']
        selected=lambda t:t[0,0].tolist()
        ops=logical_operations(np.zeros(c.qk_dim),np.zeros(c.qk_dim),circuit_method)
        img=Image.fromarray(array[:,:,0] if array.shape[-1]==1 else array)
        buffer=io.BytesIO()
        img.resize((224,224)).save(buffer,format='PNG')
        return jsonify({'prediction':int(p.argmax()),'probabilities':p.tolist(),'ground_truth':label,
                        'sample_split':'validation','calibration':calibration_label,'temperature':calibration['temperature'],
                        'mode':audit['mode'] if audit else 'classical_analytical','latency_seconds':seconds,
                        'circuit_method':circuit_method,
                        'image':'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode(),
                        'attention':selected(last['attention']), 'ideal_fidelity':selected(last['ideal']) if c.model=='fidelity' else None,
                        'estimated_fidelity':selected(last['estimated']) if c.model=='fidelity' else None,
                        'shot_allocation':audit['shots'][0,0].tolist() if audit else None,
                        'fidelity_standard_error':audit['standard_error'][0,0].tolist() if audit else None,
                        'total_shots':estimator.total_shots if estimator else 0,'actual_circuit_calls':estimator.circuit_calls if estimator else 0,
                        'circuit_resource_reference':resource_counts(basis_operations(ops)),
                        'probe_q':last['q'][0,0,0].tolist(),'probe_k':last['k'][0,0,1].tolist(),
                        'research_only':True,'heatmap_scope':'last block, head 0; model attention, not a lesion explanation'})
    except (ValueError,TypeError,KeyError) as e:
        return jsonify(error=str(e)),400


@app.post('/api/probe')
def probe():
    try:
        data=request.get_json() or {}
        q=np.asarray(data.get('q',[]),dtype=float)
        k=np.asarray(data.get('k',[]),dtype=float)
        if q.shape!=(4,) or k.shape!=(4,) or not np.isfinite(q).all() or not np.isfinite(k).all():
            return jsonify(error='Provide four finite query/key features from the fidelity model'),400
        method=data.get('method','swap')
        strength=float(data.get('strength',.01))
        if not 0<=strength<=.1:
            return jsonify(error='Noise strength must be between 0 and 0.1'),400
        noise_type=data.get('noise','none')
        if noise_type=='calibration_proxy':
            from research.noise_probe import calibration_profile
            noise,profile=calibration_profile(ROOT/'calibrations'/'props_guadalupe.json')
        elif noise_type=='none':
            noise=Noise()
        elif noise_type in {'readout','one_qubit','two_qubit','amplitude','phase','bitflip','phaseflip','preparation'}:
            noise=Noise(**{noise_type:strength})
        else:
            return jsonify(error='Unknown noise model'),400
        shots=int(data.get('shots',1024))
        if shots not in [16,32,64,128,256,512,1024]:
            return jsonify(error='Unsupported shots'),400
        qa,ka=(np.tanh(q)+1)*np.pi/2,(np.tanh(k)+1)*np.pi/2
        r=run_circuit(qa,ka,method,shots,123,noise)
        r.pop('values')
        r['ideal_fidelity']=float(np.prod(np.cos((qa-ka)/2)**2))
        r['scope']='One selected CLS-to-patch overlap only; classifier prediction above is unchanged'
        mitigation=data.get('mitigation','none')
        if mitigation=='readout_inverse':
            if method!='swap' or not noise.readout:
                return jsonify(error='Readout inversion requires ancilla SWAP with readout noise'),400
            r['mitigated_fidelity']=r['fidelity']/(1-2*noise.readout)
            r['mitigation_scope']='Known-channel inversion; no calibration-shot cost'
        elif mitigation!='none':
            return jsonify(error='Unknown mitigation'),400
        return jsonify(r)
    except (ValueError,TypeError) as e:
        return jsonify(error=str(e)),400


if __name__=='__main__':
    app.run(host='127.0.0.1',port=5050,debug=False,threaded=False)
