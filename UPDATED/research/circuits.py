"""PennyLane SWAP, destructive SWAP and compute-uncompute circuit backends.

Circuit measurements, basis decomposition and optional line routing are executed.
Noise is an explicitly specified simulator channel, never a physical-device claim.
"""
from dataclasses import dataclass, asdict
from functools import lru_cache
from itertools import product
import time
import numpy as np
import pennylane as qml


@dataclass(frozen=True)
class Noise:
    preparation: float = 0.
    one_qubit: float = 0.
    two_qubit: float = 0.
    amplitude: float = 0.
    phase: float = 0.
    bitflip: float = 0.
    phaseflip: float = 0.
    readout: float = 0.

    def __post_init__(self):
        if any(not 0 <= v < 1 for v in asdict(self).values()):
            raise ValueError('Noise probabilities must lie in [0,1)')


def _prepare(theta, wires, entangle=False):
    for i,w in enumerate(wires):
        qml.RY(theta[i], w)
    if entangle:
        for i in range(len(wires)-1):
            qml.CNOT([wires[i], wires[i+1]])
        # Data re-uploading after entanglement prevents a common unitary cancelling from the overlap.
        for i,w in enumerate(wires):
            qml.RZ(theta[i], w)


def logical_operations(q, k, method='swap', entangle=False):
    d=len(q)
    if method not in {'swap','destructive','uncompute'}:
        raise ValueError('Unknown overlap circuit')
    with qml.queuing.AnnotatedQueue() as queue:
        if method=='uncompute':
            _prepare(q,range(d),entangle)
            qml.adjoint(_prepare)(k,range(d),entangle)
        else:
            offset=1 if method=='swap' else 0
            _prepare(q,range(offset,offset+d),entangle)
            _prepare(k,range(offset+d,offset+2*d),entangle)
            if method=='swap':
                qml.Hadamard(0)
                for i in range(d):
                    qml.CSWAP([0,i+1,i+d+1])
                qml.Hadamard(0)
            else:
                for i in range(d):
                    qml.CNOT([i,i+d])
                    qml.Hadamard(i)
    return list(qml.tape.QuantumScript.from_queue(queue).operations)


def basis_operations(ops, route_line=False):
    expanded=qml.tape.QuantumScript(ops).expand(depth=10,stop_at=lambda op:len(op.wires)<=2).operations
    out=[]
    for op in expanded:
        if not route_line or len(op.wires)<2 or abs(int(op.wires[0])-int(op.wires[1]))==1:
            out.append(op)
            continue
        control,target=map(int,op.wires)
        direction=1 if control>target else -1
        swaps=[]
        while abs(target-control)>1:
            adjacent=target+direction
            swaps.append((target,adjacent))
            target=adjacent
        def add_swap(a,b):
            out.extend([qml.CNOT([a,b]),qml.CNOT([b,a]),qml.CNOT([a,b])])
        for a,b in swaps:
            add_swap(a,b)
        out.append(qml.map_wires(op,{int(op.wires[0]):control,int(op.wires[1]):target}))
        for a,b in reversed(swaps):
            add_swap(a,b)
    return out


def resource_counts(ops):
    clocks={}
    for op in ops:
        t=1+max([clocks.get(int(w),0) for w in op.wires],default=0)
        clocks.update({int(w):t for w in op.wires})
    return {'depth':max(clocks.values(),default=0), 'one_qubit_gates':sum(len(op.wires)==1 for op in ops),
            'two_qubit_gates':sum(len(op.wires)==2 for op in ops),
            'three_qubit_gates':sum(len(op.wires)==3 for op in ops), 'gate_count':len(ops)}


@lru_cache(maxsize=32)
def depolarizing_two(p):
    paulis=[np.eye(2),np.array([[0,1],[1,0]]),np.array([[0,-1j],[1j,0]]),np.diag([1,-1])]
    return [np.sqrt(1-p if i==j==0 else p/15)*np.kron(a,b)
            for (i,a),(j,b) in product(enumerate(paulis),repeat=2)]


def run_circuit(q, k, method='swap', shots=None, seed=0, noise=Noise(), route_line=False, entangle=False):
    q,k=np.asarray(q),np.asarray(k)
    if q.ndim!=1 or q.shape!=k.shape or len(q)<1:
        raise ValueError('Pass equal nonempty angle vectors')
    d=len(q)
    wires=d if method=='uncompute' else 2*d+int(method=='swap')
    logical=logical_operations(q,k,method,entangle)
    ops=basis_operations(logical,route_line)
    noisy=any(asdict(noise).values())
    dev=qml.device('default.mixed' if noisy else 'default.qubit', wires=wires, seed=seed)
    @qml.qnode(dev)
    def circuit():
        if noise.preparation:
            for w in range(wires):
                qml.BitFlip(noise.preparation,w)
        for op in ops:
            qml.apply(op)
            if len(op.wires)==2 and noise.two_qubit:
                qml.QubitChannel(depolarizing_two(noise.two_qubit),wires=op.wires)
            for w in op.wires:
                if len(op.wires)==1 and noise.one_qubit:
                    qml.DepolarizingChannel(noise.one_qubit,w)
                for rate,channel in ((noise.amplitude,qml.AmplitudeDamping),(noise.phase,qml.PhaseDamping),
                                     (noise.bitflip,qml.BitFlip),(noise.phaseflip,qml.PhaseFlip)):
                    if rate:
                        channel(rate,w)
        measured=[0] if method=='swap' else list(range(wires))
        if noise.readout:
            for w in measured:
                qml.BitFlip(noise.readout,w)
        return qml.sample(wires=measured) if shots else qml.probs(wires=measured)
    start=time.perf_counter()
    raw=np.asarray(qml.set_shots(circuit,shots=shots)() if shots else circuit())
    elapsed=time.perf_counter()-start
    if shots:
        bits=raw.reshape(shots,-1)
        if method=='swap':
            values=1-2*bits[:,0]
        elif method=='destructive':
            values=1-2*(np.sum(bits[:,:d]*bits[:,d:],axis=1)%2)
        else:
            values=np.all(bits==0,axis=1).astype(int)
        fidelity=float(values.mean())
    else:
        values=None
        if method=='swap':
            fidelity=float(raw[0]-raw[1])
        elif method=='uncompute':
            fidelity=float(raw[0])
        else:
            indices=np.arange(2**wires)
            bits=(indices[:,None] >> np.arange(wires-1,-1,-1)) & 1
            fidelity=float(raw @ (1-2*(np.sum(bits[:,:d]*bits[:,d:],axis=1)%2)))
    return {'fidelity':fidelity,'values':values,'seconds':elapsed,'shots':shots or 0,'circuit_calls':1,
            'logical_qubits':wires,'mapped_qubits':wires if route_line else None,
            'logical':resource_counts(logical),'compiled':resource_counts(ops),
            'connectivity':'synthetic_line' if route_line else 'all_to_all',
            'mode':('noisy_' if noisy else 'ideal_')+('finite_shot_circuit_simulation' if shots else 'circuit_simulation'),
            'noise':asdict(noise), 'physical_hardware':False}


class CircuitSampler:
    def __init__(self,q,k,rng,method='swap',noise=Noise(),route_line=False):
        from .models import angles
        self.q=angles(q).detach().cpu().numpy()
        self.k=angles(k).detach().cpu().numpy()
        self.p=np.empty((*self.q.shape[:-1],self.k.shape[-2])) # shape only; no exact oracle available
        self.rng,self.method,self.noise,self.route_line=rng,method,noise,route_line
        if method=='uncompute':
            raise ValueError('Signed SWAP allocation requires swap or destructive measurements')
        self.circuit_calls=self.modeled_circuit_executions=0
        self.mode=('noisy_' if any(asdict(noise).values()) else 'ideal_')+'finite_shot_circuit_simulation'

    def draw(self,counts):
        successes=np.zeros_like(counts)
        for index in zip(*np.nonzero(counts)):
            *batch,i,j=index
            r=run_circuit(self.q[tuple(batch)+(i,)],self.k[tuple(batch)+(j,)],self.method,
                          int(counts[index]),int(self.rng.integers(2**31)),self.noise,self.route_line)
            successes[index]=np.count_nonzero(r['values']==1)
            self.circuit_calls+=1
        return successes


class IdealCircuitEstimator:
    def __init__(self, method='swap'):
        self.method=method
        self.total_shots=self.circuit_calls=self.modeled_circuit_executions=0

    def __call__(self,q,k,v,ideal,beta):
        import torch
        from .models import angles
        qa,ka=angles(q).detach().cpu().numpy(),angles(k).detach().cpu().numpy()
        result=np.empty(ideal.shape)
        for index in np.ndindex(result.shape):
            *batch,i,j=index
            result[index]=run_circuit(qa[tuple(batch)+(i,)],ka[tuple(batch)+(j,)],self.method)['fidelity']
        self.circuit_calls+=result.size
        audit={'mode':'ideal_statevector_circuit_simulation','circuit_calls':result.size,'total_shots':0,
               'shots':np.zeros(result.shape,dtype=int),'standard_error':np.zeros_like(result),
               'modeled_circuit_executions':0}
        return torch.as_tensor(result,dtype=q.dtype,device=q.device),audit
