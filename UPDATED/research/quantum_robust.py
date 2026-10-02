"""Differentiable four-qubit circuit attention and generalization utilities.

Torch executes gates on small statevectors for training. PennyLane can execute
the matching compute-uncompute circuit for every inference overlap. Both are
classical quantum-circuit simulators; neither implies physical QPU execution.
"""
import math
from functools import lru_cache
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .models import Attention,MatchedViT,angles


def rotation(state,theta,wire,d,kind):
    shape=state.shape
    blocks=state.reshape(*shape[:-1],2**wire,2,2**(d-wire-1))
    a,b=blocks.select(-2,0),blocks.select(-2,1)
    theta=theta[...,None,None]
    if kind=='y':
        c,s=torch.cos(theta/2),torch.sin(theta/2)
        out=torch.stack([c*a-s*b,s*a+c*b],dim=-2)
    else:
        out=torch.stack([torch.exp(-.5j*theta)*a,torch.exp(.5j*theta)*b],dim=-2)
    return out.reshape(shape)


@lru_cache(maxsize=32)
def cnot_permutation(d,control,target):
    i=torch.arange(2**d)
    return i^(((i>>(d-control-1))&1)<<(d-target-1))


def feature_states(theta,entangled=True):
    d=theta.shape[-1]
    if d!=4:
        raise ValueError('This follow-up uses exactly four qubits')
    dtype=torch.complex128 if theta.dtype==torch.float64 else torch.complex64
    state=torch.zeros((*theta.shape[:-1],2**d),device=theta.device,dtype=dtype)
    state[...,0]=1
    for wire in range(d):
        state=rotation(state,theta[...,wire],wire,d,'y')
    if entangled:
        for wire in range(d-1):
            state=state.index_select(-1,cnot_permutation(d,wire,wire+1).to(theta.device))
        for wire in range(d):
            state=rotation(state,theta[...,wire],wire,d,'z')
            state=rotation(state,.5*theta[...,wire],wire,d,'y')
    return state


def feature_circuit(theta,entangled=True):
    import pennylane as qml
    for wire in range(4):
        qml.RY(theta[...,wire],wire)
    if entangled:
        for wire in range(3):
            qml.CNOT([wire,wire+1])
        for wire in range(4):
            qml.RZ(theta[...,wire],wire)
            qml.RY(.5*theta[...,wire],wire)


class PennyLaneOverlap:
    def __init__(self,entangled=True,chunk_size=4096):
        import pennylane as qml
        self.entangled=entangled
        self.chunk_size=chunk_size
        self.qnode_calls=0
        self.circuit_settings=0
        self.shots=0
        device=qml.device('default.qubit',wires=4)
        @qml.qnode(device)
        def overlap(q,k):
            feature_circuit(q,self.entangled)
            qml.adjoint(feature_circuit)(k,self.entangled)
            return qml.probs(wires=range(4))
        self.circuit=overlap

    def __call__(self,qa,ka):
        qa,ka=np.broadcast_arrays(np.asarray(qa,dtype=np.float64),np.asarray(ka,dtype=np.float64))
        shape=qa.shape[:-1]
        qa,ka=qa.reshape(-1,4),ka.reshape(-1,4)
        result=[]
        for start in range(0,len(qa),self.chunk_size):
            p=np.asarray(self.circuit(qa[start:start+self.chunk_size],ka[start:start+self.chunk_size]))
            result.append(p[:,0])
            self.qnode_calls+=1
            self.circuit_settings+=len(p)
        return np.concatenate(result).reshape(shape)


class CircuitAttention(Attention):
    def __init__(self,c,kernel):
        super().__init__(c)
        self.kernel=kernel
        self.circuit_backend=None
        self.prepared_states=0

    def forward(self,x):
        b,n,e=x.shape
        q=self.q(x).view(b,n,self.heads,self.dk).transpose(1,2)
        k=self.k(x).view(b,n,self.heads,self.dk).transpose(1,2)
        v=self.v(x).view(b,n,self.heads,self.dv).transpose(1,2)
        if self.kernel=='dot':
            scores=(q@k.transpose(-1,-2))/math.sqrt(self.dk)
        else:
            qa,ka=angles(q),angles(k)
            if self.circuit_backend is None:
                qs=feature_states(qa,self.kernel=='entangled')
                ks=feature_states(ka,self.kernel=='entangled')
                self.prepared_states+=2*b*self.heads*n
                scores=(qs.conj()@ks.transpose(-1,-2)).abs().square()
            else:
                if self.training:
                    raise ValueError('PennyLane override is inference-only')
                f=self.circuit_backend(qa.detach().numpy()[:,:,:,None,:],ka.detach().numpy()[:,:,None,:,:])
                scores=torch.as_tensor(f,dtype=x.dtype,device=x.device)
        attention=(self.beta*scores).softmax(-1)
        output=(self.drop(attention)@v).transpose(1,2).reshape(b,n,e)
        return self.out(output)


class QuantumRobustViT(MatchedViT):
    def __init__(self,c,kernel):
        super().__init__(c)
        for block in self.blocks:
            block.attn=CircuitAttention(c,kernel)
        self.kernel=kernel

    def set_circuit_backend(self,backend):
        if self.kernel=='dot':
            raise ValueError('Dot is the classical matched control')
        for block in self.blocks:
            block.attn.circuit_backend=backend


def augment(x,enabled=True):
    if not enabled:
        return x
    b=len(x)
    # Per-image affine transforms with border padding (no cyclic wraparound).
    angle=(torch.rand(b,device=x.device)*2-1)*(10*math.pi/180)
    scale=.9+.2*torch.rand(b,device=x.device)
    shift=(torch.rand(b,2,device=x.device)*2-1)*.1
    affine=torch.zeros(b,2,3,device=x.device,dtype=x.dtype)
    affine[:,0,0]=torch.cos(angle)*scale
    affine[:,1,1]=torch.cos(angle)*scale
    affine[:,0,1]=-torch.sin(angle)*scale
    affine[:,1,0]=torch.sin(angle)*scale
    affine[:,:,2]=shift
    grid=F.affine_grid(affine,x.shape,align_corners=False)
    transformed=F.grid_sample(x,grid,mode='bilinear',padding_mode='border',align_corners=False)
    flip=torch.rand(b,device=x.device)<.5
    transformed=torch.where(flip[:,None,None,None],transformed.flip(-1),transformed)
    # Independent brightness/contrast changes act on [0,1] intensity.
    z=(transformed+1)/2
    contrast=.8+.4*torch.rand(b,1,1,1,device=x.device)
    brightness=(torch.rand(b,1,1,1,device=x.device)*2-1)*.1
    z=((z-.5)*contrast+.5+brightness).clamp(0,1)
    return z*2-1


class SAMStep:
    """Two-pass SAM around a base optimizer; restores parameters before updating."""
    def __init__(self,optimizer,rho=.05):
        self.optimizer=optimizer
        self.rho=rho

    def step(self,closure):
        self.optimizer.zero_grad(set_to_none=True)
        rng=torch.get_rng_state()
        loss=closure();loss.backward()
        params=[p for group in self.optimizer.param_groups for p in group['params'] if p.grad is not None]
        norm=torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in params]))
        backups=[]
        with torch.no_grad():
            for p in params:
                backups.append((p,p.detach().clone()))
                p.add_(p.grad,alpha=float(self.rho/(norm+1e-12)))
        try:
            self.optimizer.zero_grad(set_to_none=True)
            torch.set_rng_state(rng) # same dropout realization in both passes
            perturbed=closure();perturbed.backward()
        finally:
            with torch.no_grad():
                for p,old in backups:
                    p.copy_(old)
        nn.utils.clip_grad_norm_(params,1.)
        self.optimizer.step()
        return loss.detach()
