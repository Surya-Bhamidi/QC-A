"""Exact factorized density-matrix simulation of product-state destructive SWAP.

The full 2*d-wire circuit splits into d disjoint two-wire components for this
feature map and local noise. This simulates their gates/channels, then returns
the mean of the signed parity observable. It is not an ideal-fidelity oracle,
a hardware run, or a factorization valid for entangling/correlated circuits.
"""
from functools import lru_cache
import numpy as np
from .circuits import Noise

I=np.eye(2,dtype=complex)
X=np.array([[0,1],[1,0]],dtype=complex)
Y=np.array([[0,-1j],[1j,0]])
Z=np.diag([1.,-1.]).astype(complex)
H=np.array([[1,1],[1,-1]],dtype=complex)/np.sqrt(2)
CX=np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],dtype=complex)


def expand(matrix,wire):
    if wire==0:
        return np.einsum('...ij,kl->...ikjl',matrix,I).reshape(*matrix.shape[:-2],4,4)
    return np.einsum('ij,...kl->...ikjl',I,matrix).reshape(*matrix.shape[:-2],4,4)


def conjugate(rho,matrix):
    return matrix@rho@np.swapaxes(matrix.conj(),-1,-2)


def channel(rho,operators,wire):
    return sum(conjugate(rho,expand(k,wire)) for k in operators)


@lru_cache(maxsize=256)
def kraus(kind,p):
    if kind=='depolarizing':
        return (np.sqrt(1-p)*I,*(np.sqrt(p/3)*s for s in (X,Y,Z)))
    if kind=='amplitude':
        return (np.diag([1,np.sqrt(1-p)]),np.array([[0,np.sqrt(p)],[0,0]]))
    if kind=='phase':
        return (np.diag([1,np.sqrt(1-p)]),np.diag([0,np.sqrt(p)]))
    return (np.sqrt(1-p)*I,np.sqrt(p)*(X if kind=='bitflip' else Z))


def after_gate(rho,wires,noise):
    if len(wires)==2 and noise.two_qubit:
        # Exactly the uniform 15-nonidentity-Pauli channel in circuits.py.
        rho=(1-16*noise.two_qubit/15)*rho+(4*noise.two_qubit/15)*np.eye(4)
    for wire in wires:
        if len(wires)==1 and noise.one_qubit:
            rho=channel(rho,kraus('depolarizing',noise.one_qubit),wire)
        for kind in ['amplitude','phase','bitflip','phaseflip']:
            p=getattr(noise,kind)
            if p:
                rho=channel(rho,kraus(kind,p),wire)
    return rho


def destructive_means(q,k,noise=Noise()):
    q,k=np.broadcast_arrays(np.asarray(q,dtype=float),np.asarray(k,dtype=float))
    if q.ndim<1 or not np.isfinite(q).all() or not np.isfinite(k).all():
        raise ValueError('Finite angle arrays ending in a feature dimension required')
    rho=np.zeros((*q.shape,4,4),dtype=complex)
    rho[...,0,0]=1
    if noise.preparation:
        for wire in [0,1]:
            rho=channel(rho,kraus('bitflip',noise.preparation),wire)
    for wire,theta in [(0,q),(1,k)]:
        u=np.empty((*theta.shape,2,2),dtype=complex)
        u[...,0,0]=u[...,1,1]=np.cos(theta/2)
        u[...,0,1]=-np.sin(theta/2)
        u[...,1,0]=np.sin(theta/2)
        rho=after_gate(conjugate(rho,expand(u,wire)),[wire],noise)
    rho=after_gate(conjugate(rho,CX),[0,1],noise)
    rho=after_gate(conjugate(rho,expand(H,0)),[0],noise)
    if noise.readout:
        for wire in [0,1]:
            rho=channel(rho,kraus('bitflip',noise.readout),wire)
    means=(1-2*rho[...,3,3].real).prod(-1)
    return np.clip(means,-1,1)


def reference_angles(d):
    """Known F values use controlled angle differences; no target/test labels."""
    target=np.array([0.,.25,.65,1.])
    q=np.full((len(target),d),.37)
    delta=2*np.arccos(target**(1/(2*d)))
    k=q+delta[:,None]
    return q,k,target


class NoisyParitySampler:
    """Sample simulated circuit parities. The allocation API exposes no means."""
    mode='factorized_noisy_density_matrix_simulation_with_parity_sampling'

    def __init__(self,means,rng):
        self._probabilities=np.clip((np.asarray(means)+1)/2,0,1)
        self.shape=self._probabilities.shape
        self.rng=rng
        self.shots=self.measurement_jobs=0

    def draw(self,counts):
        counts=np.asarray(counts,dtype=np.int64)
        if counts.shape!=self.shape or np.any(counts<0):
            raise ValueError('Counts must match simulator shape and be nonnegative')
        self.shots+=int(counts.sum())
        self.measurement_jobs+=int(np.count_nonzero(counts))
        return self.rng.binomial(counts,self._probabilities)
