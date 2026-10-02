"""Attention-output selective mitigation with charged calibration and pilots.

All allocation choices see measured outcomes and values, not true target
fidelities, true target noisy means, labels or simulator noise parameters.
See BIAS_AWARE_METHOD.md for assumptions and prior-art boundaries.
"""
from dataclasses import dataclass
import numpy as np
from .shots import softmax,integer_allocation,estimate,ShotConfig

POLICIES=['raw_uniform','raw_variance','raw_sensitivity','raw_combined',
          'calibrated_uniform','calibrated_pooled','calibrated_neyman','risk_uniform','risk_joint','risk_pooled',
          'risk_diagonal','risk_no_cal_cov']


@dataclass(frozen=True)
class RiskConfig:
    shots: int=64
    pilot: int=8
    calibration_fraction: float=.125
    slope_floor: float=.15
    alternations: int=4
    coordinate_sweeps: int=15

    def __post_init__(self):
        if self.shots<16 or self.pilot<1 or not 0<self.calibration_fraction<.5:
            raise ValueError('Unsupported production/pilot/calibration budget')


def attention_gram(f,values,beta):
    a=softmax(beta*f)
    y=np.einsum('rn,rnd->rd',a,values)
    j=beta*a[...,None]*(values-y[:,None,:])
    return j@np.swapaxes(j,-1,-2)


def continuous_counts(coeff,total,floor=1.):
    """Exact continuous floor-constrained square-root target by bisection."""
    coeff=np.maximum(np.asarray(coeff,dtype=float),1e-20)
    totals=np.broadcast_to(np.asarray(total,dtype=float),coeff.shape[:-1])
    if np.any(totals<floor*coeff.shape[-1]):
        raise ValueError('Total smaller than mandatory production floor')
    root=np.sqrt(coeff)
    low=np.zeros_like(totals)
    high=totals/np.maximum(root.min(-1),1e-30)
    for _ in range(55):
        scale=(low+high)/2
        used=np.maximum(floor,root*scale[...,None]).sum(-1)
        low=np.where(used<=totals,scale,low)
        high=np.where(used>totals,scale,high)
    return np.maximum(floor,root*((low+high)/2)[...,None])


def rounded_counts(target,total):
    out=np.floor(target+1e-10).astype(np.int64)
    totals=np.broadcast_to(np.asarray(total,dtype=np.int64),target.shape[:-1])
    remainder=totals-out.sum(-1)
    order=np.argsort(-(target-out),axis=-1,kind='stable')
    ranks=np.argsort(order,axis=-1)
    out+=(ranks<remainder[...,None])
    if np.any(out.sum(-1)!=totals) or np.any(out<1):
        raise ArithmeticError('Invalid rounded production allocation')
    return out


def fit_response(successes,counts,targets):
    """OLS calibration plus outcome-estimated heteroscedastic covariance."""
    means=2*successes/counts-1
    design=np.column_stack([targets,np.ones_like(targets)])
    inverse=np.linalg.pinv(design)
    coefficients=means@inverse.T
    p=(successes+.5)/(counts+1)
    variance=4*p*(1-p)/counts
    covariance=np.einsum('ik,bk,jk->bij',inverse,variance,inverse)
    return coefficients,covariance


def risk_components(f,mu,values,beta,slopes,offsets,covariance):
    gram=attention_gram(f,values,beta)
    bias=(slopes[:,None]-1)*f+offsets[:,None]
    bias_q=gram*bias[:,:,None]*bias[:,None,:]
    response=np.stack([f,np.ones_like(f)],axis=-1)/slopes[:,None,None]
    cal_q=gram*np.einsum('rni,rij,rmj->rnm',response,covariance,response)
    return gram,bias_q,cal_q


def solve_weights(bias_q,cal_q,variance_term,correction,initial,sweeps):
    # R=t'Mt+2*l't+constant, with M PSD. Batched cyclic coordinate descent.
    n=initial.shape[-1]
    matrix=bias_q+cal_q
    matrix=matrix.copy()
    matrix[:,np.arange(n),np.arange(n)]+=variance_term*correction[:,None]**2
    linear=-bias_q.sum(-1)+variance_term*correction[:,None]
    t=initial.copy()
    for _ in range(sweeps):
        for j in range(n):
            grad=np.einsum('rn,rn->r',matrix[:,j,:],t)+linear[:,j]
            diagonal=matrix[:,j,j]
            t[:,j]=np.where(diagonal>1e-18,
                           np.clip(t[:,j]-grad/np.maximum(diagonal,1e-18),0,1),t[:,j])
    return t


def surrogate(t,counts,bias_q,cal_q,coefficient,correction):
    r=1-t
    return (np.einsum('ri,rij,rj->r',r,bias_q,r)
            +np.einsum('ri,rij,rj->r',t,cal_q,t)
            +(coefficient*(1+correction[:,None]*t)**2/counts).sum(-1))


def design(f,mu,values,beta,slopes,offsets,covariance,total,policy,cfg):
    gram,bias_q,cal_q=risk_components(f,mu,values,beta,slopes,offsets,covariance)
    if policy=='risk_diagonal':
        n=f.shape[-1]
        mask=np.eye(n)[None]
        gram,bias_q,cal_q=gram*mask,bias_q*mask,cal_q*mask
    if policy=='risk_no_cal_cov':
        cal_q=np.zeros_like(cal_q)
    correction=1/slopes-1
    # Jeffreys-smoothed pilot means avoid spuriously zero outcome variance.
    coefficient=np.maximum(np.diagonal(gram,axis1=-2,axis2=-1)*(1-mu**2),1e-18)
    counts=np.broadcast_to(np.asarray(total)[:,None]/f.shape[-1],f.shape).copy()
    t=np.ones_like(f) if policy=='calibrated_neyman' else np.zeros_like(f)
    for _ in range(cfg.alternations):
        if policy!='calibrated_neyman':
            t=solve_weights(bias_q,cal_q,coefficient/counts,correction,t,cfg.coordinate_sweeps)
        if policy!='risk_uniform':
            counts=continuous_counts(coefficient*(1+correction[:,None]*t)**2,total)
    counts=rounded_counts(counts,total)
    return t,counts,surrogate(t,counts,bias_q,cal_q,coefficient,correction)


class _LegacySampler:
    def __init__(self,sampler):
        self.sampler=sampler
        self.p=np.empty((sampler.shape[0],1,sampler.shape[-1])) # shape only
        self.mode=sampler.mode
        self.circuit_calls=self.modeled_circuit_executions=0

    def draw(self,counts):
        self.circuit_calls+=int(np.count_nonzero(counts))
        return self.sampler.draw(counts[:,0,:])[:,None,:]


def run_policy(sampler,reference_sampler,values,beta,policy,cfg,heads=2):
    if policy not in POLICIES:
        raise ValueError('Unknown risk policy')
    rows,n=sampler.shape
    if rows%heads:
        raise ValueError('Rows must group all heads of each image')
    images=rows//heads
    full=cfg.shots*n*heads
    if policy=='raw_uniform':
        counts=np.full((rows,n),cfg.shots,dtype=np.int64)
        hits=sampler.draw(counts)
        return 2*hits/counts-1,{'production_counts':counts,'production_successes':hits,
            'shots_per_image':np.full(images,full),'calibration_shots_per_image':np.zeros(images,dtype=int),
            'pilot_shots_per_image':np.zeros(images,dtype=int)}
    if policy.startswith('raw_'):
        f,audit=estimate(_LegacySampler(sampler),values,beta,
                         ShotConfig(shots=cfg.shots,pilot=cfg.pilot,policy=policy.removeprefix('raw_')))
        return f[:,0,:],{'production_counts':audit['shots'][:,0,:],
            'production_successes':audit['successes'][:,0,:], 'legacy_rounds':np.stack(audit['decisions']),
            'shots_per_image':np.full(images,full),'calibration_shots_per_image':np.zeros(images,dtype=int),
            'pilot_shots_per_image':np.full(images,n*heads*cfg.pilot)}
    cal_total=max(32,4*int(round(full*cfg.calibration_fraction/4)))
    cal_counts=np.full((images,4),cal_total//4,dtype=np.int64)
    cal_hits=reference_sampler.draw(cal_counts)
    coefficients,cov=fit_response(cal_hits,cal_counts,np.array([0.,.25,.65,1.]))
    if policy in {'calibrated_pooled','risk_pooled'}:
        # Stationary-device control: pool paid reference outcomes across this batch.
        # No target labels, logits, ideal fidelities or target outcomes are pooled.
        pooled,pool_cov=fit_response(cal_hits.sum(0,keepdims=True),cal_counts.sum(0,keepdims=True),
                                     np.array([0.,.25,.65,1.]))
        coefficients=np.repeat(pooled,images,axis=0)
        cov=np.repeat(pool_cov,images,axis=0)
    failed=coefficients[:,0]<cfg.slope_floor
    slopes=np.repeat(np.maximum(coefficients[:,0],cfg.slope_floor),heads)
    offsets=np.repeat(coefficients[:,1],heads)
    covariance=np.repeat(cov,heads,axis=0)
    pilot=np.zeros((rows,n),dtype=np.int64)
    if policy in {'calibrated_uniform','calibrated_pooled'}:
        available=full-cal_total
        target=np.full((images,heads*n),available/(heads*n))
        counts=rounded_counts(target,np.full(images,available)).reshape(rows,n)
        t=np.ones((rows,n))
        risk=np.zeros(rows)
        pilot_hits=np.zeros_like(pilot)
    else:
        pilot[:]=cfg.pilot
        pilot_hits=sampler.draw(pilot)
        pilot_mu=2*(pilot_hits+.5)/(pilot+1)-1
        f=np.clip((pilot_mu-offsets[:,None])/slopes[:,None],0,1)
        available=full-cal_total-n*heads*cfg.pilot
        totals=np.full(rows,available//heads,dtype=int)
        totals.reshape(images,heads)[:,:available%heads]+=1
        t,counts,risk=design(f,pilot_mu,values,beta,slopes,offsets,covariance,totals,policy,cfg)
    t[np.repeat(failed,heads)]=0 # measured calibration failed; retain raw estimator
    hits=sampler.draw(counts)
    mean=2*hits/counts-1
    result=mean+t*((mean-offsets[:,None])/slopes[:,None]-mean)
    spent=counts.reshape(images,-1).sum(-1)+pilot.reshape(images,-1).sum(-1)+cal_total
    if np.any(spent!=full):
        raise ArithmeticError('Budget violation')
    audit={'production_counts':counts,'production_successes':hits,'pilot_counts':pilot,
        'pilot_successes':pilot_hits,'mitigation_weights':t,'calibration_counts':cal_counts,
        'calibration_successes':cal_hits,'calibration_coefficients':coefficients,'calibration_covariance':cov,
        'calibration_failed':failed,'estimated_surrogate':risk,'shots_per_image':spent,
        'calibration_shots_per_image':np.full(images,cal_total),
        'pilot_shots_per_image':pilot.reshape(images,-1).sum(-1)}
    return result,audit
