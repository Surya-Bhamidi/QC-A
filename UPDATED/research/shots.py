"""Sequential inference estimators. Probability emulation is explicitly not circuit execution."""
from dataclasses import dataclass
import numpy as np
import torch


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def integer_allocation(weights, totals):
    weights = np.maximum(weights, 1e-15)
    raw = weights / weights.sum(-1, keepdims=True) * totals[..., None]
    out = np.floor(raw).astype(np.int64)
    remainder = totals - out.sum(-1)
    order = np.argsort(-(raw - out), axis=-1, kind='stable')
    ranks = np.argsort(order, axis=-1)
    return out + (ranks < remainder[..., None])


@dataclass(frozen=True)
class ShotConfig:
    shots: int = 64  # Mean pair budget; every query row has N*shots ceiling.
    pilot: int = 8
    rounds: int = 3
    policy: str = 'combined'
    tolerance: float = 0.0  # Approximate output RMSE threshold; 0 disables early stopping.

    def __post_init__(self):
        if self.policy not in {'uniform', 'variance', 'sensitivity', 'combined'}:
            raise ValueError('Unknown allocation policy')
        if self.shots < 1 or not 1 <= self.pilot <= self.shots or self.rounds < 1 or self.tolerance < 0:
            raise ValueError('Invalid budget, pilot, rounds or tolerance')


class BernoulliSampler:
    mode = 'classical_probability_sampling_emulator'

    def __init__(self, fidelity, rng):
        self.p = np.clip((np.asarray(fidelity) + 1) / 2, 0, 1)
        self.rng = rng
        self.circuit_calls = 0
        self.modeled_circuit_executions = 0

    def draw(self, counts):
        self.modeled_circuit_executions += int(np.count_nonzero(counts))
        return self.rng.binomial(counts, self.p)


def estimate(sampler, values, beta, cfg):
    """Allocator sees only pilot counts/values, never ideal fidelities or test labels.

    Values have shape (..., keys, dv); measurement probabilities (..., queries, keys).
    Variance is Jeffreys-smoothed plug-in variance of a signed SWAP outcome.
    Adaptive optional stopping may bias the sample mean; no unbiasedness claim is made.
    """
    shape = sampler.p.shape
    n = shape[-1]
    counts = np.full(shape, cfg.pilot, dtype=np.int64)
    successes = sampler.draw(counts)
    decisions = [counts.copy()]
    stopped = np.zeros(shape[:-1], dtype=bool)
    last_rmse = np.zeros(shape[:-1])
    for step in range(cfg.rounds):
        f = 2 * successes / counts - 1  # Do not clip: clipping introduces boundary bias.
        a = softmax(beta * f)
        y = a @ values
        distance = np.maximum((values**2).sum(-1)[..., None, :] + (y**2).sum(-1)[..., :, None]
                              - 2 * (y @ np.swapaxes(values, -1, -2)), 0)
        sensitivity = (beta * a)**2 * distance
        p = (successes + .5) / (counts + 1)
        variance = 4 * p * (1 - p)
        last_rmse = np.sqrt((sensitivity * variance / counts).sum(-1))
        if cfg.tolerance:
            stopped |= last_rmse <= cfg.tolerance
        remaining = n * cfg.shots - counts.sum(-1)
        remaining[stopped] = 0
        spend = (remaining + (cfg.rounds-step) - 1) // (cfg.rounds-step)
        if not np.any(spend):
            break
        if cfg.policy == 'uniform':
            # Restore equal total counts, including round-off from previous rounds.
            target = integer_allocation(np.ones(shape), counts.sum(-1) + spend)
            extra = np.maximum(target - counts, 0)
        else:
            c = variance if cfg.policy == 'variance' else sensitivity
            if cfg.policy == 'combined':
                c = sensitivity * variance
            # Sequential square-root target with already-paid pilot lower bounds.
            weights = np.sqrt(np.maximum(c, 1e-15))
            target = weights / weights.sum(-1, keepdims=True) * (counts.sum(-1) + spend)[..., None]
            extra = integer_allocation(np.maximum(target - counts, 1e-12), spend)
        successes += sampler.draw(extra)
        counts += extra
        decisions.append(extra.copy())
    f = 2 * successes / counts - 1
    p = (successes + .5) / (counts + 1)
    return f, {'shots': counts, 'successes': successes, 'standard_error': np.sqrt(4*p*(1-p)/counts),
               'decisions': decisions, 'approx_output_rmse': last_rmse, 'stopped_rows': stopped,
               'total_shots': int(counts.sum()), 'circuit_calls': sampler.circuit_calls,
               'modeled_circuit_executions': sampler.modeled_circuit_executions, 'mode': sampler.mode}


class ShotEstimator:
    def __init__(self, cfg, seed=0, sampler_factory=None):
        self.cfg, self.rng = cfg, np.random.default_rng(seed)
        self.sampler_factory = sampler_factory
        self.total_shots = self.circuit_calls = self.modeled_circuit_executions = 0
        self.records = []

    def __call__(self, q, k, v, ideal, beta):
        sampler = (self.sampler_factory(q, k, self.rng) if self.sampler_factory else
                   BernoulliSampler(ideal.detach().cpu().numpy(), self.rng))
        f, audit = estimate(sampler, v.detach().cpu().numpy(), beta, self.cfg)
        self.total_shots += audit['total_shots']
        self.circuit_calls += audit['circuit_calls']
        self.modeled_circuit_executions += audit['modeled_circuit_executions']
        return torch.as_tensor(f, dtype=q.dtype, device=q.device), audit
