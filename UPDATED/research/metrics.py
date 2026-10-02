import numpy as np
from scipy import optimize, stats
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score, log_loss)


def probabilities(logits, temperature=1.0):
    z = logits / temperature
    p = np.exp(z - z.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def fit_temperature(logits, labels):
    f = lambda t: log_loss(labels, probabilities(logits, np.exp(t)), labels=np.arange(logits.shape[1]))
    return float(np.exp(optimize.minimize_scalar(f, bounds=(-3, 3), method='bounded').x))


def classification_metrics(y, p, bins=10):
    y, p = np.asarray(y), np.asarray(p)
    c = p.shape[1]
    pred = p.argmax(1)
    cm = confusion_matrix(y, pred, labels=np.arange(c))
    tp = np.diag(cm)
    fn, fp = cm.sum(1) - tp, cm.sum(0) - tp
    tn = cm.sum() - tp - fn - fp
    divide = lambda a, b: np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b > 0)
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, labels=np.arange(c), zero_division=0)
    specificity, npv = divide(tn, tn+fp), divide(tn, tn+fn)
    confidence, correct = p.max(1), (pred == y).astype(float)
    reliability, ece = [], 0.
    memberships = np.minimum((confidence * bins).astype(int), bins-1)
    for b in range(bins):
        mask = memberships == b
        if mask.any():
            acc, conf = float(correct[mask].mean()), float(confidence[mask].mean())
            ece += mask.mean() * abs(acc-conf)
            reliability.append({'bin': b, 'n': int(mask.sum()), 'accuracy': acc, 'confidence': conf})
    try:
        auc = roc_auc_score(y, p[:, 1]) if c == 2 else roc_auc_score(y, p, multi_class='ovr', average='macro')
    except ValueError:
        auc = None
    # Binary Brier convention; multiclass is sum of squared errors (not divided by C).
    brier = np.mean((p[:, 1] - y)**2) if c == 2 else np.mean(((p - np.eye(c)[y])**2).sum(1))
    avg = lambda x: float(x[1]) if c == 2 else float(x.mean())
    return {'n': len(y), 'accuracy': float(accuracy_score(y, pred)),
            'balanced_accuracy': float(balanced_accuracy_score(y, pred)), 'auroc': None if auc is None else float(auc),
            'sensitivity': avg(recall), 'specificity': avg(specificity), 'precision': avg(precision),
            'f1': avg(f1), 'npv': avg(npv), 'ece': float(ece), 'brier': float(brier),
            'nll': float(log_loss(y, p, labels=np.arange(c))), 'confusion_matrix': cm.tolist(),
            'per_class': {'recall': recall.tolist(), 'specificity': specificity.tolist(), 'f1': f1.tolist()},
            'averaging': 'positive_class_1' if c == 2 else 'macro_one_vs_rest', 'reliability': reliability}


def delong_paired(y, score_a, score_b):
    """Binary DeLong covariance using per-positive/per-negative U-statistic contributions."""
    components, aucs = [], []
    for s in (score_a, score_b):
        pos, neg = s[y == 1], s[y == 0]
        if min(len(pos), len(neg)) < 2:
            return {'p': None, 'reason': 'Insufficient cases per class'}
        delta = pos[:, None] - neg[None, :]
        kernel = (delta > 0) + .5*(delta == 0)
        components.append((kernel.mean(1), kernel.mean(0)))
        aucs.append(kernel.mean())
    covariance = np.cov(np.stack([v[0] for v in components])) / len(components[0][0])
    covariance += np.cov(np.stack([v[1] for v in components])) / len(components[0][1])
    var = float(covariance[0,0] + covariance[1,1] - 2*covariance[0,1])
    diff = float(aucs[0]-aucs[1])
    p = 1. if var <= 1e-15 and abs(diff) <= 1e-15 else (None if var <= 1e-15 else float(2*stats.norm.sf(abs(diff)/np.sqrt(var))))
    return {'difference': diff, 'variance': var, 'p': p}


def paired_predictions(y, a, b):
    good_a, good_b = a.argmax(1) == y, b.argmax(1) == y
    n01, n10 = int((~good_a & good_b).sum()), int((good_a & ~good_b).sum())
    result = {'mcnemar_discordant': [n01,n10],
              'mcnemar_exact_p': float(stats.binomtest(n01, n01+n10, .5).pvalue) if n01+n10 else 1.}
    if a.shape[1] == 2:
        result['delong'] = delong_paired(y, a[:,1], b[:,1])
    return result


def bootstrap_seed_mean(values, seed=123, repeats=10000):
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, (repeats,len(values)), replace=True).mean(1)
    return {'mean': float(values.mean()), 'sd': float(values.std(ddof=1)) if len(values)>1 else None,
            'ci95_seed_bootstrap': np.quantile(draws,[.025,.975]).tolist(), 'seeds': len(values)}


def holm(pvalues):
    p = np.asarray(pvalues)
    order = np.argsort(p)
    adjusted = np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p))))
    out = np.empty_like(p)
    out[order] = np.minimum(adjusted, 1)
    return out.tolist()
