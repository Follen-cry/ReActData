"""Statistics and the selection rule. Unit = datapoint; tasks are strata (reactdata-v1).

s_{m,i}: mean rule score of model m on datapoint i over the generation-seed set S.
d_i = s_{c,i} - s_{base,i};  D = equal-weight mean over tasks of mean_i d_i (frozen dev aggregation: dev scale and dev
move weigh 0.5 each; a single-task set reduces to the plain mean);  bootstrap resamples datapoints within each task
(B = 10,000, fixed seed; seeds not resampled);
SE = std of the bootstrap distribution;  delta from the null comparison of the fixed baseline with itself
(S vs S' = disjoint generation seeds): delta = max(|D_null|, |q97.5 of the null bootstrap|, |q2.5|).
"""
from __future__ import annotations
import math
import os
import sys
from typing import Dict, List, Optional
import numpy as np
from experiment.runtime.io import read_json

def single_group(item_id):
    return "all"

B = 10000
BOOT_SEED = 20261025
ORDER = {'fixed': 0, 'static': 1, 'schedule': 2, 'feedback': 3}

def item_means(run_dir: str, split: str, seeds: List[int]) -> Dict[str, float]:
    per = {}
    for s in seeds:
        sc = read_json(os.path.join(run_dir, 'eval', split, f'g{s}', 'scores.json'))
        for x in sc['items']:
            per.setdefault(x['id'], []).append(x['score'])
    bad = [k for k, v in per.items() if len(v) != len(seeds)]
    if bad:
        raise ValueError(f'{run_dir}: missing seeds for {bad[:3]}')
    return {k: float(np.mean(v)) for k, v in per.items()}


def agg(values: Dict[str, float], *, group_of=single_group) -> float:
    groups = {}
    for k, v in values.items():
        groups.setdefault(group_of(k), []).append(v)
    return float(np.mean([np.mean(g) for g in groups.values()]))

def paired(a: Dict[str, float], b: Dict[str, float], *, group_of=single_group) -> Dict[str, float]:
    ids = sorted(set(a) & set(b))
    if len(ids) != len(a) or len(ids) != len(b):
        raise ValueError('paired comparison needs the same datapoints')
    groups = {}
    for i in ids:
        groups.setdefault(group_of(i), []).append(a[i] - b[i])
    rng = np.random.default_rng(BOOT_SEED)
    boot = np.zeros(B)
    per_task = {}
    for t, d in sorted(groups.items()):
        d = np.array(d)
        bt = d[rng.integers(0, len(d), size=(B, len(d)))].mean(1)
        boot += bt / len(groups)
        per_task[t] = {'n': len(d), 'D': float(d.mean()), 'ci_lo': float(np.percentile(bt, 2.5)), 'ci_hi': float(np.percentile(bt, 97.5))}
    D = float(np.mean([np.mean(g) for g in groups.values()]))
    return {'n': len(ids), 'D': D, 'SE': float(boot.std(ddof=1)), 'ci_lo': float(np.percentile(boot, 2.5)), 'ci_hi': float(np.percentile(boot, 97.5)), 'mean_a': agg({i: a[i] for i in ids}, group_of=group_of), 'mean_b': agg({i: b[i] for i in ids}, group_of=group_of), 'per_task': per_task}

def null_delta(base_S: Dict[str, float], base_Sp: Dict[str, float], *, group_of=single_group) -> Dict[str, float]:
    p = paired(base_S, base_Sp, group_of=group_of)
    p['delta'] = float(max(abs(p['D']), abs(p['ci_hi']), abs(p['ci_lo'])))
    return p

def select(cands: List[Dict], baseline: Dict, delta: float, *, group_of=single_group) -> Dict:
    """cands: [{"id", "class_label", "compute_ratio", "items": {id: s}}] (valid, trained, audited candidates);
    baseline: {"id", "items"}. Returns the decision with every comparison (dev numbers are selection-biased)."""
    comps = []
    for c in cands:
        p = paired(c['items'], baseline['items'], group_of=group_of)
        comps.append({'id': c['id'], 'class_label': c['class_label'], 'compute_ratio': c['compute_ratio'], **p})
    best: Optional[Dict] = max(comps, key=lambda x: x['D']) if comps else None
    if best and best['D'] >= delta and (best['ci_lo'] > 0):
        return {'selected': best['id'], 'rule': 'best candidate: D >= delta and CI lower bound > 0', 'delta': delta, 'comparisons': comps}
    pool = [{'id': baseline['id'], 'class_label': 'fixed', 'compute_ratio': 1.0, 'D': 0.0}]
    top = max([0.0] + [x['D'] for x in comps])
    pool += [x for x in comps if top - x['D'] <= delta]
    pool = [x for x in pool if top - x['D'] <= delta]
    pick = min(pool, key=lambda x: (ORDER.get(x['class_label'], 3), round(x['compute_ratio'], 1), x['id'] != baseline['id'], -x['D']))
    return {'selected': pick['id'], 'rule': 'no candidate passed; simplest/cheapest program within delta of the best (baseline D = 0)', 'delta': delta, 'within_delta': [x['id'] for x in pool], 'comparisons': comps}

def run_means(run_dirs: List[str], split: str, seeds: List[int]) -> List[Dict[str, float]]:
    return [item_means(d, split, seeds) for d in run_dirs]

def training_noise(replicates: List[Dict[str, float]], *, group_of=single_group) -> Dict[str, float]:
    ms = [agg(r, group_of=group_of) for r in replicates]
    sd = float(np.std(ms, ddof=1)) if len(ms) > 1 else float('nan')
    return {'k': len(ms), 'run_scores': ms, 'mean': float(np.mean(ms)), 'sd_train': sd, 'delta': sd, 'range': float(max(ms) - min(ms)) if ms else float('nan')}

def _z(alpha: float, m: int) -> float:
    from statistics import NormalDist
    return float(NormalDist().inv_cdf(1 - alpha / (2 * max(1, m))))

def confirm_compare(cand: List[Dict[str, float]], base: List[Dict[str, float]], sd_train: float, *, m: int=1, alpha: float=0.05, group_of=single_group) -> Dict[str, float]:
    """cand[r], base[r]: per-item means of the r-th seed pair (same seeds, same datapoints)."""
    if len(cand) != len(base) or not cand:
        raise ValueError('confirmation needs paired runs')
    R = len(cand)
    D_runs = [agg(c, group_of=group_of) - agg(b, group_of=group_of) for c, b in zip(cand, base)]
    D = float(np.mean(D_runs))
    ids = sorted(cand[0])
    avg_c = {i: float(np.mean([c[i] for c in cand])) for i in ids}
    avg_b = {i: float(np.mean([b[i] for b in base])) for i in ids}
    item = paired(avg_c, avg_b, group_of=group_of)
    se_train = math.sqrt(2.0) * sd_train / math.sqrt(R)
    se = math.sqrt(se_train ** 2 + item['SE'] ** 2)
    z = _z(alpha, m)
    return {'R': R, 'D': D, 'D_per_seed': D_runs, 'SE_train': se_train, 'SE_item': item['SE'], 'SE': se, 'z': z, 'ci_lo': D - z * se, 'ci_hi': D + z * se, 'per_task': item['per_task'], 'm_comparisons': m, 'mean_cand': float(np.mean([agg(c, group_of=group_of) for c in cand])), 'mean_base': float(np.mean([agg(b, group_of=group_of) for b in base]))}

class Statistics:
    """Task supplies item strata; the design owns the same statistical operations."""
    def __init__(self, group_of=single_group):
        self.group_of = group_of

    item_means = staticmethod(item_means)
    run_means = staticmethod(run_means)

    def task_of(self, item_id):
        return self.group_of(item_id)

    def agg(self, values):
        return agg(values, group_of=self.group_of)

    def paired(self, a, b):
        return paired(a, b, group_of=self.group_of)

    def training_noise(self, replicates):
        return training_noise(replicates, group_of=self.group_of)

    def null_delta(self, a, b):
        return null_delta(a, b, group_of=self.group_of)

    def confirm_compare(self, cand, base, sd_train, **options):
        return confirm_compare(cand, base, sd_train, group_of=self.group_of, **options)

    def select(self, candidates, baseline, delta):
        return select(candidates, baseline, delta, group_of=self.group_of)
