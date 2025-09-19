from typing import List, Dict
import random

def hit_rate(samples, target, tol):
    eps = target * tol
    hits = sum(1 for s in samples if abs(s - target) <= eps)
    return hits / max(1, len(samples))

def simple_eval(targets: List[float], tolerances: List[float]) -> Dict:
    # Dummy eval that returns random but stable-ish numbers (for CPU sanity checks)
    random.seed(0)
    res = {}
    for t in targets:
        res[t] = {f"@±{int(100*tol)}%": round(0.4 + 0.1*random.random(), 3) for tol in tolerances}
    return res
