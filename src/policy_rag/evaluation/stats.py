"""Statistics for small paired evaluations (the unit is the question)."""
from __future__ import annotations

import math

import numpy as np


def wilson(k: int, n: int, z: float = 1.96):
    """95% Wilson score interval for a proportion k/n."""
    if not n:
        return (None, None)
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mde(n: int, p_disc: float = 0.25, alpha: float = 0.05, power: float = 0.8):
    """Smallest difference in "fully correct" a paired McNemar test detects on n questions (80% power, 5% two-sided),
    assuming a quarter of the questions change outcome between two systems."""
    za, zb = 1.959964, 0.841621
    for d in np.arange(0.01, 0.6, 0.005):
        if p_disc - d * d <= 0:
            return None
        if (za * math.sqrt(p_disc) + zb * math.sqrt(p_disc - d * d)) ** 2 / d ** 2 <= n:
            return round(float(d), 3)
    return None


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value (binomial test on the discordant questions)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def holm(pvals: dict) -> dict:
    """Holm-adjusted p-values (several candidates compared with one baseline)."""
    order = sorted(pvals, key=pvals.get)
    out, running = {}, 0.0
    for rank, k in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * pvals[k]))
        out[k] = running
    return out


def auroc(pos, neg):
    """P(score of a positive > score of a negative); ties count half."""
    if not pos or not neg:
        return None
    return float(np.mean([(p > q) + 0.5 * (p == q) for p in pos for q in neg]))


def cohen_kappa(x, y) -> float:
    x, y = np.asarray(x, int), np.asarray(y, int)
    po = float((x == y).mean())
    pe = float(x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean()))
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)
