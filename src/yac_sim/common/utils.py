
from __future__ import annotations
import math
import numpy as np

def set_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed) & 0xFFFFFFFF)

def norm2(x: np.ndarray) -> float:
    return float(np.linalg.norm(x.reshape(-1), ord=2))

def mean_ci95(values: np.ndarray) -> tuple[float, float]:
    """Return (mean, halfwidth) for 95% CI using normal approx."""
    v = np.asarray(values, dtype=float)
    m = float(v.mean())
    if v.size <= 1:
        return m, 0.0
    s = float(v.std(ddof=1))
    hw = 1.96 * s / math.sqrt(v.size)
    return m, hw

def knee_point(x: np.ndarray, y: np.ndarray) -> int:
    """Heuristic knee: max distance to line between endpoints in (x,y)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3:
        return int(np.argmax(y))
    x1, y1 = x[0], y[0]
    x2, y2 = x[-1], y[-1]
    dx = x2 - x1
    dy = y2 - y1
    denom = math.hypot(dx, dy) + 1e-12
    # distance from point to line
    d = np.abs(dy*x - dx*y + x2*y1 - y2*x1) / denom
    return int(np.argmax(d))

def dlqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, iters: int = 500, eps: float = 1e-10) -> np.ndarray:
    """Discrete-time LQR via iterative Riccati (no SciPy dependency)."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)

    P = Q.copy()
    for _ in range(iters):
        BT_P = B.T @ P
        S = R + BT_P @ B
        K = np.linalg.solve(S, BT_P @ A)  # K = (R+B'PB)^{-1} B'PA
        Pn = Q + A.T @ P @ A - A.T @ P @ B @ K
        if np.max(np.abs(Pn - P)) < eps:
            P = Pn
            break
        P = Pn
    # final gain
    BT_P = B.T @ P
    S = R + BT_P @ B
    K = np.linalg.solve(S, BT_P @ A)
    return K
