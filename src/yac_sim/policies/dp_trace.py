from __future__ import annotations

import numpy as np


def scalar_kalman_step(
    A: np.ndarray,
    C: np.ndarray,
    Qw: np.ndarray,
    Rv: np.ndarray,
    s: float,
) -> tuple[float, float]:
    """One-step trace surrogate using isotropic covariance P ≈ (s/n)I.

    Returns (s_pred, s_upd), where s_pred is the predicted trace and s_upd is the
    trace after a (hypothetical) successful measurement update.
    """
    n = A.shape[0]
    P = (float(s) / n) * np.eye(n, dtype=float)
    P_pred = A @ P @ A.T + Qw
    S = C @ P_pred @ C.T + Rv
    Kk = P_pred @ C.T @ np.linalg.pinv(S)
    P_upd = (np.eye(n) - Kk @ C) @ P_pred
    return float(np.trace(P_pred)), float(np.trace(P_upd))


def trace_bounds(
    P0: np.ndarray,
    A: np.ndarray,
    C: np.ndarray,
    Qw: np.ndarray,
    Rv: np.ndarray,
    horizon: int,
) -> tuple[float, float]:
    s = float(np.trace(P0))
    s_min, s_max = s, s
    for _ in range(int(horizon)):
        s_pred, s_upd = scalar_kalman_step(A, C, Qw, Rv, s)
        s_min = min(s_min, s_upd)
        s_max = max(s_max, s_pred)
        s = s_pred
    return s_min, s_max


def dp_trace_policy(
    P0: np.ndarray,
    A: np.ndarray,
    C: np.ndarray,
    Qw: np.ndarray,
    Rv: np.ndarray,
    p_success: float,
    lamb: float,
    horizon: int,
    grid_size: int = 180,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the finite-horizon DP-trace policy table.

    Uses the scalar information state s_k = tr(P_k) with an isotropic covariance
    approximation P_k ≈ (s_k/n)I. Solves the finite-horizon Bellman recursion
    for the stage cost s_k + lamb * gamma_k under Bernoulli delivery with
    probability p_success.

    Returns:
      - s_grid: grid of scalar states s
      - actions: array with actions[k, i] ∈ {0,1} approximating gamma_k^*(s_grid[i])
    """
    s_min, s_max = trace_bounds(P0, A, C, Qw, Rv, horizon)
    s_grid = np.linspace(s_min * 0.9, s_max * 1.1, int(grid_size))

    V_next = np.zeros_like(s_grid)
    actions = np.zeros((int(horizon), s_grid.size), dtype=int)

    s_lo, s_hi = s_grid[0], s_grid[-1]
    p_succ = float(p_success)

    for k in range(int(horizon) - 1, -1, -1):
        V_curr = np.zeros_like(s_grid)
        for i, s_i in enumerate(s_grid):
            s_pred, s_upd = scalar_kalman_step(A, C, Qw, Rv, float(s_i))
            s_pred_c = float(np.clip(s_pred, s_lo, s_hi))
            s_upd_c = float(np.clip(s_upd, s_lo, s_hi))
            V_pred = float(np.interp(s_pred_c, s_grid, V_next))
            V_upd = float(np.interp(s_upd_c, s_grid, V_next))
            q0 = V_pred
            q1 = float(lamb) + ((1.0 - p_succ) * V_pred + p_succ * V_upd)
            if q1 < q0:
                actions[k, i] = 1
                V_curr[i] = float(s_i) + q1
            else:
                actions[k, i] = 0
                V_curr[i] = float(s_i) + q0
        V_next = V_curr

    return s_grid, actions


def make_trace_policy_fn(s_grid: np.ndarray, actions: np.ndarray):
    """Return policy_fn(k, P) implementing the DP-trace action table."""
    def policy_fn(k: int, P: np.ndarray) -> bool:
        s = float(np.trace(P))
        idx = int(np.searchsorted(s_grid, s))
        if idx <= 0:
            idx = 0
        elif idx >= s_grid.size:
            idx = s_grid.size - 1
        else:
            if abs(s - s_grid[idx - 1]) < abs(s - s_grid[idx]):
                idx -= 1
        return bool(actions[k, idx])

    return policy_fn
