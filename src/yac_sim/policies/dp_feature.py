from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..common.kalman import cov_step


def feature_trace_logdet(P: np.ndarray, eps: float = 1e-6) -> tuple[float, float]:
    n = P.shape[0]
    P_eps = P + float(eps) * np.eye(n, dtype=float)
    sign, logdet = np.linalg.slogdet(P_eps)
    if sign <= 0:
        logdet = float(np.log(np.maximum(np.linalg.det(P_eps), eps)))
    return float(np.trace(P)), float(logdet)


@dataclass(frozen=True)
class FeatureGrid:
    f1_grid: np.ndarray
    f2_grid: np.ndarray
    reps: np.ndarray  # shape (n1, n2, n, n)


def _grid_centers(f1_grid: np.ndarray, f2_grid: np.ndarray) -> np.ndarray:
    f1c, f2c = np.meshgrid(f1_grid, f2_grid, indexing="ij")
    return np.stack([f1c, f2c], axis=-1)


def build_feature_grid(
    samples: list[np.ndarray],
    grid_shape: tuple[int, int],
    expand: float = 0.1,
) -> FeatureGrid:
    feats = np.array([feature_trace_logdet(P) for P in samples], dtype=float)
    f1_min, f1_max = feats[:, 0].min(), feats[:, 0].max()
    f2_min, f2_max = feats[:, 1].min(), feats[:, 1].max()
    f1_span = f1_max - f1_min
    f2_span = f2_max - f2_min
    f1_min -= float(expand) * f1_span
    f1_max += float(expand) * f1_span
    f2_min -= float(expand) * f2_span
    f2_max += float(expand) * f2_span

    f1_grid = np.linspace(f1_min, f1_max, int(grid_shape[0]))
    f2_grid = np.linspace(f2_min, f2_max, int(grid_shape[1]))
    n = samples[0].shape[0]
    reps = np.zeros((f1_grid.size, f2_grid.size, n, n), dtype=float)
    reps.fill(np.nan)

    centers = _grid_centers(f1_grid, f2_grid)
    for P, feat in zip(samples, feats, strict=False):
        i = int(np.clip(np.searchsorted(f1_grid, feat[0]) - 1, 0, f1_grid.size - 1))
        j = int(np.clip(np.searchsorted(f2_grid, feat[1]) - 1, 0, f2_grid.size - 1))
        center = centers[i, j]
        if np.isnan(reps[i, j]).any():
            reps[i, j] = P
        else:
            cur = reps[i, j]
            cur_feat = np.array(feature_trace_logdet(cur), dtype=float)
            if np.linalg.norm(feat - center) < np.linalg.norm(cur_feat - center):
                reps[i, j] = P

    reps = _fill_missing(reps, centers)
    return FeatureGrid(f1_grid=f1_grid, f2_grid=f2_grid, reps=reps)


def _fill_missing(reps: np.ndarray, centers: np.ndarray) -> np.ndarray:
    filled = reps.copy()
    n1, n2 = reps.shape[0], reps.shape[1]
    valid = ~np.isnan(reps[..., 0, 0])
    if valid.all():
        return filled

    valid_idx = np.argwhere(valid)
    for i in range(n1):
        for j in range(n2):
            if valid[i, j]:
                continue
            target = centers[i, j]
            best = None
            best_dist = np.inf
            for vi, vj in valid_idx:
                dist = float(np.linalg.norm(centers[vi, vj] - target))
                if dist < best_dist:
                    best_dist = dist
                    best = (vi, vj)
            if best is not None:
                filled[i, j] = reps[best[0], best[1]]
    return filled


def sample_covariances(
    P0: np.ndarray,
    A: np.ndarray,
    C: np.ndarray,
    Qw: np.ndarray,
    Rv: np.ndarray,
    p_success: float,
    horizon: int,
    n_rollouts: int,
    rng: np.random.Generator,
    tx_prob: float = 0.5,
) -> list[np.ndarray]:
    samples: list[np.ndarray] = []
    for _ in range(int(n_rollouts)):
        P = P0.copy()
        for _ in range(int(horizon)):
            samples.append(P.copy())
            do_tx = float(rng.random()) < float(tx_prob)
            P_pred, P_upd = cov_step(P, A, C, Qw, Rv)
            if do_tx and (float(rng.random()) < float(p_success)):
                P = P_upd
            else:
                P = P_pred
    return samples


def _feature_to_index(f1: float, f2: float, grid: FeatureGrid) -> tuple[int, int]:
    i = int(np.searchsorted(grid.f1_grid, f1))
    j = int(np.searchsorted(grid.f2_grid, f2))
    i = int(np.clip(i, 0, grid.f1_grid.size - 1))
    j = int(np.clip(j, 0, grid.f2_grid.size - 1))
    if i > 0 and abs(f1 - grid.f1_grid[i - 1]) < abs(f1 - grid.f1_grid[i]):
        i -= 1
    if j > 0 and abs(f2 - grid.f2_grid[j - 1]) < abs(f2 - grid.f2_grid[j]):
        j -= 1
    return i, j


def dp_feature_policy(
    grid: FeatureGrid,
    A: np.ndarray,
    C: np.ndarray,
    Qw: np.ndarray,
    Rv: np.ndarray,
    p_success: float,
    lamb: float,
    horizon: int,
) -> np.ndarray:
    n1, n2 = grid.f1_grid.size, grid.f2_grid.size
    actions = np.zeros((int(horizon), n1, n2), dtype=int)
    V_next = np.zeros((n1, n2), dtype=float)

    p_succ = float(p_success)
    for k in range(int(horizon) - 1, -1, -1):
        V_curr = np.zeros((n1, n2), dtype=float)
        for i in range(n1):
            for j in range(n2):
                P = grid.reps[i, j]
                P_pred, P_upd = cov_step(P, A, C, Qw, Rv)
                f_pred = feature_trace_logdet(P_pred)
                f_upd = feature_trace_logdet(P_upd)
                i_pred, j_pred = _feature_to_index(f_pred[0], f_pred[1], grid)
                i_upd, j_upd = _feature_to_index(f_upd[0], f_upd[1], grid)
                stage = float(np.trace(P))
                v0 = stage + V_next[i_pred, j_pred]
                v1 = stage + float(lamb) + (
                    (1.0 - p_succ) * V_next[i_pred, j_pred]
                    + p_succ * V_next[i_upd, j_upd]
                )
                if v1 < v0:
                    actions[k, i, j] = 1
                    V_curr[i, j] = v1
                else:
                    actions[k, i, j] = 0
                    V_curr[i, j] = v0
        V_next = V_curr
    return actions


def make_feature_policy_fn(grid: FeatureGrid, actions: np.ndarray):
    def policy_fn(k: int, P: np.ndarray) -> bool:
        f1, f2 = feature_trace_logdet(P)
        i, j = _feature_to_index(f1, f2, grid)
        return bool(actions[k, i, j])

    return policy_fn
