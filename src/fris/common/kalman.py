from __future__ import annotations

import numpy as np


def cov_predict(P: np.ndarray, A: np.ndarray, Qw: np.ndarray) -> np.ndarray:
    """Predict covariance: P^- = A P A^T + Qw."""
    P_pred = A @ P @ A.T + Qw
    return 0.5 * (P_pred + P_pred.T)


def cov_update(P_pred: np.ndarray, C: np.ndarray, Rv: np.ndarray) -> np.ndarray:
    """Update covariance after measurement."""
    S = C @ P_pred @ C.T + Rv
    Kk = P_pred @ C.T @ np.linalg.pinv(S)
    P_upd = (np.eye(P_pred.shape[0]) - Kk @ C) @ P_pred
    return 0.5 * (P_upd + P_upd.T)


def cov_step(
    P: np.ndarray,
    A: np.ndarray,
    C: np.ndarray,
    Qw: np.ndarray,
    Rv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (P_pred, P_upd) for a single Kalman step."""
    P_pred = cov_predict(P, A, Qw)
    P_upd = cov_update(P_pred, C, Rv)
    return P_pred, P_upd
