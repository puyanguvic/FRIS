
from __future__ import annotations
import numpy as np

def double_integrator_2d(Ts: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Planar double integrator discretized with ZOH."""
    Ts = float(Ts)
    A1 = np.array([[1.0, Ts],
                   [0.0, 1.0]], dtype=float)
    B1 = np.array([[0.5 * Ts * Ts],
                   [Ts]], dtype=float)
    A = np.block([[A1, np.zeros((2,2))],
                  [np.zeros((2,2)), A1]])
    B = np.block([[B1, np.zeros((2,1))],
                  [np.zeros((2,1)), B1]])
    return A, B
