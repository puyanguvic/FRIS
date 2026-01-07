
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class SimConfig:
    # timing
    Ts: float = 0.1
    T_steps: int = 400
    mc_runs: int = 30
    seed: int = 7

    # channel model: "iid" (Bernoulli) or "ge" (Gilbert-Elliott)
    channel_model: str = "iid"
    p_success: float = 0.85

    # trigger / periodic
    delta: float = 0.4
    period_M: int = 10

    # disturbances / noise (paper: Gaussian with covariance)
    w_bar: float = 0.06   # legacy: L_infty bound (kept for compatibility; not used in new experiments)
    v_bar: float = 0.00   # legacy: L_infty bound (kept for compatibility; not used in new experiments)
    # Gaussian noise stds for process and measurement (used in Kalman covariance updates)
    sigma_w: float = 0.03
    sigma_v: float = 0.02
    bits_per_value: int = 32

    # measurement model (paper uses y = Cx + v; default is full-state)
    C_full_state: bool = True

    # channel (Gilbert–Elliott)
    p_good_to_bad: float =  0.03
    p_bad_to_good: float =  0.08
    loss_good: float =  0.02
    loss_bad: float =  0.45

    # model mismatch (predictor uses A_hat = A + eps*I)
    mismatch_eps: float = 0.0

    # quantizer range
    q_min: float = -20.0
    q_max: float = 20.0

    # cost matrices (as in paper)
    Q_px: float = 10.0
    Q_vx: float = 1.0
    Q_py: float = 10.0
    Q_vy: float = 1.0
    R_u: float = 0.1
    # initial estimation covariance scale (P0 = P0_scale * I)
    P0_scale: float = 1.0

    def force_theory(self) -> None:
        """Configure idealized baseline consistent with the paper assumptions."""
        self.channel_model = "iid"
        self.p_success = 1.0
        self.v_bar = 0.0
        self.bits_per_value = 32
        self.loss_good = 0.0
        self.loss_bad = 0.0
        self.mismatch_eps = 0.0
