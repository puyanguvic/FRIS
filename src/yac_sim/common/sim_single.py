from __future__ import annotations

import numpy as np

from .config import SimConfig
from .models import double_integrator_2d
from .utils import dlqr, norm2
from .channels import GilbertElliottChannel


def _cost_matrices(cfg: SimConfig) -> tuple[np.ndarray, np.ndarray]:
    Q = np.diag([cfg.Q_px, cfg.Q_vx, cfg.Q_py, cfg.Q_vy]).astype(float)
    R = (cfg.R_u * np.eye(2)).astype(float)
    return Q, R


def _gaussian_noise(rng: np.random.Generator, sigma: float, size: int) -> np.ndarray:
    """Zero-mean Gaussian noise with isotropic standard deviation."""
    s = float(sigma)
    if s <= 0.0:
        return np.zeros((size,), dtype=float)
    return rng.normal(0.0, s, size=(size,)).astype(float)


def _measurement_matrices(cfg: SimConfig, n: int) -> tuple[np.ndarray, int]:
    """Return (C, p). Default is full-state measurement (C = I)."""
    if getattr(cfg, "C_full_state", True):
        C = np.eye(n, dtype=float)
        p = n
        return C, p

    # If you ever want partial measurement, customize here.
    # For now, we keep it simple and explicit: measure positions only.
    # State is [p_x, v_x, p_y, v_y]^T -> measure [p_x, p_y]
    C = np.array([[1.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.0]], dtype=float)
    p = C.shape[0]
    return C, p


def simulate(
    cfg: SimConfig,
    policy: str = "ET",
    rng: np.random.Generator | None = None,
    policy_fn=None,
) -> dict:
    """Simulate one rollout over a finite horizon.

    Policies:
      - ET: covariance-triggered (trace(P_k) threshold)
      - PER: periodic with period_M
      - ALWAYS: always attempt transmission
      - NONE: never attempt transmission
      - DP: use policy_fn(k, P) supplied by experiment code
    """
    if rng is None:
        rng = np.random.default_rng(int(cfg.seed) & 0xFFFFFFFF)

    A, B = double_integrator_2d(cfg.Ts)
    n = A.shape[0]
    C, p = _measurement_matrices(cfg, n)

    Q, R = _cost_matrices(cfg)
    K = dlqr(A, B, Q, R)  # control law uses estimate: u = -K x_hat

    # Predictor may have mismatch (robust experiment)
    A_hat = A + float(cfg.mismatch_eps) * np.eye(n, dtype=float)

    ch = GilbertElliottChannel(cfg.p_good_to_bad, cfg.p_bad_to_good, cfg.loss_good, cfg.loss_bad, rng)

    # initial state: random position/velocity
    x = np.array([rng.normal(0, 5.0), rng.normal(0, 1.0), rng.normal(0, 5.0), rng.normal(0, 1.0)], dtype=float)
    x_hat = x.copy()
    u_prev = np.zeros((2,), dtype=float)
    P = float(cfg.P0_scale) * np.eye(n, dtype=float)

    Qw = (float(cfg.sigma_w) ** 2) * np.eye(n, dtype=float)
    Rv = (float(cfg.sigma_v) ** 2) * np.eye(p, dtype=float)

    # logs
    x_norm = np.zeros(cfg.T_steps)
    tilde_x_norm = np.zeros(cfg.T_steps)          # ||x - x_hat||
    innovation_norm = np.zeros(cfg.T_steps)       # ||y - C x_hat^-||
    tx_attempt = np.zeros(cfg.T_steps, dtype=int)
    tx_deliv = np.zeros(cfg.T_steps, dtype=int)
    P_trace = np.zeros(cfg.T_steps)

    J_P = 0.0
    J_C = 0.0
    J_X = 0.0

    for k in range(cfg.T_steps):
        # decide whether to transmit based on current covariance
        trace_curr = float(np.trace(P))
        if policy_fn is not None:
            do_tx = bool(policy_fn(k, P))
        elif policy == "ET":
            do_tx = trace_curr > float(cfg.delta)
        elif policy == "PER":
            do_tx = (k % max(int(cfg.period_M), 1) == 0)
        elif policy == "ALWAYS":
            do_tx = True
        elif policy == "NONE":
            do_tx = False
        else:
            raise ValueError(f"Unknown policy: {policy}")

        # --- plant update (process noise) ---
        w = _gaussian_noise(rng, cfg.sigma_w, n)
        x = A @ x + B @ u_prev + w

        # --- measurement (paper: y = Cx + v) ---
        v = _gaussian_noise(rng, cfg.sigma_v, p)
        y = (C @ x) + v

        # --- remote prediction at controller ---
        x_hat_pred = A_hat @ x_hat + B @ u_prev
        P_pred = A_hat @ P @ A_hat.T + Qw

        # innovation is logged; trigger uses trace(P_k)
        innovation = y - (C @ x_hat_pred)
        innovation_norm[k] = norm2(innovation)

        if do_tx:
            tx_attempt[k] = 1
            if cfg.channel_model == "iid":
                delivered = float(rng.random()) < float(cfg.p_success)
            elif cfg.channel_model == "ge":
                delivered = ch.deliver()
            else:
                raise ValueError(f"Unknown channel model: {cfg.channel_model}")

            if delivered:
                tx_deliv[k] = 1
                innovation_rx = y - (C @ x_hat_pred)
                S = C @ P_pred @ C.T + Rv
                Kk = P_pred @ C.T @ np.linalg.pinv(S)
                x_hat = x_hat_pred + (Kk @ innovation_rx)
                P = (np.eye(n) - Kk @ C) @ P_pred
            else:
                x_hat = x_hat_pred
                P = P_pred
        else:
            x_hat = x_hat_pred
            P = P_pred

        # Log covariance trace used by the trigger.
        P_trace[k] = trace_curr

        # control uses remote estimate only (paper eq. (control))
        u = -(K @ x_hat).reshape(-1)

        # prediction error (paper eq. (tilde))
        tilde_x = x - x_hat
        tilde_x_norm[k] = norm2(tilde_x)

        x_norm[k] = norm2(x)
        u_prev = u

        J_P += trace_curr
        J_C += float(do_tx)
        J_X += float(x_norm[k] ** 2)

    # packet counts are primary in the paper; bits are auxiliary (for legacy plots)
    N_attempt = int(tx_attempt.sum())
    N_deliv = int(tx_deliv.sum())
    bits_attempt = N_attempt * n * int(cfg.bits_per_value)
    bits_deliv = N_deliv * n * int(cfg.bits_per_value)

    return dict(
        J_P=J_P,
        J_C=J_C,
        J_X=J_X,
        x_norm=x_norm,
        tilde_x_norm=tilde_x_norm,
        innovation_norm=innovation_norm,
        tx_attempt=tx_attempt,
        tx_deliv=tx_deliv,
        P_trace=P_trace,
        N_attempt=N_attempt,
        N_deliv=N_deliv,
        bits_attempt=bits_attempt,
        bits_deliv=bits_deliv,
    )


def monte_carlo(
    cfg: SimConfig,
    policy: str,
    deltas: list[float] | None = None,
    periods: list[int] | None = None,
    seeds: list[int] | None = None,
) -> list[dict]:
    """Run Monte Carlo for a sweep of a single policy knob.

    Returns a list of dicts, one per sweep point, matching the plotting helpers in experiments/paper1.py.
    Each dict contains:
      - param: sweep value (delta / period / p)
      - J_P: array of sum trace(P_k) over MC runs
      - J_C: array of sum a_k over MC runs
      - J_X: array of sum ||x_k||^2 over MC runs
      - N_deliv: array of delivered packet counts over MC runs
      - bits_deliv: array of delivered bits over MC runs (auxiliary)
      - N_attempt / bits_attempt also included for completeness
    """
    if seeds is None:
        rng_master = np.random.default_rng(int(cfg.seed) & 0xFFFFFFFF)
        seeds = rng_master.integers(0, 2**31 - 1, size=int(cfg.mc_runs), dtype=np.int64).tolist()

    results: list[dict] = []

    if policy == "ET":
        assert deltas is not None
        for d in deltas:
            Jp, Jc, Jx, Nd, Bd, Na, Ba = [], [], [], [], [], [], []
            for s in seeds:
                rng = np.random.default_rng(int(s))
                cfg2 = SimConfig(**cfg.__dict__)
                cfg2.delta = float(d)
                out = simulate(cfg2, "ET", rng=rng)
                Jp.append(out["J_P"])
                Jc.append(out["J_C"])
                Jx.append(out["J_X"])
                Nd.append(out["N_deliv"])
                Bd.append(out["bits_deliv"])
                Na.append(out["N_attempt"])
                Ba.append(out["bits_attempt"])
            results.append(
                dict(
                    param=float(d),
                    J_P=np.asarray(Jp, dtype=float),
                    J_C=np.asarray(Jc, dtype=float),
                    J_X=np.asarray(Jx, dtype=float),
                    N_deliv=np.asarray(Nd, dtype=float),
                    bits_deliv=np.asarray(Bd, dtype=float),
                    N_attempt=np.asarray(Na, dtype=float),
                    bits_attempt=np.asarray(Ba, dtype=float),
                )
            )

    elif policy == "PER":
        assert periods is not None
        for M in periods:
            Jp, Jc, Jx, Nd, Bd, Na, Ba = [], [], [], [], [], [], []
            for s in seeds:
                rng = np.random.default_rng(int(s))
                cfg2 = SimConfig(**cfg.__dict__)
                cfg2.period_M = int(M)
                out = simulate(cfg2, "PER", rng=rng)
                Jp.append(out["J_P"])
                Jc.append(out["J_C"])
                Jx.append(out["J_X"])
                Nd.append(out["N_deliv"])
                Bd.append(out["bits_deliv"])
                Na.append(out["N_attempt"])
                Ba.append(out["bits_attempt"])
            results.append(
                dict(
                    param=int(M),
                    J_P=np.asarray(Jp, dtype=float),
                    J_C=np.asarray(Jc, dtype=float),
                    J_X=np.asarray(Jx, dtype=float),
                    N_deliv=np.asarray(Nd, dtype=float),
                    bits_deliv=np.asarray(Bd, dtype=float),
                    N_attempt=np.asarray(Na, dtype=float),
                    bits_attempt=np.asarray(Ba, dtype=float),
                )
            )

    else:
        raise ValueError(f"Unknown policy: {policy}")

    return results
