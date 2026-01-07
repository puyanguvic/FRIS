from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ..common.config import SimConfig
from ..common.utils import dlqr, mean_ci95, norm2
from ..policies.dp_feature import (
    build_feature_grid,
    dp_feature_policy,
    feature_trace_logdet,
    make_feature_policy_fn,
    sample_covariances,
)
from ..policies.dp_trace import dp_trace_policy, make_trace_policy_fn


def apply_ieee_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.5,
        "grid.linewidth": 0.4,
    })


def savefig(fig, outdir: Path, name: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{name}.png", dpi=300, bbox_inches="tight")


def _mean_ci(values: np.ndarray) -> tuple[float, float]:
    return mean_ci95(values)


def _simulate_rollout(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    Qw: np.ndarray,
    Rv: np.ndarray,
    K: np.ndarray,
    P0: np.ndarray,
    horizon: int,
    p_success: float,
    rng: np.random.Generator,
    policy_fn,
) -> dict:
    n = A.shape[0]
    p = C.shape[0]
    x = rng.normal(0.0, 1.0, size=(n,))
    x_hat = x.copy()
    u_prev = np.zeros((B.shape[1],), dtype=float)
    P = P0.copy()

    J_P = 0.0
    J_C = 0.0
    J_X = 0.0

    for k in range(int(horizon)):
        do_tx = bool(policy_fn(k, P))

        w = rng.normal(0.0, 1.0, size=(n,))
        v = rng.normal(0.0, 1.0, size=(p,))
        x = A @ x + B @ u_prev + (w * np.sqrt(np.diag(Qw)))
        y = C @ x + (v * np.sqrt(np.diag(Rv)))

        x_hat_pred = A @ x_hat + B @ u_prev
        P_pred = A @ P @ A.T + Qw

        if do_tx and (float(rng.random()) < float(p_success)):
            innovation = y - (C @ x_hat_pred)
            S = C @ P_pred @ C.T + Rv
            Kk = P_pred @ C.T @ np.linalg.pinv(S)
            x_hat = x_hat_pred + (Kk @ innovation)
            P = (np.eye(n) - Kk @ C) @ P_pred
        else:
            x_hat = x_hat_pred
            P = P_pred

        P = 0.5 * (P + P.T)

        u = -(K @ x_hat).reshape(-1)
        u_prev = u

        J_P += float(np.trace(P))
        J_C += float(do_tx)
        J_X += float(norm2(x) ** 2)

    return dict(J_P=J_P, J_C=J_C, J_X=J_X)


def _eval_policy(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    Qw: np.ndarray,
    Rv: np.ndarray,
    K: np.ndarray,
    P0: np.ndarray,
    horizon: int,
    p_success: float,
    seeds: list[int],
    policy_fn,
) -> dict:
    Jp, Jc, Jx = [], [], []
    for s in seeds:
        rng = np.random.default_rng(int(s))
        out = _simulate_rollout(A, B, C, Qw, Rv, K, P0, horizon, p_success, rng, policy_fn)
        Jp.append(out["J_P"])
        Jc.append(out["J_C"])
        Jx.append(out["J_X"])
    return dict(J_P=np.asarray(Jp), J_C=np.asarray(Jc), J_X=np.asarray(Jx))


def _summarize_eval(eval_out: dict) -> dict:
    jp, jp_ci = _mean_ci(eval_out["J_P"])
    jc, jc_ci = _mean_ci(eval_out["J_C"])
    jx, jx_ci = _mean_ci(eval_out["J_X"])
    return dict(J_P=jp, J_P_ci=jp_ci, J_C=jc, J_C_ci=jc_ci, J_X=jx, J_X_ci=jx_ci)


def _make_et_policy_fn(delta: float):
    def policy_fn(k: int, P: np.ndarray) -> bool:
        return float(np.trace(P)) > float(delta)

    return policy_fn


def _lambda_max(P: np.ndarray) -> float:
    return float(np.linalg.eigvalsh(P).max())


def run(outdir: str = "figs", mc_runs: int | None = None, t_steps: int | None = None, fast: bool = False) -> None:
    apply_ieee_style()
    cfg = SimConfig()
    cfg.T_steps = int(t_steps or (60 if fast else 120))
    cfg.mc_runs = int(mc_runs or (10 if fast else 40))
    cfg.p_success = 0.7
    cfg.sigma_w = 0.04
    cfg.sigma_v = 0.03

    # anisotropic 2D system to expose trace insufficiency
    A = np.array([[1.08, 0.10],
                  [0.00, 0.90]], dtype=float)
    B = np.array([[0.10],
                  [0.05]], dtype=float)
    C = np.array([[1.0, 0.25]], dtype=float)

    Q = np.diag([10.0, 1.0]).astype(float)
    R = np.array([[0.1]], dtype=float)
    K = dlqr(A, B, Q, R)

    Qw = (float(cfg.sigma_w) ** 2) * np.eye(A.shape[0], dtype=float)
    Rv = (float(cfg.sigma_v) ** 2) * np.eye(C.shape[0], dtype=float)
    P0 = float(cfg.P0_scale) * np.eye(A.shape[0], dtype=float)

    rng_master = np.random.default_rng(int(cfg.seed) & 0xFFFFFFFF)
    seeds = rng_master.integers(0, 2**31 - 1, size=int(cfg.mc_runs), dtype=np.int64).tolist()

    # feature grid built from reachable covariances
    samples = sample_covariances(
        P0=P0,
        A=A,
        C=C,
        Qw=Qw,
        Rv=Rv,
        p_success=float(cfg.p_success),
        horizon=int(cfg.T_steps),
        n_rollouts=260 if not fast else 100,
        rng=rng_master,
        tx_prob=0.5,
    )
    grid = build_feature_grid(samples, grid_shape=(20, 20))

    lambdas = np.linspace(0.0, 1.0, 10)
    trace_points = []
    feat_points = []

    for lamb in lambdas:
        s_grid, actions_trace = dp_trace_policy(
            P0=P0,
            A=A,
            C=C,
            Qw=Qw,
            Rv=Rv,
            p_success=float(cfg.p_success),
            lamb=float(lamb),
            horizon=int(cfg.T_steps),
            grid_size=120,
        )
        policy_trace = make_trace_policy_fn(s_grid, actions_trace)
        trace_eval = _eval_policy(A, B, C, Qw, Rv, K, P0, cfg.T_steps, cfg.p_success, seeds, policy_trace)

        actions_feat = dp_feature_policy(
            grid=grid,
            A=A,
            C=C,
            Qw=Qw,
            Rv=Rv,
            p_success=float(cfg.p_success),
            lamb=float(lamb),
            horizon=int(cfg.T_steps),
        )
        policy_feat = make_feature_policy_fn(grid, actions_feat)
        feat_eval = _eval_policy(A, B, C, Qw, Rv, K, P0, cfg.T_steps, cfg.p_success, seeds, policy_feat)

        trace_points.append(dict(lamb=float(lamb), raw=trace_eval, **_summarize_eval(trace_eval)))
        feat_points.append(dict(lamb=float(lamb), raw=feat_eval, **_summarize_eval(feat_eval)))

    traces = np.array([float(np.trace(P)) for P in samples], dtype=float)
    t_min, t_max = float(traces.min()), float(traces.max())
    if t_max <= t_min:
        t_max = t_min + 1.0
    deltas = np.linspace(t_min * 0.9, t_max * 1.05, 12)
    et_points = []
    for delta in deltas:
        policy_et = _make_et_policy_fn(float(delta))
        et_eval = _eval_policy(A, B, C, Qw, Rv, K, P0, cfg.T_steps, cfg.p_success, seeds, policy_et)
        et_points.append(dict(delta=float(delta), raw=et_eval, **_summarize_eval(et_eval)))

    def _sorted_curve(points: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        jc = np.array([p["J_C"] for p in points], dtype=float)
        jp = np.array([p["J_P"] for p in points], dtype=float)
        order = np.argsort(jc)
        return jc[order], jp[order]

    jc_trace, jp_trace = _sorted_curve(trace_points)
    jc_feat, jp_feat = _sorted_curve(feat_points)
    jc_et, jp_et = _sorted_curve(et_points)

    jc_min = max(jc_trace.min(), jc_feat.min())
    jc_max = min(jc_trace.max(), jc_feat.max())
    jc_grid = np.linspace(jc_min, jc_max, 30)
    jp_trace_i = np.interp(jc_grid, jc_trace, jp_trace)
    jp_feat_i = np.interp(jc_grid, jc_feat, jp_feat)
    gap_jp = jp_trace_i - jp_feat_i

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7))
    ax0, ax1 = axes

    ax0.plot(jc_trace, jp_trace, "o-", label="DP-trace")
    ax0.plot(jc_feat, jp_feat, "o-", label="DP-2feature")
    ax0.plot(jc_et, jp_et, "o--", label="ET")
    ax0.set_xlabel(r"$J_C$")
    ax0.set_ylabel(r"$J_P$")
    ax0.grid(True, alpha=0.3)
    ax0.legend(frameon=False)

    ax1.plot(jc_grid, gap_jp, "o-", color="tab:red")
    ax1.axhline(0.0, color="k", linewidth=0.6, alpha=0.6)
    ax1.set_xlabel(r"$J_C$")
    ax1.set_ylabel(r"$J_P^{trace} - J_P^{2feat}$")
    ax1.grid(True, alpha=0.3)

    fig.suptitle("Trace is not enough: gap under matched $J_C$", y=1.02)
    fig.tight_layout()
    savefig(fig, Path(outdir), "fig2_gap_curve")

    # Counterexample scatter: fix trace band, show action split by lambda_max
    mid_k = int(cfg.T_steps // 2)
    best_feat = min(feat_points, key=lambda r: abs(r["J_C"] - np.median(jc_feat)))
    best_trace = min(trace_points, key=lambda r: abs(r["J_C"] - np.median(jc_trace)))
    actions_feat_mid = dp_feature_policy(
        grid=grid,
        A=A,
        C=C,
        Qw=Qw,
        Rv=Rv,
        p_success=float(cfg.p_success),
        lamb=float(best_feat["lamb"]),
        horizon=int(cfg.T_steps),
    )
    policy_feat_mid = make_feature_policy_fn(grid, actions_feat_mid)
    policy_trace_mid = make_trace_policy_fn(
        *dp_trace_policy(
            P0=P0,
            A=A,
            C=C,
            Qw=Qw,
            Rv=Rv,
            p_success=float(cfg.p_success),
            lamb=float(best_trace["lamb"]),
            horizon=int(cfg.T_steps),
            grid_size=120,
        )
    )

    feats = np.array([feature_trace_logdet(P) for P in samples], dtype=float)
    target_trace = float(np.median(traces))
    band = 0.03 * (t_max - t_min)
    if band <= 0:
        band = 1e-3
    mask = np.abs(traces - target_trace) <= band
    while mask.sum() < 60 and band < (t_max - t_min):
        band *= 1.5
        mask = np.abs(traces - target_trace) <= band

    x_tx, y_tx = [], []
    x_hold, y_hold = [], []
    trace_actions = []
    for P, feat, use in zip(samples, feats, mask, strict=False):
        if not use:
            continue
        a_trace = policy_trace_mid(mid_k, P)
        a_feat = policy_feat_mid(mid_k, P)
        trace_actions.append(int(a_trace))
        x_val = _lambda_max(P)
        y_val = float(feat[1])
        if a_feat:
            x_tx.append(x_val)
            y_tx.append(y_val)
        else:
            x_hold.append(x_val)
            y_hold.append(y_val)

    trace_action = "tx" if (np.mean(trace_actions) > 0.5) else "hold"
    fig2, ax = plt.subplots(1, 1, figsize=(3.4, 2.6))
    ax.scatter(x_hold, y_hold, s=10, alpha=0.6, label="DP-2feature: hold")
    ax.scatter(x_tx, y_tx, s=12, alpha=0.8, label="DP-2feature: transmit")
    ax.set_xlabel(r"$\lambda_{\max}(P)$")
    ax.set_ylabel(r"$\log\det(P)$")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    ax.set_title(f"Trace band action split (trace-DP: {trace_action})")
    combined_x = np.array(x_tx + x_hold, dtype=float)
    combined_y = np.array(y_tx + y_hold, dtype=float)
    if combined_x.size > 0 and combined_y.size > 0:
        ax.annotate(
            "larger $\\lambda_{\\max}$\n=> higher risk",
            xy=(np.percentile(combined_x, 85), np.percentile(combined_y, 50)),
            xytext=(0.02, 0.98),
            textcoords="axes fraction",
            ha="left",
            va="top",
            arrowprops=dict(arrowstyle="->", linewidth=0.6),
        )
    fig2.tight_layout()
    savefig(fig2, Path(outdir), "fig2_counterexample_scatter")


if __name__ == "__main__":
    run()
