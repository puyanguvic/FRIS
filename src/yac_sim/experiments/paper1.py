from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from ..common.config import SimConfig
from ..common.models import double_integrator_2d
from ..common.sim_single import simulate, monte_carlo
from ..common.utils import mean_ci95, knee_point


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


def _measurement_matrices(cfg: SimConfig, n: int) -> tuple[np.ndarray, int]:
    if getattr(cfg, "C_full_state", True):
        C = np.eye(n, dtype=float)
        p = n
        return C, p
    C = np.array([[1.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.0]], dtype=float)
    p = C.shape[0]
    return C, p


def _summarize_curve(
    res_list: list[dict],
    metric_key: str,
    xkey: str = "N_deliv",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, yci = [], [], []
    for r in res_list:
        x_m, _ = mean_ci95(r[xkey])
        y_m, y_hw = mean_ci95(r[metric_key])
        xs.append(x_m)
        ys.append(y_m)
        yci.append(y_hw)
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    yci = np.asarray(yci, dtype=float)
    order = np.argsort(xs)
    return xs[order], ys[order], yci[order]


def _best_under_budget(curve_bits: np.ndarray, curve_y: np.ndarray, budget: float) -> float:
    if budget <= 0:
        return float(curve_y.min())
    mask = curve_bits <= budget
    if np.any(mask):
        return float(curve_y[mask].min())
    return float(curve_y[np.argmin(np.abs(curve_bits - budget))])


def figure_A_pareto(cfg: SimConfig, outdir: Path) -> float:
    """Figure A: Pareto curves for performance/energy vs communication."""
    apply_ieee_style()

    deltas = np.logspace(np.log10(0.05), np.log10(2.0), 12).tolist()
    periods = [1, 2, 3, 4, 5, 8, 10, 15, 20, 30, 40]
    ps = np.linspace(0.05, 0.5, 10).tolist()

    sigma_levels = [
        max(cfg.sigma_w * 0.5, 0.01),
        cfg.sigma_w,
        max(cfg.sigma_w * 1.5, 0.02),
    ]

    et_curves = []
    for sigma in sigma_levels:
        cfg_i = SimConfig(**cfg.__dict__)
        cfg_i.sigma_w = float(sigma)
        et_res = monte_carlo(cfg_i, "ET", deltas=deltas)
        et_curves.append((sigma, et_res))

    per_res = monte_carlo(cfg, "PER", periods=periods)
    rand_res = monte_carlo(cfg, "RAND", random_ps=ps)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.4))

    # ET curves for different disturbance levels
    for sigma, et_res in et_curves:
        x_et, y_et, _ = _summarize_curve(et_res, "Jx")
        ax1.plot(x_et, y_et, marker="o", label=fr"ET ($\sigma_w$={sigma:.3f})")
        x_et_e, y_et_e, _ = _summarize_curve(et_res, "Eu")
        ax2.plot(x_et_e, y_et_e, marker="o", label=fr"ET ($\sigma_w$={sigma:.3f})")

    # Baselines (nominal disturbance)
    x_per, y_per, _ = _summarize_curve(per_res, "Jx")
    x_rd, y_rd, _ = _summarize_curve(rand_res, "Jx")
    ax1.plot(x_per, y_per, "s--", markerfacecolor="white", markeredgewidth=0.7, label="PER")
    ax1.plot(x_rd, y_rd, "^:", markerfacecolor="white", markeredgewidth=0.7, label="RAND")

    x_per_e, y_per_e, _ = _summarize_curve(per_res, "Eu")
    x_rd_e, y_rd_e, _ = _summarize_curve(rand_res, "Eu")
    ax2.plot(x_per_e, y_per_e, "s--", markerfacecolor="white", markeredgewidth=0.7, label="PER")
    ax2.plot(x_rd_e, y_rd_e, "^:", markerfacecolor="white", markeredgewidth=0.7, label="RAND")

    ax1.set_xlabel("Delivered updates")
    ax1.set_ylabel(r"$J_x$")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 150)
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))

    ax2.set_xlabel("Delivered updates")
    ax2.set_ylabel(r"$E_u$")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 150)
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=5))

    ax1.legend(loc="best", frameon=True, borderpad=0.3)
    ax2.legend(loc="best", frameon=True, borderpad=0.3)

    fig.tight_layout(pad=0.6)
    savefig(fig, outdir, "fig_A_pareto_tradeoff")
    plt.close(fig)

    # knee point on nominal ET curve (Jx vs N_tx)
    et_nom = et_curves[1][1]
    x_et_nom, y_et_nom, _ = _summarize_curve(et_nom, "Jx")
    knee_idx = knee_point(x_et_nom, y_et_nom)
    et_x_means = np.asarray([mean_ci95(r["N_deliv"])[0] for r in et_nom], dtype=float)
    et_order = np.argsort(et_x_means)
    et_params_sorted = np.asarray([r["param"] for r in et_nom], dtype=float)[et_order]
    delta_knee = float(et_params_sorted[knee_idx])
    return delta_knee


def figure_B_time_response(cfg: SimConfig, outdir: Path, delta_knee: float) -> None:
    """Figure B: grow-and-reset visualization for trace(P) and transmission error."""
    apply_ieee_style()

    cfg1 = SimConfig(**cfg.__dict__)
    cfg1.delta = float(delta_knee)

    rng = np.random.default_rng(int(cfg.seed) + 123)
    out = simulate(cfg1, "ET", rng=rng)

    P_trace = out["P_trace"]
    e_tilde = out["tilde_x_norm"]
    tx = out["tx_deliv"]
    t = np.arange(cfg1.T_steps) * cfg1.Ts

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 2.6), sharex=True)

    ax1.plot(t, P_trace, label=r"$\mathrm{tr}(P_k)$")
    ax1.axhline(cfg1.delta, linestyle="--", linewidth=0.8, label=r"threshold $\delta$")
    idx = np.where(tx > 0)[0]
    if idx.size > 0:
        ax1.vlines(t[idx], ymin=0, ymax=np.max(P_trace), linewidth=0.6, alpha=0.5)
    ax1.set_ylabel(r"$\mathrm{tr}(P_k)$")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", frameon=True, borderpad=0.3)

    ax2.plot(t, e_tilde, label=r"$\|x_k-\hat{x}_k\|_2$")
    if idx.size > 0:
        ax2.vlines(t[idx], ymin=0, ymax=np.max(e_tilde), linewidth=0.6, alpha=0.5)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Error norm")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", frameon=True, borderpad=0.3)

    fig.tight_layout(pad=0.6)
    savefig(fig, outdir, "fig_B_time_response")
    plt.close(fig)


def figure_C_budget_curves(cfg: SimConfig, outdir: Path) -> None:
    """Figure C: budgeted operation under communication constraints."""
    apply_ieee_style()

    deltas = np.logspace(np.log10(0.05), np.log10(2.0), 14).tolist()
    periods = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40]

    et_res = monte_carlo(cfg, "ET", deltas=deltas)
    per_res = monte_carlo(cfg, "PER", periods=periods)

    x_et, y_et, _ = _summarize_curve(et_res, "Jx")
    x_et_e, y_et_e, _ = _summarize_curve(et_res, "Eu")
    x_per, y_per, _ = _summarize_curve(per_res, "Jx")
    x_per_e, y_per_e, _ = _summarize_curve(per_res, "Eu")

    budgets = np.unique(np.round(x_et)).astype(float)
    budgets = budgets[budgets > 0]

    et_best = np.array([_best_under_budget(x_et, y_et, b) for b in budgets], dtype=float)
    per_best = np.array([_best_under_budget(x_per, y_per, b) for b in budgets], dtype=float)
    et_best_e = np.array([_best_under_budget(x_et_e, y_et_e, b) for b in budgets], dtype=float)
    per_best_e = np.array([_best_under_budget(x_per_e, y_per_e, b) for b in budgets], dtype=float)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.6, 3.0), sharex=True)

    ax1.plot(budgets, et_best, marker="o", label="ET")
    ax1.plot(budgets, per_best, marker="s", label="PER")
    ax1.set_ylabel(r"$J_x$")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", frameon=True, borderpad=0.3)

    ax2.plot(budgets, et_best_e, marker="o", label="ET")
    ax2.plot(budgets, per_best_e, marker="s", label="PER")
    ax2.set_xlabel(r"Budget $N_{\max}$ (updates)")
    ax2.set_ylabel(r"$E_u$")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", frameon=True, borderpad=0.3)

    fig.tight_layout(pad=0.6)
    savefig(fig, outdir, "fig_C_budget_curves")
    plt.close(fig)


def _scalar_kalman_step(A: np.ndarray, C: np.ndarray, Qw: np.ndarray, Rv: np.ndarray, s: float) -> tuple[float, float]:
    n = A.shape[0]
    P = (s / n) * np.eye(n, dtype=float)
    P_pred = A @ P @ A.T + Qw
    S = C @ P_pred @ C.T + Rv
    Kk = P_pred @ C.T @ np.linalg.pinv(S)
    P_upd = (np.eye(n) - Kk @ C) @ P_pred
    return float(np.trace(P_pred)), float(np.trace(P_upd))


def _scalar_rollout(cfg: SimConfig, policy: str, param: float | int, lamb: float = 0.0) -> tuple[float, int]:
    A, _ = double_integrator_2d(cfg.Ts)
    n = A.shape[0]
    C, p = _measurement_matrices(cfg, n)
    Qw = (float(cfg.sigma_w) ** 2) * np.eye(n, dtype=float)
    Rv = (float(cfg.sigma_v) ** 2) * np.eye(p, dtype=float)

    s = float(cfg.P0_scale) * n
    Jp = 0.0
    N_tx = 0
    for k in range(cfg.T_steps):
        s_pred, s_upd = _scalar_kalman_step(A, C, Qw, Rv, s)
        if policy == "ET":
            do_tx = s_pred > float(param)
        elif policy == "PER":
            do_tx = (k % max(int(param), 1) == 0)
        else:
            raise ValueError(policy)
        Jp += s + lamb * float(do_tx)
        if do_tx:
            s = s_upd
            N_tx += 1
        else:
            s = s_pred
    return Jp, N_tx


def _dp_oracle(cfg: SimConfig, lamb: float, grid_size: int = 160) -> tuple[float, int]:
    A, _ = double_integrator_2d(cfg.Ts)
    n = A.shape[0]
    C, p = _measurement_matrices(cfg, n)
    Qw = (float(cfg.sigma_w) ** 2) * np.eye(n, dtype=float)
    Rv = (float(cfg.sigma_v) ** 2) * np.eye(p, dtype=float)

    # rough bounds from always-update vs never-update trajectories
    s = float(cfg.P0_scale) * n
    s_min, s_max = s, s
    for _ in range(cfg.T_steps):
        s_pred, s_upd = _scalar_kalman_step(A, C, Qw, Rv, s)
        s_min = min(s_min, s_upd)
        s_max = max(s_max, s_pred)
        s = s_pred
    s_grid = np.linspace(s_min * 0.8, s_max * 1.2, grid_size)

    V_next = np.zeros_like(s_grid)
    for _ in range(cfg.T_steps - 1, -1, -1):
        V_curr = np.zeros_like(s_grid)
        for i, s_i in enumerate(s_grid):
            s_pred, s_upd = _scalar_kalman_step(A, C, Qw, Rv, float(s_i))
            v0 = s_i + np.interp(s_pred, s_grid, V_next)
            v1 = s_i + lamb + np.interp(s_upd, s_grid, V_next)
            V_curr[i] = min(v0, v1)
        V_next = V_curr

    # rollout with DP policy
    s = float(cfg.P0_scale) * n
    Jp = 0.0
    N_tx = 0
    for _ in range(cfg.T_steps):
        s_pred, s_upd = _scalar_kalman_step(A, C, Qw, Rv, s)
        v0 = s + np.interp(s_pred, s_grid, V_next)
        v1 = s + lamb + np.interp(s_upd, s_grid, V_next)
        if v1 < v0:
            Jp += s + lamb
            s = s_upd
            N_tx += 1
        else:
            Jp += s
            s = s_pred
    return Jp, N_tx


def figure_D_oracle_comparison(cfg: SimConfig, outdir: Path) -> None:
    """Figure D: DP oracle vs threshold ET and periodic scheduling."""
    apply_ieee_style()

    deltas = np.logspace(np.log10(0.05), np.log10(2.0), 18).tolist()
    periods = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40]
    lambdas = np.logspace(-2, 0.7, 10)

    # ET and PER curves on the reduced-order trace model
    et_points = np.array([_scalar_rollout(cfg, "ET", d) for d in deltas], dtype=float)
    per_points = np.array([_scalar_rollout(cfg, "PER", m) for m in periods], dtype=float)

    x_et, y_et = et_points[:, 1], et_points[:, 0]
    x_per, y_per = per_points[:, 1], per_points[:, 0]
    order_et = np.argsort(x_et)
    order_per = np.argsort(x_per)

    # DP oracle sweep over lambda
    dp_points = np.array([_dp_oracle(cfg, float(lmb)) for lmb in lambdas], dtype=float)
    x_dp, y_dp = dp_points[:, 1], dp_points[:, 0]
    order_dp = np.argsort(x_dp)

    fig = plt.figure(figsize=(3.5, 2.4))
    ax = fig.add_subplot(111)
    ax.plot(x_dp[order_dp], y_dp[order_dp], "D-", label="DP oracle")
    ax.plot(x_et[order_et], y_et[order_et], "o-", label="ET")
    ax.plot(x_per[order_per], y_per[order_per], "s--", label="PER")
    ax.set_xlabel("Delivered updates")
    ax.set_ylabel(r"$\sum \mathrm{tr}(P_k) + \lambda N_{\mathrm{tx}}$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True, borderpad=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    fig.tight_layout(pad=0.6)
    savefig(fig, outdir, "fig_D_oracle")
    plt.close(fig)


def run_all(outdir: str = "figs", mc_runs: int | None = None, t_steps: int | None = None, fast: bool = False) -> None:
    outdir = Path(outdir)
    cfg = SimConfig()

    if fast:
        if mc_runs is None:
            mc_runs = 6
        if t_steps is None:
            t_steps = 200
    if mc_runs is not None:
        cfg.mc_runs = int(mc_runs)
    if t_steps is not None:
        cfg.T_steps = int(t_steps)
    if mc_runs is not None or t_steps is not None:
        print(f"[yac] running with mc_runs={cfg.mc_runs}, T_steps={cfg.T_steps}")

    delta_knee = figure_A_pareto(cfg, outdir)
    figure_B_time_response(cfg, outdir, delta_knee)
    figure_C_budget_curves(cfg, outdir)
    figure_D_oracle_comparison(cfg, outdir)
