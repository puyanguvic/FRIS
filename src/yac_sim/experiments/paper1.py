from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from ..common.config import SimConfig
from ..common.models import double_integrator_2d
from ..common.sim_single import simulate, monte_carlo
from ..common.utils import mean_ci95
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


def _measurement_matrices(cfg: SimConfig, n: int) -> tuple[np.ndarray, int]:
    if getattr(cfg, "C_full_state", True):
        C = np.eye(n, dtype=float)
        p = n
        return C, p
    C = np.array([[1.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.0]], dtype=float)
    p = C.shape[0]
    return C, p


def _dp_trace_policy_fn(cfg: SimConfig, lamb: float, grid_size: int = 180):
    A, _ = double_integrator_2d(cfg.Ts)
    n = A.shape[0]
    C, p = _measurement_matrices(cfg, n)
    Qw = (float(cfg.sigma_w) ** 2) * np.eye(n, dtype=float)
    Rv = (float(cfg.sigma_v) ** 2) * np.eye(p, dtype=float)
    P0 = float(cfg.P0_scale) * np.eye(n, dtype=float)
    s_grid, actions = dp_trace_policy(
        P0=P0,
        A=A,
        C=C,
        Qw=Qw,
        Rv=Rv,
        p_success=float(cfg.p_success),
        lamb=float(lamb),
        horizon=int(cfg.T_steps),
        grid_size=int(grid_size),
    )
    return make_trace_policy_fn(s_grid, actions)


def _make_seeds(cfg: SimConfig) -> list[int]:
    rng_master = np.random.default_rng(int(cfg.seed) & 0xFFFFFFFF)
    return rng_master.integers(0, 2**31 - 1, size=int(cfg.mc_runs), dtype=np.int64).tolist()


def _mc_eval_policy(
    cfg: SimConfig,
    seeds: list[int],
    policy: str,
    policy_fn=None,
) -> dict:
    Jp, Jc, Jx, Nd = [], [], [], []
    for s in seeds:
        rng = np.random.default_rng(int(s))
        out = simulate(cfg, policy, rng=rng, policy_fn=policy_fn)
        Jp.append(out["J_P"])
        Jc.append(out["J_C"])
        Jx.append(out["J_X"])
        Nd.append(out["N_deliv"])
    return dict(
        J_P=np.asarray(Jp, dtype=float),
        J_C=np.asarray(Jc, dtype=float),
        J_X=np.asarray(Jx, dtype=float),
        N_deliv=np.asarray(Nd, dtype=float),
    )


def _mean_ci(values: np.ndarray) -> tuple[float, float]:
    return mean_ci95(values)


def _closest_by_jc(res_list: list[dict], target: float) -> dict:
    diffs = []
    for r in res_list:
        m, _ = _mean_ci(r["J_C"])
        diffs.append(abs(m - target))
    idx = int(np.argmin(diffs))
    return res_list[idx]


def figure_A_tradeoff(cfg: SimConfig, outdir: Path) -> float:
    """Experiment A: finite-horizon trade-off curves."""
    apply_ieee_style()

    deltas = np.logspace(np.log10(0.05), np.log10(3.0), 18).tolist()
    periods = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40]
    lambdas = np.logspace(-2, 0.7, 9)

    seeds = _make_seeds(cfg)
    et_res = monte_carlo(cfg, "ET", deltas=deltas, seeds=seeds)
    per_res = monte_carlo(cfg, "PER", periods=periods, seeds=seeds)

    dp_points = []
    for lamb in lambdas:
        policy_fn = _dp_trace_policy_fn(cfg, float(lamb))
        res = _mc_eval_policy(cfg, seeds, "DP", policy_fn=policy_fn)
        res["lambda"] = float(lamb)
        dp_points.append(res)

    et_match = []
    per_match = []
    for dp in dp_points:
        jc_mean, _ = _mean_ci(dp["J_C"])
        et_match.append(_closest_by_jc(et_res, jc_mean))
        per_match.append(_closest_by_jc(per_res, jc_mean))

    def _collect(points: list[dict], key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xs, ys, yci = [], [], []
        for p in points:
            x_m, _ = _mean_ci(p["J_C"])
            y_m, y_hw = _mean_ci(p[key])
            xs.append(x_m)
            ys.append(y_m)
            yci.append(y_hw)
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        yci = np.asarray(yci, dtype=float)
        order = np.argsort(xs)
        return xs[order], ys[order], yci[order]

    x_dp, y_dp, yci_dp = _collect(dp_points, "J_P")
    x_et, y_et, yci_et = _collect(et_match, "J_P")
    x_per, y_per, yci_per = _collect(per_match, "J_P")

    x_dp_x, y_dp_x, yci_dp_x = _collect(dp_points, "J_X")
    x_et_x, y_et_x, yci_et_x = _collect(et_match, "J_X")
    x_per_x, y_per_x, yci_per_x = _collect(per_match, "J_X")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.4))

    ax1.errorbar(x_dp, y_dp, yerr=yci_dp, marker="D", label="DP-trace")
    ax1.errorbar(x_et, y_et, yerr=yci_et, marker="o", label="ET")
    ax1.errorbar(x_per, y_per, yerr=yci_per, marker="s", label="PER")
    ax1.set_xlabel(r"$J_C$")
    ax1.set_ylabel(r"$J_P$")
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax1.legend(loc="best", frameon=True, borderpad=0.3)

    ax2.errorbar(x_dp_x, y_dp_x, yerr=yci_dp_x, marker="D", label="DP-trace")
    ax2.errorbar(x_et_x, y_et_x, yerr=yci_et_x, marker="o", label="ET")
    ax2.errorbar(x_per_x, y_per_x, yerr=yci_per_x, marker="s", label="PER")
    ax2.set_xlabel(r"$J_C$")
    ax2.set_ylabel(r"$J_X$")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax2.legend(loc="best", frameon=True, borderpad=0.3)

    fig.tight_layout(pad=0.6)
    savefig(fig, outdir, "fig_A_tradeoff_curves")
    plt.close(fig)

    mid_idx = len(dp_points) // 2
    delta_mid = float(et_match[mid_idx]["param"])
    return delta_mid


def figure_B_time_response(cfg: SimConfig, outdir: Path, delta: float) -> None:
    """Experiment B: grow-and-reset visualization."""
    apply_ieee_style()

    cfg1 = SimConfig(**cfg.__dict__)
    cfg1.delta = float(delta)

    rng = np.random.default_rng(int(cfg.seed) + 123)
    out = simulate(cfg1, "ET", rng=rng)

    P_trace = out["P_trace"]
    e_tilde = out["tilde_x_norm"]
    tx = out["tx_attempt"]
    rx = out["tx_deliv"]
    t = np.arange(cfg1.T_steps) * cfg1.Ts

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 2.6), sharex=True)

    ax1.plot(t, P_trace, label=r"$\mathrm{tr}(P_k)$")
    ax1.axhline(cfg1.delta, linestyle="--", linewidth=0.8, label=r"threshold $\delta$")
    idx_tx = np.where(tx > 0)[0]
    idx_rx = np.where(rx > 0)[0]
    if idx_tx.size > 0:
        ax1.vlines(t[idx_tx], ymin=0, ymax=np.max(P_trace), linewidth=0.5, alpha=0.4, label="attempt")
    if idx_rx.size > 0:
        ax1.vlines(t[idx_rx], ymin=0, ymax=np.max(P_trace), linewidth=0.8, alpha=0.6, label="reception")
    ax1.set_ylabel(r"$\mathrm{tr}(P_k)$")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", frameon=True, borderpad=0.3)

    ax2.plot(t, e_tilde, label=r"$\|x_k-\hat{x}_k\|_2$")
    if idx_rx.size > 0:
        ax2.vlines(t[idx_rx], ymin=0, ymax=np.max(e_tilde), linewidth=0.6, alpha=0.5)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Error norm")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", frameon=True, borderpad=0.3)

    fig.tight_layout(pad=0.6)
    savefig(fig, outdir, "fig_B_time_response")
    plt.close(fig)


def figure_C_sensitivity(cfg: SimConfig, outdir: Path, lamb: float) -> None:
    """Experiment C: sensitivity to packet success probability."""
    apply_ieee_style()

    p_vals = np.linspace(0.4, 0.95, 7)
    cfg_nom = SimConfig(**cfg.__dict__)
    cfg_nom.p_success = float(cfg.p_success)

    seeds = _make_seeds(cfg_nom)
    policy_nom = _dp_trace_policy_fn(cfg_nom, float(lamb))

    # match ET to nominal DP communication
    dp_nom = _mc_eval_policy(cfg_nom, seeds, "DP", policy_fn=policy_nom)
    jc_nom, _ = _mean_ci(dp_nom["J_C"])

    deltas = np.logspace(np.log10(0.05), np.log10(3.0), 18).tolist()
    et_sweep = monte_carlo(cfg_nom, "ET", deltas=deltas, seeds=seeds)
    et_nom = _closest_by_jc(et_sweep, jc_nom)
    delta_nom = float(et_nom["param"])

    jp_nr, jx_nr = [], []
    jp_rt, jx_rt = [], []
    jp_nr_et, jx_nr_et = [], []
    jp_rt_et, jx_rt_et = [], []

    for p_succ in p_vals:
        cfg_eval = SimConfig(**cfg.__dict__)
        cfg_eval.p_success = float(p_succ)

        # no-retune (use nominal policy)
        dp_nr = _mc_eval_policy(cfg_eval, seeds, "DP", policy_fn=policy_nom)
        jp_nr.append(_mean_ci(dp_nr["J_P"])[0])
        jx_nr.append(_mean_ci(dp_nr["J_X"])[0])

        et_nr_cfg = SimConfig(**cfg_eval.__dict__)
        et_nr_cfg.delta = delta_nom
        et_nr = _mc_eval_policy(et_nr_cfg, seeds, "ET")
        jp_nr_et.append(_mean_ci(et_nr["J_P"])[0])
        jx_nr_et.append(_mean_ci(et_nr["J_X"])[0])

        # retune
        policy_rt = _dp_trace_policy_fn(cfg_eval, float(lamb))
        dp_rt = _mc_eval_policy(cfg_eval, seeds, "DP", policy_fn=policy_rt)
        jp_rt.append(_mean_ci(dp_rt["J_P"])[0])
        jx_rt.append(_mean_ci(dp_rt["J_X"])[0])

        et_sweep_rt = monte_carlo(cfg_eval, "ET", deltas=deltas, seeds=seeds)
        et_rt = _closest_by_jc(et_sweep_rt, _mean_ci(dp_rt["J_C"])[0])
        et_rt_cfg = SimConfig(**cfg_eval.__dict__)
        et_rt_cfg.delta = float(et_rt["param"])
        et_rt_eval = _mc_eval_policy(et_rt_cfg, seeds, "ET")
        jp_rt_et.append(_mean_ci(et_rt_eval["J_P"])[0])
        jx_rt_et.append(_mean_ci(et_rt_eval["J_X"])[0])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.4))

    ax1.plot(p_vals, jp_nr, "D-", label="DP-trace no-retune")
    ax1.plot(p_vals, jp_rt, "D--", label="DP-trace retune")
    ax1.plot(p_vals, jp_nr_et, "o-", label="ET no-retune")
    ax1.plot(p_vals, jp_rt_et, "o--", label="ET retune")
    ax1.set_xlabel(r"$p$")
    ax1.set_ylabel(r"$J_P$")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", frameon=True, borderpad=0.3)

    ax2.plot(p_vals, jx_nr, "D-", label="DP-trace no-retune")
    ax2.plot(p_vals, jx_rt, "D--", label="DP-trace retune")
    ax2.plot(p_vals, jx_nr_et, "o-", label="ET no-retune")
    ax2.plot(p_vals, jx_rt_et, "o--", label="ET retune")
    ax2.set_xlabel(r"$p$")
    ax2.set_ylabel(r"$J_X$")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", frameon=True, borderpad=0.3)

    fig.tight_layout(pad=0.6)
    savefig(fig, outdir, "fig_C_sensitivity_p")
    plt.close(fig)


def figure_D_robustness(cfg: SimConfig, outdir: Path, lamb: float) -> None:
    """Experiment D: robustness to mismatch and bursty losses."""
    apply_ieee_style()

    seeds = _make_seeds(cfg)
    policy_nom = _dp_trace_policy_fn(cfg, float(lamb))

    dp_nom = _mc_eval_policy(cfg, seeds, "DP", policy_fn=policy_nom)
    jp_nom = _mean_ci(dp_nom["J_P"])[0]
    jx_nom = _mean_ci(dp_nom["J_X"])[0]

    deltas = np.logspace(np.log10(0.05), np.log10(3.0), 18).tolist()
    et_sweep = monte_carlo(cfg, "ET", deltas=deltas, seeds=seeds)
    et_nom = _closest_by_jc(et_sweep, _mean_ci(dp_nom["J_C"])[0])
    delta_nom = float(et_nom["param"])

    # channel mismatch (iid p)
    p_vals = np.linspace(0.4, 0.95, 7)
    dp_ratio_p, et_ratio_p = [], []
    dp_ratio_x, et_ratio_x = [], []
    for p_succ in p_vals:
        cfg_eval = SimConfig(**cfg.__dict__)
        cfg_eval.p_success = float(p_succ)
        dp_eval = _mc_eval_policy(cfg_eval, seeds, "DP", policy_fn=policy_nom)
        dp_ratio_p.append(_mean_ci(dp_eval["J_P"])[0] / jp_nom)
        dp_ratio_x.append(_mean_ci(dp_eval["J_X"])[0] / jx_nom)

        et_eval_cfg = SimConfig(**cfg_eval.__dict__)
        et_eval_cfg.delta = delta_nom
        et_eval = _mc_eval_policy(et_eval_cfg, seeds, "ET")
        et_ratio_p.append(_mean_ci(et_eval["J_P"])[0] / jp_nom)
        et_ratio_x.append(_mean_ci(et_eval["J_X"])[0] / jx_nom)

    # bursty losses (Gilbert-Elliott)
    betas = np.array([0.02, 0.05, 0.1, 0.2, 0.35, 0.5])
    dp_ratio_b, et_ratio_b = [], []
    for beta in betas:
        cfg_ge = SimConfig(**cfg.__dict__)
        cfg_ge.channel_model = "ge"
        cfg_ge.p_bad_to_good = float(beta)
        dp_eval = _mc_eval_policy(cfg_ge, seeds, "DP", policy_fn=policy_nom)
        dp_ratio_b.append(_mean_ci(dp_eval["J_P"])[0] / jp_nom)

        et_eval_cfg = SimConfig(**cfg_ge.__dict__)
        et_eval_cfg.delta = delta_nom
        et_eval = _mc_eval_policy(et_eval_cfg, seeds, "ET")
        et_ratio_b.append(_mean_ci(et_eval["J_P"])[0] / jp_nom)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.4))

    ax1.plot(p_vals, dp_ratio_p, "D-", label="DP-trace $J_P$")
    ax1.plot(p_vals, et_ratio_p, "o-", label="ET $J_P$")
    ax1.plot(p_vals, dp_ratio_x, "D--", label="DP-trace $J_X$")
    ax1.plot(p_vals, et_ratio_x, "o--", label="ET $J_X$")
    ax1.set_xlabel(r"$p$")
    ax1.set_ylabel("Normalized degradation")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", frameon=True, borderpad=0.3)

    ax2.plot(betas, dp_ratio_b, "D-", label="DP-trace (bursty)")
    ax2.plot(betas, et_ratio_b, "o-", label="ET (bursty)")
    ax2.set_xlabel(r"$\beta$ (bad$\to$good)")
    ax2.set_ylabel(r"$J_P/J_P^{nom}$")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", frameon=True, borderpad=0.3)

    fig.tight_layout(pad=0.6)
    savefig(fig, outdir, "fig_D_robustness")
    plt.close(fig)


def run_all(outdir: str = "figs", mc_runs: int | None = None, t_steps: int | None = None, fast: bool = False) -> None:
    outdir = Path(outdir)
    cfg = SimConfig()

    if fast:
        if mc_runs is None:
            mc_runs = 6
        if t_steps is None:
            t_steps = 120
    if mc_runs is not None:
        cfg.mc_runs = int(mc_runs)
    if t_steps is not None:
        cfg.T_steps = int(t_steps)
    if mc_runs is not None or t_steps is not None:
        print(f"[fris] running with mc_runs={cfg.mc_runs}, T_steps={cfg.T_steps}")

    delta_mid = figure_A_tradeoff(cfg, outdir)
    figure_B_time_response(cfg, outdir, delta_mid)
    figure_C_sensitivity(cfg, outdir, lamb=0.2)
    figure_D_robustness(cfg, outdir, lamb=0.2)
