from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from .common.config import SimConfig
from .common.models import double_integrator_2d
from .common.sim_single import monte_carlo, simulate
from .common.utils import dlqr, mean_ci95, norm2
from .policies.dp_feature import (
    build_feature_grid,
    dp_feature_policy,
    feature_trace_logdet,
    make_feature_policy_fn,
    sample_covariances,
)
from .policies.dp_trace import dp_trace_policy, make_trace_policy_fn


def apply_ieee_style() -> None:
    plt.rcParams.update(
        {
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
        }
    )


def savefig(fig, outdir: Path, name: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{name}.png", dpi=300, bbox_inches="tight")


def _measurement_matrices(cfg: SimConfig, n: int) -> tuple[np.ndarray, int]:
    if getattr(cfg, "C_full_state", True):
        C = np.eye(n, dtype=float)
        p = n
        return C, p
    C = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
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


def _mean_ci(values: np.ndarray) -> tuple[float, float]:
    return mean_ci95(values)


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


def _closest_by_jc(res_list: list[dict], target: float) -> dict:
    diffs = []
    for r in res_list:
        m, _ = _mean_ci(r["J_C"])
        diffs.append(abs(m - target))
    idx = int(np.argmin(diffs))
    return res_list[idx]


def figure_A_tradeoff(cfg: SimConfig, outdir: Path) -> float:
    """Figure A: finite-horizon trade-off curves with DP-gap annotations."""
    apply_ieee_style()

    # Curves are shown as mean values (no error bars) to avoid clutter.
    # Include lambda=0 as the always-transmit endpoint of the Lagrangian relaxation.
    lambdas = np.concatenate(([0.0], np.logspace(-2, 2, 10)))
    deltas = [0.0] + np.logspace(np.log10(0.05), np.log10(2.0), 14).tolist()
    periods = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]

    seeds = _make_seeds(cfg)
    et_res = monte_carlo(cfg, "ET", deltas=deltas, seeds=seeds)
    per_res = monte_carlo(cfg, "PER", periods=periods, seeds=seeds)

    dp_points = []
    for lamb in lambdas:
        policy_fn = _dp_trace_policy_fn(cfg, float(lamb))
        res = _mc_eval_policy(cfg, seeds, "DP", policy_fn=policy_fn)
        res["lambda"] = float(lamb)
        dp_points.append(res)

    def _collect(points: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for pnt in points:
            x_m, _ = _mean_ci(pnt["J_C"])
            y_m, _ = _mean_ci(pnt[key])
            xs.append(x_m)
            ys.append(y_m)
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        order = np.argsort(xs)
        return xs[order], ys[order]

    x_dp, y_dp = _collect(dp_points, "J_P")
    x_et, y_et = _collect(et_res, "J_P")
    x_per, y_per = _collect(per_res, "J_P")

    # Gap curves at matched communication usage (interpolate baselines at DP x-axis).
    gap_et = np.full_like(y_dp, np.nan)
    if x_et.size > 1:
        gap_et = np.interp(x_dp, x_et, y_et, left=np.nan, right=np.nan) - y_dp
    gap_per = np.full_like(y_dp, np.nan)
    if x_per.size > 1:
        gap_per = np.interp(x_dp, x_per, y_per, left=np.nan, right=np.nan) - y_dp

    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.6))

    ax.plot(x_dp, y_dp, "D-", label="DP-trace")
    ax.plot(x_et, y_et, "o-", label="ET")
    ax.plot(x_per, y_per, "s--", label="PER")

    gap_indices = np.arange(len(x_dp), dtype=int)
    if gap_indices.size > 0:
        n_marks = min(8, gap_indices.size)
        gap_indices = np.linspace(0, gap_indices.size - 1, n_marks, dtype=int)
    x_span = float(np.max(x_dp) - np.min(x_dp)) if x_dp.size > 0 else 0.0
    x_offset = 0.012 * x_span

    def _gap_lines(x_ref, y_ref, y_base, color, label, offset):
        shown = False
        for idx in gap_indices:
            if not np.isfinite(y_base[idx]):
                continue
            x = x_ref[idx] + offset
            y0 = y_ref[idx]
            y1 = y_base[idx]
            ax.vlines(
                x,
                min(y0, y1),
                max(y0, y1),
                color=color,
                alpha=0.35,
                linewidth=0.9,
                label=label if not shown else None,
            )
            shown = True

    _gap_lines(x_dp, y_dp, y_dp + gap_et, "tab:orange", "ET gap to DP", -x_offset)
    _gap_lines(x_dp, y_dp, y_dp + gap_per, "tab:green", "PER gap to DP", x_offset)

    ax.set_xlabel(r"$J_C$")
    ax.set_ylabel(r"$J_P$")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.legend(loc="best", frameon=True, borderpad=0.3, ncol=2)

    fig.tight_layout(pad=0.6)
    savefig(fig, outdir, "fig_A_tradeoff_curves")
    plt.close(fig)

    mid_idx = len(lambdas) // 2
    return float(lambdas[mid_idx])


def figure_B_time_response(cfg: SimConfig, outdir: Path, lamb: float) -> None:
    """Figure B: rollout showing finite-horizon thresholds and event timing."""
    apply_ieee_style()

    cfg1 = SimConfig(**cfg.__dict__)

    rng = np.random.default_rng(int(cfg.seed) + 123)
    # Compute a representative DP-trace policy and its (approximate) time-varying
    # trace threshold s_k^* from the action table.
    A, _ = double_integrator_2d(cfg1.Ts)
    n = A.shape[0]
    C, p = _measurement_matrices(cfg1, n)
    Qw = (float(cfg1.sigma_w) ** 2) * np.eye(n, dtype=float)
    Rv = (float(cfg1.sigma_v) ** 2) * np.eye(p, dtype=float)
    P0 = float(cfg1.P0_scale) * np.eye(n, dtype=float)
    s_grid, actions = dp_trace_policy(
        P0=P0,
        A=A,
        C=C,
        Qw=Qw,
        Rv=Rv,
        p_success=float(cfg1.p_success),
        lamb=float(lamb),
        horizon=int(cfg1.T_steps),
        grid_size=180,
    )
    thresholds = np.full(int(cfg1.T_steps), np.nan, dtype=float)
    for k in range(int(cfg1.T_steps)):
        idx = np.where(actions[k] == 1)[0]
        if idx.size == 0:
            continue
        i1 = int(idx[0])
        if i1 <= 0:
            thresholds[k] = float(s_grid[0])
        else:
            thresholds[k] = 0.5 * float(s_grid[i1 - 1] + s_grid[i1])

    policy_fn = make_trace_policy_fn(s_grid, actions)
    out = simulate(cfg1, "DP", rng=rng, policy_fn=policy_fn)

    P_trace = out["P_trace"]
    e_tilde = out["tilde_x_norm"]
    gamma = out["gamma"]
    delta_arr = out["delta"]
    t = np.arange(cfg1.T_steps) * cfg1.Ts

    # Drop the first sample (large initial covariance) for readability.
    k0 = 1
    t_plot = t[k0:]
    P_plot = P_trace[k0:]
    th_plot = thresholds[k0:]
    e_plot = e_tilde[k0:]
    gamma_plot = gamma[k0:]
    delta_plot = delta_arr[k0:]

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(3.5, 2.8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.3]},
    )

    ax1.plot(t_plot, P_plot, label=r"$\mathrm{tr}(P_k)$")
    ax1.plot(t_plot, th_plot, "--", linewidth=0.9, label=r"DP threshold $s_k^\star$")
    ax1.set_ylabel(r"$\mathrm{tr}(P_k)$")
    ax1.grid(True, alpha=0.3)

    idx_tx = np.where(gamma_plot > 0)[0]
    idx_rx = np.where(delta_plot > 0)[0]
    times_tx = t_plot[idx_tx] if idx_tx.size > 0 else np.array([], dtype=float)
    times_rx = t_plot[idx_rx] if idx_rx.size > 0 else np.array([], dtype=float)

    y_min, y_max = ax1.get_ylim()
    rug_base = y_min + 0.02 * (y_max - y_min)
    rug_height = 0.06 * (y_max - y_min)
    ax1.vlines(
        times_tx,
        rug_base,
        rug_base + rug_height,
        color="tab:gray",
        alpha=0.7,
        linewidth=0.8,
    )
    ax1.vlines(
        times_rx,
        rug_base,
        rug_base + 2.0 * rug_height,
        color="tab:blue",
        alpha=0.8,
        linewidth=1.0,
    )
    ax1.plot([], [], color="tab:gray", linewidth=1.0, label="attempt")
    ax1.plot([], [], color="tab:blue", linewidth=1.2, label="ACK")
    ax1.legend(loc="best", frameon=True, borderpad=0.3)

    ax2.plot(t_plot, e_plot, label=r"$\|x_k-\hat{x}_k\|_2$")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Error norm")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", frameon=True, borderpad=0.3)

    fig.tight_layout(pad=0.6)
    savefig(fig, outdir, "fig_B_time_response")
    plt.close(fig)


def figure_C_channel_impairments(cfg: SimConfig, outdir: Path, lamb: float) -> None:
    """Figure C: channel impairments summary (p sensitivity, mismatch, burstiness)."""
    apply_ieee_style()

    p_vals = np.linspace(0.4, 0.95, 7)
    cfg_nom = SimConfig(**cfg.__dict__)
    cfg_nom.p_success = float(cfg.p_success)

    seeds = _make_seeds(cfg_nom)
    policy_nom = _dp_trace_policy_fn(cfg_nom, float(lamb))

    deltas = [0.0] + np.logspace(np.log10(0.05), np.log10(2.0), 14).tolist()
    et_nom_sweep = monte_carlo(cfg_nom, "ET", deltas=deltas, seeds=seeds)
    obj_means = [float((r["J_P"] + float(lamb) * r["J_C"]).mean()) for r in et_nom_sweep]
    delta_nom_obj = float(et_nom_sweep[int(np.argmin(obj_means))]["param"])

    obj_dp_nr, obj_dp_rt = [], []
    obj_et_nr, obj_et_rt = [], []
    jc_dp_nr, jc_dp_rt = [], []
    jc_et_nr, jc_et_rt = [], []
    obj_dp_gain, obj_et_gain = [], []

    for p_succ in p_vals:
        cfg_eval = SimConfig(**cfg.__dict__)
        cfg_eval.p_success = float(p_succ)

        dp_nr = _mc_eval_policy(cfg_eval, seeds, "DP", policy_fn=policy_nom)
        obj_dp_nr.append(float((dp_nr["J_P"] + float(lamb) * dp_nr["J_C"]).mean()))
        jc_dp_nr.append(float(dp_nr["J_C"].mean()))

        policy_rt = _dp_trace_policy_fn(cfg_eval, float(lamb))
        dp_rt = _mc_eval_policy(cfg_eval, seeds, "DP", policy_fn=policy_rt)
        obj_dp_rt.append(float((dp_rt["J_P"] + float(lamb) * dp_rt["J_C"]).mean()))
        jc_dp_rt.append(float(dp_rt["J_C"].mean()))

        et_sweep = monte_carlo(cfg_eval, "ET", deltas=deltas, seeds=seeds)
        idx_nom = int(np.argmin([abs(float(r["param"]) - delta_nom_obj) for r in et_sweep]))
        et_nr = et_sweep[idx_nom]
        obj_et_nr.append(float((et_nr["J_P"] + float(lamb) * et_nr["J_C"]).mean()))
        jc_et_nr.append(float(et_nr["J_C"].mean()))

        objs = [float((r["J_P"] + float(lamb) * r["J_C"]).mean()) for r in et_sweep]
        et_rt = et_sweep[int(np.argmin(objs))]
        obj_et_rt.append(float((et_rt["J_P"] + float(lamb) * et_rt["J_C"]).mean()))
        jc_et_rt.append(float(et_rt["J_C"].mean()))

        obj_dp_gain.append((obj_dp_nr[-1] - obj_dp_rt[-1]) / max(1e-9, obj_dp_nr[-1]))
        obj_et_gain.append((obj_et_nr[-1] - obj_et_rt[-1]) / max(1e-9, obj_et_nr[-1]))

    dp_nom = _mc_eval_policy(cfg_nom, seeds, "DP", policy_fn=policy_nom)
    jp_nom = _mean_ci(dp_nom["J_P"])[0]
    jx_nom = _mean_ci(dp_nom["J_X"])[0]

    et_nom_match = _closest_by_jc(et_nom_sweep, _mean_ci(dp_nom["J_C"])[0])
    delta_nom_match = float(et_nom_match["param"])

    dp_ratio_p, et_ratio_p = [], []
    dp_ratio_x, et_ratio_x = [], []
    for p_succ in p_vals:
        cfg_eval = SimConfig(**cfg.__dict__)
        cfg_eval.p_success = float(p_succ)
        dp_eval = _mc_eval_policy(cfg_eval, seeds, "DP", policy_fn=policy_nom)
        dp_ratio_p.append(_mean_ci(dp_eval["J_P"])[0] / jp_nom)
        dp_ratio_x.append(_mean_ci(dp_eval["J_X"])[0] / jx_nom)

        et_eval_cfg = SimConfig(**cfg_eval.__dict__)
        et_eval_cfg.delta = delta_nom_match
        et_eval = _mc_eval_policy(et_eval_cfg, seeds, "ET")
        et_ratio_p.append(_mean_ci(et_eval["J_P"])[0] / jp_nom)
        et_ratio_x.append(_mean_ci(et_eval["J_X"])[0] / jx_nom)

    alpha0 = 0.02
    pi_good = float(cfg.p_success)
    beta0 = (alpha0 * pi_good) / max(1e-6, (1.0 - pi_good))
    scales = np.array([0.25, 0.5, 1.0, 2.0, 5.0], dtype=float)
    bad_run = 1.0 / np.clip(beta0 * scales, 1e-6, None)
    dp_ratio_b, et_ratio_b = [], []
    for s in scales:
        cfg_ge = SimConfig(**cfg.__dict__)
        cfg_ge.channel_model = "ge"
        cfg_ge.loss_good = 0.0
        cfg_ge.loss_bad = 1.0
        cfg_ge.p_good_to_bad = float(alpha0 * s)
        cfg_ge.p_bad_to_good = float(beta0 * s)

        dp_eval = _mc_eval_policy(cfg_ge, seeds, "DP", policy_fn=policy_nom)
        dp_ratio_b.append(_mean_ci(dp_eval["J_P"])[0] / jp_nom)

        et_eval_cfg = SimConfig(**cfg_ge.__dict__)
        et_eval_cfg.delta = delta_nom_match
        et_eval = _mc_eval_policy(et_eval_cfg, seeds, "ET")
        et_ratio_b.append(_mean_ci(et_eval["J_P"])[0] / jp_nom)

    fig, axs = plt.subplots(2, 2, figsize=(6.8, 4.8))
    ax1, ax2 = axs[0, 0], axs[0, 1]
    ax3, ax4 = axs[1, 0], axs[1, 1]

    ax1.plot(p_vals, obj_dp_nr, "D-", label="DP-trace no-retune")
    ax1.plot(p_vals, obj_dp_rt, "D--", label="DP-trace retune")
    ax1.plot(p_vals, obj_et_nr, "o-", label="ET no-retune")
    ax1.plot(p_vals, obj_et_rt, "o--", label="ET retune")
    ax1.set_xlabel(r"$p$")
    ax1.set_ylabel(r"$J_{\lambda}=J_P+\lambda J_C$")
    ax1.set_title("(a) Lagrangian objective vs $p$")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", frameon=True, borderpad=0.3)
    ax1.set_xlim(float(p_vals.min()), float(p_vals.max()))

    ax2.plot(p_vals, 100.0 * np.array(obj_dp_gain), "D-", label="DP-trace gain")
    ax2.plot(p_vals, 100.0 * np.array(obj_et_gain), "o-", label="ET gain")
    ax2.axhline(0.0, color="k", linewidth=0.8, alpha=0.6)
    ax2.set_xlabel(r"$p$")
    ax2.set_ylabel(r"Retune gain in $J_{\lambda}$ (%)")
    ax2.set_title("(b) Retune gain (percent)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", frameon=True, borderpad=0.3)
    ax2.set_xlim(float(p_vals.min()), float(p_vals.max()))

    ax3.plot(p_vals, dp_ratio_p, "D-", label="DP-trace $J_P$")
    ax3.plot(p_vals, et_ratio_p, "o-", label="ET $J_P$")
    ax3.plot(p_vals, dp_ratio_x, "D--", label="DP-trace $J_X$")
    ax3.plot(p_vals, et_ratio_x, "o--", label="ET $J_X$")
    ax3.set_xlabel(r"$p$")
    ax3.set_ylabel("Normalized degradation")
    ax3.set_title("(c) Mismatch degradation vs $p$")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="best", frameon=True, borderpad=0.3)
    ax3.set_xlim(float(p_vals.min()), float(p_vals.max()))

    ax4.plot(bad_run, dp_ratio_b, "D-", label="DP-trace (bursty)")
    ax4.plot(bad_run, et_ratio_b, "o-", label="ET (bursty)")
    ax4.set_xlabel(r"Mean loss-burst length $1/\beta$ (steps)")
    ax4.set_ylabel(r"$J_P/J_P^{nom}$")
    ax4.set_title("(d) Bursty losses")
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc="best", frameon=True, borderpad=0.3)

    fig.tight_layout(pad=0.6, w_pad=0.6, h_pad=0.6)
    savefig(fig, outdir, "fig_C_channel_impairments")
    plt.close(fig)


def figure_C_sensitivity(cfg: SimConfig, outdir: Path, lamb: float) -> None:
    """Figure C: sensitivity to packet success probability."""
    apply_ieee_style()

    p_vals = np.linspace(0.4, 0.95, 7)
    cfg_nom = SimConfig(**cfg.__dict__)
    cfg_nom.p_success = float(cfg.p_success)

    seeds = _make_seeds(cfg_nom)
    policy_nom = _dp_trace_policy_fn(cfg_nom, float(lamb))

    # Define a nominal ET threshold by minimizing the same Lagrangian objective
    # J_lambda = J_P + lambda J_C at the nominal success probability.
    deltas = [0.0] + np.logspace(np.log10(0.05), np.log10(2.0), 14).tolist()
    et_nom_sweep = monte_carlo(cfg_nom, "ET", deltas=deltas, seeds=seeds)
    obj_means = [float((r["J_P"] + float(lamb) * r["J_C"]).mean()) for r in et_nom_sweep]
    delta_nom = float(et_nom_sweep[int(np.argmin(obj_means))]["param"])

    obj_dp_nr, obj_dp_rt = [], []
    obj_et_nr, obj_et_rt = [], []
    jc_dp_nr, jc_dp_rt = [], []
    jc_et_nr, jc_et_rt = [], []

    for p_succ in p_vals:
        cfg_eval = SimConfig(**cfg.__dict__)
        cfg_eval.p_success = float(p_succ)

        # no-retune (use nominal policy)
        dp_nr = _mc_eval_policy(cfg_eval, seeds, "DP", policy_fn=policy_nom)
        obj_dp_nr.append(float((dp_nr["J_P"] + float(lamb) * dp_nr["J_C"]).mean()))
        jc_dp_nr.append(float(dp_nr["J_C"].mean()))

        # retune
        policy_rt = _dp_trace_policy_fn(cfg_eval, float(lamb))
        dp_rt = _mc_eval_policy(cfg_eval, seeds, "DP", policy_fn=policy_rt)
        obj_dp_rt.append(float((dp_rt["J_P"] + float(lamb) * dp_rt["J_C"]).mean()))
        jc_dp_rt.append(float(dp_rt["J_C"].mean()))

        # ET: reuse a sweep to get both no-retune and retune variants.
        et_sweep = monte_carlo(cfg_eval, "ET", deltas=deltas, seeds=seeds)
        # no-retune = deploy delta_nom under p_succ
        idx_nom = int(np.argmin([abs(float(r["param"]) - delta_nom) for r in et_sweep]))
        et_nr = et_sweep[idx_nom]
        obj_et_nr.append(float((et_nr["J_P"] + float(lamb) * et_nr["J_C"]).mean()))
        jc_et_nr.append(float(et_nr["J_C"].mean()))

        # retune = choose delta minimizing the Lagrangian objective at p_succ
        objs = [float((r["J_P"] + float(lamb) * r["J_C"]).mean()) for r in et_sweep]
        et_rt = et_sweep[int(np.argmin(objs))]
        obj_et_rt.append(float((et_rt["J_P"] + float(lamb) * et_rt["J_C"]).mean()))
        jc_et_rt.append(float(et_rt["J_C"].mean()))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.4, 4.6), sharex=True)

    ax1.plot(p_vals, obj_dp_nr, "D-", label="DP-trace no-retune")
    ax1.plot(p_vals, obj_dp_rt, "D--", label="DP-trace retune")
    ax1.plot(p_vals, obj_et_nr, "o-", label="ET no-retune")
    ax1.plot(p_vals, obj_et_rt, "o--", label="ET retune")
    ax1.set_xlabel("")
    ax1.set_ylabel(r"$J_{\lambda}=J_P+\lambda J_C$")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", frameon=True, borderpad=0.3)

    ax2.plot(p_vals, jc_dp_nr, "D-", label="DP-trace no-retune")
    ax2.plot(p_vals, jc_dp_rt, "D--", label="DP-trace retune")
    ax2.plot(p_vals, jc_et_nr, "o-", label="ET no-retune")
    ax2.plot(p_vals, jc_et_rt, "o--", label="ET retune")
    ax2.set_xlabel(r"$p$")
    ax2.set_ylabel(r"$J_C$")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", frameon=True, borderpad=0.3)

    fig.tight_layout(pad=0.6, h_pad=0.5)
    savefig(fig, outdir, "fig_C_sensitivity_p")
    plt.close(fig)


def figure_D_robustness(cfg: SimConfig, outdir: Path, lamb: float) -> None:
    """Figure D: robustness to mismatch and bursty losses."""
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

    # bursty losses (Gilbert-Elliott) while keeping the *mean* success rate fixed.
    # We use a 2-state channel with success in the good state and loss in the bad
    # state. Scaling both transition probabilities changes burstiness (temporal
    # correlation) without changing the stationary success probability.
    alpha0 = 0.02  # good->bad
    pi_good = float(cfg.p_success)
    beta0 = (alpha0 * pi_good) / max(1e-6, (1.0 - pi_good))  # bad->good
    scales = np.array([0.25, 0.5, 1.0, 2.0, 5.0], dtype=float)
    bad_run = 1.0 / np.clip(beta0 * scales, 1e-6, None)
    dp_ratio_b, et_ratio_b = [], []
    for s in scales:
        cfg_ge = SimConfig(**cfg.__dict__)
        cfg_ge.channel_model = "ge"
        cfg_ge.loss_good = 0.0
        cfg_ge.loss_bad = 1.0
        cfg_ge.p_good_to_bad = float(alpha0 * s)
        cfg_ge.p_bad_to_good = float(beta0 * s)

        dp_eval = _mc_eval_policy(cfg_ge, seeds, "DP", policy_fn=policy_nom)
        dp_ratio_b.append(_mean_ci(dp_eval["J_P"])[0] / jp_nom)

        et_eval_cfg = SimConfig(**cfg_ge.__dict__)
        et_eval_cfg.delta = delta_nom
        et_eval = _mc_eval_policy(et_eval_cfg, seeds, "ET")
        et_ratio_b.append(_mean_ci(et_eval["J_P"])[0] / jp_nom)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.4, 4.8))

    ax1.plot(p_vals, dp_ratio_p, "D-", label="DP-trace $J_P$")
    ax1.plot(p_vals, et_ratio_p, "o-", label="ET $J_P$")
    ax1.plot(p_vals, dp_ratio_x, "D--", label="DP-trace $J_X$")
    ax1.plot(p_vals, et_ratio_x, "o--", label="ET $J_X$")
    ax1.set_xlabel(r"$p$")
    ax1.set_ylabel("Normalized degradation")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", frameon=True, borderpad=0.3)

    ax2.plot(bad_run, dp_ratio_b, "D-", label="DP-trace (bursty)")
    ax2.plot(bad_run, et_ratio_b, "o-", label="ET (bursty)")
    ax2.set_xlabel(r"Mean loss-burst length $1/\beta$ (steps)")
    ax2.set_ylabel(r"$J_P/J_P^{nom}$")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", frameon=True, borderpad=0.3)

    fig.tight_layout(pad=0.6, h_pad=0.6)
    savefig(fig, outdir, "fig_D_robustness")
    plt.close(fig)


def _simulate_rollout_counterexample(
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


def _eval_policy_counterexample(
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
        out = _simulate_rollout_counterexample(A, B, C, Qw, Rv, K, P0, horizon, p_success, rng, policy_fn)
        Jp.append(out["J_P"])
        Jc.append(out["J_C"])
        Jx.append(out["J_X"])
    return dict(J_P=np.asarray(Jp), J_C=np.asarray(Jc), J_X=np.asarray(Jx))


def _summarize_eval(eval_out: dict) -> dict:
    jp, jp_ci = _mean_ci(eval_out["J_P"])
    jc, jc_ci = _mean_ci(eval_out["J_C"])
    jx, jx_ci = _mean_ci(eval_out["J_X"])
    return dict(J_P=jp, J_P_ci=jp_ci, J_C=jc, J_C_ci=jc_ci, J_X=jx, J_X_ci=jx_ci)


def figure_2_trace_not_enough(outdir: Path, fast: bool = False) -> None:
    """Optional counterexample: trace-only DP can be suboptimal."""
    apply_ieee_style()
    cfg = SimConfig()
    cfg.T_steps = int(60 if fast else 120)
    cfg.mc_runs = int(10 if fast else 40)
    cfg.p_success = 0.7
    cfg.sigma_w = 0.04
    cfg.sigma_v = 0.03

    # anisotropic 2D system to expose trace insufficiency
    A = np.array([[1.08, 0.10], [0.00, 0.90]], dtype=float)
    B = np.array([[0.10], [0.05]], dtype=float)
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
        trace_eval = _eval_policy_counterexample(A, B, C, Qw, Rv, K, P0, cfg.T_steps, cfg.p_success, seeds, policy_trace)

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
        feat_eval = _eval_policy_counterexample(A, B, C, Qw, Rv, K, P0, cfg.T_steps, cfg.p_success, seeds, policy_feat)

        trace_points.append(dict(lamb=float(lamb), raw=trace_eval, **_summarize_eval(trace_eval)))
        feat_points.append(dict(lamb=float(lamb), raw=feat_eval, **_summarize_eval(feat_eval)))

    def _sorted_curve(points: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        jc = np.array([pnt["J_C"] for pnt in points], dtype=float)
        jp = np.array([pnt["J_P"] for pnt in points], dtype=float)
        order = np.argsort(jc)
        return jc[order], jp[order]

    jc_trace, jp_trace = _sorted_curve(trace_points)
    jc_feat, jp_feat = _sorted_curve(feat_points)

    jc_min = max(jc_trace.min(), jc_feat.min())
    jc_max = min(jc_trace.max(), jc_feat.max())
    jc_grid = np.linspace(jc_min, jc_max, 30)
    jp_trace_i = np.interp(jc_grid, jc_trace, jp_trace)
    jp_feat_i = np.interp(jc_grid, jc_feat, jp_feat)
    gap_jp = jp_trace_i - jp_feat_i

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(3.4, 4.8))

    ax0.plot(jc_trace, jp_trace, "o-", label="DP-trace")
    ax0.plot(jc_feat, jp_feat, "o-", label="DP-2feature")
    ax0.set_xlabel(r"$J_C$")
    ax0.set_ylabel(r"$J_P$")
    ax0.grid(True, alpha=0.3)
    ax0.legend(frameon=False)

    ax1.plot(jc_grid, gap_jp, "o-", color="tab:red")
    ax1.axhline(0.0, color="k", linewidth=0.6, alpha=0.6)
    ax1.set_xlabel(r"$J_C$")
    ax1.set_ylabel(r"$J_P^{trace} - J_P^{2feat}$")
    ax1.grid(True, alpha=0.3)

    fig.tight_layout(pad=0.6, h_pad=0.6)
    savefig(fig, outdir, "fig2_gap_curve")
    plt.close(fig)


def run_all(
    outdir: str = "figs",
    mc_runs: int | None = None,
    t_steps: int | None = None,
    fast: bool = False,
    with_counterexample: bool = False,
) -> None:
    outdir_path = Path(outdir)
    cfg = SimConfig()
    # Paper defaults: short horizon to reveal finite-horizon nonstationarity.
    cfg.T_steps = 120
    cfg.mc_runs = 40

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

    lamb_mid = figure_A_tradeoff(cfg, outdir_path)
    figure_B_time_response(cfg, outdir_path, lamb_mid)
    figure_C_channel_impairments(cfg, outdir_path, lamb=lamb_mid)
    if with_counterexample:
        figure_2_trace_not_enough(outdir_path, fast=fast)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate FRIS paper figures.")
    parser.add_argument("--outdir", default="figs", help="Output directory for figures.")
    parser.add_argument("--mc-runs", type=int, default=None, help="Override Monte Carlo runs per sweep.")
    parser.add_argument("--t-steps", type=int, default=None, help="Override number of simulation steps.")
    parser.add_argument("--fast", action="store_true", help="Quick run with reduced MC runs and steps.")
    parser.add_argument(
        "--with-counterexample",
        action="store_true",
        help="Also generate trace-not-enough counterexample figures.",
    )
    args = parser.parse_args(argv)
    run_all(
        outdir=args.outdir,
        mc_runs=args.mc_runs,
        t_steps=args.t_steps,
        fast=args.fast,
        with_counterexample=args.with_counterexample,
    )


if __name__ == "__main__":
    main()
