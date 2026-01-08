from __future__ import annotations
import argparse

from .paper_figures import run_all

def main() -> None:
    parser = argparse.ArgumentParser(description="Run FRIS paper figure generation.")
    parser.add_argument("--outdir", default="figs", help="Output directory for figures.")
    parser.add_argument("--mc-runs", type=int, default=None, help="Override Monte Carlo runs per sweep.")
    parser.add_argument("--t-steps", type=int, default=None, help="Override number of simulation steps.")
    parser.add_argument("--fast", action="store_true", help="Quick run with reduced MC runs and steps.")
    parser.add_argument(
        "--with-counterexample",
        action="store_true",
        help="Also generate trace-not-enough counterexample figures.",
    )
    args = parser.parse_args()
    run_all(
        outdir=args.outdir,
        mc_runs=args.mc_runs,
        t_steps=args.t_steps,
        fast=args.fast,
        with_counterexample=args.with_counterexample,
    )

if __name__ == "__main__":
    main()
