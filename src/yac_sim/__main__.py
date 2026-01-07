
from __future__ import annotations
import argparse

from .experiments.paper1 import run_all
from .experiments.exp2_trace_not_enough import run as run_exp2

def main() -> None:
    parser = argparse.ArgumentParser(description="Run FRIS simulation experiments.")
    parser.add_argument("--outdir", default="figs", help="Output directory for figures.")
    parser.add_argument("--mc-runs", type=int, default=None, help="Override Monte Carlo runs per sweep.")
    parser.add_argument("--t-steps", type=int, default=None, help="Override number of simulation steps.")
    parser.add_argument("--fast", action="store_true", help="Quick run with reduced MC runs and steps.")
    parser.add_argument(
        "--exp",
        default="paper1",
        choices=["paper1", "exp2"],
        help="Select which experiment suite to run.",
    )
    args = parser.parse_args()
    if args.exp == "exp2":
        run_exp2(args.outdir, mc_runs=args.mc_runs, t_steps=args.t_steps, fast=args.fast)
    else:
        run_all(args.outdir, mc_runs=args.mc_runs, t_steps=args.t_steps, fast=args.fast)

if __name__ == "__main__":
    main()
