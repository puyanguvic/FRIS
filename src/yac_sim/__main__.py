
from __future__ import annotations
import argparse
from .experiments.paper1 import run_all

def main() -> None:
    parser = argparse.ArgumentParser(description="Run FRIS simulation experiments.")
    parser.add_argument("--outdir", default="figs", help="Output directory for figures.")
    parser.add_argument("--mc-runs", type=int, default=None, help="Override Monte Carlo runs per sweep.")
    parser.add_argument("--t-steps", type=int, default=None, help="Override number of simulation steps.")
    parser.add_argument("--fast", action="store_true", help="Quick run with reduced MC runs and steps.")
    args = parser.parse_args()
    run_all(args.outdir, mc_runs=args.mc_runs, t_steps=args.t_steps, fast=args.fast)

if __name__ == "__main__":
    main()
