"""
Run the surrogate for one set of control parameters.

    python simulate.py                       # uses params.json
    python simulate.py --out vectors.csv

Prints the transmission count, the objective, and summary statistics of the
transmitted-atom distribution, and optionally writes the sampled vectors.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from aion_model import Surrogate, THETA_COLUMNS, THETA_RANGES
from objective import OBJECTIVES

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=HERE / "params.json")
    ap.add_argument("--out", type=Path, default=None, help="write the sampled vectors here")
    ap.add_argument("--objective", choices=list(OBJECTIVES), default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    theta = cfg["params"]
    n_atoms = float(cfg.get("total_atoms", 5e6))
    obj_name = args.objective or cfg.get("objective", "capturable_flux")
    obj_kwargs = cfg.get("objective_parameters", {}).get(obj_name, {})

    out_of_range = Surrogate.check_ranges(theta)
    if out_of_range:
        print("WARNING: outside the trained range, results are extrapolation:")
        for k in out_of_range:
            lo, hi = THETA_RANGES[k]
            print(f"  {k} = {theta[k]}  (trained on [{lo}, {hi}])")

    t0 = time.time()
    model = Surrogate(device=args.device)
    print(f"loaded on {model.device} in {time.time()-t0:.1f} s")

    t0 = time.time()
    count, vectors = model.evaluate(theta, n_atoms=n_atoms,
                                    n_estimate=cfg.get("n_estimate", 20000))
    dt = time.time() - t0

    print(f"\nparameter set:")
    for c in THETA_COLUMNS:
        print(f"  {c:24} {theta[c]:>12.4f}")
    print(f"\ntransmitted atoms (of {n_atoms:.2e} launched) : {count:,.1f}")
    print(f"vectors sampled                             : {len(vectors):,}")
    print(f"evaluation time                             : {dt*1e3:.1f} ms")

    result = OBJECTIVES[obj_name](count, vectors, **obj_kwargs)
    print(f"\nobjective '{obj_name}':")
    for k, v in result.items():
        print(f"  {k:20} {v:>14.4f}")

    if len(vectors):
        print("\ntransmitted-atom distribution:")
        print(vectors.describe().loc[["mean", "std", "min", "50%", "max"]].round(3).to_string())

    if args.out:
        vectors.to_csv(args.out, index=False)
        print(f"\nwrote {len(vectors):,} vectors to {args.out}")


if __name__ == "__main__":
    main()
