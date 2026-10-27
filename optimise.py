"""
Bayesian optimisation of the nine 2D MOT control parameters against the capturable-flux
objective, using the tree-structured Parzen estimator from HyperOpt.

    python optimise.py --iterations 1000

The search space is the Sobol range each parameter was trained over, so the optimiser
cannot leave the region where the surrogate is valid. Every evaluation and the running
best are written to the output directory as the run proceeds, so a long run can be
inspected or resumed from its trials file.
"""

from __future__ import annotations

import argparse
import json
import pickle
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
    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--out-dir", type=Path, default=HERE / "optimisation_output")
    ap.add_argument("--objective", choices=list(OBJECTIVES), default=None)
    ap.add_argument("--n-estimate", type=int, default=20_000,
                    help="vectors sampled per evaluation to estimate the accepted fraction")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--resume", action="store_true",
                    help="continue from trials.pkl in the output directory")
    args = ap.parse_args()

    from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

    cfg = json.loads(args.config.read_text())
    n_atoms = float(cfg.get("total_atoms", 5e6))
    obj_name = args.objective or cfg.get("objective", "capturable_flux")
    obj_kwargs = cfg.get("objective_parameters", {}).get(obj_name, {})
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = Surrogate(device=args.device)
    objective_fn = OBJECTIVES[obj_name]
    space = {c: hp.uniform(c, *THETA_RANGES[c]) for c in THETA_COLUMNS}

    log_path = args.out_dir / "evaluations.csv"
    trials_path = args.out_dir / "trials.pkl"
    rows: list[dict] = []
    state = {"n": 0, "best": -np.inf, "t0": time.time()}

    def loss(theta):
        count, vectors = model.evaluate(theta, n_atoms=n_atoms, n_estimate=args.n_estimate)
        probability = count / n_atoms
        # capturable_flux is a delivered *fraction*, so it takes P(theta); the legacy
        # weighted_penalty is defined on the count and keeps it.
        yield_arg = probability if obj_name == "capturable_flux" else count
        result = objective_fn(yield_arg, vectors, **obj_kwargs)
        value = float(result["objective"])
        state["n"] += 1
        if value > state["best"]:
            state["best"] = value
            (args.out_dir / "best.json").write_text(json.dumps(
                {"objective": value, "objective_name": obj_name, "iteration": state["n"],
                 "transmission_count": count, "transmission_probability": probability,
                 "params": theta, "metrics": result}, indent=2))
        rows.append({**theta, "transmission_count": count,
                     "transmission_probability": probability, **result})
        if state["n"] % 25 == 0 or state["n"] == 1:
            rate = state["n"] / max(time.time() - state["t0"], 1e-9)
            pd.DataFrame(rows).to_csv(log_path, index=False)
            print(f"  [{state['n']:>5}/{args.iterations}] "
                  f"current {value:12.2f}   best {state['best']:12.2f}   "
                  f"{rate:5.2f} eval/s", flush=True)
        return {"loss": -value, "status": STATUS_OK}

    trials = Trials()
    if args.resume and trials_path.exists():
        trials = pickle.loads(trials_path.read_bytes())
        print(f"resumed from {len(trials.trials)} previous evaluations")

    print(f"objective '{obj_name}' with {obj_kwargs or 'default parameters'}")
    print(f"{args.iterations} iterations on {model.device}\n")
    best = fmin(fn=loss, space=space, algo=tpe.suggest, max_evals=args.iterations,
                trials=trials, rstate=np.random.default_rng(args.seed),
                show_progressbar=False)

    frame = pd.DataFrame(rows).sort_values("objective", ascending=False)
    frame.to_csv(log_path, index=False)
    frame.head(10).to_csv(args.out_dir / "top10.csv", index=False)
    trials_path.write_bytes(pickle.dumps(trials))

    print(f"\ncompleted {len(frame):,} evaluations in "
          f"{(time.time()-state['t0'])/60:.1f} min")
    print(f"\ntop 10 by {obj_name}:")
    cols = ["objective", "transmission_count", "transmission_probability"] + \
           [c for c in ("accepted_fraction", "mean_vz", "mean_speed", "radial_std_mm")
            if c in frame.columns]
    print(frame.head(10)[cols].to_string(index=False, float_format=lambda v: f"{v:10.4f}"))
    print(f"\nbest parameter set:")
    for c in THETA_COLUMNS:
        print(f"  {c:24} {frame.iloc[0][c]:>12.4f}")
    print(f"\nwrote {log_path}, top10.csv, best.json and trials.pkl to {args.out_dir}")


if __name__ == "__main__":
    main()
