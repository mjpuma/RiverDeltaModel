"""Robustness runs for the signatures paper: the 5 km semicircle with other initial beds
and with other values of the avulsion threshold and the floodplain deposition rate.

    DELTA_RUNS=/path python paper/signatures/run_robustness.py [case ...]

Each run takes about 6 minutes on one core. The base case is paper/run_paper.py's
"semicircle" (seed 2026, beta = 1, floodplain deposition 0.5 x subsidence). These runs skip
the stratigraphy record, which does not affect the bed, and keep the bed every 20 yr for
the compensation analysis.
"""
import os
import pickle
import sys
import time

import numpy as np

from deltamodel import Params, initial_bed, run

OUT = os.environ.get("DELTA_RUNS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs"))
QS = 0.0025 / (50 * 71) * np.pi * 50 ** 2 / 2       # same supply per unit area as version 1

CASES = {
    "seed2027": dict(seed=2027),
    "seed2028": dict(seed=2028),
    "beta0.75": dict(beta=0.75),
    "beta1.5": dict(beta=1.5),
    "floodplain0.25": dict(floodplain_fraction=0.25),
    "floodplain0.75": dict(floodplain_fraction=0.75),
}


def setup(case):
    kw = dict(CASES[case])
    seed = kw.pop("seed", 2026)
    p = Params(domain="semicircle", ymax=50, Qs=QS, **kw)
    return p, dict(eta0=initial_bed(p, seed=seed), save_snapshots=True, stratigraphy=False)


def main(cases):
    os.makedirs(OUT, exist_ok=True)
    for case in cases:
        p, kw = setup(case)
        t0 = time.time()
        r = run(p, **kw)
        print(f"{case}: {r.num_avulsions} avulsions, {time.time() - t0:.0f} s", flush=True)
        with open(os.path.join(OUT, f"semicircle_{case}.pkl"), "wb") as f:
            pickle.dump((p, r), f)


if __name__ == "__main__":
    main(sys.argv[1:] or list(CASES))
