"""Simulations for version 2 of Puma et al. (arXiv:1606.04558).

    python paper/run_paper.py            # writes paper/runs/{base,large}.pkl

base   : the 5 km x 7 km delta of Table 2, 14,200 yr (710 flood years)
large  : 10 km x 14 km, four times the sediment feed, 28,400 yr
         (stratigraphy not recorded, to save memory; it does not affect the dynamics)

The initial bed of base is tests/data/eta_init.txt.gz (seed 2007). The large
domain uses initial_bed(seed=2016, roughness=0.5): with the full random roughness the
first channel, once cut to its normal depth, has reversed slopes on such a long domain
and the run cannot start.
Each run takes 5-10 minutes (large: about an hour) on one core.
"""
import gzip
import os
import pickle
import sys
import time

import numpy as np

from deltamodel import Params, initial_bed, run

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("DELTA_RUNS", os.path.join(HERE, "runs"))


def eta_init():
    path = os.path.join(HERE, "..", "tests", "data", "eta_init.txt.gz")
    with gzip.open(path, "rt") as f:
        return np.array([[float(x) for x in line.split()] for line in f])


def main(cases):
    os.makedirs(OUT, exist_ok=True)
    for case in cases:
        if case == "base":
            p = Params()
            kw = dict(eta0=eta_init())
        elif case == "large":
            p = Params(xmax=141, ymax=100, imidx=71, Qs=0.01, tmax=1420.0, teq=220.0)
            kw = dict(eta0=initial_bed(p, seed=2016, roughness=0.5), stratigraphy=False)
        else:
            raise SystemExit(f"unknown case {case}")
        t0 = time.time()
        r = run(p, **kw)
        print(f"{case}: {r.num_avulsions} avulsions, {time.time() - t0:.0f} s", flush=True)
        with open(os.path.join(OUT, case + ".pkl"), "wb") as f:
            pickle.dump((p, r), f)


if __name__ == "__main__":
    main(sys.argv[1:] or ["base", "large"])
