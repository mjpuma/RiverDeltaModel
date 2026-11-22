"""Simulations for version 2 of Puma et al. (arXiv:1606.04558).

    python paper/run_paper.py                    # all cases, written to paper/runs/<case>.pkl
    python paper/run_paper.py semicircle         # one case

semicircle       : land within L = 5 km of the source, open shoreline on the arc,
                   14,200 yr (710 flood years). The main configuration of version 2.
semicircle_large : L = 10 km, four times the sediment feed, 28,400 yr
rectangle        : the 5 km x 7 km domain of version 1 (no-flow side walls, straight
                   shoreline), 14,200 yr
rectangle_large  : 10 km x 14 km, four times the sediment feed, 28,400 yr

The sediment feed is set so that supply per unit area of delta plain is the same in
every case (version 1's 2.5e-3 m3/s over 50 x 71 cells). The rectangle runs start from
tests/data/eta_init.txt.gz (seed 2007); the others from initial_bed with the seeds below.
rectangle_large uses roughness=0.5: with the full random roughness the first channel,
once cut to its normal depth, has reversed slopes on such a long straight domain and the
run cannot start. The large runs skip the stratigraphy record (it does not affect the
dynamics) to save memory. Each 5 km run takes 5-10 minutes on one core, each 10 km run
about an hour.
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

QS_V1 = 0.0025                      # m3/s on 50 x 71 cells
QS_PER_CELL = QS_V1 / (50 * 71)


def eta_init():
    path = os.path.join(HERE, "..", "tests", "data", "eta_init.txt.gz")
    with gzip.open(path, "rt") as f:
        return np.array([[float(x) for x in line.split()] for line in f])


def setup(case):
    half_disk = lambda R: np.pi * R ** 2 / 2
    if case == "semicircle":
        p = Params(domain="semicircle", ymax=50, Qs=QS_PER_CELL * half_disk(50))
        return p, dict(eta0=initial_bed(p, seed=2026), save_snapshots=True)
    if case == "semicircle_large":
        p = Params(domain="semicircle", ymax=100, Qs=QS_PER_CELL * half_disk(100), tmax=1420.0, teq=220.0)
        return p, dict(eta0=initial_bed(p, seed=2016), stratigraphy=False)
    if case == "rectangle":
        return Params(), dict(eta0=eta_init(), save_snapshots=True)
    if case == "rectangle_large":
        p = Params(xmax=141, ymax=100, imidx=71, Qs=4 * QS_V1, tmax=1420.0, teq=220.0)
        return p, dict(eta0=initial_bed(p, seed=2016, roughness=0.5), stratigraphy=False)
    raise SystemExit(f"unknown case {case}")


def main(cases):
    os.makedirs(OUT, exist_ok=True)
    for case in cases:
        p, kw = setup(case)
        t0 = time.time()
        r = run(p, **kw)
        print(f"{case}: {r.num_avulsions} avulsions, {time.time() - t0:.0f} s", flush=True)
        with open(os.path.join(OUT, case + ".pkl"), "wb") as f:
            pickle.dump((p, r), f)


if __name__ == "__main__":
    main(sys.argv[1:] or ["semicircle", "rectangle", "semicircle_large", "rectangle_large"])
