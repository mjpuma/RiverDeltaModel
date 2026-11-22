"""The Python model must reproduce the Fortran in fortran/ exactly.

The reference (tests/data/fortran_30yr_dump.txt.gz) is a 30-year run of fortran/ made by
tests/make_fortran_reference.sh. A 710-year run was also compared: same avulsions and
identical final state (see README)."""
import gzip
import os

import numpy as np
import pytest

from deltamodel import Params, run

HERE = os.path.dirname(__file__)


def eta_init():
    with gzip.open(os.path.join(HERE, "data", "eta_init.txt.gz"), "rt") as f:
        return np.array([[float(x) for x in line.split()] for line in f])


@pytest.fixture(scope="module")
def pair():
    res = run(Params(tmax=30.0, teq=5.0), eta0=eta_init())
    with gzip.open(os.path.join(HERE, "data", "fortran_30yr_dump.txt.gz"), "rt") as f:
        lines = f.read().splitlines()
    nav = int(lines[0].split()[1])
    X = res.eta.shape[0]
    rows = lines[3:]

    def block(a, b, typ=float):
        return np.array([[typ(x) for x in r.split()] for r in rows[a:b]])

    ref = dict(
        num_avul=nav, t=float(lines[2].split()[1]),
        eta=block(0, X), levee=block(X, 2 * X), top_lay=block(2 * X, 3 * X, int),
        avul=block(3 * X, 3 * X + nav), depth_chan=block(3 * X + nav, 4 * X + nav),
        visit=block(4 * X + nav, 5 * X + nav),
    )
    return res, ref


def test_avulsions_identical(pair):
    res, ref = pair
    assert res.num_avulsions == ref["num_avul"] > 100
    assert np.array_equal(res.avulsions, ref["avul"])
    assert res.t == ref["t"]


def test_final_state_identical(pair):
    res, ref = pair
    assert np.array_equal(res.eta, ref["eta"])
    assert np.array_equal(res.levee, ref["levee"])
    assert np.array_equal(res.top_layer + 1, ref["top_lay"])


def test_stratigraphy_and_visits_identical(pair):
    res, ref = pair
    assert np.array_equal(res.depth_channel, ref["depth_chan"])
    assert np.array_equal(res.visit_mean, ref["visit"])
