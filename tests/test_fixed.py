"""Physical consistency checks of the model."""
import numpy as np
import pytest

from deltamodel import Params, run, initial_bed


@pytest.fixture(scope="module")
def fixed_run():
    p = Params(tmax=20.0, teq=2.0)
    return p, run(p, seed=3)


def test_stratigraphy_matches_bed(fixed_run):
    """Every column's stratigraphy (base elevation plus deposits minus erosion) must equal
    the final bed elevation plus the subsidence applied, the same in every cell
    (corrections 4, 5, 8 and 9)."""
    p, r = fixed_run
    col = np.array([[r.strat_thickness[i, j, :r.top_layer[i, j] + 1].sum()
                     for j in range(r.eta.shape[1])] for i in range(r.eta.shape[0])])
    resid = col - r.eta
    assert resid.max() - resid.min() < 1e-6
    # total subsidence = subside * t / I_tavg (time steps add up to just over tmax)
    assert resid.mean() == pytest.approx(p.subside * p.tmax / p.I_tavg, rel=1e-3)


def test_avulsions_and_channel(fixed_run):
    p, r = fixed_run
    assert r.num_avulsions > 10
    assert np.all(np.diff(r.avulsions[:, 0]) >= 0)            # in time order
    assert np.all((r.avulsions[:, 1] >= p.a) & (r.avulsions[:, 1] <= p.a * p.xmax))
    # one connected channel from the source to the shoreline row
    ch = r.channel
    assert ch[p.imidx - 1, 0] == 1
    assert ch[:, -1].sum() >= 1


def test_stratigraphy_types(fixed_run):
    _, r = fixed_run
    tops = r.top_layer
    for i in range(0, r.eta.shape[0], 10):
        for j in range(0, r.eta.shape[1], 10):
            types = r.strat_type[i, j, 1:tops[i, j] + 1]
            assert np.all(np.isin(types, (1, 2)))               # overbank or channel above the base
            assert np.all(r.strat_thickness[i, j, 1:tops[i, j] + 1] >= -1e-12)


def test_channel_history(fixed_run):
    """The logged paths rebuild the final channel, and the visits they imply match the
    model's own visit statistics."""
    from deltamodel.history import channel_paths, occupancy_intervals
    p, r = fixed_run
    *_, (t, path) = channel_paths(r)
    ch = np.zeros_like(r.channel)
    for i, j in path:
        ch[i - 1, j - 1] = 1
    assert np.array_equal(ch, r.channel)
    occ, still_open = occupancy_intervals(r)
    done = occ[~still_open]
    assert (done[:, 3] > p.teq).sum() == r.visit_num.sum()


def test_stratigraphy_switch_changes_nothing_else():
    p = Params(tmax=10.0, teq=2.0)
    eta0 = initial_bed(p, seed=7)
    a = run(p, eta0=eta0)
    b = run(p, eta0=eta0, stratigraphy=False)
    assert np.array_equal(a.avulsions, b.avulsions)
    assert np.array_equal(a.eta, b.eta)
