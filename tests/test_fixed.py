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


@pytest.fixture(scope="module")
def semi_run():
    p = Params(domain="semicircle", ymax=20, Qs=0.0025 / (50 * 71) * np.pi * 20 ** 2 / 2,
               tmax=20.0, teq=2.0)
    return p, run(p, eta0=initial_bed(p, seed=5), seed=5)


def test_semicircle_geometry():
    from deltamodel import geometry
    p = Params(domain="semicircle", ymax=20)
    xmax, ymax, imidx, active, terminal = geometry(p)     # padded, 1-based
    assert active[imidx, 1] and not terminal[imidx, 1]       # source is on land
    land = (active == 1) & (terminal == 0)
    # every land cell lies within L of the source, every shoreline cell at L to L + 1.5
    i, j = np.nonzero(land)
    assert np.hypot(i - imidx, j - 1).max() < 20
    i, j = np.nonzero(terminal == 1)
    r = np.hypot(i - imidx, j - 1)
    assert r.min() >= 20 and r.max() < 21.5
    # the plain is symmetric about the source row
    a = active[1:xmax + 1]
    assert np.array_equal(a, a[::-1, :])


def test_semicircle_run(semi_run):
    """Stratigraphy balances, paths stay in the domain and end on the shoreline."""
    from deltamodel.history import channel_paths
    p, r = semi_run
    assert r.num_avulsions >= 3
    col = np.array([[r.strat_thickness[i, j, :r.top_layer[i, j] + 1].sum()
                     for j in range(r.eta.shape[1])] for i in range(r.eta.shape[0])])
    resid = (col - r.eta)[r.land]
    assert resid.max() - resid.min() < 1e-6
    for t, path in channel_paths(r):
        ii = np.array([q[0] for q in path]) - 1
        jj = np.array([q[1] for q in path]) - 1
        assert r.land[ii[:-1], jj[:-1]].all()          # every cell but the last is land
        assert r.shoreline[ii[-1], jj[-1]]             # the path ends on the shoreline
