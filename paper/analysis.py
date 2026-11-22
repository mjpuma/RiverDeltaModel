"""Analysis of model runs for version 2 of the paper: channel visits and residence times,
stratigraphy, boundary effects, and the three comparisons with recent work
(compensation index, reworking time scales, Hack-like scaling).

Works for both domains: the rectangle of version 1 and the semicircle with an open
shoreline. Time is in years: model time (flood years) divided by the intermittency.
"""
import pickle
from collections import defaultdict

import numpy as np
from scipy import ndimage, stats

from deltamodel import geometry
from deltamodel.history import channel_paths, occupancy_intervals

SQ2 = np.sqrt(2.0)


class Case:
    """A finished run with its geometry and derived channel statistics."""

    def __init__(self, path, step=10.0):
        self.p, self.r = pickle.load(open(path, "rb"))
        p, r = self.p, self.r
        self.domain = getattr(p, "domain", "rectangle")
        self.If = p.I_tavg
        self.teq = p.teq
        self.a = p.a
        self.Rc = p.ymax                                  # L in cells
        self.L = p.a * p.ymax
        X, Y = r.eta.shape
        self.X, self.Y = X, Y
        if getattr(r, "land", None) is None:              # runs saved before the masks existed
            _, _, imidx, active, terminal = geometry(p)
            act = active[1:X + 1, 1:Y + 1] == 1
            ter = terminal[1:X + 1, 1:Y + 1] == 1
            r.land, r.shoreline, r.source = act & ~ter, ter, (imidx - 1, 0)
        self.land, self.shore = r.land, r.shoreline
        self.i0 = r.source[0]
        ii, jj = np.meshgrid(np.arange(X), np.arange(Y), indexing="ij")
        self.xL = jj / self.Rc
        self.yL = (ii - self.i0) / self.Rc
        self.R = np.hypot(self.xL, self.yL)
        self.theta = np.degrees(np.arctan2(ii - self.i0, jj))   # 0 = straight downstream
        self.paths = list(channel_paths(r))
        self.occ, self.open = occupancy_intervals(r)
        self.Tfv, self.fv_curve, self.Tfv_starts = self._t_fv(step)

    def years(self, t):
        return np.asarray(t) / self.If

    # ----------------------------------------------------------------- analysis regions
    def region(self, name="plain"):
        """Boolean mask of cells for statistics. plain: the delta plain (no shoreline
        cells). square: version 1's L x L subsection of the rectangle (the plain, for the
        semicircle). interior: the plain without the 3 cells nearest a side wall."""
        m = self.land.copy()
        if self.domain == "rectangle" and name in ("square", "interior"):
            k = self.Rc // 2 if name == "square" else 3
            cols = np.zeros(self.X, bool)
            if name == "square":
                cols[self.i0 - k:self.i0 + k] = True
            else:
                cols[k:self.X - k] = True
            m &= cols[:, None]
            if name == "square":
                m &= (np.arange(self.Y) < self.Rc)[None, :]
        return m

    # ----------------------------------------------------------------- visits
    def _t_fv(self, step):
        """T_fv=1: time for the channel to visit every cell of the delta plain from a
        start time after the spin-up; the median over start times every `step` (model
        time). Also the mean fraction visited against time interval (10-90% range)."""
        o = self.occ
        keep = self.land[o[:, 0].astype(int), o[:, 1].astype(int)]
        o = o[keep]
        n = self.land.sum()
        done, curves = [], []
        grid = np.linspace(0, self.r.t - self.teq, 400)
        for s in np.arange(self.teq, self.r.t, step):
            live = o[:, 3] > s
            first = np.full((self.X, self.Y), np.inf)
            np.minimum.at(first, (o[live, 0].astype(int), o[live, 1].astype(int)),
                          np.maximum(o[live, 2], s) - s)
            f = np.sort(first[self.land])
            if not np.isfinite(f[-1]):
                break
            done.append(f[-1])
            curves.append(np.searchsorted(f, grid, side="right") / n)
        if not done:
            raise RuntimeError("the channel never visits every cell; run longer")
        C = np.array(curves)
        done = self.years(np.array(done))
        return (float(np.median(done)),
                (self.years(grid), C.mean(0), np.quantile(C, 0.1, 0), np.quantile(C, 0.9, 0)),
                done)

    def visits(self):
        """Completed visits ending after the spin-up: (i, j, duration in years)."""
        o = self.occ[~self.open]
        o = o[o[:, 3] > self.teq]
        return o[:, 0].astype(int), o[:, 1].astype(int), self.years(o[:, 3] - o[:, 2])

    def mean_residence(self):
        i, j, d = self.visits()
        s = np.zeros((self.X, self.Y)); n = np.zeros((self.X, self.Y))
        np.add.at(s, (i, j), d); np.add.at(n, (i, j), 1)
        with np.errstate(invalid="ignore"):
            return np.where(n > 0, s / n, np.nan)

    def total_residence(self):
        o = self.occ
        t_on = np.maximum(o[:, 2], self.teq)
        d = np.clip(o[:, 3] - t_on, 0, None)
        s = np.zeros((self.X, self.Y))
        np.add.at(s, (o[:, 0].astype(int), o[:, 1].astype(int)), self.years(d))
        return s

    def window_residence(self, t0, dt):
        o = self.occ
        on = np.clip(o[:, 2], t0, t0 + dt)
        off = np.clip(o[:, 3], t0, t0 + dt)
        s = np.zeros((self.X, self.Y))
        np.add.at(s, (o[:, 0].astype(int), o[:, 1].astype(int)), self.years(off - on))
        return s

    def occupancy(self):
        """Time-weighted channel occupancy of each cell after the spin-up (fraction of time)."""
        return self.total_residence() / self.years(self.r.t - self.teq)

    # ----------------------------------------------------------------- stratigraphy
    def column(self, i, j):
        r = self.r
        k = r.top_layer[i, j]
        th = r.strat_thickness[i, j, :k + 1]
        top = np.cumsum(th)
        return top - th, top, r.strat_type[i, j, :k + 1], r.strat_time[i, j, :k + 1]

    def section_cells(self, xf):
        """Cells of a section at distance xf*L from the source: the row x = xf L in the
        rectangle, the arc r = xf L in the semicircle (ordered by angle)."""
        if self.domain == "rectangle":
            j = int(round(xf * self.Rc))
            cells = [(i, j) for i in range(self.X)]
            coord = self.yL[:, j]
        else:
            m = self.land & (np.abs(self.R * self.Rc - xf * self.Rc) < 0.5)
            ii, jj = np.nonzero(m)
            order = np.argsort(self.theta[ii, jj])
            cells = list(zip(ii[order], jj[order]))
            coord = np.radians(self.theta[ii[order], jj[order]]) * xf   # arc length / L
        return cells, np.asarray(coord)

    def channel_bodies(self, cells):
        """Thickness of each channel deposit laid after the spin-up in the given cells
        (consecutive channel layers merged)."""
        out = []
        for i, j in cells:
            bot, top, typ, tim = self.column(i, j)
            run = 0.0
            for b, t, ty, tm in zip(bot, top, typ, tim):
                if ty == 2 and tm > self.teq:
                    run += t - b
                else:
                    if run > 0:
                        out.append(run)
                    run = 0.0
            if run > 0:
                out.append(run)
        return np.array(out)

    def section_shifts(self, xf):
        """When the channel moves to new cells of a section, the fraction of moves to an
        adjacent cell (successive occupation), and the number of moves."""
        cells, _ = self.section_cells(xf)
        index = {c: k for k, c in enumerate(cells)}
        prev = None
        n = adj = 0
        for t, path in self.paths:
            now = {index[(i - 1, j - 1)] for i, j in path if (i - 1, j - 1) in index}
            if t > self.teq and prev and now and not (now & prev):
                d = min(abs(a - b) for a in now for b in prev)
                n += 1
                adj += d == 1
            prev = now
        return adj / max(n, 1), n


# ===================================================================== comparisons

def exceedance(v):
    v = np.sort(v[np.isfinite(v)])
    return v, 1.0 - np.arange(v.size) / v.size


def fit_exceedance(v, p_hi=0.5, p_lo=0.01):
    t, P = exceedance(v)
    m = (P <= p_hi) & (P >= p_lo)
    return -np.polyfit(np.log10(t[m]), np.log10(P[m]), 1)[0]


def chi_with_range(v):
    fits = [fit_exceedance(v, a, b) for a in (0.7, 0.5, 0.3) for b in (0.03, 0.01, 0.003)]
    return float(np.median(fits)), float(min(fits)), float(max(fits))


def compensation(c, mask=None, nT=24):
    """Straub et al. (2009): sigma_ss(T), the standard deviation over the delta plain of
    sedimentation over a window T divided by the long-term rate, for windows T. Uses the
    yearly bed snapshots after the spin-up; sedimentation = change in bed elevation plus
    subsidence. Returns T (yr), sigma_ss, and the mean channel depth (m)."""
    sn = c.r.snapshots
    if not sn:
        raise RuntimeError("run has no snapshots")
    eta, ts = sn["eta"], c.years(np.asarray(sn["t"]))
    mask = c.region("plain") if mask is None else mask
    keep = ts > c.years(c.teq)
    eta, ts = eta[keep][:, mask], ts[keep]
    sigma = c.p.subside                                   # m per year (real time)
    dt0 = np.median(np.diff(ts))
    lags = np.unique(np.round(np.logspace(0, np.log10(len(ts) // 3), nT)).astype(int))
    T, s = [], []
    for k in lags:
        idx = np.arange(0, len(ts) - k, k)                # non-overlapping windows
        dep = eta[idx + k] - eta[idx] + sigma * (ts[idx + k] - ts[idx])[:, None]
        ratio = dep / (sigma * (ts[idx + k] - ts[idx])[:, None])
        T.append(k * dt0)
        s.append(np.sqrt(np.mean((ratio - 1.0) ** 2)))
    H = c.r.Havg[mask & (c.r.Havg > 0)]
    return np.array(T), np.array(s), float(np.mean(H))


def kappa_fit(T, s, Tc):
    """Compensation index: slope of log sigma_ss against log T for T below Tc, and above."""
    lo, hi = T < Tc, T > Tc
    k_lo = -np.polyfit(np.log10(T[lo]), np.log10(s[lo]), 1)[0] if lo.sum() > 2 else np.nan
    k_hi = -np.polyfit(np.log10(T[hi]), np.log10(s[hi]), 1)[0] if hi.sum() > 2 else np.nan
    return float(k_lo), float(k_hi)


def reworking(c, step=10.0, mask=None):
    """Wickert et al. (2013) metrics. Unreworked fraction: share of the plain not yet
    visited by the channel a time dt after a start time. Channel overlap: share of the
    cells occupied at the start that are occupied again at start + dt, rescaled so that
    0 means no more overlap than with a random set of cells of the same size. Both are
    averaged over start times every `step`; e-folding times from exponential fits."""
    mask = c.region("plain") if mask is None else mask
    tg, fv, _, _ = c.fv_curve
    unrew = 1.0 - fv
    # channel overlap
    times = np.array([t for t, _ in c.paths])
    sets = [frozenset((i - 1, j - 1) for i, j in path if mask[i - 1, j - 1]) for _, path in c.paths]

    def at(t):
        return sets[max(np.searchsorted(times, t, side="right") - 1, 0)]

    lag_max = min(0.5 * (c.r.t - c.teq), 4 * c.Tfv * c.If)
    lags = np.concatenate([[0.0], np.logspace(np.log10(c.p.dtmax), np.log10(lag_max), 150)])
    starts = np.arange(c.teq, c.r.t - lags[-1], step)
    nplain = mask.sum()
    ov = np.zeros(lags.size)
    for s in starts:
        a = at(s)
        fw = len(a) / nplain
        for k, lag in enumerate(lags):
            b = at(s + lag)
            raw = len(a & b) / len(a)
            ov[k] += (raw - fw) / (1 - fw)
    ov /= len(starts)
    lag_y = c.years(lags)

    def efold(t, f, fmin=0.05, fmax=0.9):
        """f ~ A exp(-t / T) fitted where fmin < f < fmax; returns T, A and the r^2."""
        m = (f > fmin) & (f < fmax) & (t > 0)
        slope, icpt = np.polyfit(t[m], np.log(f[m]), 1)
        r2 = np.corrcoef(t[m], np.log(f[m]))[0, 1] ** 2
        return float(-1.0 / slope), float(np.exp(icpt)), float(r2)

    T_R, A_R, r2_R = efold(tg, unrew)
    T_M, A_M, r2_M = efold(lag_y, ov)
    return dict(t_unreworked=tg, unreworked=unrew, T_R=T_R, A_R=A_R, r2_R=r2_R,
                t_overlap=lag_y, overlap=ov, T_M=T_M, A_M=A_M, r2_M=r2_M)


def _islands_area(c, nodes, outlets):
    """Nourishment area: the cells of a sub-network plus the interdistributary areas
    enclosed between it and the shoreline segment spanning its outlets (cells)."""
    m = np.zeros((c.X, c.Y), bool)
    for i, j in nodes:
        m[i, j] = True
    shore = np.argwhere(c.shore)
    if c.domain == "rectangle":
        key = shore[:, 0]
        okey = [i for i, j in outlets]
    else:
        key = c.theta[shore[:, 0], shore[:, 1]]
        okey = [c.theta[i, j] for i, j in outlets]
    seg = shore[(key >= min(okey)) & (key <= max(okey))]
    m[seg[:, 0], seg[:, 1]] = True
    filled = ndimage.binary_fill_holes(m)
    return int((filled & (c.land | c.shore)).sum())


def hack(c, frac, starts):
    """Hack-like scaling of the network of channel paths active during a window of
    length frac * T_fv=1, following Dong et al. (2026): for every bifurcation of the
    time-integrated network, the longest flow path from it to the shoreline (L, along
    the realized channel paths) and its nourishment area (A: the downstream network and
    the interdistributary areas it encloses with the shoreline). Returns arrays L, A (m,
    m^2) pooled over the start times."""
    times = np.array([t for t, _ in c.paths])
    Ls, As = [], []
    for s in starts:
        e = s + frac * c.Tfv * c.If
        k0 = max(np.searchsorted(times, s, side="right") - 1, 0)
        k1 = np.searchsorted(times, e, side="right")
        paths = [[(i - 1, j - 1) for i, j in c.paths[k][1]] for k in range(k0, k1)]
        succ = defaultdict(set)
        for pth in paths:
            for u, v in zip(pth[:-1], pth[1:]):
                succ[u].add(v)
        # downstream length along each realized path from every cell on it
        down = defaultdict(float)
        for pth in paths:
            acc = 0.0
            for u, v in zip(reversed(pth[:-1]), reversed(pth[1:])):
                acc += c.a * (SQ2 if (u[0] != v[0] and u[1] != v[1]) else 1.0)
                down[u] = max(down[u], acc)
        for apex, nxt in succ.items():
            if len(nxt) < 2:
                continue
            seen, stack = {apex}, [apex]
            while stack:
                u = stack.pop()
                for v in succ.get(u, ()):
                    if v not in seen:
                        seen.add(v)
                        stack.append(v)
            outlets = [v for v in seen if c.shore[v]]
            if len(outlets) < 2:
                continue
            Ls.append(down[apex])
            As.append(_islands_area(c, seen, outlets) * c.a ** 2)
    return np.array(Ls), np.array(As)


def hack_exponent(L, A):
    """h in L ~ A^h by least squares on log-log, with a bootstrap 95% interval."""
    x, y = np.log10(A), np.log10(L)
    h = np.polyfit(x, y, 1)[0]
    rng = np.random.default_rng(0)
    bs = [np.polyfit(x[k], y[k], 1)[0] for k in (rng.integers(0, x.size, x.size) for _ in range(500))]
    return float(h), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def boundary_stats(c):
    """How much the channel sits on the domain edges."""
    occ = c.occupancy()
    plain = c.region("plain")
    tot = occ[plain].sum()
    out = {}
    if c.domain == "rectangle":
        wall = np.zeros_like(plain)
        wall[[0, -1], :] = True
        near = np.zeros_like(plain)
        near[:3, :] = True
        near[-3:, :] = True
        out["wall_area_share"] = float((wall & plain).sum() / plain.sum())
        out["wall_occupancy_share"] = float(occ[wall & plain].sum() / tot)
        out["near3_area_share"] = float((near & plain).sum() / plain.sum())
        out["near3_occupancy_share"] = float(occ[near & plain].sum() / tot)
        touch = 0.0; total = 0.0
        times = [t for t, _ in c.paths] + [c.r.t]
        for (t, path), t1 in zip(c.paths, times[1:]):
            if t1 <= c.teq:
                continue
            w = t1 - max(t, c.teq)
            total += w
            touch += w * any(i - 1 in (0, c.X - 1) for i, j in path)
        out["time_touching_wall"] = touch / total
    # occupancy against angle from the source, by 10-degree bins (plain cells)
    th = c.theta[plain]
    bins = np.arange(-90, 91, 10)
    occ_b = np.histogram(th, bins, weights=occ[plain])[0]
    area_b = np.histogram(th, bins)[0]
    out["angle_bins"] = bins.tolist()
    out["occupancy_by_angle_rel"] = (occ_b / occ_b.sum() / (area_b / area_b.sum())).tolist()
    return out


def residence_vs_thickness(c, mask):
    T = c.total_residence()[mask] / c.Tfv
    h = c.r.depth_channel[mask]
    ok = T > 0
    T, h = T[ok], h[ok]
    q = np.quantile(T, [0, 0.25, 0.5, 0.75, 0.9, 1.0])
    k = np.digitize(T, q[1:-1])
    return dict(spearman=float(stats.spearmanr(T, h).correlation),
                mean_h_by_T_bin=[float(h[k == b].mean()) for b in range(5)],
                T_bins=q.tolist(), n=int(T.size))
