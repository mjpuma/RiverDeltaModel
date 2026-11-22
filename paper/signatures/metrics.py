"""Summary statistics of one run for the signatures paper: T_fv=1, the exponent of the
mean residence-time distribution, Hack-like scaling, reworking and channel overlap, and
the compensation index. Uses paper/analysis.py."""
import os
import sys

import numpy as np
from scipy import optimize

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from analysis import (Case, chi_with_range, compensation, hack, hack_exponent,  # noqa: E402
                      kappa_fit, reworking)

FRACS = (0.035, 0.174, 0.35)


class NullCase(Case):
    """A null run, normalized by the T_fv=1 of the model run it was built from (the
    random walk need not visit every cell within the run, so its own T_fv=1 may not
    exist). Also reports the null's own median time to visit 99% of the plain."""

    def __init__(self, path, Tfv_years, step=10.0):
        self._fixed = Tfv_years
        super().__init__(path, step)

    def _t_fv(self, step):
        o = self.occ
        o = o[self.land[o[:, 0].astype(int), o[:, 1].astype(int)]]
        n = self.land.sum()
        grid = np.linspace(0, self.r.t - self.teq, 400)
        curves, t99 = [], []
        for s in np.arange(self.teq, self.r.t, step):
            live = o[:, 3] > s
            first = np.full((self.X, self.Y), np.inf)
            np.minimum.at(first, (o[live, 0].astype(int), o[live, 1].astype(int)),
                          np.maximum(o[live, 2], s) - s)
            f = np.sort(first[self.land])
            if not np.isfinite(f[int(0.99 * n) - 1]):
                break
            t99.append(f[int(0.99 * n) - 1])
            curves.append(np.searchsorted(f, grid, side="right") / n)
        C = np.array(curves)
        self.T99 = float(np.median(self.years(np.array(t99)))) if t99 else np.nan
        return (self._fixed, (self.years(grid), C.mean(0), np.quantile(C, 0.1, 0), np.quantile(C, 0.9, 0)),
                self.years(np.array(t99)))


def stretched(t, T, b):
    return np.exp(-(t / T) ** b)


def hack_starts(c, n=8, fracs=FRACS):
    return np.linspace(c.teq + 20, c.r.t - max(fracs) * c.Tfv * c.If - 1, n)


def hack_stats(c, fracs=FRACS, n_starts=8, keep_points=0.174):
    out, pts = {}, None
    starts = hack_starts(c, n_starts, fracs)
    for f in fracs:
        L, A = hack(c, f, starts)
        h, lo, hi = hack_exponent(L, A)
        out[str(f)] = dict(h=h, h_lo=lo, h_hi=hi, n=int(L.size))
        if f == keep_points:
            pts = (L, A)
    return out, pts


def overlap_stats(c):
    rw = reworking(c)
    t, o = rw["t_overlap"], rw["overlap"]
    ok = t > 0
    (Ts, beta), _ = optimize.curve_fit(stretched, t[ok], o[ok], p0=[20, 0.6], maxfev=20000)
    half = float(np.interp(0.5, o[ok][::-1], t[ok][::-1]))
    return dict(T_R_years=rw["T_R"], T_R_over_Tfv=rw["T_R"] / c.Tfv, r2_exp_unreworked=rw["r2_R"],
                r2_exp_overlap=rw["r2_M"], stretched_T=float(Ts), stretched_beta=float(beta),
                half_overlap_years=half), rw


def compensation_stats(c):
    T, s, H = compensation(c)
    Tc = H / c.p.subside
    k_lo, k_hi = kappa_fit(T, s, Tc)
    return dict(mean_channel_depth_m=H, Tc_years=Tc, kappa_below=k_lo, kappa_above=k_hi), (T, s, Tc)


def summarize(c, with_compensation=True):
    """All summary numbers for one case, and the curves the figures need."""
    v = (c.mean_residence() / c.Tfv)[c.region("plain")]
    v = v[np.isfinite(v)]
    chi, lo, hi = chi_with_range(v)
    h, hpts = hack_stats(c)
    ov, rw = overlap_stats(c)
    avs = c.years(c.r.avulsions[:, 0])
    after = avs > c.years(c.teq)
    s = dict(T_fv1_years=c.Tfv, avulsions_after_spinup=int(after.sum()),
             years_between_avulsions=float((c.years(c.r.t) - c.years(c.teq)) / max(after.sum(), 1)),
             chi=chi, chi_min=lo, chi_max=hi, hack=h, **ov)
    curves = dict(hack_points=hpts, reworking=rw, residence=v)
    if with_compensation and getattr(c.r, "snapshots", None):
        cs, ccurve = compensation_stats(c)
        s.update(cs)
        curves["compensation"] = ccurve
    return s, curves


if __name__ == "__main__":
    import json
    for path in sys.argv[1:]:
        c = NullCase(path, 2905.6) if "null" in path else Case(path)
        s, _ = summarize(c)
        print(os.path.basename(path), json.dumps({k: v for k, v in s.items() if k != "hack"}, default=float))
        print("  hack", {k: round(v["h"], 3) for k, v in s["hack"].items()})
