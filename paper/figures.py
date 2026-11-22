"""Figures 2-9 of Puma et al., "On the connections between surficial processes and
stratigraphy in river deltas" (arXiv:1606.04558), regenerated with the corrected model.

    python paper/run_paper.py          # the simulations (about 1 hour on one core)
    python paper/figures.py            # figures and paper/results.json

Time is reported in years: model time (effective flood years) divided by the flood
intermittency I_f. Residence times are normalized by T_fv=1, the time the channel takes
to visit every cell of the domain after the spin-up.
"""
import json
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from scipy import stats

from deltamodel.history import channel_paths, occupancy_intervals

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.environ.get("DELTA_RUNS", os.path.join(HERE, "runs"))
OUT = os.environ.get("DELTA_FIGS", os.path.join(HERE, "figures"))

INK, MUTED, GRID = "#1f2328", "#57606a", "#d0d7de"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # 0.2L, 0.5L, 0.8L
STYLES = ["-", "--", ":"]
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.titlesize": 9, "legend.frameon": False,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})


# ----------------------------------------------------------------------------- helpers

class Case:
    """A finished run with its geometry and derived channel statistics."""

    def __init__(self, path):
        self.p, self.r = pickle.load(open(path, "rb"))
        p = self.p
        self.If = p.I_tavg
        self.X, self.Y = p.xmax, p.ymax + 1             # cells across, cells downstream
        self.L = p.a * p.ymax                           # downstream length, m
        self.i0 = p.imidx - 1                           # source column (0-based)
        self.occ, self.open = occupancy_intervals(self.r)
        self.teq = p.teq
        ii, jj = np.meshgrid(np.arange(self.X), np.arange(self.Y), indexing="ij")
        self.xL = jj * p.a / self.L                     # downstream distance / L
        self.yL = (ii - self.i0) * p.a / self.L         # transverse distance / L
        self.R = np.hypot(self.xL, self.yL)             # radial distance from the source / L
        self.Tfv, self.fv_curve = self._t_fv()

    def years(self, t):
        return np.asarray(t) / self.If

    def _t_fv(self, step=10.0):
        """T_fv=1: time for the channel to visit every cell of the delta plain (the L x W
        cells upstream of the shoreline row), from a start time after the spin-up. The
        value depends on the start time, so it is computed for start times every `step`
        (model time) through the run and the median is used. Returns the median (years)
        and the mean fraction visited against time interval, with its 10-90% range."""
        o = self.occ
        keep = o[:, 1] < self.Y - 1
        o = o[keep]
        done, curves = [], []
        grid = np.linspace(0, self.r.t - self.teq, 400)
        for s in np.arange(self.teq, self.r.t, step):
            live = o[:, 3] > s
            first = np.full((self.X, self.Y - 1), np.inf)
            np.minimum.at(first, (o[live, 0].astype(int), o[live, 1].astype(int)),
                          np.maximum(o[live, 2], s) - s)
            f = np.sort(first.ravel())
            if not np.isfinite(f[-1]):
                break                                   # not every cell reached before the end
            done.append(f[-1])
            curves.append(np.searchsorted(f, grid, side="right") / f.size)
        self.all_visited = len(done) > 0
        if not done:                                    # short test runs
            f = f[np.isfinite(f)]
            done = [f[-1]]
            curves = [np.searchsorted(f, grid, side="right") / (self.X * (self.Y - 1))]
        done = self.years(np.array(done))
        C = np.array(curves)
        self.Tfv_starts = done
        return float(np.median(done)), (self.years(grid), C.mean(0), np.quantile(C, 0.1, 0),
                                        np.quantile(C, 0.9, 0))

    def visits(self, after_teq=True):
        """Completed visits (i, j, duration in years), counting those that end after the
        spin-up, as the model's own visit statistics do."""
        o = self.occ[~self.open]
        if after_teq:
            o = o[o[:, 3] > self.teq]
        return o[:, 0].astype(int), o[:, 1].astype(int), self.years(o[:, 3] - o[:, 2])

    def mean_residence(self):
        i, j, d = self.visits()
        s = np.zeros((self.X, self.Y)); n = np.zeros((self.X, self.Y))
        np.add.at(s, (i, j), d); np.add.at(n, (i, j), 1)
        with np.errstate(invalid="ignore"):
            return np.where(n > 0, s / n, np.nan)

    def total_residence(self):
        """Total time each cell is occupied after the spin-up (open visits included)."""
        o = self.occ
        t_on = np.maximum(o[:, 2], self.teq)
        d = np.clip(o[:, 3] - t_on, 0, None)
        s = np.zeros((self.X, self.Y))
        np.add.at(s, (o[:, 0].astype(int), o[:, 1].astype(int)), self.years(d))
        return s

    def window_residence(self, t0, dt):
        """Time each cell is occupied in [t0, t0 + dt] (model time)."""
        o = self.occ
        on = np.clip(o[:, 2], t0, t0 + dt)
        off = np.clip(o[:, 3], t0, t0 + dt)
        s = np.zeros((self.X, self.Y))
        np.add.at(s, (o[:, 0].astype(int), o[:, 1].astype(int)), self.years(off - on))
        return s

    def square(self):
        """Index masks of the L x L subsection centred on the source."""
        half = self.p.ymax // 2
        cols = np.arange(self.i0 - half, self.i0 + half)
        return cols, np.arange(self.p.ymax)

    def column(self, i, j):
        """Layers of the stratigraphic column (i, j): bottoms, tops, types, times."""
        r = self.r
        k = r.top_layer[i, j]
        th = r.strat_thickness[i, j, :k + 1]
        top = np.cumsum(th)
        return top - th, top, r.strat_type[i, j, :k + 1], r.strat_time[i, j, :k + 1]

    def channel_bodies(self, j, after_teq=True):
        """Thickness of each channel deposit in row j (consecutive channel layers merged)."""
        out = []
        for i in range(self.X):
            bot, top, typ, tim = self.column(i, j)
            run = 0.0
            for b, t, ty, tm in zip(bot, top, typ, tim):
                if ty == 2 and (tm > self.teq or not after_teq):
                    run += t - b
                else:
                    if run > 0:
                        out.append(run)
                    run = 0.0
            if run > 0:
                out.append(run)
        return np.array(out)


def style(ax, grid_y=False):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid_y:
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)


def panel(ax, letter, title=""):
    ax.set_title(f"({letter}) {title}".rstrip(), loc="left", color=INK)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".png"))
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    plt.close(fig)


def step_hist(ax, data, bins, **kw):
    h, e = np.histogram(data, bins=bins, density=True)
    ax.stairs(h, e, lw=1.4, **kw)
    return h, e


def path_xy(c, path):
    a = np.array(path) - 1
    return c.xL[a[:, 0], a[:, 1]], c.yL[a[:, 0], a[:, 1]]


# ----------------------------------------------------------------------------- figures

def fig2(c, res):
    fig = plt.figure(figsize=(7.0, 6.4))
    ax = fig.add_axes([0.08, 0.56, 0.36, 0.40])
    times = []
    paths = list(channel_paths(c.r))
    tend = c.r.t
    span = min(c.Tfv * c.If, tend - c.teq)              # model time
    targets = [tend - 0.10 * span, tend - 0.05 * span, tend]
    shades = ["#c9ced4", "#8c959f", INK]
    for tt, col in zip(targets, shades):
        k = max(n for n, (t, _) in enumerate(paths) if t <= tt)
        x, y = path_xy(c, paths[k][1])
        ax.plot(x, y, "s-", ms=2.5, lw=1, color=col)
        times.append(c.years(tt))
    ax.set_xlim(0, 1); ax.set_ylim(0.7, -0.7)
    ax.set_xlabel("x / L"); ax.set_ylabel("y / L")
    ax.set_aspect("equal"); panel(ax, "A"); style(ax)

    ax = fig.add_axes([0.58, 0.56, 0.38, 0.40])
    tg, mean, lo, hi = c.fv_curve
    ax.fill_between(tg, lo, hi, color=SERIES[0], alpha=0.18, lw=0, label="10-90% range")
    ax.plot(tg, mean, color=SERIES[0], lw=1.6, label="mean")
    ax.axvline(c.Tfv, color=MUTED, lw=0.8, ls="--")
    ax.text(c.Tfv * 1.03, 0.12, f"$T_{{f_v=1}}$ ≈ {round(c.Tfv, -1):,.0f} yr", ha="left", color=INK)
    ax.set_xlim(0, 2.2 * c.Tfv); ax.set_ylim(0, 1.02)
    ax.set_xlabel(r"$\Delta t$ [yr]"); ax.set_ylabel(r"Fraction of domain visited, $f_v$")
    ax.legend(loc="center right", fontsize=8)
    panel(ax, "B"); style(ax, grid_y=True)

    ax = fig.add_axes([0.10, -0.02, 0.80, 0.46], projection="3d")
    ax.patch.set_alpha(0)
    ax.plot_surface(c.xL, c.yL, c.r.eta, cmap="copper", linewidth=0, antialiased=True,
                    rcount=c.X, ccount=c.Y)
    ax.set_xlabel("x / L"); ax.set_ylabel("y / L"); ax.set_zlabel("Bed elevation [m]")
    ax.view_init(elev=28, azim=-60)
    ax.text2D(0.0, 0.95, "(C)", transform=ax.transAxes, color=INK)
    save(fig, "fig2")
    res["fig2_snapshot_times_yr"] = [round(float(t)) for t in times]
    res["T_fv1_yr"] = float(c.Tfv)
    res["T_max_over_T_fv1"] = float(c.years(c.r.t) / c.Tfv)


def n_channels_at_outlet(occupied_row):
    """Number of separate runs of occupied cells along the shoreline row."""
    b = np.r_[0, (occupied_row > 0).astype(int), 0]
    return int(np.sum(np.diff(b) == 1))


def fig3(c, res, t0=None):
    fracs = [0.0174, 0.035, 0.174, 0.35]
    t0 = c.teq + 0.5 * (c.r.t - c.teq) if t0 is None else t0
    cmap = ListedColormap(plt.get_cmap("Blues")(np.linspace(0.25, 1.0, 256)))
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 6.2))
    out = []
    for ax, f, L in zip(axs.ravel(), fracs, "ABCD"):
        dt = f * min(c.Tfv * c.If, c.r.t - c.teq)
        w = c.window_residence(t0, dt) / c.Tfv
        lw = np.log10(np.where(w > 0, w, np.nan))
        vmin, vmax = (-3.6, -1.6) if f < 0.1 else (-5.0, -0.5)
        im = ax.pcolormesh(c.xL[0], c.yL[:, 0], lw, cmap=cmap, vmin=vmin, vmax=vmax,
                           shading="nearest", edgecolors="face", linewidth=0, rasterized=True)
        ax.set_xlim(0, 1); ax.set_ylim(0.7, -0.7); ax.set_aspect("equal")
        ax.set_xlabel("x / L"); ax.set_ylabel("y / L")
        panel(ax, L, rf"$\Delta t/T_{{f_v=1}}$ = {f}")
        cb = fig.colorbar(im, ax=ax, shrink=0.8)
        cb.set_label(r"log$_{10}(T/T_{f_v=1})$")
        vals = w[w > 0]
        out.append(dict(frac=f, n_channels_outlet=n_channels_at_outlet(w[:, -1]),
                        decades=float(np.log10(vals.max() / vals.min()))))
    fig.tight_layout()
    save(fig, "fig3")
    res["fig3_t0_yr"] = float(c.years(t0))
    res["fig3"] = out


SECTIONS = {"A": 0.04, "B": 0.2, "C": 0.5, "D": 0.8}


def strat_image(c, j, cols, zlo, zhi, dz=0.01):
    """Raster of a transverse section: 0 = empty, 1 = floodplain, 2 = channel, 3 = base."""
    z = np.arange(zlo, zhi, dz)
    img = np.zeros((z.size, len(cols)), dtype=int)
    surf = []
    for n, i in enumerate(cols):
        bot, top, typ, _ = c.column(i, j)
        surf.append(top[-1])
        for b, t, ty in zip(bot, top, typ):
            if t - b <= 0:
                continue
            k0, k1 = np.searchsorted(z, [b, t])
            img[k0:k1, n] = {0: 3, 1: 1, 2: 2}.get(int(ty), 0)
    return z, img, np.array(surf)


STRAT_CMAP = ListedColormap(["white", "#9aa3ad", "black", "#e6e8eb"])


def fig4(c, res):
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.0), sharey=True)
    cols = np.arange(c.X)
    tops = [c.column(i, j)[1][-1] for i in cols for j in range(c.Y)]
    zhi = np.ceil(max(tops) + 0.5)
    zlo = zhi - 16
    for ax, (L, xf) in zip(axs.ravel(), SECTIONS.items()):
        j = int(round(xf * c.p.ymax))
        z, img, _ = strat_image(c, j, cols, zlo, zhi)
        y = c.yL[:, 0]
        ax.imshow(img, origin="lower", aspect="auto", cmap=STRAT_CMAP, vmin=0, vmax=3,
                  extent=[y[0] - 0.01, y[-1] + 0.01, zlo, zhi], interpolation="nearest")
        ax.set_xlabel("y / L"); panel(ax, L, f"x = {xf}L")
        if L in "AC":
            ax.set_ylabel("Elevation [m]")
    fig.tight_layout()
    save(fig, "fig4")
    res["fig4_elevation_range_m"] = [float(zlo), float(zhi)]


def local_extrema(s):
    mx = [k for k in range(1, len(s) - 1) if s[k] > s[k - 1] and s[k] > s[k + 1]]
    mn = [k for k in range(1, len(s) - 1) if s[k] < s[k - 1] and s[k] < s[k + 1]]
    return mx, mn


def fig5(c, res):
    fig, axs = plt.subplots(2, 1, figsize=(6.0, 6.0))
    cols = np.arange(c.i0, c.i0 + int(round(0.2 * c.p.ymax)) + 1)
    out = {}
    for ax, (L, xf) in zip(axs, [("A", 0.5), ("B", 0.8)]):
        j = int(round(xf * c.p.ymax))
        _, _, surf = strat_image(c, j, cols, 0, 1)
        zhi = np.ceil(surf.max() + 0.5)
        zlo = zhi - 10
        z, img, surf = strat_image(c, j, cols, zlo, zhi)
        y = c.yL[cols, 0]
        ax.imshow(img, origin="lower", aspect="auto", cmap=STRAT_CMAP, vmin=0, vmax=3,
                  extent=[y[0] - 0.01, y[-1] + 0.01, zlo, zhi], interpolation="nearest")
        mx, mn = local_extrema(surf)
        ax.plot(y[mx], surf[mx] + 0.35, "v", color=SERIES[1], ms=6, label="local maximum")
        ax.plot(y[mn], surf[mn] + 0.35, "v", color=SERIES[0], ms=6, mfc="white",
                label="local minimum")
        ax.set_xlabel("y / L"); ax.set_ylabel("Elevation [m]")
        panel(ax, L, f"x = {xf}L")
        ax.legend(loc="lower right", fontsize=8, ncol=2, bbox_to_anchor=(1, 1.0))
        # relief of the whole section, from the surface across all columns
        _, _, s_all = strat_image(c, j, np.arange(c.X), 0, 1)
        ma, mi = local_extrema(s_all)
        out[str(xf)] = dict(mean_max_minus_min_m=float(np.mean(s_all[ma]) - np.mean(s_all[mi])),
                            std_surface_m=float(np.std(s_all)))
    fig.tight_layout()
    save(fig, "fig5")
    res["fig5_relief"] = out


def fig6(c, res):
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    bins = np.arange(0, 2.75, 0.25)
    out = {}
    for xf, col, ls in zip([0.2, 0.5, 0.8], SERIES, STYLES):
        h = c.channel_bodies(int(round(xf * c.p.ymax)))
        step_hist(ax, h, bins, color=col, ls=ls, label=f"{xf}L")
        out[str(xf)] = dict(n=int(h.size), mean_m=float(h.mean()), median_m=float(np.median(h)))
    ax.set_xlabel(r"$h_{ch}$ [m]"); ax.set_ylabel("# in bin / total # / bin width")
    ax.set_xlim(0, 2.5); ax.legend(); style(ax, grid_y=True)
    save(fig, "fig6")
    res["fig6_channel_deposits"] = out


def fig7(c, res):
    T = c.total_residence()[:, :-1] / c.Tfv          # delta plain: without the shoreline row
    h = c.r.depth_channel[:, :-1]
    m = T > 0
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.2))
    ax = axs[0]
    ax.scatter(T[m], h[m], s=3, color=SERIES[0], alpha=0.5, lw=0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(1, 100)
    ax.set_xlabel(r"$\sum T/T_{f_v=1}$"); ax.set_ylabel(r"$\sum h_{ch}$ [m]")
    panel(ax, "A"); style(ax)
    ax = axs[1]
    ranges = [(0.08, 0.10), (0.16, 0.18), (0.25, 0.27)]      # the ranges of version 1
    bins = np.arange(0, np.ceil(h[m].max()) + 1, 1.0)
    out = []
    for (lo, hi), col, ls in zip(ranges, SERIES, STYLES):
        sel = (T >= lo) & (T < hi)
        step_hist(ax, h[sel], bins, color=col, ls=ls, label=f"{lo:.2f} – {hi:.2f}")
        out.append(dict(range=[lo, hi], n=int(sel.sum()), mean_h_m=float(h[sel].mean()),
                        std_h_m=float(h[sel].std()), skew=float(stats.skew(h[sel]))))
    ax.set_xlabel(r"$\sum h_{ch}$ [m]"); ax.set_ylabel("# in bin / total # / bin width")
    ax.legend(title=r"$\sum T/T_{f_v=1}$", fontsize=8, title_fontsize=8)
    panel(ax, "B"); style(ax, grid_y=True)
    fig.tight_layout()
    save(fig, "fig7")
    rho = stats.spearmanr(T[m], h[m])
    res["fig7"] = dict(spearman_rho=float(rho.correlation), n_cells=int(m.sum()),
                       n_below_0p1m=int((h[m] < 0.1).sum()),
                       T_range=[float(T[m].min()), float(T[m].max())],
                       h_range_m=[float(h[m].min()), float(h[m].max())], bins=out)


def exceedance(v):
    v = np.sort(v[np.isfinite(v)])
    return v, 1.0 - np.arange(v.size) / v.size


def fit_chi(v, p_hi=0.5, p_lo=0.01):
    """Exponent of P[T >= t] ~ t^-chi, fitted on the part with p_lo <= P <= p_hi."""
    t, P = exceedance(v)
    m = (P <= p_hi) & (P >= p_lo)
    slope = np.polyfit(np.log10(t[m]), np.log10(P[m]), 1)[0]
    return -slope


def chi_with_range(v):
    fits = [fit_chi(v, a, b) for a in (0.7, 0.5, 0.3) for b in (0.03, 0.01, 0.003)]
    return float(np.median(fits)), float(min(fits)), float(max(fits))


def fig8(c, big, res):
    cols, rows = c.square()
    Tm = c.mean_residence()[np.ix_(cols, rows)] / c.Tfv
    fig = plt.figure(figsize=(5.2, 8.0))
    ax = fig.add_axes([0.14, 0.58, 0.62, 0.38])
    cmap = ListedColormap(plt.get_cmap("Blues")(np.linspace(0.1, 1.0, 256)))
    im = ax.pcolormesh(c.xL[0, rows], c.yL[cols, 0], np.log10(Tm), cmap=cmap, shading="nearest",
                       edgecolors="face", linewidth=0, rasterized=True)
    ax.set_ylim(0.5, -0.5); ax.set_aspect("equal")
    ax.set_xlabel("x / L"); ax.set_ylabel("y / L"); panel(ax, "A")
    cb = fig.colorbar(im, ax=ax, shrink=0.9)
    cb.set_label(r"log$_{10}(\bar T/T_{f_v=1})$")

    ax = fig.add_axes([0.14, 0.06, 0.78, 0.42])
    sets = [(f"{len(rows)} × {len(rows)} cells (L = {c.L/1000:g} km)", Tm, SERIES[0], "o")]
    if big is not None:
        bc, br = big.square()
        Tb = big.mean_residence()[np.ix_(bc, br)] / big.Tfv
        sets.append((f"{len(br)} × {len(br)} cells (L = {big.L/1000:g} km)", Tb, SERIES[1], "+"))
    out = {}
    for lab, v, col, mk in sets:
        t, P = exceedance(v.ravel())
        ax.loglog(t, P, mk, ms=3, color=col, mfc="none", label=lab)
        chi, lo, hi = chi_with_range(v.ravel())
        out[lab] = dict(chi=chi, chi_min=lo, chi_max=hi, n_cells=int(np.isfinite(v).sum()))
    chi0 = out[sets[0][0]]["chi"]
    tt = np.logspace(-2.3, -1.0, 10)
    t0, P0 = exceedance(Tm.ravel())
    k = np.argmin(np.abs(P0 - 0.1))
    ax.loglog(tt, P0[k] * (tt / t0[k]) ** (-chi0), color=MUTED, lw=0.8, ls="--")
    ax.text(0.08, 0.25, rf"$\chi$ = {chi0:.1f}", transform=ax.transAxes, color=INK, fontsize=11)
    ax.set_xlabel("t"); ax.set_ylabel(r"$P[\bar T/T_{f_v=1} \geq t]$")
    ax.set_xlim(1e-4, 1); ax.set_ylim(1e-4, 1.5)
    ax.legend(loc="upper right", fontsize=8); panel(ax, "B"); style(ax)
    save(fig, "fig8")
    res["fig8"] = out


def fig9(c, res):
    cols, rows = c.square()
    i, j, d = c.visits()
    inside = np.isin(i, cols) & np.isin(j, rows)
    Rv = c.R[i, j]
    T = d / c.Tfv
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    bins = np.arange(0, 0.0525, 0.0025)
    dR = 0.1
    out = {}
    for Rm, col, ls in zip([0.2, 0.5, 0.8], SERIES, STYLES):
        sel = inside & (Rv >= Rm - dR / 2) & (Rv < Rm + dR / 2)
        step_hist(ax, T[sel], bins, color=col, ls=ls, label=f"{Rm}L")
        s, loc, scale = stats.lognorm.fit(T[sel], floc=0)
        ks = stats.kstest(T[sel], "lognorm", args=(s, loc, scale))
        out[str(Rm)] = dict(n=int(sel.sum()), median=float(np.median(T[sel])),
                            mean=float(T[sel].mean()), lognorm_mu=float(np.log(scale)),
                            lognorm_sigma=float(s), ks_stat=float(ks.statistic))
    ax.set_xlim(0, 0.05)
    ax.set_xlabel(r"$T/T_{f_v=1}$"); ax.set_ylabel("# in bin / total # / bin width")
    ax.legend(loc="upper right", bbox_to_anchor=(1, 1)); style(ax, grid_y=True)
    ins = ax.inset_axes([0.60, 0.12, 0.30, 0.42])
    th = np.linspace(-np.pi / 2, np.pi / 2, 200)
    for Rm, col in zip([0.2, 0.5, 0.8], SERIES):
        for rr in (Rm - dR / 2, Rm + dR / 2):
            ins.plot(rr * np.cos(th), rr * np.sin(th), color=col, lw=0.8)
    ins.plot([0, 1, 1, 0, 0], [-0.5, -0.5, 0.5, 0.5, -0.5], color=MUTED, lw=0.6)
    ins.set_xlim(0, 1); ins.set_ylim(0.5, -0.5); ins.set_aspect("equal")
    ins.set_xticks([0, 0.5, 1]); ins.set_yticks([-0.5, 0, 0.5]); ins.tick_params(labelsize=6)
    ins.set_xlabel("x / L", fontsize=6); ins.set_ylabel("y / L", fontsize=6)
    save(fig, "fig9")
    res["fig9"] = dict(dR_over_L=dR, annuli=out)


def text_stats(c, res):
    """Numbers quoted in the text that are not read off a figure."""
    out = {}
    # local relief of the bed in transverse sections (rows +-2 around each position)
    rel = {}
    for xf in (0.2, 0.5, 0.8):
        vals = []
        for j in range(int(round(xf * c.p.ymax)) - 2, int(round(xf * c.p.ymax)) + 3):
            s = c.r.eta[:, j]
            mx, mn = local_extrema(s)
            ext = sorted(mx + mn)
            vals.append(np.abs(np.diff(s[ext])).mean())
        rel[str(xf)] = float(np.mean(vals))
    out["relief_adjacent_extrema_m"] = rel
    # successive occupation: when the channel moves in a section, how often to an adjacent cell
    rows = {xf: int(round(xf * c.p.ymax)) for xf in (0.04, 0.2, 0.5, 0.8)}
    prev = {xf: None for xf in rows}
    cnt = {xf: [0, 0] for xf in rows}
    for t, path in channel_paths(c.r):
        for xf, j in rows.items():
            now = {i for i, jj in path if jj - 1 == j}
            if t > c.teq and prev[xf] and now and not (now & prev[xf]):
                d = min(abs(a - b) for a in now for b in prev[xf])
                cnt[xf][0] += 1
                cnt[xf][1] += d == 1
            prev[xf] = now
    out["section_shifts"] = {str(k): dict(n=v[0], adjacent_fraction=v[1] / v[0]) for k, v in cnt.items()}
    # reoccupation of recently abandoned channels by new paths, by distance below the avulsion
    win = 0.1 * c.Tfv * c.If
    last_on = np.full((c.X, c.Y), -np.inf)
    prevp = set()
    by = {}
    n_av = n_re = 0
    for t, path in channel_paths(c.r):
        now = set(path)
        for (i, j) in prevp - now:
            last_on[i - 1, j - 1] = t
        if t > c.teq and prevp:
            new = [k for k, cell in enumerate(path) if cell not in prevp]
            if new:
                n_av += 1
                any_re = False
                for off, cell in enumerate(path[new[0]:], start=1):
                    if cell in prevp:
                        continue
                    re = (t - last_on[cell[0] - 1, cell[1] - 1]) <= win
                    b = min((off - 1) // 5, 6)
                    by.setdefault(b, [0, 0])
                    by[b][0] += 1
                    by[b][1] += re
                    any_re |= re
                n_re += any_re
        prevp = now
    out["reoccupation"] = dict(
        window_T_fv1=0.1, avulsions=n_av, fraction_with_reoccupation=n_re / n_av,
        fraction_of_new_cells_by_offset={f"{5*b+1}-{5*b+5}" if b < 6 else "31+": v[1] / v[0]
                                         for b, v in sorted(by.items())})
    # visits in single cells against the annulus ensemble (Fig. 9)
    cols, rows_ = c.square()
    i, j, d = c.visits()
    T = d / c.Tfv
    inside = np.isin(i, cols) & np.isin(j, rows_)
    ks = {}
    for Rm in (0.2, 0.5, 0.8):
        sel = inside & (np.abs(c.R[i, j] - Rm) < 0.05)
        ps = []
        for a, b in set(zip(i[sel], j[sel])):
            m = sel & (i == a) & (j == b)
            if m.sum() >= 20:
                ps.append(stats.ks_2samp(T[m], T[sel]).pvalue)
        ps = np.array(ps)
        ks[str(Rm)] = dict(cells_with_20_visits=int(ps.size), fraction_p_above_0p05=float((ps > 0.05).mean()))
    out["fig9_cells_vs_ensemble"] = ks
    res["text"] = out


def summary(c, res, name):
    r = c.r
    after = r.avulsions[:, 0] > c.teq
    res[name] = dict(
        avulsions=int(r.num_avulsions), avulsions_after_spinup=int(after.sum()),
        run_years=float(c.years(r.t)), spinup_years=float(c.years(c.teq)),
        T_fv1_yr=float(c.Tfv), T_fv1_starts=len(c.Tfv_starts),
        T_fv1_quartiles_yr=[float(v) for v in np.quantile(c.Tfv_starts, [0.25, 0.75])],
        mean_years_between_avulsions=float(c.years(r.t - c.teq) / after.sum()),
        n_visits=int(len(c.visits()[2])),
    )


def main():
    base = Case(os.path.join(RUNS, "base.pkl"))
    big = Case(os.path.join(RUNS, "large.pkl")) if os.path.exists(os.path.join(RUNS, "large.pkl")) else None
    res = {}
    summary(base, res, "base")
    if big is not None:
        summary(big, res, "large")
    fig2(base, res); fig3(base, res); fig4(base, res); fig5(base, res)
    fig6(base, res); fig7(base, res); fig8(base, big, res); fig9(base, res)
    text_stats(base, res)
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
