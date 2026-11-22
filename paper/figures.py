"""Figures 2-12 of version 2 of Puma et al. (arXiv:1606.04558), and every number quoted in
the text (paper/figures/results.json).

    python paper/run_paper.py        # the simulations (paper/runs/)
    python paper/figures.py

The main configuration is the semicircular delta plain with an open shoreline; the
rectangle of version 1 is used to show the effect of its side walls.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from scipy import optimize, stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis import (Case, boundary_stats, chi_with_range, compensation, exceedance,  # noqa: E402
                      hack, hack_exponent, kappa_fit, residence_vs_thickness, reworking)

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.environ.get("DELTA_RUNS", os.path.join(HERE, "runs"))
OUT = os.environ.get("DELTA_FIGS", os.path.join(HERE, "figures"))

INK, MUTED, GRID = "#1f2328", "#57606a", "#d0d7de"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SERIES = [BLUE, ORANGE, AQUA]
STYLES = ["-", "--", ":"]
SEQ = ListedColormap(plt.get_cmap("Blues")(np.linspace(0.2, 1.0, 256)))
STRAT = ListedColormap(["white", "#9aa3ad", "black", "#e6e8eb"])
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
})


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


def map_axes(ax, c):
    """Plan view: x/L downstream, y/L across, with the domain outline."""
    if c.domain == "semicircle":
        th = np.linspace(-np.pi / 2, np.pi / 2, 200)
        ax.plot(np.cos(th), np.sin(th), color=MUTED, lw=0.8)
        ax.plot([0, 0], [-1, 1], color=MUTED, lw=0.8)
        ax.set_xlim(-0.03, 1.05); ax.set_ylim(1.05, -1.05)
    else:
        ax.set_xlim(0, 1); ax.set_ylim(0.72, -0.72)
    ax.set_aspect("equal")
    ax.set_xlabel("x / L"); ax.set_ylabel("y / L")


def cellmap(ax, c, Z, mask, **kw):
    Z = np.where(mask, Z, np.nan)
    return ax.pcolormesh(c.xL[0], c.yL[:, 0], Z, shading="nearest", edgecolors="face",
                         linewidth=0, rasterized=True, **kw)


def path_xy(c, path):
    a = np.array(path) - 1
    return c.xL[a[:, 0], a[:, 1]], c.yL[a[:, 0], a[:, 1]]


# ===================================================================== figures

def fig2(c, res):
    fig = plt.figure(figsize=(7.0, 7.2))
    ax = fig.add_axes([0.06, 0.55, 0.40, 0.42])
    span = c.Tfv * c.If
    times = [c.r.t - 0.10 * span, c.r.t - 0.05 * span, c.r.t]
    for tt, col in zip(times, ["#c9ced4", "#8c959f", INK]):
        k = max(n for n, (t, _) in enumerate(c.paths) if t <= tt)
        x, y = path_xy(c, c.paths[k][1])
        ax.plot(x, y, "s-", ms=2, lw=0.8, color=col)
    map_axes(ax, c); panel(ax, "A", "channel at three times"); style(ax)
    res["fig2_snapshot_years"] = [round(float(c.years(t))) for t in times]

    ax = fig.add_axes([0.58, 0.58, 0.38, 0.37])
    tg, mean, lo, hi = c.fv_curve
    ax.fill_between(tg, lo, hi, color=BLUE, alpha=0.18, lw=0, label="10-90% range")
    ax.plot(tg, mean, color=BLUE, lw=1.6, label="mean")
    ax.axvline(c.Tfv, color=MUTED, lw=0.8, ls="--")
    ax.text(c.Tfv * 1.03, 0.12, f"$T_{{f_v=1}}$ ≈ {round(c.Tfv, -1):,.0f} yr", color=INK)
    ax.set_xlim(0, 2.0 * c.Tfv); ax.set_ylim(0, 1.02)
    ax.set_xlabel(r"$\Delta t$ [yr]"); ax.set_ylabel(r"Fraction of delta plain visited, $f_v$")
    ax.legend(loc="center right", fontsize=8); panel(ax, "B"); style(ax, grid_y=True)

    ax = fig.add_axes([0.06, 0.02, 0.48, 0.42])
    m = cellmap(ax, c, c.r.eta, c.land, cmap="copper")
    map_axes(ax, c); panel(ax, "C", "bed elevation at the end"); style(ax)
    cb = fig.colorbar(m, ax=ax, shrink=0.8); cb.set_label("Bed elevation [m]")

    ax = fig.add_axes([0.66, 0.08, 0.30, 0.34])
    plain = c.region("plain")
    rb = np.arange(0, 1.0001, 0.05)
    k = np.digitize(c.R[plain], rb) - 1
    z = c.r.eta[plain]
    prof = [z[k == b].mean() for b in range(len(rb) - 1)]
    ax.plot(0.5 * (rb[1:] + rb[:-1]), prof, "o-", color=BLUE, ms=3)
    ax.set_xlabel("r / L"); ax.set_ylabel("Mean bed elevation [m]")
    panel(ax, "D", "radial profile"); style(ax, grid_y=True)
    save(fig, "fig2")


def n_outlets(c, w):
    """Separate groups of occupied shoreline cells (ordered along the shoreline)."""
    sh = np.argwhere(c.shore)
    key = c.theta[sh[:, 0], sh[:, 1]] if c.domain == "semicircle" else sh[:, 0]
    occ = w[sh[:, 0], sh[:, 1]] > 0
    occ = occ[np.argsort(key)]
    b = np.r_[0, occ.astype(int), 0]
    return int(np.sum(np.diff(b) == 1))


def fig3(c, res):
    fracs = [0.0174, 0.035, 0.174, 0.35]
    t0 = c.teq + 0.5 * (c.r.t - c.teq)
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 7.0))
    out = []
    for ax, f, L in zip(axs.ravel(), fracs, "ABCD"):
        w = c.window_residence(t0, f * c.Tfv * c.If) / c.Tfv
        vmin, vmax = (-3.6, -1.6) if f < 0.1 else (-5.0, -0.5)
        im = cellmap(ax, c, np.log10(np.where(w > 0, w, np.nan)), c.land | c.shore,
                     cmap=SEQ, vmin=vmin, vmax=vmax)
        map_axes(ax, c); panel(ax, L, rf"$\Delta t/T_{{f_v=1}}$ = {f}")
        cb = fig.colorbar(im, ax=ax, shrink=0.75); cb.set_label(r"log$_{10}(T/T_{f_v=1})$")
        v = w[w > 0]
        out.append(dict(frac=f, outlets=n_outlets(c, w), decades=float(np.log10(v.max() / v.min()))))
    fig.tight_layout()
    save(fig, "fig3")
    res["fig3_t0_years"] = float(c.years(t0))
    res["fig3"] = out


def fig4(semi, rect, res):
    """Boundary effects: where the channel spends its time, and the residence-thickness
    relation with and without the side walls."""
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw=dict(width_ratios=[1, 1, 1.1]))
    ax = axs[0]
    occ = rect.occupancy(); plain = rect.region("plain")
    prof = occ[:, :-1].sum(1) / occ[:, :-1].sum() * rect.X
    ax.plot(rect.yL[:, 0], prof, color=ORANGE, lw=1.4)
    ax.axhline(1, color=MUTED, lw=0.6, ls="--")
    ax.set_xlabel("y / L"); ax.set_ylabel("Channel occupancy / uniform")
    panel(ax, "A", "rectangle, by column"); style(ax, grid_y=True)
    ax = axs[1]
    b = res["boundary"]["semicircle"]
    bins = np.array(b["angle_bins"])
    ax.stairs(b["occupancy_by_angle_rel"], bins, color=BLUE, lw=1.4, baseline=None)
    ax.axhline(1, color=MUTED, lw=0.6, ls="--")
    ax.set_ylim(0, max(2.0, ax.get_ylim()[1]))
    ax.set_xlabel("Angle from downstream [°]"); ax.set_ylabel("Channel occupancy / uniform")
    panel(ax, "B", "semicircle, by angle"); style(ax, grid_y=True)
    ax = axs[2]
    cases = [("rectangle, whole plain", res["boundary"]["rt_rect_plain"], ORANGE, "-"),
             ("rectangle, central L×L", res["boundary"]["rt_rect_square"], ORANGE, ":"),
             ("semicircle", res["boundary"]["rt_semi"], BLUE, "-")]
    for lab, d, col, ls in cases:
        ax.plot(["<25", "25-50", "50-75", "75-90", ">90"], d["mean_h_by_T_bin"], ls, marker="o",
                ms=3, color=col, label=f"{lab} (ρ = {d['spearman']:.2f})")
    ax.set_xlabel(r"Percentile of $\sum T$"); ax.set_ylabel(r"Mean $\sum h_{ch}$ [m]")
    lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + 0.45 * (hi - lo))
    ax.legend(fontsize=6.5, loc="upper left"); panel(ax, "C", "deposit vs residence"); style(ax, grid_y=True)
    fig.tight_layout()
    save(fig, "fig4")


SECTIONS = {"A": 0.04, "B": 0.2, "C": 0.5, "D": 0.8}


def strat_image(c, cells, zlo, zhi, dz=0.01):
    z = np.arange(zlo, zhi, dz)
    img = np.zeros((z.size, len(cells)), dtype=int)
    for n, (i, j) in enumerate(cells):
        bot, top, typ, _ = c.column(i, j)
        for b0, t0, ty in zip(bot, top, typ):
            if t0 - b0 <= 0:
                continue
            k0, k1 = np.searchsorted(z, [b0, t0])
            img[k0:k1, n] = {0: 3, 1: 1, 2: 2}.get(int(ty), 0)
    return img


def fig5(c, res):
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.0), sharey=True)
    tops = [c.column(i, j)[1][-1] for i, j in zip(*np.nonzero(c.land))]
    zhi = np.ceil(max(tops) + 0.5); zlo = zhi - 16
    for ax, (L, xf) in zip(axs.ravel(), SECTIONS.items()):
        cells, coord = c.section_cells(xf)
        img = strat_image(c, cells, zlo, zhi)
        ax.imshow(img, origin="lower", aspect="auto", cmap=STRAT, vmin=0, vmax=3,
                  extent=[coord[0], coord[-1], zlo, zhi], interpolation="nearest")
        ax.set_xlabel("Distance along the arc / L" if c.domain == "semicircle" else "y / L")
        panel(ax, L, f"r = {xf}L")
        if L in "AC":
            ax.set_ylabel("Elevation [m]")
    fig.tight_layout()
    save(fig, "fig5")


def fig6(semi, rect, res):
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0))
    ax = axs[0]
    bins = np.arange(0, 2.75, 0.25)
    out = {}
    for xf, col, ls in zip([0.2, 0.5, 0.8], SERIES, STYLES):
        h = semi.channel_bodies(semi.section_cells(xf)[0])
        step_hist(ax, h, bins, color=col, ls=ls, label=f"{xf}L")
        out[str(xf)] = dict(n=int(h.size), median_m=float(np.median(h)), mean_m=float(h.mean()))
    ax.set_xlabel(r"$h_{ch}$ [m]"); ax.set_ylabel("# in bin / total # / bin width")
    ax.set_xlim(0, 2.5); ax.legend(); panel(ax, "A", "channel deposit thickness"); style(ax, grid_y=True)
    res["fig6_channel_deposits"] = out
    ax = axs[1]
    xs = [0.04, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]
    succ = {}
    for c, lab, col, mk in [(semi, "semicircle", BLUE, "o"), (rect, "rectangle", ORANGE, "s")]:
        f = [c.section_shifts(x) for x in xs]
        ax.plot(xs, [a for a, n in f], mk + "-", color=col, ms=4, label=lab)
        succ[lab] = {str(x): dict(adjacent=a, moves=n) for x, (a, n) in zip(xs, f)}
    ax.set_xlabel("Distance from the source / L"); ax.set_ylabel("Moves to an adjacent cell")
    ax.set_ylim(0, None); ax.legend(); panel(ax, "B", "successive occupation"); style(ax, grid_y=True)
    res["successive_occupation"] = succ
    fig.tight_layout()
    save(fig, "fig6")


def fig7(c, res):
    plain = c.region("plain")
    T = c.total_residence()[plain] / c.Tfv
    h = c.r.depth_channel[plain]
    m = T > 0
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.2))
    ax = axs[0]
    ax.scatter(T[m], h[m], s=3, color=BLUE, alpha=0.5, lw=0)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1, 100)
    ax.set_xlabel(r"$\sum T/T_{f_v=1}$"); ax.set_ylabel(r"$\sum h_{ch}$ [m]")
    panel(ax, "A"); style(ax)
    ax = axs[1]
    q = np.quantile(T[m], [0.3, 0.7, 0.95])
    ranges = [(round(v * 0.9, 3), round(v * 1.1, 3)) for v in q]
    bins = np.arange(0, np.ceil(h[m].max()) + 1, 1.0)
    out = []
    for (lo, hi), col, ls in zip(ranges, SERIES, STYLES):
        sel = (T >= lo) & (T < hi)
        step_hist(ax, h[sel], bins, color=col, ls=ls, label=f"{lo:.3f} – {hi:.3f}")
        out.append(dict(range=[lo, hi], n=int(sel.sum()), mean_h_m=float(h[sel].mean()),
                        std_h_m=float(h[sel].std())))
    ax.set_xlabel(r"$\sum h_{ch}$ [m]"); ax.set_ylabel("# in bin / total # / bin width")
    ax.legend(title=r"$\sum T/T_{f_v=1}$", fontsize=7, title_fontsize=7)
    panel(ax, "B"); style(ax, grid_y=True)
    fig.tight_layout()
    save(fig, "fig7")
    res["fig7"] = dict(spearman=float(stats.spearmanr(T[m], h[m]).correlation), n=int(m.sum()),
                       h_median_m=float(np.median(h[m])), ranges=out)


def fig8(cases, res):
    """cases: list of (label, Case, region, colour, marker); the first is mapped."""
    lab0, c0, reg0 = cases[0][:3]
    fig = plt.figure(figsize=(5.4, 8.2))
    ax = fig.add_axes([0.12, 0.56, 0.68, 0.40])
    mask = c0.region(reg0)
    Tm = c0.mean_residence() / c0.Tfv
    im = cellmap(ax, c0, np.log10(Tm), mask, cmap=SEQ)
    map_axes(ax, c0); panel(ax, "A", lab0)
    cb = fig.colorbar(im, ax=ax, shrink=0.85); cb.set_label(r"log$_{10}(\bar T/T_{f_v=1})$")
    ax = fig.add_axes([0.14, 0.05, 0.80, 0.42])
    out = {}
    for lab, c, reg, col, mk in cases:
        v = (c.mean_residence() / c.Tfv)[c.region(reg)]
        t, P = exceedance(v)
        ax.loglog(t, P, mk, ms=2.5, color=col, mfc="none", label=lab)
        chi, lo, hi = chi_with_range(v)
        out[lab] = dict(chi=chi, chi_min=lo, chi_max=hi, cells=int(np.isfinite(v).sum()))
    chi0 = out[lab0]["chi"]
    v0 = (c0.mean_residence() / c0.Tfv)[c0.region(reg0)]
    t0, P0 = exceedance(v0)
    k = np.argmin(np.abs(P0 - 0.1))
    tt = np.logspace(np.log10(t0[k]) - 0.7, np.log10(t0[k]) + 0.7, 10)
    ax.loglog(tt, P0[k] * (tt / t0[k]) ** (-chi0), color=MUTED, lw=0.8, ls="--")
    ax.text(0.06, 0.2, rf"$\chi$ ≈ {chi0:.1f}", transform=ax.transAxes, fontsize=11, color=INK)
    ax.set_xlabel("t"); ax.set_ylabel(r"$P[\bar T/T_{f_v=1} \geq t]$")
    ax.set_ylim(1e-4, 1.5); ax.legend(fontsize=7, loc="upper right"); panel(ax, "B"); style(ax)
    save(fig, "fig8")
    res["fig8"] = out


def fig9(c, res):
    i, j, d = c.visits()
    plain = c.region("plain")
    inside = plain[i, j]
    Rv, T = c.R[i, j], d / c.Tfv
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    bins = np.linspace(0, 0.03, 25)
    out = {}
    for Rm, col, ls in zip([0.2, 0.5, 0.8], SERIES, STYLES):
        sel = inside & (np.abs(Rv - Rm) < 0.05)
        step_hist(ax, T[sel], bins, color=col, ls=ls, label=f"{Rm}L")
        s, loc, sc = stats.lognorm.fit(T[sel], floc=0)
        xx = np.linspace(bins[1] / 4, bins[-1], 200)
        ax.plot(xx, stats.lognorm.pdf(xx, s, 0, sc), color=col, lw=0.7, alpha=0.8)
        out[str(Rm)] = dict(n=int(sel.sum()), median=float(np.median(T[sel])), lognorm_mu=float(np.log(sc)),
                            lognorm_sigma=float(s), ks=float(stats.kstest(T[sel], "lognorm", args=(s, 0, sc)).statistic))
    ax.set_xlim(0, bins[-1]); ax.set_xlabel(r"$T/T_{f_v=1}$"); ax.set_ylabel("# in bin / total # / bin width")
    ax.legend(); style(ax, grid_y=True)
    save(fig, "fig9")
    # individual cells against the annulus ensemble
    cmp = {}
    for Rm in (0.2, 0.5, 0.8):
        sel = inside & (np.abs(Rv - Rm) < 0.05)
        ps = []
        for a, b in set(zip(i[sel], j[sel])):
            mm = sel & (i == a) & (j == b)
            if mm.sum() >= 20:
                ps.append(stats.ks_2samp(T[mm], T[sel]).pvalue)
        ps = np.array(ps)
        cmp[str(Rm)] = dict(cells=int(ps.size), share_p_above_0p05=float((ps > 0.05).mean()) if ps.size else None)
    res["fig9"] = dict(dR_over_L=0.1, annuli=out, cells_vs_ensemble=cmp)


def fig10(cases, res):
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    out = {}
    for lab, c, col, mk in cases:
        T, s, H = compensation(c)
        Tc = H / c.p.subside
        k_lo, k_hi = kappa_fit(T, s, Tc)
        ax.loglog(T / Tc, s, mk + "-", color=col, ms=3, lw=1, label=f"{lab}: κ = {k_lo:.2f} below $T_c$")
        out[lab] = dict(mean_channel_depth_m=H, Tc_years=Tc, kappa_below=k_lo, kappa_above=k_hi)
    x = np.logspace(-1.5, 0.6, 10)
    y0 = 1.0
    ax.loglog(x, y0 * x ** -0.5, color=MUTED, lw=0.7, ls=":")
    ax.loglog(x, y0 * x ** -1.0, color=MUTED, lw=0.7, ls="--")
    ax.text(x[-1] * 1.1, y0 * x[-1] ** -0.5, "κ = 0.5", color=MUTED, fontsize=7, va="center")
    ax.text(x[-1] * 1.1, y0 * x[-1] ** -1.0, "κ = 1", color=MUTED, fontsize=7, va="center")
    ax.axvline(1, color=MUTED, lw=0.6)
    ax.set_xlabel(r"$T / T_c$,  $T_c = \bar H / \sigma$"); ax.set_ylabel(r"$\sigma_{ss}$")
    ax.legend(fontsize=7, loc="lower left"); style(ax)
    save(fig, "fig10")
    res["compensation"] = out


def stretched(t, T, b):
    return np.exp(-(t / T) ** b)


def fig11(cases, res):
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0))
    out = {}
    for lab, c, col, ls in cases:
        rw = reworking(c)
        axs[0].plot(rw["t_unreworked"] / c.Tfv, rw["unreworked"], ls, color=col, lw=1.4, label=lab)
        t, o = rw["t_overlap"], rw["overlap"]
        ok = t > 0
        (Ts, beta), _ = optimize.curve_fit(stretched, t[ok], o[ok], p0=[20, 0.6], maxfev=20000)
        axs[1].semilogx(t[ok], o[ok], ls, color=col, lw=1.4, label=lab)
        axs[1].semilogx(t[ok], stretched(t[ok], Ts, beta), color=col, lw=0.6, alpha=0.7)
        out[lab] = dict(T_R=rw["T_R"], T_R_over_Tfv=rw["T_R"] / c.Tfv, r2_exp_unreworked=rw["r2_R"],
                        T_M_exp=rw["T_M"], r2_exp_overlap=rw["r2_M"], stretched_T=float(Ts), stretched_beta=float(beta),
                        half_overlap_years=float(t[np.argmax(o < 0.5)]))
    axs[0].set_yscale("log"); axs[0].set_ylim(1e-3, 1.1); axs[0].set_xlim(0, 1.5)
    axs[0].set_xlabel(r"$\Delta t / T_{f_v=1}$"); axs[0].set_ylabel("Fraction of the plain not yet visited")
    axs[0].legend(fontsize=7); panel(axs[0], "A", "surface reworking"); style(axs[0], grid_y=True)
    axs[1].set_xlabel(r"$\Delta t$ [yr]"); axs[1].set_ylabel("Channel overlap (normalized)")
    axs[1].set_ylim(-0.02, 1.02); panel(axs[1], "B", "channel overlap"); style(axs[1], grid_y=True)
    fig.tight_layout()
    save(fig, "fig11")
    res["reworking"] = out


def fig12(cases, res, fracs=(0.035, 0.174, 0.35), n_starts=8):
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.2))
    out = {}
    for k, (lab, c, col) in enumerate(cases):
        starts = np.linspace(c.teq + 20, c.r.t - max(fracs) * c.Tfv * c.If - 1, n_starts)
        out[lab] = {}
        for f in fracs:
            L, A = hack(c, f, starts)
            h, lo, hi = hack_exponent(L, A)
            out[lab][str(f)] = dict(h=h, h_lo=lo, h_hi=hi, n=int(L.size))
            if k == 0 and f == 0.174:
                axs[0].loglog(A / 1e6, L / 1e3, ".", ms=1.5, color=col, alpha=0.4)
                aa = np.logspace(np.log10(A.min()), np.log10(A.max()), 10)
                cfit = np.polyfit(np.log10(A), np.log10(L), 1)
                axs[0].loglog(aa / 1e6, 10 ** np.polyval(cfit, np.log10(aa)) / 1e3, color=INK, lw=1,
                              label=f"fit: h = {h:.2f}")
                axs[0].loglog(aa / 1e6, 10 ** (np.polyval(cfit, np.log10(aa.mean())) + 0.6 * (np.log10(aa) - np.log10(aa.mean()))) / 1e3,
                              color=ORANGE, lw=1, ls="--", label="h = 0.6 (Dong et al. 2026)")
        axs[1].errorbar(np.arange(len(fracs)) + 0.08 * (k - 1), [out[lab][str(f)]["h"] for f in fracs],
                        yerr=[[out[lab][str(f)]["h"] - out[lab][str(f)]["h_lo"] for f in fracs],
                              [out[lab][str(f)]["h_hi"] - out[lab][str(f)]["h"] for f in fracs]],
                        fmt="o-", color=col, ms=4, lw=1, label=lab)
    axs[0].set_xlabel(r"Nourishment area, $A$ [km$^2$]"); axs[0].set_ylabel(r"Longest flow path, $L$ [km]")
    axs[0].legend(fontsize=7); panel(axs[0], "A", r"semicircle, $\Delta t/T_{f_v=1}$ = 0.174"); style(axs[0])
    axs[1].axhline(0.6, color=ORANGE, lw=0.8, ls="--")
    axs[1].axhline(0.5, color=MUTED, lw=0.6, ls=":")
    axs[1].set_xticks(range(len(fracs)), [str(f) for f in fracs])
    axs[1].set_xlabel(r"Window $\Delta t / T_{f_v=1}$"); axs[1].set_ylabel("Hack exponent h")
    axs[1].legend(fontsize=7); panel(axs[1], "B"); style(axs[1], grid_y=True)
    fig.tight_layout()
    save(fig, "fig12")
    res["hack"] = out


def summary(c, name, res):
    r = c.r
    after = r.avulsions[:, 0] > c.teq
    res.setdefault("runs", {})[name] = dict(
        domain=c.domain, L_m=c.L, Qs_m3s=c.p.Qs, plain_cells=int(c.land.sum()),
        avulsions=int(r.num_avulsions), avulsions_after_spinup=int(after.sum()),
        years=float(c.years(r.t)), spinup_years=float(c.years(c.teq)),
        T_fv1_years=c.Tfv, T_fv1_quartiles=[float(q) for q in np.percentile(c.Tfv_starts, [25, 75])],
        T_max_over_T_fv1=float(c.years(r.t) / c.Tfv),
        years_between_avulsions=float(c.years(r.t - c.teq) / after.sum()),
    )


def main():
    load = lambda n: Case(os.path.join(RUNS, n + ".pkl"))
    semi, rect = load("semicircle"), load("rectangle")
    semi_l, rect_l = load("semicircle_large"), load("rectangle_large")
    res = {}
    for n, c in [("semicircle", semi), ("rectangle", rect), ("semicircle_large", semi_l), ("rectangle_large", rect_l)]:
        summary(c, n, res)
    res["boundary"] = dict(
        rectangle=boundary_stats(rect), semicircle=boundary_stats(semi), rectangle_large=boundary_stats(rect_l),
        rt_rect_plain=residence_vs_thickness(rect, rect.region("plain")),
        rt_rect_interior=residence_vs_thickness(rect, rect.region("interior")),
        rt_rect_square=residence_vs_thickness(rect, rect.region("square")),
        rt_semi=residence_vs_thickness(semi, semi.region("plain")),
    )
    # bed relief along sections (mean difference between adjacent local extrema)
    rel = {}
    for c, lab in [(semi, "semicircle"), (rect, "rectangle")]:
        rel[lab] = {}
        for xf in (0.2, 0.5, 0.8):
            cells, _ = c.section_cells(xf)
            s = np.array([c.r.eta[i, j] for i, j in cells])
            ext = [k for k in range(1, len(s) - 1) if (s[k] - s[k - 1]) * (s[k + 1] - s[k]) < 0]
            rel[lab][str(xf)] = float(np.abs(np.diff(s[ext])).mean())
    res["relief_m"] = rel
    fig2(semi, res); fig3(semi, res); fig4(semi, rect, res); fig5(semi, res); fig6(semi, rect, res)
    fig7(semi, res)
    fig8([("semicircle, L = 5 km", semi, "plain", BLUE, "o"),
          ("semicircle, L = 10 km", semi_l, "plain", AQUA, "+"),
          ("rectangle L×L, L = 5 km", rect, "square", ORANGE, "s"),
          ("rectangle L×L, L = 10 km", rect_l, "square", YELLOW, "x")], res)
    fig9(semi, res)
    fig10([("semicircle", semi, BLUE, "o"), ("rectangle", rect, ORANGE, "s")], res)
    fig11([("semicircle, 5 km", semi, BLUE, "-"), ("semicircle, 10 km", semi_l, AQUA, "--"),
           ("rectangle, 5 km", rect, ORANGE, ":")], res)
    fig12([("semicircle, 5 km", semi, BLUE), ("semicircle, 10 km", semi_l, AQUA),
           ("rectangle, 5 km", rect, ORANGE)], res)
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
