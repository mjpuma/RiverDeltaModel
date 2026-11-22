"""Figures 1-6 and S1 of the signatures paper, and every number quoted in its text
(results.json).

    DELTA_RUNS=...  (paper/run_paper.py: semicircle, semicircle_large, rectangle)
    SIG_RUNS=...    (run_robustness.py and null_model.py output)
    python paper/signatures/figures.py

The question: which statistical signatures of a delta's channel come from the timing and
location of avulsions, which from the topography that steers new channels, and which from
geometry alone? The model is compared with two random-walk nulls built from its own
avulsion record: one keeps the distance from the source of every avulsion, the other
places avulsions anywhere along the channel with equal probability.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import figures as F                                                       # noqa: E402
from analysis import Case, boundary_stats, exceedance, residence_vs_thickness  # noqa: E402
from metrics import FRACS, NullCase, stretched, summarize                 # noqa: E402

plt = F.plt
RUNS = os.environ.get("DELTA_RUNS", os.path.join(HERE, "..", "runs"))
SIG = os.environ.get("SIG_RUNS", os.path.join(HERE, "runs"))
F.OUT = os.environ.get("SIG_FIGS", os.path.join(HERE, "figures"))
INK, MUTED, BLUE, ORANGE, AQUA, YELLOW = F.INK, F.MUTED, F.BLUE, F.ORANGE, F.AQUA, F.YELLOW
GREY = "#8c959f"

MODEL, NULL_R, NULL_U = "model", "null, same avulsion sites", "null, sites uniform along the channel"
COL = {MODEL: BLUE, NULL_R: ORANGE, NULL_U: AQUA}
VARIANTS = [("seed2027", "other initial bed (2)"), ("seed2028", "other initial bed (3)"),
            ("beta0.75", "β = 0.75"), ("beta1.5", "β = 1.5"),
            ("floodplain0.25", "floodplain 0.25σ"), ("floodplain0.75", "floodplain 0.75σ")]


def sites_r(c, after_spinup=True):
    """Distance from the source (r/L) of each avulsion."""
    av = c.r.avulsions
    keep = av[:, 0] > c.teq if after_spinup else np.ones(len(av), bool)
    i = av[keep, 1] / c.a - 1
    j = av[keep, 2] / c.a - 1
    return np.hypot(i - c.i0, j) / c.Rc


def path_lengths(c, every=20):
    """Length of the channel from the source to the shoreline, in units of L."""
    out = []
    for t, path in c.paths[::every]:
        if t <= c.teq:
            continue
        a = np.array(path)
        d = np.abs(np.diff(a, axis=0))
        out.append(np.sum(np.where((d[:, 0] > 0) & (d[:, 1] > 0), np.sqrt(2), 1.0)) / c.Rc)
    return np.array(out)


# ======================================================================== figures

def fig1(cases, res):
    c = cases[MODEL]
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 5.6))
    ax = axs[0, 0]
    m = F.cellmap(ax, c, c.r.eta, c.land, cmap="copper")
    x, y = F.path_xy(c, c.paths[-1][1])
    ax.plot(x, y, "-", color="white", lw=1.0)
    F.map_axes(ax, c); F.panel(ax, "A", "bed and channel")
    cb = fig.colorbar(m, ax=ax, shrink=0.7); cb.set_label("Bed elevation [m]")
    t0 = c.teq + 0.5 * (c.r.t - c.teq)
    f = 0.174
    res["fig1"] = dict(window=f, window_years=f * c.Tfv, t0_years=float(c.years(t0)))
    res["Qs_m3s"] = c.p.Qs
    for ax, name, L in zip([axs[0, 1], axs[0, 2], axs[1, 0]], [MODEL, NULL_R, NULL_U], "BCD"):
        cc = cases[name]
        w = cc.window_residence(t0, f * c.Tfv * c.If) / c.Tfv
        im = F.cellmap(ax, cc, np.log10(np.where(w > 0, w, np.nan)), cc.land | cc.shore,
                       cmap=F.SEQ, vmin=-5.0, vmax=-0.5)
        F.map_axes(ax, cc)
        F.panel(ax, L, {MODEL: "model", NULL_R: "null: same sites", NULL_U: "null: uniform sites"}[name])
        res["fig1"][name] = dict(outlets=F.n_outlets(cc, w), visited_fraction=float((w[cc.land] > 0).mean()))
    for ax in axs[0, 1:].tolist() + [axs[1, 0]]:
        ax.set_ylabel("")
    cb = fig.colorbar(im, ax=axs[1, 0], shrink=0.7); cb.set_label(r"log$_{10}(T/T_{f_v=1})$")

    ax = axs[1, 1]
    bins = np.linspace(0, 1, 21)
    for name in (MODEL, NULL_U):
        r = sites_r(cases[name])
        F.step_hist(ax, r, bins, color=COL[name], ls="-" if name == MODEL else ":",
                    label="model (and null, same sites)" if name == MODEL else "null, uniform sites")
        res.setdefault("sites", {})[name] = dict(median_r_over_L=float(np.median(r)),
                                                 share_beyond_half=float((r > 0.5).mean()))
    ax.set_xlabel("Distance of avulsion from source, r / L"); ax.set_ylabel("Probability density")
    ax.legend(fontsize=6.5, loc="upper left"); F.panel(ax, "E", "avulsion sites"); F.style(ax, grid_y=True)

    ax = axs[1, 2]
    bins = np.linspace(1.0, 1.8, 33)
    for name in (MODEL, NULL_R, NULL_U):
        pl = path_lengths(cases[name])
        F.step_hist(ax, pl, bins, color=COL[name], ls={MODEL: "-", NULL_R: "--", NULL_U: ":"}[name],
                    label={MODEL: "model", NULL_R: "null, same sites", NULL_U: "null, uniform sites"}[name])
        res.setdefault("path_length_over_L", {})[name] = float(np.median(pl))
    ax.set_xlabel("Channel length / L"); ax.set_ylabel("Probability density")
    ax.legend(fontsize=6.5, loc="upper right")
    F.panel(ax, "F", "channel length"); F.style(ax, grid_y=True)
    fig.tight_layout()
    F.save(fig, "fig1")


def fig2(cases, summ, curves, res):
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.2))
    ax = axs[0]
    L, A = curves[MODEL]["hack_points"]
    ax.loglog(A / 1e6, L / 1e3, ".", ms=1.5, color=BLUE, alpha=0.35)
    aa = np.logspace(np.log10(A.min()), np.log10(A.max()), 10)
    cf = np.polyfit(np.log10(A), np.log10(L), 1)
    ax.loglog(aa / 1e6, 10 ** np.polyval(cf, np.log10(aa)) / 1e3, color=INK, lw=1.2,
              label=f"model: h = {cf[0]:.2f}")
    k = np.exp(np.mean(np.log(L)) - 0.6 * np.mean(np.log(A)))
    ax.loglog(aa / 1e6, k * aa ** 0.6 / 1e3, "--", color=MUTED, lw=1.0, label="h = 0.6 (natural deltas)")
    ax.set_xlabel(r"Nourishment area, $A$ [km$^2$]"); ax.set_ylabel(r"Longest flow path, $L$ [km]")
    ax.legend(fontsize=7, loc="lower right")
    F.panel(ax, "A", r"model, $\Delta t/T_{f_v=1}$ = 0.174"); F.style(ax)

    ax = axs[1]
    ax.axhspan(0.58, 0.62, color=GREY, alpha=0.18, lw=0)
    ax.text(2.35, 0.6, "natural\ndeltas", fontsize=6.5, color=MUTED, va="center")
    rows = [(MODEL, BLUE, "o", "model"), (NULL_R, ORANGE, "s", "null, same sites"),
            (NULL_U, AQUA, "^", "null, uniform sites"), ("rectangle", GREY, "D", "model, rectangle")]
    for q, (name, col, mk, lab) in enumerate(rows):
        hs = summ[name]["hack"]
        x = np.arange(len(FRACS)) + 0.09 * (q - 1.5)
        y = [hs[str(f)]["h"] for f in FRACS]
        lo = [hs[str(f)]["h"] - hs[str(f)]["h_lo"] for f in FRACS]
        hi = [hs[str(f)]["h_hi"] - hs[str(f)]["h"] for f in FRACS]
        ax.errorbar(x, y, yerr=[lo, hi], fmt=mk + "-", color=col, ms=4, lw=1, capsize=0, label=lab)
    ax.set_xticks(range(len(FRACS)), [str(f) for f in FRACS]); ax.set_xlim(-0.4, 2.8)
    ax.set_xlabel(r"Window $\Delta t / T_{f_v=1}$"); ax.set_ylabel("Hack exponent h")
    ax.legend(fontsize=6.5, loc="upper right", ncol=2); F.panel(ax, "B"); F.style(ax, grid_y=True)
    lo_, hi_ = ax.get_ylim(); ax.set_ylim(lo_, hi_ + 0.35 * (hi_ - lo_))
    fig.tight_layout()
    F.save(fig, "fig2")


def fig3(cases, summ, curves, res):
    c = cases[MODEL]
    fig = plt.figure(figsize=(7.0, 3.4))
    ax = fig.add_axes([0.02, 0.12, 0.40, 0.78])
    Tm = c.mean_residence() / c.Tfv
    im = F.cellmap(ax, c, np.log10(Tm), c.region("plain"), cmap=F.SEQ)
    F.map_axes(ax, c); F.panel(ax, "A", "model: mean residence time"); F.style(ax)
    cb = fig.colorbar(im, ax=ax, shrink=0.8); cb.set_label(r"log$_{10}(\bar T/T_{f_v=1})$")
    ax = fig.add_axes([0.58, 0.15, 0.40, 0.75])
    for name, mk in ((MODEL, "o"), (NULL_R, "s"), (NULL_U, "^")):
        t, P = exceedance(curves[name]["residence"])
        ax.loglog(t, P, mk, ms=2, mfc="none", color=COL[name],
                  label=f"{ {MODEL: 'model', NULL_R: 'null, same sites', NULL_U: 'null, uniform sites'}[name]}"
                        f": χ = {summ[name]['chi']:.2f}")
    ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$P[\bar T/T_{f_v=1} \geq t]$"); ax.set_ylim(1e-4, 1.5)
    ax.legend(fontsize=6.5, loc="lower left"); F.panel(ax, "B"); F.style(ax)
    F.save(fig, "fig3")


def fig4(cases, summ, curves, res):
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.1))
    lab = {MODEL: "model", NULL_R: "null, same sites", NULL_U: "null, uniform sites"}
    ls = {MODEL: "-", NULL_R: "--", NULL_U: ":"}
    for name in (MODEL, NULL_R, NULL_U):
        rw = curves[name]["reworking"]
        s = summ[name]
        t, o = rw["t_overlap"], rw["overlap"]
        ok = t > 0
        axs[0].semilogx(t[ok], o[ok], ls[name], color=COL[name], lw=1.4,
                        label=f"{lab[name]}: β = {s['stretched_beta']:.2f}")
        axs[0].semilogx(t[ok], stretched(t[ok], s["stretched_T"], s["stretched_beta"]), color=COL[name], lw=0.6, alpha=0.7)
        Tfv = cases[MODEL].Tfv
        axs[1].semilogy(rw["t_unreworked"] / Tfv, rw["unreworked"], ls[name], color=COL[name], lw=1.4,
                        label=f"{lab[name]}: {s['T_R_over_Tfv']:.2f}")
    axs[0].set_xlabel(r"$\Delta t$ [yr]"); axs[0].set_ylabel("Channel overlap (normalized)")
    axs[0].legend(fontsize=6.5, loc="lower left"); F.panel(axs[0], "A", "channel overlap"); F.style(axs[0], grid_y=True)
    axs[1].set_ylim(1e-3, 1.05); axs[1].set_xlim(0, 1.5)
    axs[1].set_xlabel(r"$\Delta t/T_{f_v=1}$ (model)"); axs[1].set_ylabel("Fraction of plain not yet visited")
    axs[1].legend(fontsize=6.5, loc="upper right", title=r"e-folding time / $T_{f_v=1}$", title_fontsize=6.5)
    F.panel(axs[1], "B", "reworking"); F.style(axs[1], grid_y=True)
    fig.tight_layout()
    F.save(fig, "fig4")


def fig5(cases, summ, curves, res):
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    names = [MODEL] + [k for k, _ in VARIANTS if k.startswith("seed")]
    for q, name in enumerate(names):
        T, s, Tc = curves[name]["compensation"]
        ax.loglog(T / Tc, s, "o-", ms=2.5, lw=0.9, color=BLUE, alpha=1.0 if q == 0 else 0.45,
                  label=f"model: κ = {summ[MODEL]['kappa_below']:.2f} below, "
                        f"{summ[MODEL]['kappa_above']:.2f} above $T_c$" if q == 0 else
                  ("other initial beds" if q == 1 else None))
    T, s, Tc = curves[MODEL]["compensation"]
    x = np.array([T.min() / Tc, T.max() / Tc])
    y0 = np.interp(1.0, T / Tc, s)
    ax.loglog(x, y0 * x ** -0.5, ":", color=MUTED, lw=0.9)
    ax.loglog(x, y0 * x ** -1.0, "--", color=MUTED, lw=0.9)
    ax.text(x[-1] * 1.1, y0 * x[-1] ** -0.5, "κ = 0.5", color=MUTED, fontsize=7, va="center")
    ax.text(x[-1] * 1.1, y0 * x[-1] ** -1.0, "κ = 1", color=MUTED, fontsize=7, va="center")
    ax.axvline(1.0, color=MUTED, lw=0.6)
    ax.set_xlabel(r"$T/T_c$,  $T_c = \bar H/\sigma$"); ax.set_ylabel(r"$\sigma_{ss}$")
    ax.legend(fontsize=6.5, loc="lower left"); F.style(ax)
    F.save(fig, "fig5")


def fig6(summ, res):
    """Robustness: each statistic for the model variants, with the nulls for reference."""
    rows = [(MODEL, "base")] + VARIANTS + [("semicircle_large", "L = 10 km")]
    stats_ = [("h", "h at 0.174", lambda s: s["hack"]["0.174"]["h"]),
              ("chi", "χ", lambda s: s["chi"]),
              ("beta", "overlap β", lambda s: s["stretched_beta"]),
              ("TR", r"$T_R/T_{f_v=1}$", lambda s: s["T_R_over_Tfv"]),
              ("kappa", "κ below $T_c$", lambda s: s.get("kappa_below", np.nan))]
    fig, axs = plt.subplots(1, len(stats_), figsize=(7.2, 3.0), sharey=True)
    y = np.arange(len(rows))[::-1]
    nulls = {MODEL: ([NULL_R, NULL_R + " (2)"], [NULL_U, NULL_U + " (2)"]),
             "semicircle_large": (["null large, same sites"], ["null large, uniform sites"])}
    for ax, (key, title, fn) in zip(axs, stats_):
        vals = [fn(summ[k]) for k, _ in rows]
        ax.plot(vals, y, "o", color=BLUE, ms=4, zorder=3)
        for (k, _), yy in zip(rows, y):
            for names, col, mk in zip(nulls.get(k, ()), (ORANGE, AQUA), ("s", "^")):
                v = [fn(summ[n]) for n in names]
                if np.all(np.isfinite(v)):
                    ax.plot(v, [yy] * len(v), mk, color=col, ms=4, mfc="none")
        ax.set_title(title, fontsize=8, color=INK)
        F.style(ax)
        ax.grid(axis="x", color=F.GRID, lw=0.5)
        res.setdefault("robustness", {})[key] = dict(min=float(np.nanmin(vals)), max=float(np.nanmax(vals)))
    axs[0].set_yticks(y, [lab for _, lab in rows], fontsize=7)
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], color=BLUE, marker="o", ls=""), Line2D([], [], color=ORANGE, marker="s", mfc="none", ls=""),
                Line2D([], [], color=AQUA, marker="^", mfc="none", ls="")], ["model", "null, same sites", "null, uniform sites"],
               loc="lower center", ncol=3, fontsize=7, frameon=False, bbox_to_anchor=(0.55, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    F.save(fig, "fig6")


def figS1(semi, rect, res):
    """Supplementary: the side walls of a rectangular domain (Fig. 4 of the revised preprint)."""
    sub = {"boundary": dict(
        rectangle=boundary_stats(rect), semicircle=boundary_stats(semi),
        rt_rect_plain=residence_vs_thickness(rect, rect.region("plain")),
        rt_rect_square=residence_vs_thickness(rect, rect.region("square")),
        rt_semi=residence_vs_thickness(semi, semi.region("plain")))}
    F.fig4(semi, rect, sub)
    os.replace(os.path.join(F.OUT, "fig4.png"), os.path.join(F.OUT, "figS1.png"))
    os.replace(os.path.join(F.OUT, "fig4.pdf"), os.path.join(F.OUT, "figS1.pdf"))
    res["boundary"] = sub["boundary"]


# ======================================================================== main

def main():
    res, summ, curves, cases = {}, {}, {}, {}

    def add(name, c, comp=True):
        s, cv = summarize(c, with_compensation=comp)
        summ[name], curves[name] = s, cv
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items() if k != "hack"},
              {f: round(v["h"], 3) for f, v in s["hack"].items()}, flush=True)

    import pickle
    cache = os.path.join(F.OUT, "summaries.pkl")
    base = Case(os.path.join(RUNS, "semicircle.pkl"))
    cases[MODEL] = base
    for name, fn in ((NULL_R, "null_semicircle_s1.pkl"), (NULL_U, "null_uniform_semicircle_s1.pkl")):
        cases[name] = NullCase(os.path.join(SIG, fn), base.Tfv)
    rect = Case(os.path.join(RUNS, "rectangle.pkl"))
    if os.path.exists(cache) and os.environ.get("SIG_REUSE"):
        summ, curves = pickle.load(open(cache, "rb"))
    else:
        compute_all(base, cases, rect, add)
        os.makedirs(F.OUT, exist_ok=True)
        pickle.dump((summ, curves), open(cache, "wb"))
    render(base, cases, rect, summ, curves, res)


def compute_all(base, cases, rect, add):
    add(MODEL, base)
    for name in (NULL_R, NULL_U):
        add(name, cases[name], comp=False)
    for name, fn in ((NULL_R + " (2)", "null_semicircle_s2.pkl"), (NULL_U + " (2)", "null_uniform_semicircle_s2.pkl")):
        add(name, NullCase(os.path.join(SIG, fn), base.Tfv), comp=False)
    for k, _ in VARIANTS:
        add(k, Case(os.path.join(SIG, f"semicircle_{k}.pkl")))
    big = Case(os.path.join(RUNS, "semicircle_large.pkl"))
    add("semicircle_large", big, comp=False)
    for name, fn in (("null large, same sites", "null_semicircle_large_s1.pkl"),
                     ("null large, uniform sites", "null_uniform_semicircle_large_s1.pkl")):
        add(name, NullCase(os.path.join(SIG, fn), big.Tfv), comp=False)
    del big
    add("rectangle", rect, comp=False)


def render(base, cases, rect, summ, curves, res):
    figS1(base, rect, res)                   # first: it reuses the name fig4 for a moment
    fig1(cases, res)
    fig2(cases, summ, curves, res)
    fig3(cases, summ, curves, res)
    fig4(cases, summ, curves, res)
    fig5(cases, summ, curves, res)
    fig6(summ, res)
    res["cases"] = summ
    m, nr, nu = summ[MODEL], summ[NULL_R], summ[NULL_U]
    seeds = [summ[k] for k in (MODEL, "seed2027", "seed2028")]
    res["derived"] = dict(
        reworking_ratio_same=nr["T_R_over_Tfv"] / m["T_R_over_Tfv"],
        reworking_ratio_uniform=nu["T_R_over_Tfv"] / m["T_R_over_Tfv"],
        seeds=dict((k, dict(min=float(min(f(x) for x in seeds)), max=float(max(f(x) for x in seeds))))
                   for k, f in (("h", lambda x: x["hack"]["0.174"]["h"]), ("chi", lambda x: x["chi"]),
                                ("beta", lambda x: x["stretched_beta"]), ("TR", lambda x: x["T_R_over_Tfv"]),
                                ("kappa", lambda x: x["kappa_below"]), ("kappa_above", lambda x: x["kappa_above"]),
                                ("Tfv", lambda x: x["T_fv1_years"]))),
        large=dict(TR_model=summ["semicircle_large"]["T_R_over_Tfv"],
                   TR_same=summ["null large, same sites"]["T_R_over_Tfv"],
                   TR_uniform=summ["null large, uniform sites"]["T_R_over_Tfv"],
                   ratio_same=summ["null large, same sites"]["T_R_over_Tfv"] / summ["semicircle_large"]["T_R_over_Tfv"],
                   h_model=summ["semicircle_large"]["hack"]["0.174"]["h"],
                   h_same=summ["null large, same sites"]["hack"]["0.174"]["h"],
                   h_uniform=summ["null large, uniform sites"]["hack"]["0.174"]["h"],
                   chi_model=summ["semicircle_large"]["chi"], chi_same=summ["null large, same sites"]["chi"],
                   chi_uniform=summ["null large, uniform sites"]["chi"]),
        kappa_above_all=dict(min=float(min(summ[k]["kappa_above"] for k in [MODEL] + [v for v, _ in VARIANTS])),
                             max=float(max(summ[k]["kappa_above"] for k in [MODEL] + [v for v, _ in VARIANTS]))),
        nulls=dict((k, dict(min=float(min(f(summ[n]) for n in names)), max=float(max(f(summ[n]) for n in names))))
                   for names, tag in (((NULL_R, NULL_R + " (2)"), "same"), ((NULL_U, NULL_U + " (2)"), "uniform"))
                   for k, f in ((f"h_{tag}", lambda x: x["hack"]["0.174"]["h"]), (f"chi_{tag}", lambda x: x["chi"]),
                                (f"beta_{tag}", lambda x: x["stretched_beta"]), (f"TR_{tag}", lambda x: x["T_R_over_Tfv"]))),
    )
    with open(os.path.join(F.OUT, "results.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print("wrote", F.OUT)


if __name__ == "__main__":
    main()
