"""Figures 1-4 of the short paper on Hack-like scaling in deltas, and the numbers it quotes
(added to paper/signatures/figures/results.json under "short").

    DELTA_RUNS=...  SIG_RUNS=...  SIG_CACHE=.../summaries.pkl  REAL_JSON=.../real_deltas.json \
    DONG_BREAKS=.../Break4.csv (Dong et al. 2026 code archive) \
    EXTRA_JSON=a.json,b.json (real_deltas.py --extra) SPREAD_JSON=... (real_deltas.py --spread) DIDNT_DIR=... DONG_CSV=... OUTLETS_DIR=... \
        python paper/signatures/short_figures.py

Uses the statistics cached by figures.py (summaries.pkl) and the output of real_deltas.py.
"""
import json
import os
import pickle
import sys

import matplotlib.ticker as mt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import figures as F                                                   # noqa: E402  (paper/figures.py)
from analysis import Case                                             # noqa: E402
from metrics import FRACS, NullCase                                   # noqa: E402
from real_deltas import break_fit                                     # noqa: E402

plt = F.plt
RUNS = os.environ.get("DELTA_RUNS", os.path.join(HERE, "..", "runs"))
SIG = os.environ.get("SIG_RUNS", os.path.join(HERE, "runs"))
OUT = os.environ.get("SHORT_FIGS", os.path.join(HERE, "figures", "short"))
F.OUT = OUT
MODEL, NULL_R, NULL_U = "model", "null, same avulsion sites", "null, sites uniform along the channel"
COL = {MODEL: F.BLUE, NULL_R: F.ORANGE, NULL_U: F.AQUA}
ORANGE_, GREY_, YELLOW_, AQUA_, AQUA_L = F.ORANGE, "#8c959f", F.YELLOW, F.AQUA, "hatch"
ORANGE_D, YELLOW_D = F.ORANGE, F.YELLOW
LAB = {MODEL: "model", NULL_R: "random walk, model's avulsion sites", NULL_U: "random walk, uniform avulsion sites"}


def fig1(cases, curves, res):
    """Networks traced by the channel in a window, and L against A for the model."""
    fig = plt.figure(figsize=(7.2, 5.6))
    c = cases[MODEL]
    t0 = c.teq + 0.5 * (c.r.t - c.teq)
    f = 0.174
    for k, name in enumerate((MODEL, NULL_R, NULL_U)):
        ax = fig.add_axes([0.01 + 0.33 * k, 0.50, 0.30, 0.46])
        cc = cases[name]
        w = cc.window_residence(t0, f * c.Tfv * c.If) / c.Tfv
        im = F.cellmap(ax, cc, np.log10(np.where(w > 0, w, np.nan)), cc.land | cc.shore,
                       cmap=F.SEQ, vmin=-5.0, vmax=-0.5)
        F.map_axes(ax, cc)
        ax.set_xlabel(""); ax.set_ylabel(""); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        F.panel(ax, "ABC"[k], {MODEL: "model", NULL_R: "random walk, same sites", NULL_U: "random walk, uniform sites"}[name])
    cax = fig.add_axes([0.30, 0.47, 0.40, 0.015])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label(r"log$_{10}$ of occupation time / $T_{f_v=1}$", fontsize=7)
    cb.ax.tick_params(labelsize=7)
    ax = fig.add_axes([0.10, 0.07, 0.38, 0.30])
    L, A = curves[MODEL]["hack_points"]
    ax.loglog(A / 1e6, L / 1e3, ".", ms=1.2, color=F.BLUE, alpha=0.3)
    aa = np.logspace(np.log10(A.min()), np.log10(A.max()), 10)
    cf = np.polyfit(np.log10(A), np.log10(L), 1)
    ax.loglog(aa / 1e6, 10 ** np.polyval(cf, np.log10(aa)) / 1e3, color=F.INK, lw=1.2, label=f"fit: h = {cf[0]:.2f}")
    kk = np.exp(np.mean(np.log(L)) - 0.6 * np.mean(np.log(A)))
    ax.loglog(aa / 1e6, kk * aa ** 0.6 / 1e3, "--", color=F.MUTED, lw=1.0, label="h = 0.6")
    ax.set_xlabel(r"Nourishment area $A$ [km$^2$]"); ax.set_ylabel(r"Longest flow path $L$ [km]")
    ax.legend(fontsize=7, loc="lower right"); F.panel(ax, "D", "model"); F.style(ax)
    ax = fig.add_axes([0.60, 0.07, 0.38, 0.30])
    for name, mk in ((MODEL, "o"), (NULL_R, "s"), (NULL_U, "^")):
        L, A = curves[name]["hack_points"]
        b = np.logspace(np.log10(A.min()), np.log10(A.max()), 14)
        k = np.digitize(A, b)
        x = [np.exp(np.mean(np.log(A[k == i]))) for i in range(1, len(b)) if (k == i).sum() > 5]
        y = [np.exp(np.mean(np.log(L[k == i]))) for i in range(1, len(b)) if (k == i).sum() > 5]
        ax.loglog(np.array(x) / 1e6, np.array(y) / 1e3, mk + "-", ms=3, lw=1, color=COL[name], mfc="none", label=LAB[name])
    ax.set_xlabel(r"Nourishment area $A$ [km$^2$]"); ax.set_ylabel(r"Mean $L$ [km]")
    ax.yaxis.set_minor_formatter(mt.NullFormatter())
    ax.legend(fontsize=6, loc="lower right"); F.panel(ax, "E", "binned means"); F.style(ax)
    F.save(fig, "fig1")


def fig2(summ, res):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw=dict(width_ratios=[1.1, 1]))
    ax = axs[0]
    ax.axhspan(0.58, 0.62, color="#8c959f", alpha=0.18, lw=0)
    rows = [(MODEL, F.BLUE, "o", "model"), (NULL_R, F.ORANGE, "s", "random walk, same sites"),
            (NULL_U, F.AQUA, "^", "random walk, uniform sites"), ("rectangle", "#8c959f", "D", "model, walled domain")]
    for q, (name, col, mk, lab) in enumerate(rows):
        hs = summ[name]["hack"]
        x = np.arange(len(FRACS)) + 0.09 * (q - 1.5)
        y = [hs[str(f)]["h"] for f in FRACS]
        lo = [hs[str(f)]["h"] - hs[str(f)]["h_lo"] for f in FRACS]
        hi = [hs[str(f)]["h_hi"] - hs[str(f)]["h"] for f in FRACS]
        ax.errorbar(x, y, yerr=[lo, hi], fmt=mk + "-", color=col, ms=4, lw=1, label=lab)
    ax.set_xticks(range(len(FRACS)), [str(f) for f in FRACS]); ax.set_xlim(-0.4, 2.4)
    ax.set_xlabel(r"Window $\Delta t / T_{f_v=1}$ (denser network $\rightarrow$)"); ax.set_ylabel("Hack exponent h")
    lo_, hi_ = ax.get_ylim(); ax.set_ylim(lo_, hi_ + 0.6 * (hi_ - lo_))
    ax.legend(fontsize=6.5, loc="upper right", ncol=2); F.panel(ax, "A"); F.style(ax, grid_y=True)
    ax = axs[1]
    rows = [(MODEL, "base"), ("seed2027", "initial bed 2"), ("seed2028", "initial bed 3"), ("beta0.75", "β = 0.75"),
            ("beta1.5", "β = 1.5"), ("floodplain0.25", "floodplain 0.25σ"), ("floodplain0.75", "floodplain 0.75σ"),
            ("semicircle_large", "L = 10 km")]
    nulls = {MODEL: ([NULL_R, NULL_R + " (2)"], [NULL_U, NULL_U + " (2)"]),
             "semicircle_large": (["null large, same sites"], ["null large, uniform sites"])}
    y = np.arange(len(rows))[::-1]
    for (k, lab), yy in zip(rows, y):
        ax.plot(summ[k]["hack"]["0.174"]["h"], yy, "o", color=F.BLUE, ms=4, zorder=3)
        for names, col, mk in zip(nulls.get(k, ()), (F.ORANGE, F.AQUA), ("s", "^")):
            ax.plot([summ[n]["hack"]["0.174"]["h"] for n in names], [yy] * len(names), mk, color=col, ms=4, mfc="none")
    ax.axvspan(0.58, 0.62, color="#8c959f", alpha=0.18, lw=0)
    ax.set_yticks(y, [lab for _, lab in rows], fontsize=7)
    ax.set_xlabel(r"h, window 0.174 $T_{f_v=1}$"); F.panel(ax, "B"); F.style(ax)
    ax.grid(axis="x", color=F.GRID, lw=0.5)
    fig.tight_layout()
    F.save(fig, "fig2")
    vals = [summ[k]["hack"]["0.174"]["h"] for k, _ in rows]
    nul = [summ[n]["hack"]["0.174"]["h"] for v in nulls.values() for ns in v for n in ns]
    res["h_model_variants"] = dict(min=float(min(vals)), max=float(max(vals)))
    res["h_nulls"] = dict(min=float(min(nul)), max=float(max(nul)))
    res["h_window"] = {name: {str(f): summ[name]["hack"][str(f)]["h"] for f in FRACS} for name, *_ in [(MODEL,), (NULL_R,), (NULL_U,), ("rectangle",)]}


def fig4(curves, real, res, delta="Colville"):
    """Scale breaks: one real delta and a random network in its outline, and h near the
    coast (small A) and near the apex (large A) for every delta, its random networks, and
    the model."""
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.9), gridspec_kw=dict(width_ratios=[1, 1, 1.3]))
    v = real[delta]
    B = v["B"]
    Lm, Am = curves[MODEL]["hack_points"]
    for ax, (L, A, BB), col, lab, L_ in ((axs[0], (v["real_points"][0], v["real_points"][1], B), F.BLUE, f"{delta}, mapped", "A"),
                                         (axs[1], (Lm, Am, 100.0), F.BLUE, "model (B = cell size)", "B")):
        B = BB
        L, A = np.array(L) / B, np.array(A) / B ** 2
        ax.loglog(A, L, "o", ms=2.0 if L_ == "A" else 1.0, color=col, mfc="none" if L_ == "A" else col, alpha=0.7 if L_ == "A" else 0.25)
        bf = break_fit(L * B, A * B ** 2, B)
        if np.isfinite(bf["break_L_over_B"]):
            lb = bf["break_L_over_B"]
            for sel, h in ((L < lb, bf["h1"]), (L >= lb, bf["h2"])):
                if sel.sum() > 2:
                    c = np.polyfit(np.log(A[sel]), np.log(L[sel]), 1)
                    xx = np.array([A[sel].min(), A[sel].max()])
                    ax.loglog(xx, np.exp(np.polyval(c, np.log(xx))), color=F.INK, lw=1)
            ax.axhline(lb, color=F.MUTED, lw=0.6, ls=":")
            ax.text(0.04, 0.92, f"h = {bf['h1']:.2f} below, {bf['h2']:.2f} above\nthe break (L/B = {lb:.0f})",
                    transform=ax.transAxes, fontsize=6.5, va="top")
        ax.set_xlabel(r"$A/B^2$"); ax.set_ylabel(r"$L/B$")
        ax.xaxis.set_major_locator(mt.LogLocator(numticks=5)); ax.xaxis.set_minor_formatter(mt.NullFormatter())
        ax.yaxis.set_major_locator(mt.LogLocator(numticks=5)); ax.yaxis.set_minor_formatter(mt.NullFormatter())
        F.panel(ax, L_, lab); F.style(ax)
    ax = axs[2]
    import pandas as pd
    pub = pd.read_csv(os.environ["DONG_BREAKS"]).set_index("name")      # Dong et al. (2026), Data S1
    pub.index = [{"Wax": "Wax"}.get(i, i) for i in pub.index]
    names = sorted(real, key=lambda n: (np.isnan(pub.loc[n, "h1"]), n))
    out = {}
    for k, n in enumerate(names):
        nd = real[n]["nulls"]["dense"]
        h1n = np.nanmedian([x["h1"] for x in nd]); h2n = np.nanmedian([x["h2"] for x in nd])
        if np.isfinite(pub.loc[n, "h1"]):                      # composite: two exponents
            h1, h2 = pub.loc[n, "h1"], pub.loc[n, "h2"]
            ax.plot([k - 0.12] * 2, [h1, h2], "-", color=F.BLUE, lw=0.8)
            ax.plot(k - 0.12, h1, "^", color=F.BLUE, ms=4); ax.plot(k - 0.12, h2, "v", color=F.BLUE, ms=4)
        else:                                                  # uniform: one exponent
            h1 = h2 = np.nan
            ax.plot(k - 0.12, pub.loc[n, "h"], "o", color=F.BLUE, ms=4)
        ax.plot([k + 0.12] * 2, [h1n, h2n], "-", color=F.ORANGE, lw=0.8)
        ax.plot(k + 0.12, h1n, "^", color=F.ORANGE, ms=4, mfc="none"); ax.plot(k + 0.12, h2n, "v", color=F.ORANGE, ms=4, mfc="none")
        out[n] = dict(dong_h1=float(h1), dong_h2=float(h2), dong_h=float(pub.loc[n, "h"]),
                      dong_break=float(pub.loc[n, "break"]) if np.isfinite(pub.loc[n, "break"]) else np.nan,
                      random_h1=float(h1n), random_h2=float(h2n),
                      random_break=float(np.nanmedian([x["break_L_over_B"] for x in nd])),
                      random_break_share=float(np.mean([x["davies_p"] < 0.05 for x in nd])),
                      real_raster_p=real[n]["real_raster"]["davies_p"])
    # model and nulls
    k0 = len(names) + 0.5
    for j, (name, col) in enumerate(((MODEL, F.BLUE), (NULL_R, F.ORANGE), (NULL_U, F.AQUA))):
        L, A = curves[name]["hack_points"]
        bf = break_fit(L, A, 100.0)
        x = k0 + 0.35 * j
        ax.plot([x] * 2, [bf["h1"], bf["h2"]], "-", color=col, lw=0.8)
        ax.plot(x, bf["h1"], "^", color=col, ms=4, mfc="none" if j else col)
        ax.plot(x, bf["h2"], "v", color=col, ms=4, mfc="none" if j else col)
        out[name] = bf
    ax.axvline(len(names) + 0.15, color=F.GRID, lw=0.8)
    ax.set_xticks(list(range(len(names))) + [k0 + 0.35], names + ["model,\nnulls"], fontsize=6, rotation=90)
    ax.set_ylabel("h near coast (▲), near apex (▼)"); ax.set_ylim(0, 1.6)
    ax.plot([], [], "^", color=F.BLUE, label="mapped (Dong et al. 2026)")
    ax.plot([], [], "o", color=F.BLUE, label="mapped, no break (Dong et al.)")
    ax.plot([], [], "^", color=F.ORANGE, mfc="none", label="random networks (median)")
    ax.legend(fontsize=6, loc="upper left"); F.panel(ax, "C"); F.style(ax, grid_y=True)
    fig.tight_layout()
    F.save(fig, "fig4")
    res["breaks"] = out
    dl = [n for n in names]
    res["breaks_summary"] = dict(
        random_h1_min=float(min(out[n]["random_h1"] for n in dl)), random_h1_max=float(max(out[n]["random_h1"] for n in dl)),
        random_h2_min=float(min(out[n]["random_h2"] for n in dl)), random_h2_max=float(max(out[n]["random_h2"] for n in dl)),
        random_share_min=float(min(out[n]["random_break_share"] for n in dl)),
        random_share_max=float(max(out[n]["random_break_share"] for n in dl)),
        n_composite=int(sum(np.isfinite(out[n]["dong_h1"]) for n in dl)),
        real_raster_breaks=int(sum(out[n]["real_raster_p"] < 0.05 for n in dl)))



KINDS = [("dense", "random, with loops", ORANGE_), ("tree", "random trees", GREY_),
         ("tree_radial", "random trees, radial", YELLOW_), ("laplace1", "Laplace trees, η = 1", AQUA_L),
         ("laplace2", "Laplace trees, η = 2", AQUA_)]


def fig3(real, extra, spread, res, map_delta="Mossy", seed=3):
    """Real deltas against the comparison networks grown in their outlines."""
    import real_deltas as R
    from scipy import ndimage
    from scipy.stats import binomtest
    for n in real:
        real[n]["nulls"].update(extra[n]["nulls"])
    order = sorted(real, key=lambda n: real[n]["real_raster"]["h"])
    fig = plt.figure(figsize=(7.2, 6.4))
    ax = fig.add_axes([0.09, 0.47, 0.89, 0.49])
    w = 0.15
    for k, n in enumerate(order):
        v = real[n]
        for j, (kind, lab, col) in enumerate(KINDS):
            hn = [x["h"] for x in v["nulls"][kind] if "h" in x]
            x = k + (j - 2) * w
            if not hn:
                continue
            bp = dict(facecolor=col, edgecolor=F.INK, lw=0.5) if col != "hatch" else \
                dict(facecolor="white", edgecolor=AQUA_, hatch="//////", lw=0.8)
            ax.boxplot(hn, positions=[x], widths=0.12, showfliers=False, patch_artist=True,
                       boxprops=bp, medianprops=dict(color=F.INK, lw=0.8),
                       whiskerprops=dict(color=F.INK, lw=0.5), capprops=dict(color=F.INK, lw=0.5))
        ax.plot([k - 0.42, k + 0.42], [v["real_raster"]["h"]] * 2, color=F.BLUE, lw=1.6, zorder=3)
        ax.plot(k + 0.42, v["dong"]["h"], "D", color=F.INK, mfc="none", ms=4, zorder=3)
    from matplotlib.patches import Patch
    handles = [Patch(facecolor="white", edgecolor=AQUA_, hatch="//////", label=lab) if col == "hatch"
               else Patch(facecolor=col, edgecolor=F.INK, lw=0.5, label=lab) for kind, lab, col in KINDS]
    ax.plot([], [], "-", color=F.BLUE, lw=1.6, label="mapped network")
    ax.plot([], [], "D", color=F.INK, mfc="none", ms=4, label="mapped, Dong et al. (2026)")
    ax.set_xticks(range(len(order)), [("Wax Lake" if n == "Wax" else n) for n in order], fontsize=7.5)
    ax.set_xlim(-0.55, len(order) - 0.45)
    ax.set_ylabel("Hack exponent h"); ax.set_ylim(0.15, 1.3)
    hl, ll = ax.get_legend_handles_labels()
    ax.legend(handles + hl, [h.get_label() for h in handles] + ll, fontsize=6.3, loc="upper left", ncol=4, frameon=True,
              framealpha=1, edgecolor="none", facecolor="white").set_zorder(5)
    F.panel(ax, "a"); F.style(ax, grid_y=True)

    # maps of one delta
    D = R.Delta(map_delta, os.environ["DIDNT_DIR"], pd.read_csv(os.environ["DONG_CSV"]), os.environ["OUTLETS_DIR"])
    rs = R.real_succ(D)
    nb = R.n_branches(rs)
    target = R.n_outlets(D, set(rs) | {v for x in rs.values() for v in x})
    rng = np.random.default_rng(seed)
    nets = [("mapped", None, F.BLUE),
            ("random, loops", R.tree_network(D, rng, 0, join=True, target_branches=nb), ORANGE_D),
            ("random tree", R.tree_network(D, rng, target), F.MUTED)]
    with R.field(D, "radial"):
        nets.append(("radial tree", R.tree_network(D, rng, target), YELLOW_D))
    nets.append(("Laplace, η = 2", [[tuple(int(v) for v in c) for c in p] for p in extra[map_delta]["example_paths"]["laplace2"]], AQUA_))
    for k, (lab, P, col) in enumerate(nets):
        a = fig.add_axes([0.003 + k * 0.152, 0.01, 0.148, 0.36])
        img = np.full(D.chan.shape + (3,), 1.0)
        img[D.domain] = (0.92, 0.92, 0.92)
        nm = np.zeros(D.chan.shape, bool)
        if P is None:
            nm = D.chan & D.domain
        else:
            for p in P:
                for c in p:
                    nm[c] = True
        rgb = tuple(int(col[i:i + 2], 16) / 255 for i in (1, 3, 5))
        img[ndimage.binary_dilation(nm) & D.domain] = rgb
        a.imshow(np.transpose(img, (1, 0, 2)), origin="lower", interpolation="nearest")
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
        F.panel(a, "bcdef"[k], lab)
        a._left_title.set_fontsize(7)
    # h against spread
    ag = fig.add_axes([0.80, 0.07, 0.185, 0.28])
    scol = {"dense": ORANGE_, "tree": GREY_, "tree_radial": YELLOW_, "laplace2": AQUA_, "mapped": F.BLUE}
    xs, ys = [], []
    for kind in ("tree", "tree_radial", "dense", "laplace2", "mapped"):
        pts = [(x["spread"], x["h"]) for n in spread for x in spread[n][kind]]
        if not pts:
            continue
        pts = np.array(pts)
        ag.plot(pts[:, 0], pts[:, 1], "o", ms=3.2 if kind == "mapped" else 2.4, color=scol[kind],
                mec=F.INK if kind == "mapped" else "none", mew=0.4, alpha=1 if kind == "mapped" else 0.8, zorder=3 if kind == "mapped" else 2)
        xs += list(pts[:, 0]); ys += list(pts[:, 1])
    ag.set_xscale("log")
    ag.xaxis.set_major_locator(mt.FixedLocator([1.5, 3, 6, 12]))
    ag.xaxis.set_major_formatter(mt.FixedFormatter(["1.5", "3", "6", "12"]))
    ag.xaxis.set_minor_locator(mt.NullLocator())
    ag.set_xlabel("spread of channels", fontsize=7); ag.set_ylabel("h", fontsize=7)
    ag.tick_params(labelsize=6)
    F.panel(ag, "g"); F.style(ag, grid_y=True)
    F.save(fig, "fig3")
    from scipy.stats import spearmanr
    res["spread"] = dict(rho=float(spearmanr(xs, ys)[0]), n=len(xs),
                         median={k: float(np.median([x["spread"] for n in spread for x in spread[n][k]]))
                                 for k in ("mapped", "dense", "tree", "tree_radial", "laplace2")},
                         h_median={k: float(np.median([x["h"] for n in spread for x in spread[n][k]]))
                                   for k in ("mapped", "dense", "tree", "tree_radial", "laplace2")})
    # range over deltas of each kind's median spread, and in how many deltas the mapped
    # network is closer (in log spread) to the Laplace trees than to the random networks
    rng_ = {}
    for k in ("mapped", "dense", "tree", "tree_radial", "laplace2"):
        m = [np.median([x["spread"] for x in spread[n][k]]) for n in spread if spread[n][k]]
        rng_[k] = dict(min=float(min(m)), max=float(max(m)))
    res["spread"]["range"] = rng_
    closer = 0
    below = {"dense": 0, "tree": 0, "laplace2": 0}
    for n in spread:
        sm = spread[n]["mapped"][0]["spread"]
        md = {k: np.median([x["spread"] for x in spread[n][k]]) for k in ("dense", "tree", "laplace2") if spread[n][k]}
        closer += abs(np.log(sm / md["laplace2"])) < min(abs(np.log(sm / md["dense"])), abs(np.log(sm / md["tree"])))
        for k in below:
            below[k] += sm < min(x["spread"] for x in spread[n][k])
    res["spread"]["closer_to_laplace"] = int(closer)
    res["spread"]["mapped_below_all"] = below
    res["spread"]["n_deltas"] = len(spread)

    # summary
    out = {}
    for kind, lab, col in KINDS + [("dense_radial", "", None), ("downstream", "", None)]:
        meds, above, allh, ok, nnull, outl = {}, 0, [], 0, 0, []
        for n in real:
            hn = [x["h"] for x in real[n]["nulls"][kind] if "h" in x]
            nnull += len(real[n]["nulls"][kind])
            ok += len(hn)
            outl += [x.get("outlets", np.nan) for x in real[n]["nulls"][kind]]
            if hn:
                meds[n] = float(np.median(hn))
                above += real[n]["real_raster"]["h"] > meds[n]
                allh += hn
        e = dict(median=float(np.median(allh)), q25=float(np.percentile(allh, 25)), q75=float(np.percentile(allh, 75)),
                 above=int(above), n_deltas=len(meds), valid=int(ok), total=int(nnull),
                 p=float(binomtest(int(above), len(meds), 0.5).pvalue),
                 med_min=float(min(meds.values())), med_max=float(max(meds.values())), per_delta=meds)
        # deltas where the real exponent lies within the 5-95% range of the networks
        inside = 0
        for n in real:
            hn = [x["h"] for x in real[n]["nulls"][kind] if "h" in x]
            if hn and np.percentile(hn, 5) <= real[n]["real_raster"]["h"] <= np.percentile(hn, 95):
                inside += 1
        e["inside_5_95"] = inside
        recs = [x for n in real for x in real[n]["nulls"][kind] if "h" in x]
        e["break_share"] = float(np.mean([x["davies_p"] < 0.05 for x in recs]))
        e["h1_median"] = float(np.nanmedian([x["h1"] for x in recs if x["davies_p"] < 0.05]))
        e["h2_median"] = float(np.nanmedian([x["h2"] for x in recs if x["davies_p"] < 0.05]))
        e["outlets_median"] = float(np.nanmedian(outl))
        if kind == "tree_radial":
            short = [n for n in real if np.nanmedian([x.get("outlets", np.nan) for x in real[n]["nulls"][kind]])
                     < 0.9 * extra[n]["target_outlets"]]
            e["outlets_short_n"] = len(short)
            nm = [("Wax Lake" if n == "Wax" else n) for n in sorted(short)]
            e["outlets_short_list"] = ", ".join(nm[:-1]) + " and " + nm[-1] if len(nm) > 1 else "".join(nm)
            e["per_delta_str"] = {n: ("–" if (n in short or n not in meds) else f"{meds[n]:.2f}") for n in real}
        out[kind] = e
    hr = np.array([real[n]["real_raster"]["h"] for n in real])
    for kind in ("dense", "tree", "laplace1", "laplace2", "tree_radial"):
        m = np.array([out[kind]["per_delta"].get(n, np.nan) for n in real])
        out[kind]["mean_abs_diff"] = float(np.nanmean(np.abs(m - hr)))
    for e in out.values():
        e["real_below"] = e["n_deltas"] - e["above"]
    out["laplace_eta_diff"] = float(out["laplace2"]["median"] - out["laplace1"]["median"])
    res["fields"] = out
    ok = [n for n in real if n not in out["tree_radial"]["outlets_short_list"].replace("Wax Lake", "Wax")]
    d = [out["tree_radial"]["per_delta"][n] - out["tree"]["per_delta"][n] for n in ok]
    res["radial_diff"] = dict(min=float(min(d)), max=float(max(d)), n=len(d))
    res["fields"]["laplace_minus_tree"] = [out["laplace2"]["per_delta"][n] - out["tree"]["per_delta"][n] for n in real]


def main():
    os.makedirs(OUT, exist_ok=True)
    summ, curves = pickle.load(open(os.environ["SIG_CACHE"], "rb"))
    real = json.load(open(os.environ["REAL_JSON"]))
    res = {}
    base = Case(os.path.join(RUNS, "semicircle.pkl"))
    cases = {MODEL: base,
             NULL_R: NullCase(os.path.join(SIG, "null_semicircle_s1.pkl"), base.Tfv),
             NULL_U: NullCase(os.path.join(SIG, "null_uniform_semicircle_s1.pkl"), base.Tfv)}
    fig1(cases, curves, res)
    fig2(summ, res)
    fig4(curves, real, res)
    extra = {}
    for fn in os.environ["EXTRA_JSON"].split(","):
        extra.update(json.load(open(fn)))
    fig3(json.load(open(os.environ["REAL_JSON"])), extra, json.load(open(os.environ["SPREAD_JSON"])), res)
    rp = os.path.join(HERE, "figures", "results.json")
    allres = json.load(open(rp))
    allres["short"] = res
    with open(rp, "w") as f:
        json.dump(allres, f, indent=1, default=float)
    import shutil
    for ext in ("png", "pdf"):
        shutil.copy(os.path.join(HERE, "figures", f"figS1.{ext}"), os.path.join(OUT, f"figS1.{ext}"))
    shutil.copy(rp, os.path.join(OUT, "results.json"))
    print(json.dumps(res["breaks"], indent=0, default=float)[:1500])


if __name__ == "__main__":
    main()
