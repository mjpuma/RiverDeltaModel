"""Figures 1-4 of the short paper on Hack-like scaling in deltas, and the numbers it quotes
(added to paper/signatures/figures/results.json under "short").

    DELTA_RUNS=...  SIG_RUNS=...  SIG_CACHE=.../summaries.pkl  REAL_JSON=.../real_deltas.json \
    DONG_BREAKS=.../Break4.csv (Dong et al. 2026 code archive) \
        python paper/signatures/short_figures.py

Uses the statistics cached by figures.py (summaries.pkl) and the output of real_deltas.py.
"""
import json
import os
import pickle
import sys

import matplotlib.ticker as mt
import numpy as np

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
    rp = os.path.join(HERE, "figures", "results.json")
    allres = json.load(open(rp))
    allres["short"] = res
    with open(rp, "w") as f:
        json.dump(allres, f, indent=1, default=float)
    # Figure 3 is Figure 7 of the long draft (real deltas against random networks)
    import shutil
    for ext in ("png", "pdf"):
        shutil.copy(os.path.join(HERE, "figures", f"fig7.{ext}"), os.path.join(OUT, f"fig3.{ext}"))
        shutil.copy(os.path.join(HERE, "figures", f"figS1.{ext}"), os.path.join(OUT, f"figS1.{ext}"))
    shutil.copy(rp, os.path.join(OUT, "results.json"))
    print(json.dumps(res["breaks"], indent=0, default=float)[:1500])


if __name__ == "__main__":
    main()
