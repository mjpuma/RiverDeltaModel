"""Run the delta model and plot the final delta plain and where avulsions happen.

    python examples/plot_delta.py            # 710-year run, about 5 minutes
    python examples/plot_delta.py --years 100
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np

from deltamodel import Params, run


def plot(res, p, path):
    ink, muted = "#1f2328", "#6e7781"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.6), gridspec_kw=dict(width_ratios=[1.25, 1]))
    x_km = np.arange(p.xmax) * p.a / 1000
    y_km = np.arange(p.ymax + 1) * p.a / 1000
    im = ax1.pcolormesh(x_km, y_km, res.eta.T, cmap="Blues", shading="nearest")
    ci, cj = np.nonzero(res.channel)
    ax1.plot(x_km[ci], y_km[cj], "s", ms=3, color=ink, label="active channel")
    ax1.set_xlabel("across the delta (km)")
    ax1.set_ylabel("distance from the source (km)")
    ax1.set_title("Bed elevation at the end of the run", loc="left", fontsize=11, color=ink)
    ax1.set_aspect("equal")
    ax1.legend(loc="upper left", bbox_to_anchor=(0, -0.16), fontsize=8, frameon=False)
    fig.colorbar(im, ax=ax1, label="bed elevation (m)", shrink=0.85)

    # avulsions after spin-up, by distance from the source
    av = res.avulsions[res.avulsions[:, 0] > p.teq]
    rows = np.rint(av[:, 2] / p.a).astype(int) - 1
    counts = np.bincount(rows, minlength=p.ymax + 1)
    years = p.tmax - p.teq
    ax2.bar(y_km, counts / years, width=0.08, color="#2f6fba")
    ax2.set_xlabel("distance from the source (km)")
    ax2.set_ylabel("avulsions per effective year")
    ax2.set_title("Where channels avulse", loc="left", fontsize=11, color=ink)
    for a in (ax1, ax2):
        a.tick_params(colors=muted, labelsize=8)
        for s in a.spines.values():
            s.set_color(muted)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="y", color="#d0d7de", lw=0.6)
    ax2.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=710.0)
    ap.add_argument("--spinup", type=float, default=110.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="delta.png")
    args = ap.parse_args()
    p = Params(tmax=args.years, teq=min(args.spinup, args.years / 2))
    res = run(p, seed=args.seed)
    print(f"{res.num_avulsions} avulsions in {args.years:g} effective years")
    plot(res, p, args.out)
    print("wrote", args.out)
