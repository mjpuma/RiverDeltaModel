"""Random-walk null model for the signatures paper.

The null keeps everything about the model's avulsions except where the new channel goes:
it uses the model's avulsion times and the distance from the source of each avulsion, but
the new path is a directed random walk to the shoreline instead of the path of steepest
descent on the evolving bed. Each step moves to one of the 8 neighbouring cells that are
farther from the source, in the domain and not already on the path, with probability
proportional to the gain in distance from the source (a radial version of Scheidegger's
directed random network). Equal probabilities would leave the walk biased by the grid:
cells about 45 degrees from the downstream direction would be visited several times less
often than others; with this weighting the density of visits is close to uniform in
angle. The channel ends when it reaches a shoreline cell, as in the model.

Comparing the model with the null separates what comes from the timing and location of
avulsions from what comes from the topography that steers the new channels.

    python paper/signatures/null_model.py RUN.pkl OUT.pkl [seed] [uniform]
"""
import pickle
import sys
from types import SimpleNamespace

import numpy as np

NBR = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _walk(start, r_of, active, shore, rng, avoid=None, exclude=()):
    """Directed random walk from `start` (0-based cell) to the shoreline. Returns the cells
    after `start`. `avoid` is a first step to avoid when another choice exists."""
    X, Y = active.shape
    path, cur, used = [], start, set(exclude) | {start}
    while not shore[cur]:
        i, j = cur
        opts = [(i + di, j + dj) for di, dj in NBR
                if 0 <= i + di < X and 0 <= j + dj < Y and active[i + di, j + dj]
                and r_of[i + di, j + dj] > r_of[i, j] and (i + di, j + dj) not in used]
        if not opts:                      # cannot happen inside a disk, but be safe
            raise RuntimeError(f"random walk stuck at {cur}")
        if avoid is not None and not path and len(opts) > 1 and avoid in opts:
            opts.remove(avoid)
        w = np.array([r_of[c] - r_of[i, j] for c in opts])
        cur = opts[rng.choice(len(opts), p=w / w.sum())]
        used.add(cur)
        path.append(cur)
    return path


def random_walk_null(p, r, seed=0, uniform_sites=False):
    """Null run with the avulsion times and radii of run r. Returns an object with the
    fields paper/analysis.py uses (paths are stored like the model's path log).
    uniform_sites=True also replaces the avulsion sites: each avulsion leaves the current
    path at a cell chosen with equal probability along it (shoreline cell excluded)."""
    rng = np.random.default_rng(seed)
    X, Y = r.eta.shape
    i0 = r.source[0]
    ii, jj = np.meshgrid(np.arange(X), np.arange(Y), indexing="ij")
    r_of = np.hypot(ii - i0, jj)                               # distance from the source, cells
    active, shore = r.land | r.shoreline, r.shoreline
    src = (i0, 0)
    path = [src] + _walk(src, r_of, active, shore, rng)
    log = [(-1, i + 1, j + 1) for i, j in path]
    keep_idx, av = [], []
    for e in range(r.num_avulsions):
        t, xa, ya = r.avulsions[e, :3]
        ia, ja = int(round(xa / p.a)) - 1, int(round(ya / p.a)) - 1
        ra = r_of[ia, ja]
        rs = np.array([r_of[c] for c in path])
        if uniform_sites:
            k = int(rng.integers(len(path) - 1))
        else:
            k = int(min(np.searchsorted(rs, ra), len(path) - 2))   # never avulse from the shoreline
        new = _walk(path[k], r_of, active, shore, rng, avoid=path[k + 1], exclude=path[:k])
        path = path[:k + 1] + new
        keep_idx.append(k + 1)
        av.append((t, p.a * (path[k][0] + 1), p.a * (path[k][1] + 1)))
        log += [(e, i + 1, j + 1) for i, j in new]
    return SimpleNamespace(
        t=r.t, eta=np.zeros_like(r.eta), land=r.land, shoreline=r.shoreline, source=r.source,
        num_avulsions=r.num_avulsions, avulsions=np.array(av), avulsion_index=np.array(keep_idx),
        path_log=np.array(log, dtype=np.int64), snapshots=None, Havg=None)


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    uniform = len(sys.argv) > 4 and sys.argv[4] == "uniform"
    p, r = pickle.load(open(src, "rb"))
    with open(dst, "wb") as f:
        pickle.dump((p, random_walk_null(p, r, seed, uniform)), f)
