"""Random-walk null networks inside real delta outlines.

For each delta, the outline is the area that Dong et al. (2026) call the nourishment area
of the delta apex: the mapped channel network (Hariharan et al., 2022, DIDNT dataset),
drawn at each channel's width, together with the interdistributary areas it encloses with
the shoreline. The grid is fine enough to resolve the network (about 300 cells across the
delta, and never coarser than the channel width at the apex, B).

Inside that outline we grow random distributary networks (tree_network): a directed random
walk from the apex to the shoreline, then walks from network cells chosen with equal
probability, each avoiding the existing network and kept only if it reaches the shoreline
more than B from the other outlets, until the network has as many outlets as the real
one. Each step moves to one of the 8 neighbouring cells, with probability proportional to
the gain in a harmonic potential that is 0 at the apex and 1 at the shoreline, with no
flux across the lateral margins; its gradient lines spread from the apex over the whole
shoreline, as radial lines do in the semicircular model domain.

The real networks are measured with the same raster method (L along the network, A the
network drawn at half the apex width on each side plus the enclosed islands), so their
exponents can be compared with the nulls' and with those reported by Dong et al. (2026).

    DIDNT_DIR=.../DIDNT DONG_CSV=.../master_text.csv OUTLETS_DIR=.../outlet_metrics \
        python paper/signatures/real_deltas.py [delta ...]

The input data are not part of this repository: DIDNT (Hariharan et al., 2022) and the
supplementary data and code of Dong et al. (2026).
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import shapefile
from pyproj import CRS, Transformer
from scipy import ndimage
from skimage.draw import line as draw_line

SQ2 = np.sqrt(2.0)
NBR = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
DELTAS = ["Colville", "Lena", "Mossy", "Niger", "Red", "Selenga", "Yukon", "Mississippi", "Wax"]


# ------------------------------------------------------------------ geometry of a delta

class Delta:
    def __init__(self, name, didnt, master, outlets_dir):
        self.name = name
        d = master[master.name == name]
        self.B = float(d.W.iloc[0])                                 # apex channel width, m
        self.n_outlets = int(pd.read_csv(os.path.join(outlets_dir, f"{name}_outlet.csv"))["no"][0])
        links = json.load(open(os.path.join(didnt, "NetworkNodesAndLinks", f"{name}_links.json")))
        nodes = json.load(open(os.path.join(didnt, "NetworkNodesAndLinks", f"{name}_nodes.json")))
        epsg = links["crs"]["properties"]["name"].split(":")[-1]
        raw = os.path.join(didnt, "RawData", name)
        src = CRS.from_wkt(open(os.path.join(raw, "shoreline.prj")).read())
        tr = Transformer.from_crs(src, f"EPSG:{epsg}", always_xy=True)
        shore = np.concatenate([np.array(s.points) for s in shapefile.Reader(os.path.join(raw, "shoreline")).shapes()])
        inlet = np.array(shapefile.Reader(os.path.join(raw, "inlet")).shapes()[0].points[0])
        self.shore_xy = np.column_stack(tr.transform(shore[:, 0], shore[:, 1]))
        self.inlet_xy = np.array(tr.transform(*inlet))
        self.links = [(f["properties"], np.array(f["geometry"]["coordinates"])[:, :2]) for f in links["features"]]
        self.nodes = {f["properties"]["id"]: np.array(f["geometry"]["coordinates"][:2]) for f in nodes["features"]}
        self._grid()

    def _grid(self, ncells=300):
        allxy = np.vstack([c for _, c in self.links] + [self.inlet_xy[None, :]])
        span = (allxy.max(0) - allxy.min(0)).max()
        # cell size: at most the apex channel width, and fine enough to resolve the network
        self.dx = dx = min(self.B, span / ncells)
        self.rB = max(1, int(round(0.5 * self.B / dx)))         # half the apex width, cells
        lo, hi = allxy.min(0) - (5 + 2 * self.rB) * dx, allxy.max(0) + (5 + 2 * self.rB) * dx
        self.x0, self.y0 = lo
        self.nx, self.ny = (np.ceil((hi - lo) / dx).astype(int) + 1)
        # channel centrelines, and the channel footprint (each link at its own width)
        self.chan = np.zeros((self.nx, self.ny), bool)
        foot = np.zeros_like(self.chan)
        for p, c in self.links:
            cells = self._raster_line(c, clip=True)
            self.chan[cells[:, 0], cells[:, 1]] = True
            r = max(1, int(round(0.5 * p.get("avgwidth", self.B) / dx)))
            foot |= self._thick(cells, r)
        sh = self._raster_line(self.shore_xy, clip=True)
        self.shoreline = np.zeros_like(self.chan)
        self.shoreline[sh[:, 0], sh[:, 1]] = True
        # position along the shoreline of every shoreline cell
        step = max(1, len(self.shore_xy) // 3000)
        sxy = self.shore_xy[::step]
        seg = np.r_[0, np.cumsum(np.hypot(*np.diff(sxy, axis=0).T))]
        cxy = np.column_stack([self.x0 + sh[:, 0] * dx, self.y0 + sh[:, 1] * dx])
        near = np.concatenate([np.argmin(((cxy[k:k + 2000, None, :] - sxy[None, :, :]) ** 2).sum(-1), 1)
                               for k in range(0, len(cxy), 2000)])
        self.along = np.full(self.chan.shape, np.nan)
        self.along[sh[:, 0], sh[:, 1]] = seg[near]
        # the real outlets: channel footprint touching the shoreline
        touch = foot & ndimage.binary_dilation(self.shoreline, iterations=2)
        out_along = self._along_of(np.argwhere(touch))
        lo_a, hi_a = np.nanmin(out_along), np.nanmax(out_along)
        self.seg_cells = self.shoreline & (self.along >= lo_a) & (self.along <= hi_a)
        # the outline: area enclosed by the channel footprint and that shoreline segment
        self.domain = ndimage.binary_fill_holes(foot | self.seg_cells)
        self.terminal = self.seg_cells & self.domain
        # a channel reaches the shoreline when it comes within half a channel width of it
        self.near_shore = ndimage.binary_dilation(self.terminal, iterations=self.rB + 1) & self.domain
        ia = self._cell(self.inlet_xy)
        self.apex = self._nearest(ia, self.chan & self.domain & ~self.near_shore)
        self._potential()

    def _thick(self, cells, r):
        m = np.zeros((self.nx, self.ny), bool)
        if len(cells) == 0:
            return m
        (i0, j0), (i1, j1) = cells.min(0) - r - 1, cells.max(0) + r + 2
        i0, j0, i1, j1 = max(i0, 0), max(j0, 0), min(i1, self.nx), min(j1, self.ny)
        sub = np.zeros((i1 - i0, j1 - j0), bool)
        sub[cells[:, 0] - i0, cells[:, 1] - j0] = True
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        m[i0:i1, j0:j1] = ndimage.binary_dilation(sub, structure=(xx ** 2 + yy ** 2) <= r * r)
        return m

    def _cell(self, xy):
        return (int(round((xy[0] - self.x0) / self.dx)), int(round((xy[1] - self.y0) / self.dx)))

    def _nearest(self, ij, mask):
        pts = np.argwhere(mask)
        k = np.argmin(((pts - np.array(ij)) ** 2).sum(1))
        return tuple(pts[k])

    def _raster_line(self, xy, clip=False):
        ij = np.column_stack([(xy[:, 0] - self.x0) / self.dx, (xy[:, 1] - self.y0) / self.dx]).round().astype(int)
        out = []
        for (a, b), (c, d) in zip(ij[:-1], ij[1:]):
            rr, cc = draw_line(a, b, c, d)
            out.append(np.column_stack([rr, cc]))
        out = np.unique(np.vstack(out), axis=0) if out else np.zeros((0, 2), int)
        if clip:
            ok = (out[:, 0] >= 0) & (out[:, 0] < self.nx) & (out[:, 1] >= 0) & (out[:, 1] < self.ny)
            out = out[ok]
        return out

    def _along_of(self, cells):
        """Shoreline position of cells, from the nearest shoreline cell."""
        sh = np.argwhere(self.shoreline)
        k = np.argmin(((cells[:, None, :] - sh[None, :, :]) ** 2).sum(-1), 1)
        return self.along[sh[k, 0], sh[k, 1]]

    def _potential(self):
        """Harmonic potential: phi = 0 at the apex, 1 in the shoreline zone, no flux across
        the lateral margins of the outline. Its gradient lines spread from the apex over the
        whole shoreline, as radial lines do in the semicircular model domain."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.linalg import spsolve
        idx = -np.ones(self.chan.shape, int)
        cells = np.argwhere(self.domain)
        idx[cells[:, 0], cells[:, 1]] = np.arange(len(cells))
        yy, xx = np.mgrid[-self.rB:self.rB + 1, -self.rB:self.rB + 1]
        src = np.zeros_like(self.domain)
        src[self.apex] = True
        src = ndimage.binary_dilation(src, structure=(xx ** 2 + yy ** 2) <= self.rB ** 2) & self.domain
        fixed = src | self.near_shore
        val = np.where(self.near_shore, 1.0, 0.0)
        rows, cols, data = [], [], []
        b = np.zeros(len(cells))
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = cells[:, 0] + di, cells[:, 1] + dj
            ok = (ni >= 0) & (ni < self.nx) & (nj >= 0) & (nj < self.ny)
            ok[ok] &= self.domain[ni[ok], nj[ok]]
            k = np.nonzero(ok)[0]
            rows += [k, k]
            cols += [k, idx[ni[ok], nj[ok]]]
            data += [np.ones(k.size), -np.ones(k.size)]
        rows, cols, data = np.concatenate(rows), np.concatenate(cols), np.concatenate(data)
        isfix = fixed[cells[:, 0], cells[:, 1]]
        keep = ~isfix[rows]
        A = coo_matrix((data[keep], (rows[keep], cols[keep])), shape=(len(cells), len(cells))).tolil()
        # move known neighbours to the right-hand side
        A = A.tocsr()
        Fv = val[cells[:, 0], cells[:, 1]]
        nf = np.nonzero(isfix)[0]
        Acol = A[:, nf]
        b -= Acol @ Fv[nf]
        free = np.nonzero(~isfix)[0]
        sol = spsolve(A[free][:, free].tocsc(), b[free])
        phi = np.full(self.chan.shape, np.nan)
        phi[cells[:, 0], cells[:, 1]] = Fv
        phi[cells[free, 0], cells[free, 1]] = sol
        self.phi = phi

    # ------------------------------------------------------------------ nourishment area
    def nourishment(self, cells, outlets):
        """Area (m^2) of a sub-network, drawn at half the apex width on each side, plus the
        interdistributary areas it encloses with the shoreline segment spanning its outlets."""
        c = np.array(list(cells))
        oa = self._along_of(np.array(list(outlets)))
        seg = self.terminal & (self.along >= np.nanmin(oa)) & (self.along <= np.nanmax(oa))
        sc = np.argwhere(seg)
        allc = np.vstack([c, sc]) if len(sc) else c
        r = self.rB
        (i0, j0), (i1, j1) = allc.min(0) - r - 1, allc.max(0) + r + 2
        i0, j0, i1, j1 = max(i0, 0), max(j0, 0), min(i1, self.nx), min(j1, self.ny)
        m = np.zeros((i1 - i0, j1 - j0), bool)
        m[c[:, 0] - i0, c[:, 1] - j0] = True
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        m = ndimage.binary_dilation(m, structure=(xx ** 2 + yy ** 2) <= r * r)
        m |= seg[i0:i1, j0:j1]
        return int((ndimage.binary_fill_holes(m) & self.domain[i0:i1, j0:j1]).sum()) * self.dx ** 2


# ------------------------------------------------------------------ Hack points of a network

def succ_from_paths(paths):
    succ = defaultdict(set)
    for p in paths:
        for u, v in zip(p[:-1], p[1:]):
            succ[u].add(v)
    return succ


def longest_down(D, succ):
    """Longest flow path (m) from every cell to the shoreline along the network, by
    depth-first search; edges that would close a loop are ignored."""
    down, state = {}, {}
    for root in list(succ):
        if root in down:
            continue
        stack = [(root, iter(succ.get(root, ())))]
        state[root] = 1
        while stack:
            u, it = stack[-1]
            v = next(it, None)
            if v is None:
                best = 0.0
                for w in succ.get(u, ()):
                    if w in down:
                        step = D.dx * (SQ2 if (u[0] != w[0] and u[1] != w[1]) else 1.0)
                        best = max(best, step + down[w])
                down[u] = best
                state[u] = 2
                stack.pop()
            elif state.get(v) is None:
                state[v] = 1
                stack.append((v, iter(succ.get(v, ()))))
    return down


def hack_points(D, succ):
    """L (m) and A (m^2) for every bifurcation of a network (cell successor graph), with
    the definitions of Dong et al. (2026)."""
    down = longest_down(D, succ)
    Ls, As = [], []
    for apex, nxt in list(succ.items()):
        if len(nxt) < 2:
            continue
        seen, stack = {apex}, [apex]
        while stack:
            u = stack.pop()
            for v in succ.get(u, ()):
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        outlets = [v for v in seen if D.near_shore[v]]
        if n_outlets(D, outlets) < 2:
            continue
        if down[apex] > 0:
            Ls.append(down[apex])
            As.append(D.nourishment(seen, outlets))
    return np.array(Ls), np.array(As)


# ------------------------------------------------------------------ the real network

def real_succ(D):
    """Cell successor graph of the mapped network: each link is oriented from low to high
    potential and its cells are joined in order along its centreline."""
    succ = defaultdict(set)
    for p, c in D.links:
        a, b = [int(v) for v in p["conn"].split(",")]
        ca, cb = D._cell(D.nodes[a]), D._cell(D.nodes[b])
        if not (D.domain[ca] or D.chan[ca]) or not (D.domain[cb] or D.chan[cb]):
            continue
        pa, pb = D.phi[ca], D.phi[cb]
        cells = _steps(list(dict.fromkeys(D._cell(xy) for xy in c)))
        if not (np.isfinite(pa) and np.isfinite(pb)):
            continue
        if pa > pb:
            cells = cells[::-1]
        cells = [x for x in cells if D.domain[x]]
        # stop where the channel reaches the shoreline
        for k, x in enumerate(cells):
            if D.near_shore[x]:
                cells = cells[:k + 1]
                break
        for u, v in zip(cells[:-1], cells[1:]):
            if u != v:
                succ[u].add(v)
    return succ


def _steps(cells):
    """Fill gaps between consecutive cells so a path moves one cell at a time."""
    out = [cells[0]]
    for c in cells[1:]:
        a = out[-1]
        if a == c:
            continue
        rr, cc = draw_line(a[0], a[1], c[0], c[1])
        out += [tuple(int(v) for v in x) for x in zip(rr[1:], cc[1:])]
    return out


# ------------------------------------------------------------------ the null networks

def walk(D, start, rng, exclude=(), join=None):
    """Directed random walk from `start` to the shoreline zone. With `join` (a set of
    network cells), the walk also ends when it steps onto one of them (a confluence)."""
    path, cur, used = [], start, set(exclude) | {start}
    X, Y = D.domain.shape
    while not D.near_shore[cur]:
        if join is not None and path and cur in join:
            break
        i, j = cur
        opts = [(i + a, j + b) for a, b in NBR
                if 0 <= i + a < X and 0 <= j + b < Y and D.domain[i + a, j + b] and (i + a, j + b) not in used]
        gain = np.array([D.phi[c] - D.phi[cur] for c in opts]) if opts else np.zeros(0)
        up = gain > 0
        if up.any():
            w = gain[up]
            cand = [c for c, u in zip(opts, up) if u]
            cur = cand[rng.choice(len(cand), p=w / w.sum())]
        elif opts:                                 # local maximum of phi: take the best step
            cur = opts[int(np.argmax(gain))]
        else:
            return None
        used.add(cur)
        path.append(cur)
        if len(path) > 20 * (X + Y):
            return None
    return path


def n_outlets(D, cells):
    t = [c for c in cells if D.near_shore[c]]
    if not t:
        return 0
    a = np.sort(D._along_of(np.array(t)))
    return int(1 + np.sum(np.diff(a) > 2 * D.rB * D.dx))


def n_branches(succ):
    return sum(1 for v in succ.values() if len(v) >= 2)


def tree_network(D, rng, target, max_tries=None, sites="uniform", join=False, target_branches=None):
    """Random distributary network (a snapshot), grown from the apex.

    Default (the "tree" null): a walk from the apex, then walks from network cells chosen
    with equal probability, each avoiding the existing network and kept only if it reaches
    the shoreline more than one apex width from other outlets, until the network has
    `target` outlets.
    sites="downstream": branch cells are chosen with probability proportional to phi^2,
        so that bifurcations concentrate downstream.
    join=True with target_branches: new walks may end where they meet the existing
        network (confluences, so the network has loops like real ones), and growth stops
        when the network has as many bifurcation cells as the real network."""
    first = walk(D, D.apex, rng)
    if first is None:
        return []
    paths = [[D.apex] + first]
    cells = set(paths[0])
    outs = [D._along_of(np.array([first[-1]]))[0]]
    succ = succ_from_paths(paths)
    tries = 0
    max_tries = max_tries or 300 * max(target, target_branches or 0)

    def done():
        if target_branches is not None:
            return n_branches(succ) >= target_branches
        return len(outs) >= target

    while not done() and tries < max_tries:
        tries += 1
        cand = [c for c in cells if not D.near_shore[c]]
        if sites == "downstream":
            w8 = np.array([D.phi[c] for c in cand]) ** 2 + 1e-6
            s = cand[rng.choice(len(cand), p=w8 / w8.sum())]
        else:
            s = cand[rng.integers(len(cand))]
        parent = next(p for p in paths if s in p)
        k = parent.index(s)
        if join:
            w = walk(D, s, rng, exclude=parent[:k + 1], join=cells - set(parent[:k + 1]))
        else:
            w = walk(D, s, rng, exclude=cells)
        if w is None or len(w) < 2:
            continue
        if D.near_shore[w[-1]]:
            a = D._along_of(np.array([w[-1]]))[0]
            if not join and min(abs(a - o) for o in outs) <= D.B:
                continue
            outs.append(a)
        new = parent[:k + 1] + w
        paths.append(new)
        cells.update(w)
        for u, v in zip(new[k:-1], new[k + 1:]):
            succ[u].add(v)
    return paths


class field:
    """Temporarily replace the potential that directs the walks: "harmonic" (default) or
    "radial", the distance from the apex, so that walks head straight out from the apex
    as in the semicircular model domain. The orientation of the mapped network always
    uses the harmonic potential."""
    def __init__(self, D, kind):
        self.D, self.kind = D, kind

    def __enter__(self):
        D = self.D
        self.saved = D.phi
        if self.kind == "radial":
            ii, jj = np.indices(D.chan.shape)
            r = np.hypot(ii - D.apex[0], jj - D.apex[1]).astype(float)
            D.phi = np.where(D.domain, r / r[D.near_shore].max(), np.nan)
        return D

    def __exit__(self, *a):
        self.D.phi = self.saved


class _Laplace:
    """Solver for Laplace's equation on the outline (4-neighbour stencil, no flux across
    the lateral margins) with values fixed on a changing set of cells."""
    def __init__(self, D):
        self.D = D
        self.cells = np.argwhere(D.domain)
        n = len(self.cells)
        idx = -np.ones(D.chan.shape, int)
        idx[self.cells[:, 0], self.cells[:, 1]] = np.arange(n)
        self.idx = idx
        a, b = [], []
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = self.cells[:, 0] + di, self.cells[:, 1] + dj
            ok = (ni >= 0) & (ni < D.nx) & (nj >= 0) & (nj < D.ny)
            ok[ok] &= D.domain[ni[ok], nj[ok]]
            a.append(np.nonzero(ok)[0])
            b.append(idx[ni[ok], nj[ok]])
        self.a, self.b = np.concatenate(a), np.concatenate(b)
        self.deg = np.bincount(self.a, minlength=n).astype(float)
        # cells in pockets of the outline that do not connect to the shoreline (4-neighbour)
        lab, _ = ndimage.label(D.domain)
        keep = np.unique(lab[D.near_shore])
        self.pocket = ~np.isin(lab[self.cells[:, 0], self.cells[:, 1]], keep[keep > 0])
        self.x = None

    def solve(self, fixed, val):
        """fixed, val: arrays over the outline's cells. Returns the solution there."""
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import spsolve
        fixed = fixed | self.pocket
        val = np.where(self.pocket & ~fixed, 0.0, val)
        free = np.nonzero(~fixed)[0]
        pos = -np.ones(len(fixed), int)
        pos[free] = np.arange(free.size)
        m = ~fixed[self.a]                                   # rows of free cells
        a, b = self.a[m], self.b[m]
        fb = fixed[b]
        rhs = np.bincount(pos[a[fb]], weights=val[b[fb]], minlength=free.size)
        r = np.r_[pos[a[~fb]], np.arange(free.size)]
        c = np.r_[pos[b[~fb]], np.arange(free.size)]
        d = np.r_[-np.ones((~fb).sum()), self.deg[free]]
        M = csr_matrix((d, (r, c)), shape=(free.size, free.size))
        u = val.astype(float).copy()
        u[free] = spsolve(M.tocsc(), rhs)
        return u


def laplace_network(D, rng, target, eta=2.0, gamma=1.0, max_tries=None):
    """Distributary tree grown in a Laplace field that feels the network.

    As in the moving-boundary model of Ke et al. (2019), flow over the unchannelized delta
    surface is described by Laplace's equation: here u = 0 on the network, u = 1 in the
    shoreline zone and no flux across the lateral margins, so u is screened by the
    existing channels. Each new channel starts at a network cell chosen with probability
    proportional to the flux of the field out of the network there (the sum of u over its
    4 neighbours outside the network) to the power eta, excluding cells within two channel
    widths of the shoreline zone, where the flux reflects the boundary condition; Ke et al. relate the advance of
    the channel boundary to this flux with exponents of 1.5-2.5. The new channel then
    follows the field: a walk whose steps are weighted by the gain in u to the power gamma,
    avoiding the existing network. As for the random trees, it is kept only if it reaches
    the shoreline more than one apex width from the other outlets, and the field is
    recomputed after every channel, until the tree has `target` outlets. Returns paths."""
    lap = _Laplace(D)
    ci, cj = lap.cells[:, 0], lap.cells[:, 1]
    first = walk(D, D.apex, rng)
    if first is None:
        return []
    paths = [[D.apex] + first]
    cells = set(paths[0])
    outs = [D._along_of(np.array([first[-1]]))[0]]
    net = np.zeros(D.domain.shape, bool)
    for c in cells:
        net[c] = True
    # new channels do not start within two channel widths of the shoreline zone, where
    # the flux reflects the boundary condition rather than the network
    band = ndimage.binary_dilation(D.near_shore, iterations=2 * D.rB + 2)
    tries, max_tries = 0, max_tries or 50 * target
    saved = D.phi
    try:
        while len(outs) < target and tries < max_tries:
            tries += 1
            fixed = (net | D.near_shore)[ci, cj]
            u = lap.solve(fixed, np.where(net[ci, cj], 0.0, 1.0))
            U = np.full(D.domain.shape, np.nan)
            U[ci, cj] = u
            # flux out of every network cell away from the shoreline zone
            V = np.where(D.domain & ~net, np.nan_to_num(U), 0.0)
            F = np.zeros_like(V)
            F[1:] += V[:-1]; F[:-1] += V[1:]; F[:, 1:] += V[:, :-1]; F[:, :-1] += V[:, 1:]
            cand = [c for c in cells if not band[c]]
            if not cand:
                break
            flux = np.array([F[c] for c in cand])
            if flux.sum() <= 0:
                break
            w8 = flux ** eta
            s = cand[rng.choice(len(cand), p=w8 / w8.sum())]
            D.phi = U ** gamma if gamma != 1.0 else U
            w = walk(D, s, rng, exclude=cells)
            D.phi = saved
            if w is None or len(w) < 2 or not D.near_shore[w[-1]]:
                continue
            a = D._along_of(np.array([w[-1]]))[0]
            if min(abs(a - o) for o in outs) <= D.B:
                continue
            outs.append(a)
            parent = next(p for p in paths if s in p)
            k = parent.index(s)
            paths.append(parent[:k + 1] + w)
            cells.update(w)
            for c in w:
                net[c] = True
    finally:
        D.phi = saved
    return paths


def avulsion_sequence(D, rng, n_paths):
    """Successive channel paths of a random-walk null in the outline, as for the delta
    model's null with avulsion sites uniform along the channel: each new path leaves the
    current one at a cell chosen with equal probability (excluding the shoreline zone) and
    follows a directed random walk to the shoreline."""
    first = walk(D, D.apex, rng)
    path = [D.apex] + first
    seq = [path]
    while len(seq) < n_paths:
        ok = [k for k, c in enumerate(path) if not D.near_shore[c]]
        k = ok[rng.integers(len(ok))]
        w = walk(D, path[k], rng, exclude=path[:k + 1])
        if w is None:
            continue
        path = path[:k + 1] + w
        seq.append(path)
    return seq


def null_network(D, seq, start, target):
    """The network traced by the null channel from path `start` on, over as many paths as
    it takes to reach the shoreline at `target` separate places (path ends more than one
    apex width apart along the shoreline)."""
    ends, k = [], start
    while k < len(seq):
        ends.append(D._along_of(np.array([seq[k][-1]]))[0])
        e = np.sort(ends)
        if 1 + np.sum(np.diff(e) > D.B) >= target:
            break
        k += 1
    return seq[start:k + 1]


# ------------------------------------------------------------------ fits

def tls_h(L, A):
    """Hack exponent by orthogonal regression of log A on log L (Dong et al., 2026)."""
    from scipy.odr import ODR, Data, Model
    x, y = np.log(L), np.log(A)
    o = ODR(Data(x, y), Model(lambda b, x: b[0] * x + b[1]), beta0=[1.0, 0.0]).run()
    return float(1 / o.beta[0])


def break_fit(L, A, B):
    """One-breakpoint fit of log A on log L (piecewise_regression, as Dong et al.);
    returns h below and above the break, the break in L/B, and the Davies-test p."""
    import piecewise_regression as pr
    from piecewise_regression.davies import davies_test
    x, y = np.log(L / B), np.log(A / B ** 2)
    try:
        f = pr.Fit(x, y, n_breakpoints=1, n_boot=50, max_iterations=60)
        e = f.get_results()["estimates"]
        if e is None:
            raise ValueError
        out = dict(h1=1 / e["alpha1"]["estimate"], h2=1 / e["alpha2"]["estimate"],
                   break_L_over_B=float(np.exp(e["breakpoint1"]["estimate"])))
    except Exception:
        out = dict(h1=np.nan, h2=np.nan, break_L_over_B=np.nan)
    out["davies_p"] = float(davies_test(x, y))
    return out


# ------------------------------------------------------------------ main

def main(names, n_null=40, seed=0):
    didnt = os.environ["DIDNT_DIR"]
    master = pd.read_csv(os.environ["DONG_CSV"])
    outlets_dir = os.environ["OUTLETS_DIR"]
    out_dir = os.environ.get("SIG_FIGS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures"))
    os.makedirs(out_dir, exist_ok=True)
    res = {}
    rng = np.random.default_rng(seed)
    for name in names:
        D = Delta(name, didnt, master, outlets_dir)
        d = master[master.name == name]
        m = (d.nL > 0) & (d.nA > 0)
        dong = dict(h=tls_h(d.nL[m].values, d.nA[m].values), n=int(m.sum()),
                    **break_fit(d.nL[m].values * D.B, d.nA[m].values * D.B ** 2, D.B))
        rs = real_succ(D)
        Lr, Ar = hack_points(D, rs)
        target = n_outlets(D, set(rs) | {v for x in rs.values() for v in x})
        ours = dict(h=tls_h(Lr, Ar), n=int(Lr.size), **break_fit(Lr, Ar, D.B)) if Lr.size > 5 else {}
        nb = n_branches(rs)
        kinds = {"tree": dict(),
                 "downstream": dict(sites="downstream"),
                 "dense": dict(join=True, target_branches=nb)}
        nulls, pts = {}, {}
        for kind, kw in kinds.items():
            nulls[kind] = []
            for k in range(n_null if kind == "tree" else n_null // 2):
                P = tree_network(D, rng, target, **kw)
                sc = succ_from_paths(P)
                L, A = hack_points(D, sc)
                if L.size < 5:
                    continue
                nulls[kind].append(dict(h=tls_h(L, A), n=int(L.size), branches=n_branches(sc),
                                        **break_fit(L, A, D.B)))
                if k == 0:
                    pts[kind] = (L.tolist(), A.tolist())
        res[name] = dict(B=D.B, dx=D.dx, outlets=D.n_outlets, outlets_raster=target, branches_real=nb,
                         cells=int(D.domain.sum()),
                         dong=dong, real_raster=ours,
                         nulls=nulls, null_points=pts, real_points=(Lr.tolist(), Ar.tolist()))
        msg = ", ".join(f"{k} {np.median([x['h'] for x in v]):.2f}" for k, v in nulls.items() if v)
        print(f"{name}: B={D.B:.0f} m, outlets {target}, branches {nb}; h Dong {dong['h']:.2f}, "
              f"real (raster) {ours.get('h', np.nan):.2f}; null medians: {msg}", flush=True)
        with open(os.path.join(out_dir, "real_deltas.json"), "w") as f:
            json.dump(res, f, default=float)
    return res


def spread(D, cells):
    """How evenly channels fill the outline: the mean distance from the outline to the
    nearest channel, divided by that for evenly spaced parallel channels of the same total
    length (a quarter of the spacing, outline area / total length). Near 1 for even
    spacing, larger when channels cluster and leave parts of the outline far from any
    channel."""
    m = np.zeros(D.domain.shape, bool)
    for c in cells:
        m[c] = True
    m &= D.domain
    dist = ndimage.distance_transform_edt(~m) * D.dx
    length = m.sum() * D.dx
    return float(dist[D.domain].mean() / (D.domain.sum() * D.dx ** 2 / (4 * length)))


def spread_check(names, out_json, n=4, seed=2):
    """Hack exponent and spread of the mapped network and of a few networks of each kind."""
    didnt = os.environ["DIDNT_DIR"]
    master = pd.read_csv(os.environ["DONG_CSV"])
    res = json.load(open(out_json)) if os.path.exists(out_json) else {}
    for name in names:
        rng = np.random.default_rng([seed, DELTAS.index(name)])
        D = Delta(name, didnt, master, os.environ["OUTLETS_DIR"])
        rs = real_succ(D)
        nb = n_branches(rs)
        allc = set(rs) | {v for x in rs.values() for v in x}
        target = n_outlets(D, allc)
        L, A = hack_points(D, rs)
        out = {"mapped": [dict(h=tls_h(L, A), spread=spread(D, allc))]}
        gens = {"dense": ("harmonic", lambda: tree_network(D, rng, 0, join=True, target_branches=nb)),
                "tree": ("harmonic", lambda: tree_network(D, rng, target)),
                "tree_radial": ("radial", lambda: tree_network(D, rng, target)),
                "laplace2": ("harmonic", lambda: laplace_network(D, rng, target, eta=2.0))}
        for kind, (fld, gen) in gens.items():
            out[kind] = []
            with field(D, fld):
                for k in range(n):
                    P = gen()
                    sc = succ_from_paths(P)
                    L, A = hack_points(D, sc)
                    if L.size >= 5:
                        out[kind].append(dict(h=tls_h(L, A), spread=spread(D, {c for p in P for c in p})))
        res[name] = out
        print(name, {k: (round(np.median([x["spread"] for x in v]), 2), round(np.median([x["h"] for x in v]), 2))
                     for k, v in out.items() if v}, flush=True)
        with open(out_json, "w") as f:
            json.dump(res, f, default=float)
    return res


def extra_nulls(names, out_json, n_lap=10, n_radial=20, seed=1):
    """Further comparison networks for the deltas already in real_deltas.json, written to
    `out_json`: random trees and random networks with loops directed by distance from the
    apex ("tree_radial", "dense_radial"), and trees grown in a Laplace field that feels the
    network, with flux exponents 1 and 2 ("laplace1", "laplace2")."""
    didnt = os.environ["DIDNT_DIR"]
    master = pd.read_csv(os.environ["DONG_CSV"])
    outlets_dir = os.environ["OUTLETS_DIR"]
    res = json.load(open(out_json)) if os.path.exists(out_json) else {}
    for name in names:
        rng = np.random.default_rng([seed, DELTAS.index(name)])
        D = Delta(name, didnt, master, outlets_dir)
        rs = real_succ(D)
        nb = n_branches(rs)
        target = n_outlets(D, set(rs) | {v for x in rs.values() for v in x})
        gens = {"laplace2": (n_lap, lambda: laplace_network(D, rng, target, eta=2.0)),
                "laplace1": (n_lap, lambda: laplace_network(D, rng, target, eta=1.0)),
                "tree_radial": (n_radial, lambda: tree_network(D, rng, target)),
                "dense_radial": (n_radial // 2, lambda: tree_network(D, rng, 0, join=True, target_branches=nb))}
        out, pts = {}, {}
        for kind, (n, gen) in gens.items():
            out[kind] = []
            with field(D, "radial" if kind.endswith("radial") else "harmonic"):
                for k in range(n):
                    P = gen()
                    sc = succ_from_paths(P)
                    L, A = hack_points(D, sc)
                    rec = dict(n=int(L.size), branches=n_branches(sc),
                               outlets=n_outlets(D, set(sc) | {v for x in sc.values() for v in x}))
                    if L.size >= 5:
                        rec.update(h=tls_h(L, A), **break_fit(L, A, D.B))
                    out[kind].append(rec)
                    if k == 0:
                        pts[kind] = [[list(c) for c in p] for p in P]
        res[name] = dict(target_outlets=target, target_branches=nb, nulls=out, example_paths=pts)
        msg = ", ".join(f"{k} {np.median([x['h'] for x in v if 'h' in x]) if any('h' in x for x in v) else float('nan'):.2f}"
                        f" ({sum('h' in x for x in v)}/{len(v)}, outlets {np.median([x['outlets'] for x in v]):.0f})"
                        for k, v in out.items())
        print(f"{name}: target outlets {target}, junctions {nb}; {msg}", flush=True)
        with open(out_json, "w") as f:
            json.dump(res, f, default=float)
    return res


if __name__ == "__main__":
    if sys.argv[1:2] == ["--spread"]:
        spread_check(sys.argv[3:] or DELTAS, sys.argv[2])
    elif sys.argv[1:2] == ["--extra"]:
        extra_nulls(sys.argv[3:] or DELTAS, sys.argv[2])
    else:
        main(sys.argv[1:] or DELTAS)


# ------------------------------------------------------------------ figure and summary

def figure(json_path, results_path, maps=("Yukon", "Mossy", "Lena"), seed=3):
    """Figure 7 of the signatures paper and the numbers quoted in its text (added to
    results.json under "real_deltas")."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    import figures as F
    from scipy.stats import binomtest
    plt = F.plt
    r = json.load(open(json_path))
    order = sorted(r, key=lambda n: r[n]["real_raster"]["h"])
    fig = plt.figure(figsize=(7.2, 6.6))
    ax = fig.add_axes([0.10, 0.57, 0.88, 0.39])
    style = {"tree": ("#c9ced4", F.MUTED, -0.18), "dense": ("#f6c3ad", F.ORANGE, 0.18)}
    for k, n in enumerate(order):
        v = r[n]
        for kind, (fc, ec, off) in style.items():
            hn = [x["h"] for x in v["nulls"][kind]]
            ax.boxplot(hn, positions=[k + off], widths=0.3, showfliers=False, patch_artist=True,
                       boxprops=dict(facecolor=fc, edgecolor=ec), medianprops=dict(color=ec),
                       whiskerprops=dict(color=ec), capprops=dict(color=ec))
        ax.plot(k, v["real_raster"]["h"], "o", color=F.BLUE, ms=6, zorder=3)
        ax.plot(k, v["dong"]["h"], "D", color=F.INK, mfc="none", ms=5, zorder=3)
    ax.plot([], [], "s", color="#f6c3ad", mec=F.ORANGE, label="random, matched junctions (with loops)")
    ax.plot([], [], "s", color="#c9ced4", mec=F.MUTED, label="random, matched outlets (trees)")
    ax.plot([], [], "o", color=F.BLUE, label="mapped network, same method")
    ax.plot([], [], "D", color=F.INK, mfc="none", label="mapped network, Dong et al. (2026)")
    ax.set_xticks(range(len(order)), order, fontsize=8)
    ax.set_ylabel("Hack exponent h"); ax.set_ylim(0.1, 1.15)
    ax.legend(fontsize=6.5, loc="upper left", ncol=2)
    F.panel(ax, "A"); F.style(ax, grid_y=True)

    didnt, master = os.environ["DIDNT_DIR"], pd.read_csv(os.environ["DONG_CSV"])
    rng = np.random.default_rng(seed)
    for k, n in enumerate(maps):
        D = Delta(n, didnt, master, os.environ["OUTLETS_DIR"])
        rs = real_succ(D)
        P = tree_network(D, rng, 0, join=True, target_branches=n_branches(rs))
        a = fig.add_axes([0.02 + k * 0.33, 0.02, 0.31, 0.46])
        img = np.full(D.chan.shape + (3,), 1.0)
        img[D.domain] = (0.93, 0.93, 0.93)
        img[ndimage.binary_dilation(D.chan & D.domain) & D.domain] = (0.16, 0.47, 0.84)
        nm = np.zeros(D.chan.shape, bool)
        for p in P:
            for c in p:
                nm[c] = True
        img[ndimage.binary_dilation(nm) & D.domain] = (0.92, 0.41, 0.20)
        a.imshow(np.transpose(img, (1, 0, 2)), origin="lower", interpolation="nearest")
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
        F.panel(a, "BCD"[k], n)
    fig.savefig(os.path.join(os.path.dirname(results_path), "fig7.png"))
    fig.savefig(os.path.join(os.path.dirname(results_path), "fig7.pdf"))
    plt.close(fig)

    # summary
    out = {"deltas": {}, "pooled": {}}
    for n, v in r.items():
        e = dict(B_m=v["B"], h_dong=v["dong"]["h"], h_real=v["real_raster"]["h"],
                 n_real=v["real_raster"]["n"], branches_real=v["branches_real"], outlets_real=v["outlets_raster"])
        for kind in ("tree", "downstream", "dense"):
            hn = np.array([x["h"] for x in v["nulls"][kind]])
            e[kind] = dict(median=float(np.median(hn)), p5=float(np.percentile(hn, 5)),
                           p95=float(np.percentile(hn, 95)), pct_real=float(100 * (hn < e["h_real"]).mean()),
                           n_points=float(np.median([x["n"] for x in v["nulls"][kind]])),
                           break_share=float(np.mean([x["davies_p"] < 0.05 for x in v["nulls"][kind]])))
        out["deltas"][n] = e
    for kind in ("tree", "downstream", "dense"):
        above = sum(out["deltas"][n]["h_real"] > out["deltas"][n][kind]["median"] for n in r)
        allh = np.concatenate([[x["h"] for x in r[n]["nulls"][kind]] for n in r])
        bs = [out["deltas"][n][kind]["break_share"] for n in r]
        out["pooled"][kind] = dict(above=int(above), p=float(binomtest(int(above), len(r), 0.5, alternative="greater").pvalue),
                                   median=float(np.median(allh)), q25=float(np.percentile(allh, 25)),
                                   q75=float(np.percentile(allh, 75)), break_share_min=float(min(bs)),
                                   break_share_max=float(max(bs)))
    hr = [r[n]["real_raster"]["h"] for n in r]
    hd = [r[n]["dong"]["h"] for n in r]
    out["pooled"]["real"] = dict(min=float(min(hr)), max=float(max(hr)), dong_min=float(min(hd)), dong_max=float(max(hd)),
                                 max_abs_diff=float(max(abs(a - b) for a, b in zip(hr, hd))))
    out["n_deltas"] = len(r)
    res = json.load(open(results_path))
    res["real_deltas"] = out
    with open(results_path, "w") as f:
        json.dump(res, f, indent=1, default=float)
    return out
