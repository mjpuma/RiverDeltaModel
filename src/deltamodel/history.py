"""Channel occupancy history, rebuilt from the paths logged at each avulsion."""
import numpy as np


def channel_paths(res):
    """Yield (time, path) for the initial channel and after each avulsion. A path is a list
    of 1-based (i, j) cells from the source to the shoreline."""
    log = res.path_log
    ev = log[:, 0]
    path = [tuple(c) for c in log[ev == -1, 1:]]
    yield 0.0, list(path)
    # rows of each event are contiguous and in event order
    starts = np.searchsorted(ev, np.arange(res.num_avulsions), side="left")
    ends = np.searchsorted(ev, np.arange(res.num_avulsions), side="right")
    for e in range(res.num_avulsions):
        keep = int(res.avulsion_index[e])
        path = path[:keep] + [tuple(c) for c in log[starts[e]:ends[e], 1:]]
        yield float(res.avulsions[e, 0]), list(path)


def occupancy_intervals(res, t_end=None):
    """All channel occupations ("visits") as an array of rows (i, j, t_on, t_off), with
    0-based i, j. A cell that stays in the channel through an avulsion keeps one visit.
    Visits still under way at the end of the run are closed at t_end (default: res.t) and
    flagged in the returned boolean array `open_at_end`."""
    t_end = res.t if t_end is None else t_end
    on = {}
    rows = []
    for t, path in channel_paths(res):
        now = set(path)
        for c in list(on):
            if c not in now:
                rows.append((c[0] - 1, c[1] - 1, on.pop(c), t))
        for c in now:
            if c not in on:
                on[c] = t
    n_closed = len(rows)
    for c, t0 in on.items():
        rows.append((c[0] - 1, c[1] - 1, t0, t_end))
    out = np.array(rows, dtype=float)
    open_at_end = np.zeros(len(rows), dtype=bool)
    open_at_end[n_closed:] = True
    return out, open_at_end


def first_visit_times(res, shape):
    """Time each cell is first occupied by the channel (inf if never)."""
    first = np.full(shape, np.inf)
    for t, path in channel_paths(res):
        for i, j in path:
            if first[i - 1, j - 1] == np.inf:
                first[i - 1, j - 1] = t
    return first
