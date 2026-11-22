"""
Cellular model of river delta evolution with backwater hydraulics, avulsion and
stratigraphy (Puma, Muneepeerakul, Paola, Rinaldo and Rodriguez-Iturbe, arXiv:1606.04558).

    from deltamodel import Params, run
    res = run(Params(tmax=110.0), seed=1)

Units: metres and seconds, with time in "effective years" (years of flood flow; the
intermittency I_tavg converts to calendar years).
"""
from dataclasses import dataclass, field
import numpy as np

from ._kernel import run_kernel


@dataclass
class Params:
    """Model parameters. Defaults are those of the May 2007 code. Lengths in m, time in
    effective years unless noted."""
    xmax: int = 71                 # cells across the delta
    ymax: int = 50                 # cells from source to shoreline (the grid has ymax + 1 rows)
    imidx: int = 36                # source column
    a: float = 100.0               # cell size, m
    Qw: float = 20.0               # water discharge, m3/s
    Qs: float = 0.0025             # sediment feed, m3/s
    tmax: float = 710.0            # run length
    teq: float = 110.0             # spin-up before statistics are collected
    dtmax: float = 0.0001          # time step
    dtmin: float = 1e-8
    dtredu: float = 0.5            # factor for time-step cuts when the bed would scour
    dtprint: float = 1.0           # interval of grid snapshots
    dtprint2: float = 0.001        # interval for averaging flow depth
    dtprint3: float = 0.1          # interval for the sediment flux series
    I_tavg: float = 0.05           # flood intermittency
    beta: float = 1.0              # avulsion threshold: superelevation relative to flow depth
    g: float = 9.81
    D: float = 0.0003              # grain size, m
    R: float = 1.65                # submerged specific gravity
    lam: float = 0.4               # porosity of deposits
    taustar_c: float = 0.0         # critical Shields stress
    n: float = 2.5                 # exponent of the transport relation
    alpha_r: float = 15.0          # Manning-Strickler coefficient
    nk: float = 2.0                # roughness height in grain diameters
    alpha_so: float = 11.25        # transport relation coefficients (Parker et al. 1998)
    alpha_sa: float = 1.5
    S_incise: float = 1e-6         # slope given to reaches that would otherwise be flat or reversed
    base_init: float = 50.0        # initial elevation at the shoreline, m
    H_down: float = 5.0            # flow depth at the shoreline, m
    sea_rise: float = 0.0          # sea-level rise, m per effective year scaled by I_tavg
    subside: float = 0.001         # subsidence, m/yr
    a_u: float = 1.0               # upwinding weight in the Exner equation
    small: float = 1e-7            # threshold for "flat" slopes (the Fortran's 10e-8)


def _consts(p: Params):
    f = float
    c = dict(
        a=float(p.a), Qw=float(p.Qw), Qs=f(p.Qs), tmax=float(p.tmax), teq=float(p.teq),
        dtmax=f(p.dtmax), dtmin=f(p.dtmin), dtredu=float(p.dtredu), dtprint=float(p.dtprint),
        dtprint2=f(p.dtprint2), dtprint3=f(p.dtprint3), I_tavg=f(p.I_tavg), beta=float(p.beta),
        g=f(p.g), D=f(p.D), R=f(p.R), lam=f(p.lam), taustar_c=float(p.taustar_c), n=float(p.n),
        alpha_r=float(p.alpha_r), nk=float(p.nk), alpha_so=f(p.alpha_so), alpha_sa=f(p.alpha_sa),
        S_incise=f(p.S_incise), base_init=float(p.base_init), H_down=float(p.H_down),
        sea_rise=float(p.sea_rise), subside=f(p.subside), a_u=float(p.a_u), small=f(p.small),
        k274=f(0.274), k565=f(0.565), half=f(0.5),
    )
    c["ks"] = c["nk"] * c["D"]
    c["pexp"] = 1.0 / 6.0
    # channel width from Church and Rood (1983)
    c["B"] = c["D"] * c["k274"] * (c["Qw"] / ((c["R"] * c["g"] * c["D"]) ** c["half"] * (c["D"] ** 2.0))) ** c["k565"]
    c["qw0"] = c["Qw"] / c["B"]
    c["qs0"] = c["Qs"] / c["B"]
    pe = c["pexp"]
    c["S_equil"] = ((c["R"] * ((c["Qs"] / (c["alpha_sa"] * c["alpha_so"]) / c["D"] / c["B"] / np.sqrt(c["R"] * c["g"] * c["D"])) ** (1 / c["n"])
                               + c["taustar_c"])) ** ((3 + 2 * pe) / (2 + 2 * pe))) \
        * (c["alpha_r"] ** (1 / (1 + pe))) * ((c["Qw"] / c["D"] / c["B"] / np.sqrt(c["g"] * c["D"])) ** (-1 / (1 + pe)))
    c["fine_dep"] = c["half"] * c["subside"] / c["I_tavg"]
    c["sec_yr"] = 365.0 * 24.0 * 60.0 * 60.0
    return c


def initial_bed(p: Params = None, seed=None, rng=None, roughness=1.0):
    """Initial bed: a plane at the equilibrium slope with random roughness of +-0.5 cell
    drop, as in the Fortran (i_etaflag = 0). Returns an (xmax, ymax + 1) array.

    roughness scales the random part. On domains much longer than the default 50 cells,
    the full roughness can leave reversed slopes in the first channel once it is cut to
    its normal depth, and the run stops at the first step; a smaller value avoids this."""
    p = p or Params()
    c = _consts(p)
    rng = rng or np.random.default_rng(seed)
    S = c["S_equil"]
    a = c["a"]
    Lb = a * p.ymax
    eta = np.empty((p.xmax, p.ymax + 1))
    for i in range(p.xmax):
        for j in range(p.ymax + 1):
            temp = rng.random()
            eta[i, j] = c["base_init"] + S * Lb - S * a * ((j + 1) - 1.0) + roughness * (S * a * (temp - 0.5))
    return eta


@dataclass
class Result:
    """Results of a run. Grids are (xmax, ymax + 1), rows across the delta, columns from
    the source (column 0) to the shoreline."""
    params: Params
    t: float
    eta: np.ndarray                # final bed elevation, m
    levee: np.ndarray              # final levee (bank) elevation, m
    channel: np.ndarray            # final active channel cells (1)
    avulsions: np.ndarray          # (n, 3): time, x (m) and y (m) of each avulsion
    Havg: np.ndarray               # mean channel flow depth after spin-up, m
    visit_mean: np.ndarray         # mean duration of channel occupation (residence time) per cell
    visit_num: np.ndarray          # number of completed occupations per cell
    return_mean: np.ndarray        # mean time between successive channel arrivals per cell
    return_num: np.ndarray
    visit_series: np.ndarray       # (xmax, ymax + 1, nvis) individual occupation durations
    return_series: np.ndarray
    visits_dropped: int            # occupations beyond nvis that were counted but not stored
    depth_channel: np.ndarray      # thickness of channel deposits laid down after spin-up, m
    depth_overbank: np.ndarray     # thickness of overbank deposits laid down after spin-up, m
    n_channel_layers: np.ndarray   # number of separate channel bodies in each column
    strat_thickness: np.ndarray    # (xmax, ymax + 1, nlayers) layer thickness (layer 0 is the base elevation)
    strat_type: np.ndarray         # 0 initial, 1 overbank, 2 channel, -1 none
    strat_time: np.ndarray         # time each layer started
    top_layer: np.ndarray          # index of the top layer in each column
    qs_series: np.ndarray          # (n, 8) sediment flux in the channel at rows 2,5,10,20,25,30,40,50
    avulsion_index: np.ndarray = None   # position along the old channel (1 = source) of each avulsion
    path_log: np.ndarray = None    # (event, i, j), 1-based: cells of each new channel path (event -1: initial)
    snapshots: dict = field(default_factory=dict)   # 'eta','levee','qs','H','channel' (n, xmax, ymax+1), 't'

    @property
    def num_avulsions(self):
        return self.avulsions.shape[0]


def run(p: Params = None, eta0=None, seed=None, save_snapshots=False, nvis=100,
        stratigraphy=True):
    """Run the model.

    p        : Params (defaults: the May 2007 settings)
    eta0     : initial bed, (xmax, ymax + 1); default initial_bed(p, seed)
    save_snapshots : keep the grid every dtprint (eta, levee, sediment flux, depth, channel)
    nvis     : number of individual visit durations stored per cell (the Fortran keeps 100)
    stratigraphy : False skips the stratigraphy bookkeeping (it does not affect the flow or
               the bed), which saves memory on large domains
    """
    p = p or Params()
    c = _consts(p)
    if eta0 is None:
        eta0 = initial_bed(p, seed)
    eta0 = np.ascontiguousarray(eta0, dtype=float)
    out = run_kernel(
        eta0, p.xmax, p.ymax, p.imidx, c["a"], c["qw0"], c["qs0"], c["B"],
        c["tmax"], c["teq"], c["dtmax"], c["dtmin"], c["dtredu"], c["dtprint"], c["dtmax"],
        c["dtprint2"], c["dtprint2"], c["dtprint3"], c["dtprint3"], c["I_tavg"], c["sec_yr"],
        c["beta"], c["g"], c["D"], c["R"], c["lam"], c["taustar_c"], c["n"], c["alpha_r"],
        c["ks"], c["alpha_so"], c["alpha_sa"], c["S_incise"], c["H_down"], c["sea_rise"],
        c["subside"], c["fine_dep"], c["a_u"], c["small"], bool(save_snapshots),
        256, int(nvis), bool(stratigraphy))
    (status, t, num_avul, num_act, eta, levee, latbond, top_lay, avul, strat, styp, stim, Havg,
     visit, vnum, visit2, vnum2, Tv, Tv2, vover, dch, dov, nchl, qout, snap, snap_t, nsnap, avul_ind, hist) = out
    if status != 0:
        raise RuntimeError(f"time step fell below dtmin at t = {t:.6f}")
    X, Y = p.xmax, p.ymax + 1
    g2 = lambda A: A[1:X + 1, 1:Y + 1].copy()
    res = Result(
        params=p, t=t, eta=g2(eta), levee=g2(levee), channel=g2(latbond),
        avulsions=avul, Havg=g2(Havg), visit_mean=g2(visit), visit_num=g2(vnum),
        return_mean=g2(visit2), return_num=g2(vnum2),
        visit_series=Tv[1:X + 1, 1:Y + 1, 1:].copy(), return_series=Tv2[1:X + 1, 1:Y + 1, 1:].copy(),
        visits_dropped=int(vover), depth_channel=g2(dch), depth_overbank=g2(dov),
        n_channel_layers=g2(nchl), strat_thickness=strat[1:X + 1, 1:Y + 1, 1:].copy(),
        strat_type=styp[1:X + 1, 1:Y + 1, 1:].copy(), strat_time=stim[1:X + 1, 1:Y + 1, 1:].copy(),
        top_layer=g2(top_lay) - 1, qs_series=qout[1:].copy(),
        avulsion_index=avul_ind, path_log=hist,
    )
    if save_snapshots:
        names = ["eta", "levee", "qs", "H", "channel"]
        res.snapshots = {nm: snap[:, q, 1:X + 1, 1:Y + 1].copy() for q, nm in enumerate(names)}
        res.snapshots["t"] = snap_t
    return res
