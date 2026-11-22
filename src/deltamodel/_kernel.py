"""
Time-stepping kernel of the delta model: a port of delta5_main.f95 (May 2007) and its
subroutines (slope5, backwater, incise_H, sedplain, strat3D, incise, tim_visit), with
the corrections of 2026 (marked `_fix N`, as listed in the README).

Arrays use the Fortran 1-based indices (index 0 unused) so the code can be read side by
side with fortran/. The order of floating-point operations follows the Fortran, so the
kernel reproduces the corrected Fortran exactly.
"""
import numpy as np
from numba import njit

NEG = -999.0


@njit(cache=True)
def _slope5(i, j, latbond, active, eta, L, S, ci, cj):
    """Slopes to the five downstream/lateral neighbours (SLOPE5). Channel cells and cells
    outside the domain get -999. Same arithmetic as the Fortran SLOPE5; the domain edge
    is given by the `active` mask (for the rectangle, the padding around the grid)."""
    for q in range(1, 6):
        ni = i + ci[q]
        nj = j + cj[q]
        if active[ni, nj] == 1 and latbond[ni, nj] != 1:
            S[q] = (eta[i, j] - eta[ni, nj]) / L[q]
        else:
            S[q] = NEG


@njit(cache=True)
def _boxed_in(i, j, active, eta, ci, cj):
    """Direction for a path step from a cell whose five neighbours are all channel or
    outside the domain (_fix 7): straight downstream if that cell is in the domain (always
    so in the rectangle), otherwise the lowest neighbour in the domain."""
    if active[i + ci[3], j + cj[3]] == 1:
        return 3
    best = -1
    for q in range(1, 6):
        if active[i + ci[q], j + cj[q]] == 1:
            if best < 0 or eta[i + ci[q], j + cj[q]] < eta[i + ci[best], j + cj[best]]:
                best = q
    return best


@njit(cache=True)
def _maxloc(S):
    """Fortran MAXLOC over S(1:5): first index of the maximum."""
    m = 1
    for q in range(2, 6):
        if S[q] > S[m]:
            m = q
    return m


@njit(cache=True)
def _backwater(num_act, H_0, S_act, dL_act, ks, qw, alpha_r, g, H):
    """Flow depths along the active channel from the backwater equation, integrated
    upstream from the downstream depth H_0 with a predictor-corrector step (BACKWATER and
    INCISE_H). Falls back to normal depth where backwater does not apply."""
    H[num_act] = H_0
    k = num_act
    flag_norm = 0
    while k != 1:
        H_temp = H[k]
        S_temp = S_act[k - 1]
        dL = dL_act[k - 1]
        H_norm = (ks ** (1.0 / 3.0) * qw ** 2.0 / alpha_r ** 2.0 / g / S_temp) ** (3.0 / 10.0)
        if flag_norm == 0:
            Fr2_p = qw ** 2.0 / g / (H_temp ** 3.0)
            Cf = (1.0 / alpha_r ** 2.0) * ((H_temp / ks) ** (-1.0 / 3.0))
            fnp = (S_temp - Cf * Fr2_p) * dL / (1.0 - Fr2_p)
            Hpred = H_temp - fnp
            if Hpred > 0.0:
                Fr2 = qw ** 2.0 / g / (Hpred ** 3.0)
                # _fix 6: evaluate the friction coefficient at the predicted depth
                Cf = (1.0 / alpha_r ** 2.0) * ((Hpred / ks) ** (-1.0 / 3.0))
                fn = (S_temp - Cf * Fr2) * dL / (1.0 - Fr2)
                H[k - 1] = H_temp - 0.5 * (fnp + fn)
            else:
                H[k - 1] = H_norm
                flag_norm = 1
            if H[k - 1] < H_norm:
                H[k - 1] = H_norm
                flag_norm = 1
            elif H[k - 1] > H_temp:
                H[k - 1] = H_norm
                flag_norm = 1
        else:
            H[k - 1] = H_norm
        k = k - 1


@njit(cache=True)
def _grow_layers(strat, sed_typ, sed_tim, nz):
    nx, ny, n0 = strat.shape
    s2 = np.zeros((nx, ny, nz))
    t2 = np.full((nx, ny, nz), -1, dtype=np.int32)
    m2 = np.full((nx, ny, nz), -1.0)
    s2[:, :, :n0] = strat
    t2[:, :, :n0] = sed_typ
    m2[:, :, :n0] = sed_tim
    return s2, t2, m2


@njit(cache=True)
def _strat3d(xmax, ymax, delta_strat, sed_typ, sed_tim, top_lay, strat):
    """Write the deposition (or erosion) accumulated since the last avulsion into the
    stratigraphic record (STRAT3D)."""
    for ii in range(1, xmax + 1):
        for jj in range(1, ymax + 2):
            layernow = top_lay[ii, jj]
            if delta_strat[ii, jj] > 0.0:
                strat[ii, jj, layernow] = delta_strat[ii, jj]
            else:
                elev_remain = delta_strat[ii, jj]
                while True:
                    if elev_remain > 0.0:
                        break
                    elev_remain = strat[ii, jj, layernow] + elev_remain
                    if elev_remain > 0.0:
                        strat[ii, jj, layernow] = elev_remain
                    else:
                        strat[ii, jj, layernow] = 0.0
                        sed_typ[ii, jj, layernow] = -1
                        sed_tim[ii, jj, layernow] = -1.0
                        top_lay[ii, jj] = top_lay[ii, jj] - 1
                        layernow = top_lay[ii, jj]
                    if layernow == 1:
                        if elev_remain < 0.0:
                            # _fix 8: erosion into the initial deposit lowers the base layer
                            strat[ii, jj, 1] = strat[ii, jj, 1] + elev_remain
                        break
            delta_strat[ii, jj] = 0.0


@njit(cache=True)
def _incise(ii, jj, delta_incise, sed_typ, sed_tim, top_lay, strat):
    """Remove an incised thickness from the top of the stratigraphic column (INCISE)."""
    layernow = top_lay[ii, jj]
    elev_remain = strat[ii, jj, layernow] + delta_incise
    if elev_remain > 0.0:
        strat[ii, jj, layernow] = strat[ii, jj, layernow] + delta_incise
    else:
        elev_remain = delta_incise
        while True:
            if elev_remain > 0.0:
                break
            elev_remain = strat[ii, jj, layernow] + elev_remain
            if elev_remain > 0.0:
                strat[ii, jj, layernow] = elev_remain
            else:
                strat[ii, jj, layernow] = 0.0
                sed_typ[ii, jj, layernow] = -1
                sed_tim[ii, jj, layernow] = -1.0
                top_lay[ii, jj] = top_lay[ii, jj] - 1
                layernow = top_lay[ii, jj]
            if layernow == 1:
                if elev_remain < 0.0:
                    strat[ii, jj, 1] = strat[ii, jj, 1] + elev_remain   # _fix 8
                break


@njit(cache=True)
def _next_len(i, j, i2, j2, L):
    """Length from (i, j) to an adjacent cell (i2, j2)."""
    if i2 != i and j2 != j:
        return L[2]
    return L[1]


@njit(cache=True)
def run_kernel(eta0, active, terminal, xmax, ymax, imidx, a,
               qw_zero0, qs_zero0, B_width0,
               tmax, teq, dtmax, dtmin, dtredu, dtprint, tprint0, tprint2_0, dtprint2,
               tprint3_0, dtprint3, I_tavg, sec_yr, beta, g, D, R, lam, taustar_c, n,
               alpha_r, ks, alpha_so, alpha_sa, S_incise, H_down, sea_rise, subside,
               fine_dep, a_u, small, save_snapshots, nlay0, nvis, strat_on=True):
    nx = xmax + 2
    ny = ymax + 3
    sqrt2 = np.sqrt(2.0)
    L = np.zeros(6)
    L[1] = a
    L[2] = sqrt2 * a
    L[3] = a
    L[4] = sqrt2 * a
    L[5] = a
    ci = np.array([0, -1, -1, 0, 1, 1])
    cj = np.array([0, 0, 1, 1, 1, 0])
    S = np.zeros(6)

    eta = np.zeros((nx, ny))
    levee = np.zeros((nx, ny))
    old_eta = np.zeros((nx, ny))
    old_levee = np.zeros((nx, ny))
    Hij = np.zeros((nx, ny))
    qsij = np.zeros((nx, ny))
    delta_agg = np.zeros((nx, ny))
    delta_eta = np.zeros((nx, ny))
    delta_strat = np.zeros((nx, ny))
    Havg = np.zeros((nx, ny))
    Navg = np.zeros((nx, ny))
    latbond = np.zeros((nx, ny), dtype=np.int32)
    top_lay = np.zeros((nx, ny), dtype=np.int32)
    # visit statistics
    last_t = np.zeros((nx, ny))
    last_t2 = np.zeros((nx, ny))
    visit = np.zeros((nx, ny))
    visit2 = np.zeros((nx, ny))
    flag_visit = np.zeros((nx, ny), dtype=np.int32)
    flag_visit2 = np.zeros((nx, ny), dtype=np.int32)
    visit_num = np.zeros((nx, ny), dtype=np.int32)
    visit_num2 = np.zeros((nx, ny), dtype=np.int32)
    Tv = np.zeros((nx, ny, nvis + 1))
    Tv2 = np.zeros((nx, ny, nvis + 1))
    vis_over = 0
    # stratigraphy
    nlay = nlay0
    strat = np.zeros((nx, ny, nlay))
    sed_typ = np.full((nx, ny, nlay), -1, dtype=np.int32)
    sed_tim = np.full((nx, ny, nlay), -1.0)

    ncell = xmax * ymax + 2
    S_act = np.zeros(ncell + 2)
    dL_act = np.zeros(ncell + 2)
    chan = np.zeros(ncell + 2)
    H = np.zeros(ncell + 2)
    qt = np.zeros(ncell + 2)
    S_index = np.zeros(ncell + 2, dtype=np.int64)
    eta_act = np.zeros(ncell + 2)
    old_eta_act = np.zeros(ncell + 2)
    levee_act = np.zeros(ncell + 2)
    old_levee_act = np.zeros(ncell + 2)
    i_vec = np.zeros(ncell + 2, dtype=np.int64)
    j_vec = np.zeros(ncell + 2, dtype=np.int64)

    # outputs
    nav_cap = 1024
    avul_out = np.zeros((nav_cap, 3))
    num_avul = 0
    # channel history: for each avulsion, the index along the old path where it happened
    # and the cells of the new path below it (initial path is event -1)
    avul_ind = np.zeros(nav_cap, dtype=np.int64)
    hist_cap = 65536
    hist = np.zeros((hist_cap, 3), dtype=np.int64)
    nhist = 0
    nprint_cap = int(tmax / dtprint) + 5
    if save_snapshots:
        snap = np.zeros((nprint_cap, 5, nx, ny))
    else:
        snap = np.zeros((1, 5, 1, 1))
    snap_t = np.zeros(nprint_cap)
    nsnap = 0
    qcols = np.array([2, 5, 10, 20, 25, 30, 40, 50])
    nq_cap = int(tmax / dtprint3) + 10
    qout = np.zeros((nq_cap, 8))
    i_out = 1
    status = 0

    # ---- initial bed ----
    for i in range(1, xmax + 1):
        for j in range(1, ymax + 2):
            eta[i, j] = eta0[i - 1, j - 1]
            levee[i, j] = eta[i, j]
            old_levee[i, j] = levee[i, j]
            old_eta[i, j] = eta[i, j]
            strat[i, j, 1] = eta[i, j]
            sed_typ[i, j, 1] = 0
            sed_tim[i, j, 1] = 0.0
            sed_typ[i, j, 2] = 1
            sed_tim[i, j, 2] = 0.0
            top_lay[i, j] = 2

    qw_zero = qw_zero0
    qs_zero = qs_zero0
    B_width = B_width0

    # ---- initial channel: path of steepest descent ----
    latbond[imidx, 1] = 1
    sed_typ[imidx, 1, 2] = 2
    eta_act[1] = eta[imidx, 1]
    levee_act[1] = levee[imidx, 1]
    i = imidx
    j = 1
    i_vec[1] = i
    j_vec[1] = j
    num_act = 1
    while True:
        if terminal[i, j] == 1:
            break
        _slope5(i, j, latbond, active, eta, L, S, ci, cj)
        it0 = _maxloc(S)
        if S[it0] == NEG:
            it0 = _boxed_in(i, j, active, eta, ci, cj)   # _fix 7: boxed in
        i_old = i
        j_old = j
        i = i + ci[it0]
        j = j + cj[it0]
        latbond[i, j] = 1
        S_index[num_act] = it0
        if S[it0] < small:
            S[it0] = S_incise
            new_eta = eta[i_old, j_old] - S_incise * L[it0]
            if new_eta > 0.0:
                delta_incise = new_eta - eta[i, j]
                eta[i, j] = new_eta
            else:
                delta_incise = -eta[i, j]
                eta[i, j] = 0.0
            levee[i, j] = levee[i, j] + delta_incise
        S_act[num_act] = S[it0]
        dL_act[num_act] = L[it0]
        eta_act[num_act + 1] = eta[i, j]
        levee_act[num_act + 1] = levee[i, j]
        i_vec[num_act + 1] = i
        j_vec[num_act + 1] = j
        num_act = num_act + 1

    for k in range(1, num_act + 1):
        hist[nhist, 0] = -1
        hist[nhist, 1] = i_vec[k]
        hist[nhist, 2] = j_vec[k]
        nhist += 1
    H_0 = H_down
    _backwater(num_act, H_0, S_act, dL_act, ks, qw_zero, alpha_r, g, chan)
    for k in range(1, num_act + 1):
        i = i_vec[k]
        j = j_vec[k]
        depth_temp = levee_act[k] - eta_act[k]
        if depth_temp < chan[k]:
            eta_act[k] = levee_act[k] - chan[k]
            eta[i, j] = eta_act[k]
            old_eta[i, j] = eta[i, j]                 # _fix 5
        sed_typ[i, j, 2] = 2
        strat[i, j, 1] = eta[i, j]
    for k in range(ncell + 2):
        old_eta_act[k] = eta_act[k]
        old_levee_act[k] = levee_act[k]

    t = 0.0
    tprint = tprint0
    tprint2 = tprint2_0
    tprint3 = tprint3_0
    istorm = 1
    next_t = np.array([0.0, 0.0, tmax])

    # ================= time loop =================
    while True:
        dt = dtmax
        t = t + dt
        if t > tmax:
            break
        if t > next_t[istorm]:
            qw_zero = qw_zero0
            qs_zero = qs_zero0
            B_width = B_width0
            istorm = istorm + 1
        qsij[imidx, 1] = qs_zero

        # ---- channel visits (TIM_VISIT) ----
        if t > teq:
            for i in range(1, xmax + 1):
                for j in range(1, ymax + 2):
                    if latbond[i, j] == 1 and flag_visit[i, j] == 0:
                        last_t[i, j] = t
                        flag_visit[i, j] = 1
                    elif latbond[i, j] == 0 and flag_visit[i, j] == 1:
                        visit_num[i, j] = visit_num[i, j] + 1
                        tv = t - last_t[i, j]
                        if visit_num[i, j] <= nvis:
                            Tv[i, j, visit_num[i, j]] = tv
                        else:
                            vis_over += 1
                        visit[i, j] = visit[i, j] + tv
                        flag_visit[i, j] = 0
                    if latbond[i, j] == 1 and flag_visit2[i, j] == 0:
                        last_t2[i, j] = t
                        flag_visit2[i, j] = 1
                    elif latbond[i, j] == 0 and flag_visit2[i, j] == 1:
                        flag_visit2[i, j] = 2
                    elif latbond[i, j] == 1 and flag_visit2[i, j] == 2:
                        visit_num2[i, j] = visit_num2[i, j] + 1
                        tv = t - last_t2[i, j]
                        if visit_num2[i, j] <= nvis:
                            Tv2[i, j, visit_num2[i, j]] = tv
                        else:
                            vis_over += 1
                        visit2[i, j] = visit2[i, j] + tv
                        flag_visit2[i, j] = 1
                        last_t2[i, j] = t
            H_0 = H_down + sea_rise * ((t - teq) / I_tavg)

        # ---- flow, transport and bed update, with time-step cuts on scour ----
        flag_etacheck = 0
        while True:
            if flag_etacheck == 1:
                break
            flag_scour = 0
            _backwater(num_act, H_0, S_act, dL_act, ks, qw_zero, alpha_r, g, H)
            for k in range(1, num_act + 1):
                Hij[i_vec[k], j_vec[k]] = H[k]
            for k in range(1, num_act + 1):
                i = i_vec[k]
                j = j_vec[k]
                Cf = (1.0 / alpha_r ** 2.0) * (H[k] / ks) ** (-1.0 / 3.0)
                tau_star = Cf * ((qw_zero / H[k]) ** 2.0) / (R * g * D)
                if k < num_act:
                    S_temp = S_act[k]
                    tau_norm = ((ks ** (1.0 / 3.0) * qw_zero ** 2.0 / alpha_r ** 2.0 / g) ** (3.0 / 10.0)) \
                        * (S_temp ** (7.0 / 10.0)) / R / D
                    if tau_star > tau_norm:
                        tau_star = tau_norm
                if tau_star > taustar_c:
                    qstar = alpha_so * alpha_sa * (tau_star - taustar_c) ** n
                else:
                    qstar = 0.0
                qt[k] = np.sqrt(R * g * D) * D * qstar
                qsij[i, j] = qt[k]

            for k in range(1, num_act + 1):
                i = i_vec[k]
                j = j_vec[k]
                qt_now = qt[k]
                if k == 1:
                    qtback = qs_zero
                    dL = L[2]
                else:
                    qtback = qt[k - 1]
                    dL = dL_act[k - 1]
                qtfrnt = 0.0
                if k == num_act:
                    qt_dif = qtback - qt_now
                else:
                    qtfrnt = qt[k + 1]
                    qt_dif = a_u * (qtback - qt_now) + (1.0 - a_u) * (qt_now - qtfrnt)
                delta_agg[i, j] = (dt * sec_yr) / (1.0 - lam) / dL * qt_dif
                delta_eta[i, j] = delta_agg[i, j] - subside * dt / I_tavg
                elev_temp = eta[i, j] + delta_eta[i, j]
                if elev_temp < 0.0:
                    delta_agg[i, j] = -eta[i, j] + subside * dt / I_tavg
                    if delta_agg[i, j] < 0.0:
                        delta_agg[i, j] = 0.0
                    delta_eta[i, j] = -eta[i, j]
                    qt_dif = delta_agg[i, j] * (1.0 - lam) * dL / (dt * sec_yr)
                    eta[i, j] = 0.0
                    if k == num_act:
                        qt[k] = qtback - qt_dif
                    else:
                        # _fix 1: invert qt_dif = a_u (qtback - qt) + (1 - a_u)(qt - qtfrnt)
                        qt[k] = (qt_dif - a_u * qtback + (1.0 - a_u) * qtfrnt) / (1.0 - 2.0 * a_u)
                else:
                    eta[i, j] = elev_temp
                levee[i, j] = levee[i, j] + delta_eta[i, j]
                eta_act[k] = eta[i, j]
                levee_act[k] = levee[i, j]
                if k > 1 and eta_act[k] > eta_act[k - 1]:
                    flag_scour = 1

            if flag_scour == 0:
                flag_etacheck = 1
            else:
                t = t - dt
                dt = dt * dtredu
                if dt < dtmin:
                    status = 1
                    break
                t = t + dt
                for jj in range(1, ymax + 2):
                    for ii in range(1, xmax + 1):
                        Hij[ii, jj] = 0.0
                        qsij[ii, jj] = 0.0
                        delta_eta[ii, jj] = 0.0
                        delta_agg[ii, jj] = 0.0
                for k in range(ncell + 2):
                    eta_act[k] = old_eta_act[k]
                    levee_act[k] = old_levee_act[k]
                for ii in range(nx):
                    for jj in range(ny):
                        eta[ii, jj] = old_eta[ii, jj]
                        levee[ii, jj] = old_levee[ii, jj]
        if status != 0:
            break

        # ---- floodplain (SEDPLAIN) and aggradation since the last avulsion ----
        for i in range(1, xmax + 1):
            for j in range(1, ymax + 2):
                if latbond[i, j] != 1 and active[i, j] == 1:
                    subside_now = subside * dt / I_tavg
                    delta_agg[i, j] = fine_dep * dt
                    delta_eta[i, j] = delta_agg[i, j] - subside_now
                    eta[i, j] = eta[i, j] + delta_eta[i, j]
                    levee[i, j] = levee[i, j] + delta_eta[i, j]
                delta_strat[i, j] = delta_strat[i, j] + delta_agg[i, j]

        # ---- avulsion ----
        avul_flag = 0
        k = 1
        while True:
            if k == num_act or avul_flag == 1:
                break
            i = i_vec[k]
            j = j_vec[k]
            i_act = i_vec[k + 1]
            j_act = j_vec[k + 1]
            _slope5(i, j, latbond, active, eta, L, S, ci, cj)
            it0 = _maxloc(S)
            L_act = _next_len(i, j, i_act, j_act, L)       # _fix 3
            avul_cond = beta * Hij[i, j]
            go = True
            if S[it0] == NEG:
                go = False                                 # _fix 7: nowhere to avulse to
            ni = i + ci[it0]
            nj = j + cj[it0]
            if go and ((eta[i, j] - avul_cond - eta[ni, nj]) / L[it0]) > \
                    ((eta[i, j] - eta[i_act, j_act]) / L_act):
                avul_flag = 1
                ind_avul = k
                if num_avul >= nav_cap:
                    a2 = np.zeros((2 * nav_cap, 3))
                    a2[:nav_cap] = avul_out
                    avul_out = a2
                    b2 = np.zeros(2 * nav_cap, dtype=np.int64)
                    b2[:nav_cap] = avul_ind
                    avul_ind = b2
                    nav_cap = 2 * nav_cap
                avul_out[num_avul, 0] = t
                avul_out[num_avul, 1] = a * float(i)
                avul_out[num_avul, 2] = a * float(j)
                avul_ind[num_avul] = ind_avul
                num_avul = num_avul + 1

                if strat_on:
                    _strat3d(xmax, ymax, delta_strat, sed_typ, sed_tim, top_lay, strat)

                for jj in range(ind_avul, num_act):
                    i_now = i_vec[jj + 1]
                    j_now = j_vec[jj + 1]
                    dL_act[jj] = 0.0
                    S_act[jj] = 0.0
                    S_index[jj] = 0
                    latbond[i_now, j_now] = 0
                    eta_act[jj + 1] = 0.0
                    levee_act[jj + 1] = 0.0
                    chan[jj + 1] = H[jj + 1]
                    H[jj + 1] = 0.0
                    i_vec[jj + 1] = 0
                    j_vec[jj + 1] = 0

                num_act = ind_avul
                k = ind_avul
                while True:
                    i = i_vec[k]
                    j = j_vec[k]
                    if terminal[i, j] == 1:
                        break
                    _slope5(i, j, latbond, active, eta, L, S, ci, cj)
                    it0 = _maxloc(S)
                    if S[it0] == NEG:
                        it0 = _boxed_in(i, j, active, eta, ci, cj)   # _fix 7
                    i = i + ci[it0]
                    j = j + cj[it0]
                    latbond[i, j] = 1
                    S_index[k] = it0
                    dL_act[k] = L[it0]
                    S_act[k] = S[it0]
                    eta_act[k + 1] = eta[i, j]
                    levee_act[k + 1] = levee[i, j]
                    i_vec[k + 1] = i
                    j_vec[k + 1] = j
                    num_act = num_act + 1
                    k = k + 1

                if nhist + num_act >= hist_cap:
                    h2 = np.zeros((2 * hist_cap + num_act, 3), dtype=np.int64)
                    h2[:nhist] = hist[:nhist]
                    hist = h2
                    hist_cap = h2.shape[0]
                for kk in range(ind_avul + 1, num_act + 1):
                    hist[nhist, 0] = num_avul - 1
                    hist[nhist, 1] = i_vec[kk]
                    hist[nhist, 2] = j_vec[kk]
                    nhist += 1
                for k in range(ind_avul + 1, num_act + 1):
                    i = i_vec[k]
                    j = j_vec[k]
                    if chan[k] < small:
                        chan[k] = H_0
                    depth_temp = levee_act[k] - eta_act[k]
                    if depth_temp < chan[k]:
                        eta_before = eta[i, j]
                        eta_act[k] = levee_act[k] - chan[k]
                        depth_temp = chan[k]
                        eta[i, j] = eta_act[k]
                        delta_incise = eta[i, j] - eta_before   # _fix 9: remove what was cut
                        if strat_on:
                            _incise(i, j, delta_incise, sed_typ, sed_tim, top_lay, strat)

                for k in range(ind_avul, num_act):
                    i = i_vec[k]
                    j = j_vec[k]
                    i_next = i_vec[k + 1]
                    j_next = j_vec[k + 1]
                    i_on = S_index[k]
                    dL_act[k] = L[i_on]
                    S_act[k] = (eta_act[k] - eta_act[k + 1]) / L[i_on]
                    if S_act[k] < small:
                        S_act[k] = S_incise
                        new_eta = eta[i, j] - S_incise * L[i_on]   # _fix 2
                        delta_incise = new_eta - eta[i_next, j_next]
                        if new_eta > 0.0:
                            eta[i_next, j_next] = new_eta
                        else:
                            delta_incise = -eta[i_next, j_next]
                            eta[i_next, j_next] = 0.0
                        if strat_on:
                            _incise(i_next, j_next, delta_incise, sed_typ, sed_tim, top_lay, strat)
                    eta_act[k + 1] = eta[i_next, j_next]

                # new layer everywhere
                if strat_on:
                    maxtop = 0
                    for ii in range(1, xmax + 1):
                        for jj in range(1, ymax + 2):
                            if top_lay[ii, jj] + 1 > maxtop:
                                maxtop = top_lay[ii, jj] + 1
                    if maxtop >= nlay:
                        strat, sed_typ, sed_tim = _grow_layers(strat, sed_typ, sed_tim, 2 * nlay)
                        nlay = 2 * nlay
                    for ii in range(1, xmax + 1):
                        for jj in range(1, ymax + 2):
                            top_lay[ii, jj] = top_lay[ii, jj] + 1
                            ln = top_lay[ii, jj]
                            if latbond[ii, jj] == 1:
                                sed_typ[ii, jj, ln] = 2
                            else:
                                sed_typ[ii, jj, ln] = 1
                            sed_tim[ii, jj, ln] = t
            k = k + 1

        for jx in range(1, num_act):
            i_on = S_index[jx]
            dL_act[jx] = L[i_on]
            S_act[jx] = (eta_act[jx] - eta_act[jx + 1]) / L[i_on]

        # ---- output ----
        if t - dt > tprint:
            if save_snapshots and nsnap < nprint_cap:
                for ii in range(nx):
                    for jj in range(ny):
                        snap[nsnap, 0, ii, jj] = eta[ii, jj]
                        snap[nsnap, 1, ii, jj] = levee[ii, jj]
                        snap[nsnap, 2, ii, jj] = qsij[ii, jj] * B_width
                        snap[nsnap, 3, ii, jj] = Hij[ii, jj]
                        snap[nsnap, 4, ii, jj] = latbond[ii, jj]
            if nsnap < nprint_cap:
                snap_t[nsnap] = t
            nsnap += 1
            tprint = tprint + dtprint
        if t > teq:
            if t - dt > tprint2:
                for ii in range(1, xmax + 1):
                    for jj in range(1, ymax + 2):
                        if Hij[ii, jj] > 0.0:
                            Havg[ii, jj] = Havg[ii, jj] + Hij[ii, jj]
                            Navg[ii, jj] = Navg[ii, jj] + 1.0
                tprint2 = tprint2 + dtprint2
            if t - dt > tprint3:
                if i_out >= nq_cap:
                    q2 = np.zeros((2 * nq_cap, 8))
                    q2[:nq_cap] = qout
                    qout = q2
                    nq_cap = 2 * nq_cap
                for ii in range(1, xmax + 1):
                    for c in range(8):
                        if latbond[ii, qcols[c]] == 1:
                            qout[i_out, c] = qsij[ii, qcols[c]] * B_width
                tprint3 = tprint3 + dtprint3
                i_out = i_out + 1

        for jj in range(1, ymax + 2):
            for ii in range(1, xmax + 1):
                Hij[ii, jj] = 0.0
                qsij[ii, jj] = 0.0
                delta_eta[ii, jj] = 0.0
                delta_agg[ii, jj] = 0.0
        for ii in range(nx):
            for jj in range(ny):
                old_eta[ii, jj] = eta[ii, jj]
                old_levee[ii, jj] = levee[ii, jj]
        for k in range(ncell + 2):
            old_eta_act[k] = eta_act[k]
            old_levee_act[k] = levee_act[k]
            chan[k] = 0.0
            H[k] = 0.0
            qt[k] = 0.0

    # ---- end of run ----
    if strat_on:
        _strat3d(xmax, ymax, delta_strat, sed_typ, sed_tim, top_lay, strat)   # _fix 4
    depth_chan = np.zeros((nx, ny))
    depth_over = np.zeros((nx, ny))
    num_ch_lay = np.zeros((nx, ny), dtype=np.int32)
    for ii in range(1, xmax + 1):
        for jj in range(1, ymax + 2):
            flag_layer = 1
            kmax = top_lay[ii, jj]                                     # _fix 4
            for kk in range(1, kmax + 1):
                if kk >= nlay:
                    break
                if sed_typ[ii, jj, kk] == 1 and sed_tim[ii, jj, kk] > teq:
                    depth_over[ii, jj] = depth_over[ii, jj] + strat[ii, jj, kk]
                    flag_layer = 1
                elif sed_typ[ii, jj, kk] == 2 and sed_tim[ii, jj, kk] > teq:
                    depth_chan[ii, jj] = depth_chan[ii, jj] + strat[ii, jj, kk]
                    if flag_layer == 1:
                        num_ch_lay[ii, jj] = num_ch_lay[ii, jj] + 1
                        flag_layer = 0
            if Navg[ii, jj] > 0.0:
                Havg[ii, jj] = Havg[ii, jj] / Navg[ii, jj]
            else:
                Havg[ii, jj] = 0.0
            if visit_num[ii, jj] > 0:
                visit[ii, jj] = visit[ii, jj] / visit_num[ii, jj]
            else:
                visit[ii, jj] = 0.0
            if visit_num2[ii, jj] > 0:
                visit2[ii, jj] = visit2[ii, jj] / visit_num2[ii, jj]
            else:
                visit2[ii, jj] = 0.0

    return (status, t, num_avul, num_act, eta, levee, latbond, top_lay,
            avul_out[:num_avul].copy(), strat, sed_typ, sed_tim, Havg,
            visit, visit_num, visit2, visit_num2, Tv, Tv2, vis_over,
            depth_chan, depth_over, num_ch_lay, qout[:i_out + 1].copy(),
            snap[:min(nsnap, nprint_cap)].copy(), snap_t[:min(nsnap, nprint_cap)].copy(), nsnap,
            avul_ind[:num_avul].copy(), hist[:nhist].copy())
