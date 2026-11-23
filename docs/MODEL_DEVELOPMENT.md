# RiverDeltaModel — mathematical formulation and developer guide

This document describes how the model's physical processes are represented numerically and where their calculations are implemented in the Python and Fortran codebases. It is intended for researchers reproducing results, extending model physics, or comparing implementations. The [main README](../README.md) covers model scope, installation, usage, and research applications.

The Python implementation is in [\`src/deltamodel/_kernel.py\`](../src/deltamodel/_kernel.py), with parameters, geometry and the public interface in [\`model.py\`](../src/deltamodel/model.py). The Fortran implementation is centered on [\`fortran/delta5_main.f95\`](../fortran/delta5_main.f95) and its numerical subroutines. The two implementations share the historical rectangular-domain formulation; the Python interface additionally provides a semicircular domain. Expressions below document the **implemented model**, including its effective-time conventions and discretization.

## 1. Grid, variables and numerical workflow

The delta plain is a grid of cells of nominal side length \(a\). A channel follows connected cells; a step can be axial (\(\Delta \ell=a\)) or diagonal (\(\Delta \ell=\sqrt{2}a\)). The state includes bed elevation \(\eta\), levee elevation, active-channel path, flow depth \(H\), sediment transport per unit channel width \(q_s\), and a layer-wise stratigraphic record.

| Quantity | Meaning | Code name |
|---|---|---|
| \(\eta_{i,j}\) | bed elevation | \`eta\` |
| \(H_k\) | depth at channel position \(k\) | \`H[k]\` |
| \(S_k\) | bed slope along channel segment | \`S_act[k]\` |
| \(\Delta\ell_k\) | length of that segment | \`dL_act[k]\`, \`deltaL_act(k)\` |
| \(q_w=Q_w/B\) | water flux per unit channel width | \`qw_zero\` |
| \(q_{s,k}\) | transported sediment flux per unit width | \`qt[k]\`, \`qt(k)\` |
| \(\lambda\) | sediment porosity | \`lam\`, \`lambda\` |
| \(\beta\) | avulsion threshold coefficient | \`beta\` |
| \(I\) | flood intermittency factor | \`I_tavg\` |

The run loop solves channel hydraulics, calculates sediment transport, changes bed and levee elevation, applies overbank deposition, tests for avulsion, constructs any new channel course, and records channel occupancy and stratigraphy. Parameter defaults are in [\`Params\`](../src/deltamodel/model.py); see also Fortran parameter setup in [\`delta5_main.f95\`](../fortran/delta5_main.f95).

## 2. Channel hydraulics: backwater and normal depth

The implementation uses a one-dimensional gradually varied flow formulation. The variables used in the code are

\[
Fr^2=\frac{q_w^2}{gH^3},\qquad
C_f=\frac{1}{\alpha_r^2}\left(\frac{H}{k_s}\right)^{-1/3},
\]

where \(k_s\) is roughness height and \(\alpha_r\) is the resistance coefficient. In the code's along-channel sign convention, the depth increment is calculated using

\[
\frac{dH}{dx}=\frac{S-C_f Fr^2}{1-Fr^2}.
\]

The solution is stepped *upstream* from the downstream depth \(H_0\) by a predictor–corrector method. With \(F(H,S)=(S-C_f(H)Fr^2(H))/(1-Fr^2(H))\), the implementation is

\[
H_{\rm pred}=H_k-\Delta\ell_k F(H_k,S_k),\qquad
H_{k-1}=H_k-\frac{\Delta\ell_k}{2}
\left[F(H_k,S_k)+F(H_{\rm pred},S_k)\right].
\]

Where the backwater branch is unsuitable, the code uses the normal depth

\[
H_{\rm norm}=
\left(\frac{k_s^{1/3}q_w^2}{\alpha_r^2gS}\right)^{3/10}.
\]

**Python** — [\`_backwater\`](../src/deltamodel/_kernel.py#L56-L90), representative implementation:

\`\`\`python
Fr2_p = qw ** 2.0 / g / (H_temp ** 3.0)
Cf = (1.0 / alpha_r ** 2.0) * ((H_temp / ks) ** (-1.0 / 3.0))
fnp = (S_temp - Cf * Fr2_p) * dL / (1.0 - Fr2_p)
Hpred = H_temp - fnp
# At the predicted depth, recompute both Fr2 and friction.
Fr2 = qw ** 2.0 / g / (Hpred ** 3.0)
Cf = (1.0 / alpha_r ** 2.0) * ((Hpred / ks) ** (-1.0 / 3.0))
fn = (S_temp - Cf * Fr2) * dL / (1.0 - Fr2)
H[k - 1] = H_temp - 0.5 * (fnp + fn)
\`\`\`

**Fortran** — [\`backwater.f95\`](../fortran/backwater.f95) and [\`incise_H.f95\`](../fortran/incise_H.f95), the equivalent corrector expression:

\`\`\`fortran
Cf = (1.d0/alpha_r**2.d0) * ((Hpred/ks)**(-1.d0/3.d0))
fn = (S_temp - Cf*Fr2) * deltaL_temp / (1.d0-Fr2)
\`\`\`

The normal-depth fallback and limiting checks are part of the source routine; the excerpt above does not replace those checks.

## 3. Bedload transport

The code computes dimensionless Shields stress from depth and velocity:

\[
\tau_*=\frac{C_f(q_w/H)^2}{RgD},
\]

where \(D\) is grain diameter and \(R\) is submerged specific gravity. For interior channel segments, the computed stress is capped by the normal-flow stress \(\tau_{*,\mathrm{norm}}\) (see source for the slope-dependent expression). Dimensionless transport is then

\[
q_*=
\begin{cases}
\alpha_{so}\alpha_{sa}(\tau_*-\tau_{*,c})^n,
& \tau_*>\tau_{*,c},\\
0,& \text{otherwise},
\end{cases}
\qquad
q_s=D\sqrt{RgD}\;q_*.
\]

**Python** — [transport loop](../src/deltamodel/_kernel.py#L401-L417):

\`\`\`python
Cf = (1.0 / alpha_r ** 2.0) * (H[k] / ks) ** (-1.0 / 3.0)
tau_star = Cf * ((qw_zero / H[k]) ** 2.0) / (R * g * D)
if tau_star > taustar_c:
    qstar = alpha_so * alpha_sa * (tau_star - taustar_c) ** n
else:
    qstar = 0.0
qt[k] = np.sqrt(R * g * D) * D * qstar
\`\`\`

**Fortran** — [\`delta5_main.f95\`, transport loop](../fortran/delta5_main.f95#L532-L560), with \`Cf\`, \`tau_star\`, \`qstar\` and \`qt\` calculated by the corresponding expressions. The Python snippet omits the preceding normal-stress cap solely for brevity; both programs apply it.

## 4. Bed-elevation change: discrete Exner balance

The active-channel Exner update converts the difference between incoming and outgoing sediment transport into sediment thickness. For a channel step of length \(\Delta\ell_k\),

\[
\Delta\eta_{\rm sed,k}=
\frac{\Delta t_{\rm sec}}{(1-\lambda)\Delta\ell_k}
\,\Delta q_{s,k},
\qquad
\Delta\eta_k=\Delta\eta_{\rm sed,k}
-\frac{u_{\rm sub}\Delta t}{I},
\]

where \(\Delta t_{\rm sec}=\Delta t\times\texttt{sec_yr}\), \(u_{\rm sub}\) is the subsidence parameter, and the second term follows the effective-year convention implemented in both codes.

For a nonterminal channel cell, the mixed upwind difference is

\[
\Delta q_{s,k}=a_u(q_{s,k-1}-q_{s,k})
 +(1-a_u)(q_{s,k}-q_{s,k+1});
\]

the terminal cell uses \(q_{s,k-1}-q_{s,k}\). The source sediment supply provides the upstream boundary flux.

**Python** — [Exner update](../src/deltamodel/_kernel.py#L419-L452):

\`\`\`python
qt_dif = a_u * (qtback - qt_now) + (1.0 - a_u) * (qt_now - qtfrnt)
delta_agg[i, j] = (dt * sec_yr) / (1.0 - lam) / dL * qt_dif
delta_eta[i, j] = delta_agg[i, j] - subside * dt / I_tavg
elev_temp = eta[i, j] + delta_eta[i, j]
\`\`\`

**Fortran** — [Exner update](../fortran/delta5_main.f95#L562-L611):

\`\`\`fortran
qt_dif = a_u*(qtback-qt_now) + (1.d0-a_u)*(qt_now-qtfrnt)
delta_agg(i,j) = (dt*sec_yr)/(1.d0-lambda)/dL*qt_dif
delta_eta(i,j) = delta_agg(i,j) - subside*dt/I_tavg
elev_temp = eta(i,j) + delta_eta(i,j)
\`\`\`

A nonnegative-bed condition constrains erosion at \(\eta=0\). When this constraint is activated, the routine adjusts the downstream sediment flux consistently with the Exner difference. For \(a_u\ne 1/2\), the nonterminal-cell inversion is

\[
q_{s,k}=\frac{\Delta q_{s,k}-a_u q_{s,k-1}
+(1-a_u)q_{s,k+1}}{1-2a_u}.
\]

The default is full upwinding (\(a_u=1\)); the expression is singular for \(a_u=1/2\), so that weight is not a supported choice for this constrained-flux branch.

## 5. Floodplain deposition and subsidence

Outside active channel cells and land-boundary cells, the implemented floodplain update is

\[
\Delta\eta_{\rm floodplain}=
f_{\rm dep}\Delta t-\frac{u_{\rm sub}\Delta t}{I},
\]

where \`fine_dep\` is the fine-sediment deposition rate calculated from the model parameters. The increment updates both bed and levee elevation. Accumulated deposition is also passed to the stratigraphy bookkeeping.

**Python** — [floodplain loop](../src/deltamodel/_kernel.py#L483-L492):

\`\`\`python
subside_now = subside * dt / I_tavg
delta_agg[i, j] = fine_dep * dt
delta_eta[i, j] = delta_agg[i, j] - subside_now
eta[i, j] = eta[i, j] + delta_eta[i, j]
levee[i, j] = levee[i, j] + delta_eta[i, j]
\`\`\`

**Fortran** — [\`sedplain.f95\`](../fortran/sedplain.f95#L21-L31):

\`\`\`fortran
subside_now = subside*dt/I_tavg
delta_agg(i,j) = fine_dep*dt
delta_eta(i,j) = delta_agg(i,j) - subside_now
eta(i,j) = eta(i,j) + delta_eta(i,j)
levee(i,j) = levee(i,j) + delta_eta(i,j)
\`\`\`

## 6. Avulsion criterion and channel rerouting

At each eligible channel cell, the model compares an alternative route toward the lowest available neighboring cell with the existing downstream channel gradient. Let \((i,j)\) be the current cell, \(n\) a free neighbor, and \(c\) the next cell on the existing channel. The implemented condition is

\[
\frac{\eta_{i,j}-\beta H_{i,j}-\eta_n}{\ell_{i,j\rightarrow n}}
>
\frac{\eta_{i,j}-\eta_c}{\ell_{i,j\rightarrow c}}.
\]

Available neighbors are screened for domain boundaries and already occupied channel cells. A successful test initiates a new downstream channel route, after which the code checks channel depth and imposes a small positive incising slope where necessary.

**Python** — [avulsion test](../src/deltamodel/_kernel.py#L494-L515):

\`\`\`python
L_act = _next_len(i, j, i_act, j_act, L)
avul_cond = beta * Hij[i, j]
if go and ((eta[i, j] - avul_cond - eta[ni, nj]) / L[it0]) > \
        ((eta[i, j] - eta[i_act, j_act]) / L_act):
    avul_flag = 1
\`\`\`

**Fortran** — [avulsion test](../fortran/delta5_main.f95#L708-L728):

\`\`\`fortran
avul_cond = beta*Hij(i,j)
if (S(itemp0) /= -999.d0) then
   avul_test = ((eta(i,j)-avul_cond-eta(i+itemp,j+jtemp)) / L(itemp0)) > &
               ((eta(i,j)-eta(i_act,j_act)) / L_act)
endif
\`\`\`

The source also handles cases with no free neighbor and chooses the actual axial or diagonal distance along the existing channel. See [\`slope5.f95\`](../fortran/slope5.f95), [\`_slope5\`](../src/deltamodel/_kernel.py#L17-L29), [\`_boxed_in\`](../src/deltamodel/_kernel.py#L31-L44), and the channel-building loops.

## 7. Three-dimensional stratigraphy and channel history

The model records net aggradation between avulsions and creates a new layer when the channel configuration changes. Erosion removes thickness from the top of a column, proceeding into deeper layers as necessary; incision is recorded as an actual removed thickness rather than a nominal channel depth. Every stratigraphic column can be checked against the corresponding modeled bed elevation after accounting for subsidence.

Implementation: **Python** [\`_strat3d\` and \`_incise\`](../src/deltamodel/_kernel.py#L106-L162); **Fortran** [\`strat3D.f95\`](../fortran/strat3D.f95) and [\`incise.f95\`](../fortran/incise.f95). Residence-time and revisit records are managed in [\`tim_visit.f95\`](../fortran/tim_visit.f95) and the Python kernel; reconstructed paths are available through [\`history.py\`](../src/deltamodel/history.py). Python runs can omit the stratigraphic arrays with \`run(..., stratigraphy=False)\` without changing bed evolution.

## 8. Reproducibility and implementation map

| Calculation | Python | Fortran |
|---|---|---|
| Parameters and initial bed | [\`model.py\`](../src/deltamodel/model.py) | [\`delta5_main.f95\`](../fortran/delta5_main.f95), [\`bed_elev.f95\`](../fortran/bed_elev.f95) |
| Neighbor slopes and channel routing | [\`_kernel.py\`](../src/deltamodel/_kernel.py) | [\`slope5.f95\`](../fortran/slope5.f95), [\`delta5_main.f95\`](../fortran/delta5_main.f95) |
| Hydraulics | [\`_backwater\`](../src/deltamodel/_kernel.py#L56-L90) | [\`backwater.f95\`](../fortran/backwater.f95), [\`incise_H.f95\`](../fortran/incise_H.f95) |
| Transport, Exner, avulsion | [\`run_kernel\`](../src/deltamodel/_kernel.py#L172-L735) | [\`delta5_main.f95\`](../fortran/delta5_main.f95) |
| Floodplain sedimentation | [\`_kernel.py\`](../src/deltamodel/_kernel.py#L483-L492) | [\`sedplain.f95\`](../fortran/sedplain.f95) |
| Incision and stratigraphy | [\`_kernel.py\`](../src/deltamodel/_kernel.py#L106-L162) | [\`incise.f95\`](../fortran/incise.f95), [\`strat3D.f95\`](../fortran/strat3D.f95) |

Use \`python -m pytest tests\` for regression checks. The repository includes [a fixed initial bed and stored 30-year reference output](../tests/data), with tests for Python–Fortran agreement, stratigraphic mass balance, history and domain behavior. The main README records longer-run validation results. These checks establish cross-implementation numerical agreement for the tested configurations; they are not a substitute for calibration or independent validation against field observations.

---

## Appendix — implementation refinements and technical history (2026)

The Python translation was initially checked line by line against the 2007 Fortran program and reproduced its behavior. Cross-implementation review subsequently identified nine numerical or bookkeeping cases that were revised consistently in both implementations. Source markers are \`Correction N (2026)\` in \`fortran/\` and \`_fix N\` in \`src/deltamodel/_kernel.py\`. This section documents those changes for reproducibility; the scientific formulation and ongoing research are discussed above.

1. **Constrained sediment flux.** When the nonnegative-bed condition limits erosion, recomputing downstream sediment flux from Exner balance requires the denominator \(1-2a_u\). The corrected inversion retains the appropriate sign for full upwinding.
2. **Flat reaches during channel construction.** When the new course needs an imposed \`S_incise\`, the bed drop is now based on the length of the step actually taken, including diagonal steps.
3. **Distance in the avulsion comparison.** The existing-channel gradient now uses the distance to the actual next channel cell, rather than a stored neighboring-cell distance.
4. **Final stratigraphy accounting.** Deposits accumulated since the most recent avulsion are included at the end of the run. Column summaries use the layer count of each column.
5. **Initial-channel incision reference.** The elevation reference is reset after cutting the initial channel so later stratigraphic updates begin from the incised surface.
6. **Backwater predictor–corrector.** The corrector evaluates friction at the predicted depth, consistently with the predicted Froude term.
7. **Cells without a free neighbor.** Avulsion candidates are excluded where no eligible neighboring cell exists; channel construction continues downstream without indexing a blocked or off-grid neighbor.
8. **Erosion through stored deposits.** When erosion reaches the initial deposit, its remaining thickness is adjusted so the stratigraphic column still corresponds to the modeled bed.
9. **Incision following avulsion.** Stratigraphy removes the thickness actually cut during channel creation, rather than a full nominal channel depth.

Items 4, 5, 8 and 9 concern stratigraphic recording, rather than the modeled bed trajectory. The refinements allow the stratigraphic record to remain consistent with elevation and subsidence.

**Portability and output maintenance (no intended effect on physical results).** Numeric literals were made consistently double precision; FTN95-specific random-number/seed routines were replaced with standard Fortran; an output unit collision involving \`bed_elev.f95\` was resolved; out-of-range and reset-index cases were addressed; visit-series storage is bounded at 100 recorded visits per cell (while later visits remain counted); and stratigraphic output now includes every recorded layer.

For the historical research context, see Puma et al., *On the connections between surficial processes and stratigraphy in river deltas* ([2016 preprint](https://arxiv.org/abs/1606.04558)). This repository's ongoing study is separate from that earlier preprint.
