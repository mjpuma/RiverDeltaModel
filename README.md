<p align="center">
  <img src="assets/RiverDeltaModel_logo.webp" width="128" alt="RiverDeltaModel emblem showing a branching river delta from mountains to the sea">
</p>

# RiverDeltaModel
**River-delta evolution, channel dynamics, and three-dimensional stratigraphy**

RiverDeltaModel is a computational model for exploring how river-delta channels evolve and organize into distributary networks, and how their dynamics shape the delta plain and its stratigraphic record. It represents backwater hydraulics, sediment transport, aggradation, avulsions, and three-dimensional deposition.

## Current research: what do delta-network patterns reveal about growth?

**Ongoing work tests which statistical properties of river-delta networks reflect formative processes, and which can arise from network geometry alone.** A particular focus is Hack-like scaling—the relationship between downstream channel length and the area nourished by distributary branches—and whether it can distinguish physically guided channel growth from random network formation.

An early [manuscript draft](paper/signatures/) reports three main findings:

- **Similar scaling can emerge from different growth mechanisms.** Networks traced by the model's avulsing channel show a Hack-like exponent near 0.58, while directed random-walk networks produce comparable exponents.
- **Comparisons with real deltas reinforce that ambiguity.** In nine mapped river-dominated deltas, both density-matched random networks and networks grown in a Laplace field can reproduce observed Hack-like exponents.
- **Channel spacing may better distinguish growth processes.** Observed channels spread across delta plains much more evenly than random networks and resemble networks in which existing channels screen the growth field. The draft's median normalized spacing values are 2.1 for mapped networks, 1.9 for Laplacian trees, and 5.3 for random networks with loops.

These are **preliminary findings from a manuscript in development**, not yet published conclusions. The broader aim is to identify network metrics that carry interpretable information about delta formation, using process-based simulations, carefully matched null models, and observed channel networks. The work also examines how network density, delta boundaries, and breaks in scaling affect interpretation.

The relevant numerical experiments, null models, network comparisons, and figure-generation scripts are in [`paper/`](paper/) and [`paper/signatures/`](paper/signatures/). Earlier model development also investigated channel residence time and its connection to stratigraphic architecture; this remains part of the model's scientific foundation.

## Model capabilities

Water and sediment enter the delta plain at a source. Flow follows an evolving channel, sediment transport changes bed elevation, and aggradation and avulsions create new routes. The model also records sediment deposition and erosion through time.

- **Channel hydraulics:** gradually varied flow with backwater effects.
- **Sediment transport and morphology:** bedload transport, bed-elevation change, levee and floodplain deposition, and subsidence.
- **Avulsions:** channel superelevation, alternative downstream routes, and channel incision.
- **Delta geometry:** rectangular and semicircular domains for studying boundary effects.
- **Stratigraphy and channel history:** three-dimensional sediment layers, deposition times, channel residence times, and revisits.
- **Reproducible implementations:** a Python package and a corresponding Fortran implementation for numerical cross-checks.

![Bed elevation and avulsion frequency after a 710-year run](assets/delta_710yr.png)

*A 710-effective-year simulation (\`examples/plot_delta.py\`). Left: final bed elevation and active channel. Right: spatial distribution of avulsions following the 110-year spin-up.*

## Getting started

Install the Python package from the repository root:

\`\`\`sh
pip install -e .
\`\`\`

The package requires NumPy and Numba. Optional dependencies are available with \`[test]\` and \`[examples]\`.

### Run a simulation

\`\`\`python
from deltamodel import Params, run

p = Params(tmax=100.0, teq=20.0)
res = run(p, seed=1)

print(res.num_avulsions)
res.eta, res.levee, res.channel
res.avulsions
res.visit_mean, res.return_mean
res.strat_thickness, res.strat_type, res.strat_time
\`\`\`

The outputs include bed and levee elevations, the final channel, avulsion time and location, channel occupancy statistics, and three-dimensional stratigraphy.

### Choose the delta geometry

The default \`Params(domain="rectangle")\` represents a plain with no-flow side walls and a straight shoreline. \`Params(domain="semicircle", ymax=R)\` instead represents a semicircular plain of radius \`R\` grid cells, with an open curved shoreline.

\`\`\`python
import numpy as np
from deltamodel import Params, run, initial_bed

R = 50
p = Params(domain="semicircle", ymax=R,
           Qs=0.0025 / (50 * 71) * np.pi * R**2 / 2)
res = run(p, eta0=initial_bed(p, seed=1), seed=1)
res.land, res.shoreline
\`\`\`

Here, the supply \`Qs\` is scaled by domain area to facilitate comparisons. The \`initial_bed\` helper initializes a radially sloping surface. For larger simulations, \`run(..., stratigraphy=False)\` skips the stratigraphic record without changing bed evolution.

### Tests

\`\`\`sh
python -m pytest tests
\`\`\`

The test suite checks numerical agreement against a stored Fortran reference run, stratigraphic mass balance, channel history, and semicircular-domain behavior.

## Code and documentation

| Location | Purpose |
|---|---|
| [\`src/deltamodel/\`](src/deltamodel/) | Python model, numerical kernel, interface, and channel-history utilities |
| [\`paper/\`](paper/) | Current simulations, quantitative analysis, and figures |
| [\`paper/signatures/\`](paper/signatures/) | Network statistics, null models, and comparisons with mapped deltas |
| [\`examples/\`](examples/) | Example model visualization |
| [\`tests/\`](tests/) | Regression and validation tests |
| [\`fortran/\`](fortran/) | Fortran implementation for reproducibility and comparison |
| [**Model Development Guide**](docs/MODEL_DEVELOPMENT.md) | Governing equations, numerical methods, Python–Fortran code examples, implementation notes, and technical history |

The Python and Fortran implementations have been compared using matched initial conditions, including a full 710-effective-year run. Detailed numerical formulation and development notes are available in the [Model Development Guide](docs/MODEL_DEVELOPMENT.md).

## Background and citation

The research builds on Puma et al., *On the connections between surficial processes and stratigraphy in river deltas* ([2016 preprint](https://arxiv.org/abs/1606.04558)), which examined channel residence time and its relationship to stratigraphic patterns. See [`CITATION.cff`](CITATION.cff) for repository citation metadata.

## License

MIT. See [\`LICENSE\`](LICENSE).
