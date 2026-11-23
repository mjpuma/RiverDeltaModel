<p align="center">
  <img src="assets/RiverDeltaModel_logo.webp" width="128" alt="RiverDeltaModel emblem showing a branching river delta from mountains to the sea">
</p>

# RiverDeltaModel
**River-delta evolution, channel dynamics, and three-dimensional stratigraphy**

RiverDeltaModel is a computational research model for investigating how sediment transport, channel aggradation, avulsions, and delta geometry shape river-delta landscapes and their stratigraphic records. **Current research** uses the model to examine boundary effects, channel mobility, and the statistical signatures of distributary networks.

## Current research

Two complementary analyses are in development.

### Delta geometry, boundary effects, and channel dynamics

The current study investigates how the shape of a delta plain and its boundary conditions affect the evolving channel network. Simulations compare **rectangular and semicircular domains** to assess channel residence time, avulsion patterns, lateral confinement, and the preservation of channel behavior in stratigraphy.

The analysis includes channel residence times, compensation, channel overlap and reworking, and Hack-like scaling of distributary networks. Simulation workflows and figures are in [\`paper/\`](paper/):

\`\`\`sh
python paper/run_paper.py
python paper/figures.py
\`\`\`

### Statistical signatures of delta channels

A related analysis asks which observed channel-network statistics reveal **topographic steering**, which reflect **avulsion location**, and which can emerge from **domain geometry alone**.

The [\`paper/signatures/\`](paper/signatures/) workflow compares simulated delta networks with directed random-walk null models and evaluates network patterns in nine mapped river-dominated deltas. It includes robustness experiments with varying initial beds, avulsion thresholds, and floodplain deposition rates. External observational datasets required for the real-delta comparisons are described in the \`real_deltas.py\` module.

\`\`\`sh
python paper/signatures/run_robustness.py
python paper/signatures/figures.py
\`\`\`

Both analyses are ongoing research; the repository contains the associated computational methods, experiments, and analysis scripts.

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

RiverDeltaModel builds on earlier work on the relationship between surface processes and delta stratigraphy: Puma et al., *On the connections between surficial processes and stratigraphy in river deltas* ([2016 preprint](https://arxiv.org/abs/1606.04558)). The current research extends that modeling foundation to new questions about delta geometry, network behavior, and stratigraphic signatures.

For repository citation metadata, see [\`CITATION.cff\`](CITATION.cff).

## License

MIT. See [\`LICENSE\`](LICENSE).
