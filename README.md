<p align="center">
  <img src="assets/RiverDeltaModel_logo.webp" width="128" alt="RiverDeltaModel emblem showing a branching river delta from mountains to the sea">
</p>

# RiverDeltaModel
**River-delta evolution, channel dynamics, and three-dimensional stratigraphy**

RiverDeltaModel is a computational model for studying how river-delta channels evolve and how their surface dynamics are recorded in the subsurface. It represents sediment transport, backwater hydraulics, channel aggradation and avulsion, and the development of three-dimensional stratigraphy.

**The current research is exploratory.** Building on earlier work linking channel residence times and deposition patterns, the project is investigating how delta geometry, avulsion dynamics, and channel-network organization may influence the statistical structure of delta landscapes and stratigraphy.

## Research direction

The central question is **how the movement and reorganization of delta channels translate into spatial patterns of sediment deposition—and what aspects of those processes can be inferred from the stratigraphic record.**

Current work is exploring:

- **Delta geometry and boundaries:** comparing rectangular and semicircular domains to assess how confinement and shoreline geometry influence channel paths and avulsions.
- **Channel dynamics across space and time:** examining residence times, channel reoccupation, overlap, and the spatial distribution of avulsions.
- **Statistical signatures:** testing which network and stratigraphic patterns may be associated with topographic steering, avulsion locations, or geometry alone, using model experiments and simplified null models.
- **Connections to observed deltas:** developing comparisons with mapped distributary networks to assess which model-derived patterns are informative beyond idealized domains.

These are **research questions and directions, not established findings**. Initial simulations and analyses are available in [`paper/`](paper/) and [`paper/signatures/`](paper/signatures/); their interpretation and robustness remain under investigation. The model is intended as a relatively simple framework for exploring processes and statistical relationships, rather than a detailed representation of any particular delta.

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
