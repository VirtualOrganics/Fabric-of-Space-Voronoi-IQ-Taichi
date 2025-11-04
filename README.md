# 3D Power Cell Foam Simulation

A GPU-accelerated 3D foam simulation using **analytic power cell geometry** and **Isoperimetric Quotient (IQ)** control, implemented in [Taichi](https://www.taichi-lang.org/).

## Overview

This simulation creates an "exotic foam" where cell growth/shrinkage is controlled by geometric regularity (IQ) rather than topological connectivity. The system operates in discrete **freeze-measure-adjust-relax** cycles:

1. **FREEZE**: Pause motion for exact geometry measurement
2. **MEASURE**: Compute 3D power cells using analytic half-space clipping (Sutherland-Hodgman 3D)
3. **ADJUST**: Expand irregular cells (low IQ), shrink regular cells (high IQ)
4. **RELAX**: Particles move with soft forces + PBD to resolve overlaps

## Key Features

- ✅ **Analytic Power Cell Geometry**: Exact volume, surface area, and IQ measurement via 3D clipping
- ✅ **IQ-Driven Control**: User-adjustable threshold for expansion/shrinkage
- ✅ **Interactive UI**: Real-time slider, pause/resume, camera controls
- ✅ **Periodic Boundaries**: Toroidal topology for infinite domain
- ✅ **Blue Noise Initialization**: Poisson disk sampling for even distribution
- ✅ **Hybrid Forces**: ShaderToy-style 1/r³ repulsion + overlap resolution
- ✅ **GPU Acceleration**: 10,000 particles @ 7000+ FPS (M1/M2 Mac)

## Installation

```bash
# Clone repository
git clone https://github.com/VirtualOrganics/Fabric-of-Space-Voronoi-IQ-Taichi.git
cd Fabric-of-Space-Voronoi-IQ-Taichi

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run simulation
python run_power_cell.py
```

### Controls

- **SPACE**: Pause/Resume simulation
- **SHIFT + LEFT DRAG**: Rotate camera
- **RIGHT DRAG**: Pan camera
- **SCROLL WHEEL**: Zoom
- **WASD / Arrow Keys**: Move camera
- **ESC**: Exit

### UI Panel

- **Average IQ**: Current mean IQ value across all cells
- **Expand/Shrink counts**: Live statistics showing how many cells will grow/shrink
- **IQ Threshold slider**: Adjust threshold (cells with IQ < threshold expand, IQ ≥ threshold shrink)
  - Ultra-fine mode (0.000-0.010) for precise control
  - Fine mode (0.00-0.10) for low IQ ranges
  - Coarse mode (0.0-1.0) for full range
- **Percentile buttons**: Auto-set threshold to 5th or 10th percentile

## Configuration

Key parameters in `power_cell_config.py`:

```python
# Simulation
N = 10000                    # Number of particles
DOMAIN_SIZE = 2.0            # Box size (periodic)

# IQ Control
IQ_THRESHOLD = 0.01          # Threshold for expand/shrink
BETA_S = 2.0                 # Expansion rate (200% volume)
BETA_E = 0.271               # Shrinkage rate (27% volume)
DR_MAX_FRAC = 0.30           # Max radius change per cycle (30%)

# Relaxation
RELAX_FRAMES = 120           # Frames to relax after each adjustment
K_ATTR = 30.0                # Baseline repulsion strength
K_REP = 30.0                 # Overlap repulsion strength
PBD_ITERS = 8                # Position-based dynamics iterations

# Initialization
INIT_RAD_MIN = 0.05 * MEAN_SPACING  # Minimum initial radius
INIT_RAD_MAX = 0.25 * MEAN_SPACING  # Maximum initial radius
```

## Architecture

### Core Modules

- **`power_cell_config.py`**: All simulation constants and parameters
- **`power_cell_state.py`**: Taichi field definitions (particle state, geometry buffers)
- **`power_cell_grid.py`**: Spatial hash grid + neighbor gathering
- **`power_cell_planes.py`**: Power plane construction (Minkowski bisectors)
- **`power_cell_clipper.py`**: Sutherland-Hodgman 3D clipping algorithm
- **`power_cell_measure.py`**: Volume, surface area, IQ, FSC computation
- **`power_cell_controller.py`**: IQ-driven radius adjustment logic
- **`power_cell_relax.py`**: Soft forces + PBD for overlap resolution
- **`power_cell_init.py`**: Blue noise initialization (Bridson's algorithm)
- **`power_cell_viz.py`**: Taichi UI rendering and controls
- **`power_cell_loop.py`**: Main freeze-measure-adjust-relax cycle
- **`run_power_cell.py`**: Entry point

## Theory

### Isoperimetric Quotient (IQ)

The IQ measures how "sphere-like" a 3D cell is:

```
IQ = 36πV² / S³
```

Where:
- `V` = cell volume
- `S` = cell surface area
- `IQ = 1.0` for perfect spheres
- `IQ < 1.0` for irregular/elongated cells

### Power Diagrams

Power diagrams (weighted Voronoi diagrams) use a modified distance metric:

```
d_power(x, p_i) = |x - p_i|² - r_i²
```

Where `r_i` is the Minkowski sphere radius (weight). Larger radii → larger cells.

### Control Logic

Each cycle:
1. Measure IQ for all cells
2. Compare to threshold:
   - **IQ < threshold**: Expand by `BETA_S` (irregular cells)
   - **IQ ≥ threshold**: Shrink by `BETA_E` (regular cells)
3. Convert volume change to radius change
4. Clamp to `DR_MAX_FRAC` to prevent instability
5. Relax particles to resolve overlaps

## Performance

Typical performance on M1/M2 Mac:
- **10,000 particles**: 7000+ FPS (relax phase)
- **Measurement phase**: ~0.1 ms per cycle (after initial compilation)
- **Total cycle time**: ~15-20 ms

## Diagnostics

The terminal output shows detailed statistics:

```
[MEASURE] IQ: μ=0.056 σ=0.135 range=[0.000, 1.000]
          Percentiles: p10=0.000 p50=0.013 p90=0.132
          Volume: μ=0.002098 range=[0.000000, 0.017140]
          Area: μ=0.307946 range=[0.000000, 1.352788]
          Overflow: 0 cells (0.0%)

[ADJUST] IQ Threshold: 0.002
         Actions: expand=2237 shrink=7763
         Volume change: -3.12e+00

[RELAX] Movement: μ=0.041147 max=1.997217 (mean spacing=0.092832)
        Overlaps: 8.0% max_pen=0.000000

[EVOLUTION] IQ change: -0.003076 (cycle 98→99)
```

## Known Issues

- **IQ oscillation**: System may oscillate if threshold is not well-tuned. Use percentile buttons to find stable threshold.
- **Low IQ values**: Power diagrams with heterogeneous radii naturally have lower IQ than standard Voronoi cells (this is expected).
- **Buffer overflow**: Very large radii can cause polyhedron buffers to overflow. Increase `V_MAX`, `F_MAX`, `I_MAX` if needed.

## References

- **Sutherland-Hodgman Algorithm**: 3D convex polygon clipping
- **Power Diagrams**: Aurenhammer, F. (1987). "Power diagrams: properties, algorithms and applications"
- **Isoperimetric Quotient**: Measure of geometric regularity in 3D
- **Taichi**: High-performance GPU computing framework

## License

MIT License

## Acknowledgments

Built with [Taichi](https://www.taichi-lang.org/) - A high-performance GPU-accelerated computing framework.

Inspired by foam physics, computational geometry, and the quest for exotic topological structures.

