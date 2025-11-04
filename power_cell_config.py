"""
Configuration for Analytic Power-Cell Foam Simulation (Option A)
Freeze → Measure → Adjust → Relax architecture

All tunables, caps, and constants in one place.
"""

import taichi as ti
import math

# =============================================================================
# Architecture & Platform
# =============================================================================
ARCH = ti.metal  # MacBook Pro M2
USE_GPU = True

# =============================================================================
# Particle Count & Domain
# =============================================================================
N = 10_000  # number of particles/sites (scaled up from 1k)
L = 1.0     # domain half-extent: box is [-L, L]³
BOX_SIZE = 2.0 * L

# Mean spacing (approximate, recomputed after initialization)
# For N particles in volume (2L)³, mean spacing ≈ (volume/N)^(1/3)
MEAN_SPACING = (BOX_SIZE**3 / N) ** (1.0/3.0)

# =============================================================================
# Fixed Capacity Limits (per cell)
# =============================================================================
# Increased significantly to handle large radii without overflow
K_MAX = 24      # max neighbors per cell (blueprint: 24-32)
V_MAX = 256     # max vertices per polyhedron (LARGE for complex cells)
F_MAX = 256     # max faces per polyhedron (LARGE for complex cells)
I_MAX = 1024    # max indices (for triangle fans) (LARGE for complex cells)

# =============================================================================
# Spatial Hash Grid
# =============================================================================
# Grid resolution (cells per axis)
# Cell size should be ≈ mean_spacing for efficient neighbor queries
# For N particles in box of size BOX_SIZE:
#   mean_spacing ≈ (BOX_SIZE³ / N)^(1/3)
#   grid_res ≈ BOX_SIZE / mean_spacing = (N / BOX_SIZE³)^(1/3) * BOX_SIZE
# For N=10k, BOX_SIZE=2.0: grid_res ≈ 17
GRID_RES = max(8, int((N / (BOX_SIZE**3))**(1.0/3.0) * BOX_SIZE * 0.5))  # 0.5 factor for larger cells
GRID_CELL_SIZE = BOX_SIZE / GRID_RES

# =============================================================================
# Geometry Construction
# =============================================================================
# Security radius for initial cube (half-extent)
# Start conservative; tune down if cells don't touch cube boundaries
R_SEC_ALPHA = 2.0
R_SEC = R_SEC_ALPHA * MEAN_SPACING

# Thresholds
A_MIN = 1e-8 * (MEAN_SPACING ** 2)  # minimum face area (cull tiny faces)
EPS_MERGE = 1e-6 * L                 # vertex deduplication tolerance
EPS_PLANE = 1e-7 * L                 # plane distance tolerance

# =============================================================================
# Controller (ADJUST phase) - USER SPEC (200% growth, 10% shrinkage)
# =============================================================================
# FIXED IQ threshold (USER ADJUSTABLE via slider)
# Cells with IQ < threshold: EXPAND by 700% (2× radius)
# Cells with IQ >= threshold: SHRINK by 27.1% (0.9× radius)
USE_PERCENTILE = False  # Always use fixed threshold (no dynamic recalculation!)
IQ_THRESHOLD = 0.01     # Fixed IQ threshold (USER ADJUSTABLE: 0.0-1.0) - Start at ~10th percentile
BETA_S = 2.0            # expand irregular cells (200% volume = 1.26× radius) - REDUCED for stability!
BETA_E = 0.271          # shrink regular cells (27.1% volume = 0.9× radius)
DR_MAX_FRAC = 0.30      # max radius change per cycle (30% of radius) - REDUCED to prevent wild swings!

# Volume-to-radius conversion method
# "sphere" = dr = dV / (4π r²)
# "proportional" = dr = γ * (dV/V) * r
DR_METHOD = "sphere"
DR_GAMMA = 0.5  # for proportional method

# Radius bounds
R_MIN = 0.01 * L        # Minimum radius (increased so shrinking is visible!)
R_MAX = 0.15 * L        # Maximum radius

# =============================================================================
# RELAX Phase (contact-only repulsion, no long-range forces)
# =============================================================================
RELAX_FRAMES = 120      # number of frames to relax after each adjust (INCREASED for stability!)

# Soft forces (ShaderToy-style: baseline 1/r³ + overlap push)
K_ATTR = 30.0           # baseline 1/r³ push strength (keeps spacing, per ShaderToy)
K_REP = 30.0            # overlap push strength (prevents overlap when radii grow)

# Integration
DT = 0.006              # timestep
DAMPING = 0.98          # velocity damping (per frame)
VEL_CLAMP = 0.05 * L    # max velocity magnitude (safety)

# PBD (Position-Based Dynamics) - ONLY overlap resolution
PBD_ITERS = 8           # iterations per frame (6-8 is sufficient without forces)

# =============================================================================
# Convergence & Diagnostics
# =============================================================================
MAX_CYCLES = 100        # max freeze-relax cycles
IQ_CONVERGENCE_THRESH = 5e-3  # stop if mean |ΔIQ| < this

LOG_INTERVAL = 1        # log every N cycles
OVERFLOW_WARN_THRESH = 0.01  # warn if >1% cells overflow

# =============================================================================
# Initialization
# =============================================================================
# Initial radius distribution (Minkowski sphere weights for power diagram)
# USER ADJUSTABLE: Start smaller to avoid overflow, controller will grow them
INIT_RAD_MEAN = 0.15 * MEAN_SPACING  # mean radius (Minkowski weight)
INIT_RAD_STD = 0.05 * MEAN_SPACING   # standard deviation
INIT_RAD_MIN = 0.05 * MEAN_SPACING   # minimum radius (USER ADJUSTABLE)
INIT_RAD_MAX = 0.25 * MEAN_SPACING   # maximum radius (USER ADJUSTABLE)

# Blue noise / Poisson disk sampling parameters (if used)
USE_BLUE_NOISE = True
POISSON_MIN_DIST = 0.9 * MEAN_SPACING  # ρ ∈ [0.8, 0.95] - higher = stronger blue noise
POISSON_K_ATTEMPTS = 30  # candidates per active sample

# Capacity-constrained power diagram (weights from target volumes)
USE_CAPACITY_SOLVER = False  # DISABLED - we use random radii for true heterogeneous foam
CAPACITY_INNER_ITERS = 5    # inner iterations for weight solver (3-10)
CAPACITY_SOLVER_SWEEPS = 2  # Jacobi/GS sweeps per iteration (1-3)
CAPACITY_ALPHA = 0.5        # damping factor for weight updates
CAPACITY_TOL = 1e-3         # convergence tolerance (mean |V_i - V_i*| / V_i*)

# =============================================================================
# Derived Constants (computed at runtime)
# =============================================================================
PI = math.pi

