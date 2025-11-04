"""
State buffers for Analytic Power-Cell Foam Simulation
SoA (Structure of Arrays) layout for GPU efficiency
"""

import taichi as ti
import power_cell_config as C

# =============================================================================
# Initialize Taichi
# =============================================================================
ti.init(arch=C.ARCH)

# =============================================================================
# Particle State (core)
# =============================================================================
pos = ti.Vector.field(3, dtype=ti.f32, shape=C.N)  # positions
rad = ti.field(dtype=ti.f32, shape=C.N)             # radii (weights for power diagram)
vel = ti.Vector.field(3, dtype=ti.f32, shape=C.N)  # velocities (used in RELAX only)

# =============================================================================
# Spatial Hash Grid (uniform grid for neighbor queries)
# =============================================================================
# Total grid cells
GRID_SIZE = C.GRID_RES ** 3

# Linked list structure
cell_head = ti.field(dtype=ti.i32, shape=GRID_SIZE)  # head of linked list per cell
next_idx = ti.field(dtype=ti.i32, shape=C.N)         # next particle in list

# =============================================================================
# Neighbor Lists (per particle)
# =============================================================================
nbr_ct = ti.field(dtype=ti.i32, shape=C.N)                    # count of neighbors
nbr_idx = ti.field(dtype=ti.i32, shape=(C.N, C.K_MAX))       # neighbor indices

# =============================================================================
# Plane Lists (power planes per particle)
# =============================================================================
# Each particle has up to K_MAX planes (one per neighbor)
plane_n = ti.Vector.field(3, dtype=ti.f32, shape=(C.N, C.K_MAX))    # plane normals
plane_b = ti.field(dtype=ti.f32, shape=(C.N, C.K_MAX))              # plane offsets
plane_ofs = ti.Vector.field(3, dtype=ti.i32, shape=(C.N, C.K_MAX))  # periodic image offset
plane_id = ti.field(dtype=ti.i32, shape=(C.N, C.K_MAX))             # neighbor ID

# =============================================================================
# Polyhedron Buffers (fixed capacity per particle)
# =============================================================================
# Vertices (stored relative to particle position for numerical stability)
verts = ti.Vector.field(3, dtype=ti.f32, shape=(C.N, C.V_MAX))

# Faces (each face is a polygon stored as index range into idx[])
faces_begin = ti.field(dtype=ti.i32, shape=(C.N, C.F_MAX))         # start index in idx[]
faces_count = ti.field(dtype=ti.i32, shape=(C.N, C.F_MAX))         # number of vertices
faces_normal = ti.Vector.field(3, dtype=ti.f32, shape=(C.N, C.F_MAX))  # face normals

# Index buffer (flat array of vertex indices for all faces)
idx = ti.field(dtype=ti.i32, shape=(C.N, C.I_MAX))

# Counters (per particle)
v_ct = ti.field(dtype=ti.i32, shape=C.N)  # vertex count
f_ct = ti.field(dtype=ti.i32, shape=C.N)  # face count
i_ct = ti.field(dtype=ti.i32, shape=C.N)  # index count

# =============================================================================
# Measurement Outputs (per particle)
# =============================================================================
volume = ti.field(dtype=ti.f32, shape=C.N)      # cell volume
area = ti.field(dtype=ti.f32, shape=C.N)        # cell surface area
iq = ti.field(dtype=ti.f32, shape=C.N)          # isoperimetric quotient
iq_prev = ti.field(dtype=ti.f32, shape=C.N)     # previous IQ (for EMA/convergence)
fsc = ti.field(dtype=ti.i32, shape=C.N)         # face-sharing count (number of faces)
overflow = ti.field(dtype=ti.i32, shape=C.N)    # overflow flag (1 if caps exceeded)

# =============================================================================
# Force Accumulator (for RELAX phase)
# =============================================================================
force = ti.Vector.field(3, dtype=ti.f32, shape=C.N)  # force accumulator

