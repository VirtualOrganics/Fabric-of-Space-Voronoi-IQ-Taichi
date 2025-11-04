"""
Spatial Hash Grid for Neighbor Queries
Uniform grid with linked-list structure
"""

import taichi as ti
import power_cell_config as C
from power_cell_state import pos, cell_head, next_idx, nbr_ct, nbr_idx

# =============================================================================
# Grid Helper Functions
# =============================================================================

@ti.func
def world_to_cell(p: ti.template()) -> ti.math.ivec3:
    """
    Map world position [-L, L]³ to grid cell [0, GRID_RES)³
    
    Args:
        p: world position (float3)
    
    Returns:
        grid cell coordinates (int3)
    """
    # Normalize to [0, 1]³
    q = (p + C.L) / C.BOX_SIZE
    
    # Scale to [0, GRID_RES)³
    c = ti.cast(ti.floor(q * C.GRID_RES), ti.i32)
    
    # Clamp to valid range
    c = ti.max(ti.min(c, C.GRID_RES - 1), 0)
    
    return c


@ti.func
def cell_index(cx: ti.i32, cy: ti.i32, cz: ti.i32) -> ti.i32:
    """
    Flatten 3D grid cell coordinates to 1D index
    
    Args:
        cx, cy, cz: grid cell coordinates
    
    Returns:
        flat index
    """
    return (cz * C.GRID_RES + cy) * C.GRID_RES + cx


@ti.func
def torus_delta(a: ti.template(), b: ti.template()) -> ti.math.vec3:
    """
    Compute periodic delta vector (minimum image convention)
    
    For a periodic box [-L, L]³, wraps delta to shortest distance.
    
    Args:
        a, b: positions (float3)
    
    Returns:
        wrapped delta a - b (float3)
    """
    d = a - b
    d -= ti.round(d / C.BOX_SIZE) * C.BOX_SIZE
    return d


# =============================================================================
# Grid Construction
# =============================================================================

@ti.kernel
def build_grid():
    """
    Build spatial hash grid using linked-list structure.
    
    For each particle, compute its grid cell and insert into the cell's linked list.
    This allows O(1) insertion and efficient neighbor queries.
    
    Note: We serialize insertion to avoid race conditions. For large N, 
    this could be optimized with proper atomics or a two-pass approach.
    """
    # Clear grid heads
    for cid in range(C.GRID_RES ** 3):
        cell_head[cid] = -1
    
    # Insert particles into grid (serialized to avoid races)
    for i in range(C.N):
        c = world_to_cell(pos[i])
        cid = cell_index(c[0], c[1], c[2])
        
        # Insert at head of linked list
        next_idx[i] = cell_head[cid]
        cell_head[cid] = i


# =============================================================================
# Neighbor Gathering
# =============================================================================

@ti.kernel
def gather_neighbors():
    """
    Gather neighbor candidates for each particle.
    
    For each particle, query the 3×3×3 grid neighborhood (27 cells)
    and collect up to K_MAX neighbors.
    
    Note: This is a simple brute-force gather. In Phase 4, we can add:
      - Distance-based culling (keep only K_MAX nearest)
      - Proxy metric filtering (cull obviously irrelevant neighbors)
    """
    for i in range(C.N):
        nbr_ct[i] = 0
        p_i = pos[i]
        c = world_to_cell(p_i)
        
        # Query 3×3×3 neighborhood
        for dz in ti.static(range(-1, 2)):
            for dy in ti.static(range(-1, 2)):
                for dx in ti.static(range(-1, 2)):
                    # Clamp to valid grid range
                    cx = ti.min(ti.max(c[0] + dx, 0), C.GRID_RES - 1)
                    cy = ti.min(ti.max(c[1] + dy, 0), C.GRID_RES - 1)
                    cz = ti.min(ti.max(c[2] + dz, 0), C.GRID_RES - 1)
                    
                    cid = cell_index(cx, cy, cz)
                    
                    # Walk linked list for this cell
                    j = cell_head[cid]
                    while j != -1:
                        if j != i:
                            k = nbr_ct[i]
                            if k < C.K_MAX:
                                nbr_idx[i, k] = j
                                nbr_ct[i] = k + 1
                        j = next_idx[j]


# =============================================================================
# Distance Queries (helper for debugging/validation)
# =============================================================================

@ti.func
def periodic_distance(a: ti.template(), b: ti.template()) -> ti.f32:
    """
    Compute periodic distance between two positions.
    
    Args:
        a, b: positions (float3)
    
    Returns:
        distance (float)
    """
    d = torus_delta(a, b)
    return d.norm()

