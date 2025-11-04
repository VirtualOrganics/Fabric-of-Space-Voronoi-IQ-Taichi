"""
Power Plane Construction
Compute bisector planes for power diagram (Laguerre cells)
"""

import taichi as ti
import power_cell_config as C
from power_cell_state import pos, rad, nbr_ct, nbr_idx, plane_n, plane_b, plane_id, plane_ofs
from power_cell_grid import torus_delta

# =============================================================================
# Power Plane Construction
# =============================================================================

@ti.kernel
def build_planes():
    """
    Build power planes for each particle based on its neighbors.
    
    For each particle i and its neighbor j, compute the bisector plane
    that separates their power cells:
    
        Power distance: π_i(x) = |x - p_i|² - r_i²
        
        Bisector plane: { x : π_i(x) = π_j(x) }
        
        Plane equation: n·(x - p_i) = b
        where:
            n = (p_j - p_i) / |p_j - p_i|  (unit normal)
            b = 0.5 * (|p_j|² - r_j² - |p_i|² + r_i²) / |p_j - p_i|
    
    The plane inequality for cell i is: n·(x - p_i) ≤ b
    
    Periodic boundaries: Use minimum-image convention (torus_delta)
    """
    for i in range(C.N):
        p_i = pos[i]
        r_i = rad[i]
        w_i = r_i * r_i  # weight
        
        # For each neighbor
        for k in range(nbr_ct[i]):
            j = nbr_idx[i, k]
            p_j = pos[j]
            r_j = rad[j]
            w_j = r_j * r_j
            
            # Periodic delta (minimum image)
            d = torus_delta(p_j, p_i)
            s = d.norm(1e-12)  # distance (avoid division by zero)
            
            # Unit normal (points from i toward j)
            n = d / s
            
            # Plane offset
            # Derivation: at the bisector, |x-p_i|² - w_i = |x-p_j|² - w_j
            # Expanding and simplifying: 2·n·(x - p_i) = |p_j|² - w_j - |p_i|² + w_i
            # So: n·(x - p_i) = b where b = 0.5 * (|p_j|² - w_j - |p_i|² + w_i) / s
            
            # Note: p_j here is actually p_j' (periodic image), so |p_j'|² not |p_j|²
            # But since we use torus_delta, we work with the wrapped delta d = p_j' - p_i
            # and p_j' = p_i + d, so |p_j'|² = |p_i + d|² = |p_i|² + 2·p_i·d + |d|²
            
            # Simplified: b = 0.5 * (2·p_i·d + |d|² + w_i - w_j) / s
            #              = 0.5 * (2·p_i·d + s² + w_i - w_j) / s
            
            # Even simpler using the fact that d = s·n:
            # b = 0.5 * (2·p_i·(s·n) + s² + w_i - w_j) / s
            #   = p_i·n + 0.5·s + 0.5·(w_i - w_j)/s
            
            # Most stable form (avoids large |p_i|² terms):
            b = 0.5 * (s + (w_i - w_j) / s)
            
            # Store plane
            plane_n[i, k] = n
            plane_b[i, k] = b
            plane_id[i, k] = j
            
            # Periodic image offset (for bookkeeping; not used in Phase 1)
            # In future, we can track which periodic image of j was used
            plane_ofs[i, k] = ti.Vector([0, 0, 0])


# =============================================================================
# Plane Query (helper for debugging)
# =============================================================================

@ti.func
def signed_distance_to_plane(p: ti.template(), i: ti.i32, k: ti.i32) -> ti.f32:
    """
    Compute signed distance from point p to plane k of particle i.
    
    Args:
        p: query point (float3)
        i: particle index
        k: plane index (0 to nbr_ct[i]-1)
    
    Returns:
        signed distance (negative = inside cell i, positive = outside)
    """
    p_i = pos[i]
    n = plane_n[i, k]
    b = plane_b[i, k]
    
    return n.dot(p - p_i) - b

