"""
RELAX Phase: Soft Forces + PBD
Particles reposition after radius changes
"""

import taichi as ti
import power_cell_config as C
from power_cell_state import (
    pos, rad, vel, force, nbr_ct, nbr_idx,
    volume, verts, faces_begin, faces_count, idx, f_ct, overflow
)
from power_cell_grid import torus_delta

# =============================================================================
# Soft Forces (Attraction + Repulsion)
# =============================================================================

@ti.kernel
def zero_forces():
    """Clear force accumulator."""
    for i in range(C.N):
        force[i] = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def compute_forces():
    """
    Compute soft forces between particles (ShaderToy-style).
    
    Two components (per ShaderToy code):
    1. Baseline 1/r³ push (always active, keeps spacing)
    2. Extra overlap push when dist < r_i + r_j (prevents overlap)
    
    This ensures particles respond to radius changes!
    """
    for i in range(C.N):
        F_i = ti.Vector([0.0, 0.0, 0.0])
        p_i = pos[i]
        r_i = rad[i]
        
        # Iterate over neighbors
        for k in range(nbr_ct[i]):
            j = nbr_idx[i, k]
            p_j = pos[j]
            r_j = rad[j]
            
            # Periodic delta
            d_vec = torus_delta(p_j, p_i)
            dist = d_vec.norm(1e-12)
            u = d_vec / dist
            
            # 1) Baseline 1/r³ push (always active)
            factor = C.K_ATTR / (dist * dist * dist + 1e-12)
            F_baseline = factor * u
            
            # 2) Extra "soft-sphere" overlap push
            r_sum = r_i + r_j
            F_overlap = ti.Vector([0.0, 0.0, 0.0])
            if dist < r_sum:
                pen = r_sum - dist
                F_overlap = C.K_REP * pen * u
            
            # Accumulate
            F_i += F_baseline + F_overlap
        
        force[i] = F_i


# =============================================================================
# Integration (Damped Euler)
# =============================================================================

@ti.kernel
def integrate():
    """
    Integrate velocities and positions with damping.
    
    Steps:
        1. Update velocity: v += F * dt
        2. Apply damping: v *= damping
        3. Clamp velocity (safety)
        4. Update position: x += v * dt
        5. Periodic wrap
    """
    for i in range(C.N):
        # Update velocity
        vel[i] += force[i] * C.DT
        
        # Apply damping
        vel[i] *= C.DAMPING
        
        # Clamp velocity (safety)
        v_norm = vel[i].norm()
        if v_norm > C.VEL_CLAMP:
            vel[i] = vel[i].normalized() * C.VEL_CLAMP
        
        # Update position
        pos[i] += vel[i] * C.DT
        
        # Periodic wrap
        pos[i] -= ti.round(pos[i] / C.BOX_SIZE) * C.BOX_SIZE


# =============================================================================
# PBD (Position-Based Dynamics)
# =============================================================================

@ti.kernel
def pbd_project():
    """
    PBD: Project overlapping particles apart.
    
    For each particle, check all neighbors and resolve overlaps by
    moving both particles apart by half the penetration depth.
    
    This is a Jacobi-style iteration (corrections are computed first,
    then applied), so multiple iterations are needed for convergence.
    
    Note: This is a simple equal-mass projection. For mass-weighted PBD,
    scale corrections by inverse mass (or radius).
    """
    for i in range(C.N):
        p_i = pos[i]
        r_i = rad[i]
        corr = ti.Vector([0.0, 0.0, 0.0])
        
        # Accumulate corrections from all overlapping neighbors
        for k in range(nbr_ct[i]):
            j = nbr_idx[i, k]
            
            if j <= i:
                # Avoid double-counting (process each pair once)
                continue
            
            p_j = pos[j]
            r_j = rad[j]
            
            # Periodic delta
            d_vec = torus_delta(p_j, p_i)
            dist = d_vec.norm(1e-12)
            
            # Check overlap
            r_sum = r_i + r_j
            if dist < r_sum:
                pen = r_sum - dist
                u = d_vec / dist
                
                # Correction: move apart by half penetration
                # (equal mass assumption)
                corr_val = 0.5 * pen
                
                # Apply correction (push i away from j)
                corr -= corr_val * u
        
        # Apply accumulated correction
        pos[i] += corr
        
        # Periodic wrap
        pos[i] -= ti.round(pos[i] / C.BOX_SIZE) * C.BOX_SIZE


# =============================================================================
# Overlap Diagnostics
# =============================================================================

@ti.kernel
def compute_overlap_stats() -> ti.types.vector(2, ti.f32):
    """
    Compute overlap statistics.
    
    Returns:
        (overlap_fraction, max_penetration)
    
    Where:
        overlap_fraction = fraction of particles with at least one overlap
        max_penetration = maximum penetration depth
    """
    overlap_count = 0
    max_pen = 0.0
    
    for i in range(C.N):
        p_i = pos[i]
        r_i = rad[i]
        has_overlap = 0
        
        for k in range(nbr_ct[i]):
            j = nbr_idx[i, k]
            
            if j <= i:
                continue
            
            p_j = pos[j]
            r_j = rad[j]
            
            d_vec = torus_delta(p_j, p_i)
            dist = d_vec.norm(1e-12)
            
            r_sum = r_i + r_j
            if dist < r_sum:
                pen = r_sum - dist
                has_overlap = 1
                max_pen = ti.max(max_pen, pen)
        
        overlap_count += has_overlap
    
    overlap_frac = ti.cast(overlap_count, ti.f32) / ti.cast(C.N, ti.f32)
    
    return ti.Vector([overlap_frac, max_pen])


def get_overlap_stats():
    """
    Get overlap statistics (CPU-side wrapper).
    
    Returns:
        (overlap_fraction, max_penetration)
    """
    stats = compute_overlap_stats()
    return float(stats[0]), float(stats[1])

