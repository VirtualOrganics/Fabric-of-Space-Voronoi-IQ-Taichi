"""
Measurement: Compute Volume, Area, IQ, FSC from Polyhedra
"""

import taichi as ti
import power_cell_config as C
from power_cell_state import (
    pos, verts, faces_begin, faces_count, faces_normal, idx,
    v_ct, f_ct, i_ct, overflow,
    volume, area, iq, iq_prev, fsc
)

# =============================================================================
# Area and Volume Computation
# =============================================================================

@ti.kernel
def measure_all():
    """
    Compute volume, area, IQ, and FSC for all particles.
    
    For each particle's polyhedron:
        - Volume: sum of signed tetrahedra (origin at particle position)
        - Area: sum of triangle areas (fan triangulation of faces)
        - IQ: 36π V² / S³ (isoperimetric quotient)
        - FSC: count of faces with area > A_MIN
    
    Volume formula (signed tet):
        For each face triangle (v0, vk, vk+1):
            V += (1/6) * dot(v0, cross(vk, vk+1))
        
        Vertices are already relative to particle position, so no translation needed.
    
    Area formula (triangle fan):
        For each face, fan-triangulate from first vertex:
            For triangle (v0, vk, vk+1):
                A += 0.5 * |cross(vk - v0, vk+1 - v0)|
    """
    for i in range(C.N):
        if overflow[i] != 0:
            # Skip overflowed cells (carry over previous values)
            continue
        
        S = 0.0  # total surface area
        V = 0.0  # total volume
        face_count = 0  # FSC counter
        
        # Iterate over all faces
        for f in range(f_ct[i]):
            begin = faces_begin[i, f]
            count = faces_count[i, f]
            
            if count < 3:
                # Degenerate face (not a polygon)
                continue
            
            # Get first vertex (fan center)
            v0 = verts[i, idx[i, begin]]
            
            # Fan triangulation: (v0, vk, vk+1) for k in [1, count-2]
            face_area = 0.0
            for k in range(1, count - 1):
                vk = verts[i, idx[i, begin + k]]
                vk1 = verts[i, idx[i, begin + k + 1]]
                
                # Triangle edges (for area)
                a = vk - v0
                b = vk1 - v0
                
                # Cross product
                c = ti.math.cross(a, b)
                
                # Area contribution
                tri_area = 0.5 * c.norm()
                face_area += tri_area
            
            # Accumulate face area
            S += face_area
            
            # Count face if area exceeds threshold
            if face_area > C.A_MIN:
                face_count += 1
        
        # Compute volume using signed tet method (blueprint Section 4.2)
        # For each triangle (v0, vk, vk+1) in face fan:
        # V += (1/6) * dot(v0, cross(vk, vk+1))
        # Vertices are already relative to p_i, so no translation needed
        for f in range(f_ct[i]):
            begin = faces_begin[i, f]
            count = faces_count[i, f]
            
            if count < 3:
                continue
            
            # Get first vertex (fan center)
            v0 = verts[i, idx[i, begin]]
            
            # Fan triangulation: (v0, vk, vk+1) for k in [1, count-2]
            for k in range(1, count - 1):
                vk = verts[i, idx[i, begin + k]]
                vk1 = verts[i, idx[i, begin + k + 1]]
                
                # Signed tet volume: (1/6) * dot(v0, cross(vk, vk+1))
                c = ti.math.cross(vk, vk1)
                V += (1.0 / 6.0) * v0.dot(c)
        
        # Take absolute value (poly is convex, sign should be consistent)
        V = ti.abs(V)
        
        # Store results
        volume[i] = V
        area[i] = S
        fsc[i] = face_count
        
        # Compute IQ
        if S > 1e-12 and V > 1e-12:
            iq[i] = (36.0 * C.PI * V * V) / (S * S * S + 1e-20)
            iq[i] = ti.min(iq[i], 1.0)  # clamp to physical range
        else:
            iq[i] = 0.0


# =============================================================================
# EMA Smoothing (optional, for diagnostics)
# =============================================================================

@ti.kernel
def smooth_iq(alpha: ti.f32):
    """
    Apply exponential moving average to IQ values.
    
    IQ_smooth = α * IQ_new + (1 - α) * IQ_prev
    
    Args:
        alpha: smoothing factor (0 = no change, 1 = full update)
    
    Note: For control, use raw IQ. EMA is only for visualization/diagnostics.
    """
    for i in range(C.N):
        if overflow[i] == 0:
            iq_smooth = alpha * iq[i] + (1.0 - alpha) * iq_prev[i]
            iq_prev[i] = iq[i]
            iq[i] = iq_smooth


# =============================================================================
# Statistics (CPU-side helpers)
# =============================================================================

def compute_iq_stats():
    """
    Compute IQ statistics (mean, std, min, max, percentiles).
    
    Returns:
        dict with keys: mean, std, min, max, p10, p50, p90
    """
    import numpy as np
    
    iq_np = iq.to_numpy()
    overflow_np = overflow.to_numpy()
    
    # Filter out overflowed cells
    valid_iq = iq_np[overflow_np == 0]
    
    if len(valid_iq) == 0:
        return {
            'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
            'p10': 0.0, 'p50': 0.0, 'p90': 0.0
        }
    
    return {
        'mean': np.mean(valid_iq),
        'std': np.std(valid_iq),
        'min': np.min(valid_iq),
        'max': np.max(valid_iq),
        'p10': np.percentile(valid_iq, 10),
        'p50': np.percentile(valid_iq, 50),
        'p90': np.percentile(valid_iq, 90)
    }


def compute_overflow_stats():
    """
    Compute overflow statistics.
    
    Returns:
        dict with keys: count, fraction
    """
    import numpy as np
    
    overflow_np = overflow.to_numpy()
    count = np.sum(overflow_np)
    fraction = count / C.N
    
    return {'count': int(count), 'fraction': fraction}

