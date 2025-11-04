"""
Controller: IQ-Driven Radius Adjustment
Asymmetric zero-sum decile policy
"""

import taichi as ti
import numpy as np
import power_cell_config as C
from power_cell_state import rad, volume, iq, overflow

# =============================================================================
# Controller: Asymmetric Zero-Sum Decile
# =============================================================================

def adjust_radii():
    """
    Adjust radii based on IQ (isoperimetric quotient).
    
    Policy:
        1. Compute skewness: s = 1 - IQ
        2. Identify worst decile (highest 10% skewness = most irregular)
        3. Expand worst decile aggressively (+β_s * V)
        4. Shrink others gently (-β_e * V_mean)
        5. Renormalize to enforce exact zero-sum: Σ ΔV = 0
        6. Convert ΔV to Δr and apply with clamp
    
    Returns:
        deficit: zero-sum deficit (should be ~0)
        expand_count: number of particles expanded
        shrink_count: number of particles shrunk
        mean_skew: mean skewness
    """
    # Get data from GPU
    iq_np = iq.to_numpy()
    vol_np = volume.to_numpy()
    rad_np = rad.to_numpy()
    overflow_np = overflow.to_numpy()
    
    # Filter out overflowed cells
    valid_mask = (overflow_np == 0)
    
    # If too many cells overflow, skip this cycle
    overflow_frac = 1.0 - np.sum(valid_mask) / C.N
    if overflow_frac > 0.5:
        print(f"  [Controller] Too many overflows ({overflow_frac*100:.1f}%) - skipping adjustment")
        return 0.0, 0, 0, 1.0
    
    # Compute skewness
    skew_np = 1.0 - iq_np
    
    # Guard: if all IQ values are identical, skip this cycle
    iq_min = np.min(iq_np[valid_mask])
    iq_max = np.max(iq_np[valid_mask])
    if iq_max - iq_min < 1e-6:
        print("  [Controller] All IQ values identical - skipping adjustment")
        return 0.0, 0, 0, float(np.mean(skew_np[valid_mask]))
    
    # Compute mean volume (for shrink budget)
    V_mean = np.mean(vol_np[valid_mask])
    
    # Select cells to expand/shrink using FIXED threshold
    # IQ < threshold → EXPAND (irregular)
    # IQ >= threshold → SHRINK (regular)
    thresh = C.IQ_THRESHOLD
    expand_mask = (iq_np < thresh) & valid_mask  # irregular cells (low IQ)
    shrink_mask = (iq_np >= thresh) & valid_mask  # regular cells (high IQ)
    
    # Compute volume deltas (NO RENORMALIZATION - IQ is about shape, not volume conservation!)
    dV = np.zeros(C.N, dtype=np.float32)
    dV[expand_mask] = C.BETA_S * vol_np[expand_mask]      # expand worst by their own volume (full amount!)
    dV[shrink_mask] = -C.BETA_E * vol_np[shrink_mask]    # shrink others by their own volume (full amount!)
    
    # No zero-sum enforcement - let cells grow/shrink based on IQ alone!
    # The threshold percentile naturally balances expansion vs shrinkage
    
    # Compute total volume change (for diagnostics only)
    deficit = np.sum(dV)
    
    # Convert ΔV to Δr
    if C.DR_METHOD == "sphere":
        # Heuristic: dr = dV / (4π r²)
        dr = dV / (4.0 * C.PI * rad_np**2 + 1e-12)
    elif C.DR_METHOD == "proportional":
        # Proportional: dr = γ * (dV/V) * r
        dr = C.DR_GAMMA * (dV / (vol_np + 1e-12)) * rad_np
    else:
        raise ValueError(f"Unknown DR_METHOD: {C.DR_METHOD}")
    
    # Clamp per-cycle change
    dr_max = C.DR_MAX_FRAC * rad_np
    dr = np.clip(dr, -dr_max, dr_max)
    
    # Apply radius changes
    rad_new = np.clip(rad_np + dr, C.R_MIN, C.R_MAX)
    
    # Upload back to GPU
    rad.from_numpy(rad_new)
    
    # Compute statistics
    expand_count = int(np.sum(expand_mask))
    shrink_count = int(np.sum(shrink_mask))
    mean_skew = float(np.mean(skew_np[valid_mask]))
    
    return deficit, expand_count, shrink_count, mean_skew


# =============================================================================
# Alternative Controller: Band-Based (TODO)
# =============================================================================

def adjust_radii_band():
    """
    Alternative controller using IQ bands (not implemented yet).
    
    Policy:
        - Define IQ bands (e.g., [0.22, 0.30])
        - Expand cells below band, shrink cells above band
        - Enforce zero-sum
    
    TODO: Implement if needed
    """
    raise NotImplementedError("Band-based controller not implemented yet")

