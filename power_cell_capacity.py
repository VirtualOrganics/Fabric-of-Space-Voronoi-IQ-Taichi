"""
Capacity-Constrained Power Diagram Solver

Solves for weights w_i such that power cell volumes match target volumes V_i*.
Uses local Laplacian updates based on face areas and neighbor distances.

Algorithm:
    1. Initialize weights from target volumes (sphere guess)
    2. For each inner iteration:
        - Build planes, clip cells, measure volumes and face areas
        - Compute Laplacian coefficients from face areas and distances
        - Run Jacobi/Gauss-Seidel sweeps to solve L·δw = b
        - Update weights: w ← w + α·δw
        - Re-center weights (gauge fixing)
    3. Set radii from weights: r_i = √(w_i)
"""

import taichi as ti
import numpy as np
import power_cell_config as C
from power_cell_state import (
    pos, rad, nbr_ct, nbr_idx, overflow, volume, area,
    faces_begin, faces_count, faces_normal, idx, verts, f_ct,
    plane_id, plane_n
)
import power_cell_grid as grid
import power_cell_planes as planes
import power_cell_clipper as clipper
import power_cell_measure as measure

# =============================================================================
# Capacity Solver State
# =============================================================================

# Target volumes (per particle)
target_volume = ti.field(dtype=ti.f32, shape=C.N)

# Weights (w_i = r_i²) - these define the power diagram
weights = ti.field(dtype=ti.f32, shape=C.N)

# Weight updates (δw from Laplacian solve)
delta_w = ti.field(dtype=ti.f32, shape=C.N)

# Face data for Laplacian (per neighbor pair)
# For each particle i and neighbor k, store:
#   - face_area[i, k]: area of face between i and nbr_idx[i, k]
#   - face_dist[i, k]: distance between particle centers
face_area_cap = ti.field(dtype=ti.f32, shape=(C.N, C.K_MAX))
face_dist_cap = ti.field(dtype=ti.f32, shape=(C.N, C.K_MAX))

# =============================================================================
# Face Data Extraction (from polyhedra)
# =============================================================================

@ti.kernel
def extract_face_data():
    """
    Extract face areas and neighbor distances for capacity solver.
    
    For each particle i:
        - Iterate through faces
        - Match face normal to neighbor (plane_id)
        - Store area and distance
    
    Note: This assumes faces are created in the same order as planes,
    which is true in our clipper (one face per neighbor).
    """
    # Clear face data
    for i in range(C.N):
        for k in range(C.K_MAX):
            face_area_cap[i, k] = 0.0
            face_dist_cap[i, k] = 0.0
    
    # Extract face areas
    for i in range(C.N):
        if overflow[i] != 0:
            continue
        
        p_i = pos[i]
        
        # For each face, compute area and match to neighbor
        for f in range(f_ct[i]):
            begin = faces_begin[i, f]
            count = faces_count[i, f]
            
            if count < 3:
                continue
            
            # Compute face area (triangle fan)
            v0 = verts[i, idx[i, begin]]
            face_area_val = 0.0
            for k in range(1, count - 1):
                vk = verts[i, idx[i, begin + k]]
                vk1 = verts[i, idx[i, begin + k + 1]]
                a = vk - v0
                b = vk1 - v0
                c = ti.math.cross(a, b)
                face_area_val += 0.5 * c.norm()
            
            # Match face to neighbor by normal direction
            # (faces are created in same order as planes in clipper)
            # For now, use simple matching: face f corresponds to neighbor k=f
            # (This works if clip_all creates faces in plane order)
            if f < nbr_ct[i]:
                k = f
                j = nbr_idx[i, k]
                
                # Compute distance (periodic)
                p_j = pos[j]
                delta = p_j - p_i
                # Periodic wrap
                delta = delta - ti.round(delta / C.BOX_SIZE) * C.BOX_SIZE
                dist = delta.norm()
                
                # Store
                face_area_cap[i, k] = face_area_val
                face_dist_cap[i, k] = ti.max(dist, 1e-9)

# =============================================================================
# Laplacian Solver (Jacobi/Gauss-Seidel)
# =============================================================================

@ti.kernel
def jacobi_sweep():
    """
    One Jacobi sweep for L·δw = b.
    
    L_ii = Σ_j (A_ij / d_ij)
    L_ij = -(A_ij / d_ij)  for j ∈ neighbors(i)
    b_i = 2 * (V_i* - V_i)
    
    Jacobi update:
        δw_i ← (b_i - Σ_j L_ij·δw_j) / L_ii
    """
    for i in range(C.N):
        if overflow[i] != 0:
            delta_w[i] = 0.0
            continue
        
        # Compute diagonal L_ii and off-diagonal sum
        L_ii = 0.0
        sum_Lj_dwj = 0.0
        
        for k in range(nbr_ct[i]):
            j = nbr_idx[i, k]
            A_ij = face_area_cap[i, k]
            d_ij = face_dist_cap[i, k]
            
            if A_ij < 1e-12 or d_ij < 1e-9:
                continue
            
            lij = A_ij / d_ij
            L_ii += lij
            sum_Lj_dwj += (-lij) * delta_w[j]  # use previous iteration values
        
        # Right-hand side
        b_i = 2.0 * (target_volume[i] - volume[i])
        
        # Jacobi update
        if L_ii > 1e-12:
            delta_w[i] = (b_i - sum_Lj_dwj) / L_ii
        else:
            delta_w[i] = 0.0

@ti.kernel
def update_weights(alpha: ti.f32):
    """
    Update weights: w ← w + α·δw
    
    Args:
        alpha: damping factor (0.3-0.8)
    """
    for i in range(C.N):
        weights[i] += alpha * delta_w[i]

@ti.kernel
def recenter_weights():
    """
    Re-center weights to fix gauge: w ← w - mean(w)
    
    Weights are only defined up to an additive constant.
    """
    # Compute mean
    w_sum = 0.0
    count = 0
    for i in range(C.N):
        if overflow[i] == 0:
            w_sum += weights[i]
            count += 1
    
    if count > 0:
        w_mean = w_sum / ti.cast(count, ti.f32)
        
        # Subtract mean
        for i in range(C.N):
            weights[i] -= w_mean

@ti.kernel
def weights_to_radii():
    """
    Convert weights to radii: r_i = √(max(w_i, r_min²))
    
    Clamp to [r_min, r_max] for stability.
    """
    for i in range(C.N):
        r_i = ti.sqrt(ti.max(weights[i], C.R_MIN * C.R_MIN))
        rad[i] = ti.max(C.R_MIN, ti.min(r_i, C.R_MAX))

# =============================================================================
# Capacity Solver Main Loop
# =============================================================================

def solve_capacity(target_volumes_np):
    """
    Solve for weights w_i such that cell volumes match target_volumes.
    
    Args:
        target_volumes_np: numpy array of target volumes (shape: N)
    
    Returns:
        converged: True if converged within tolerance
        mean_error: mean |V_i - V_i*| / V_i*
    """
    # Upload target volumes
    target_volume.from_numpy(target_volumes_np.astype(np.float32))
    
    # Initialize weights from targets (sphere guess)
    r0 = ((3.0 * target_volumes_np) / (4.0 * np.pi)) ** (1.0/3.0)
    w0 = r0 * r0
    w0 -= w0.mean()  # center
    weights.from_numpy(w0.astype(np.float32))
    
    # Convert to radii for first iteration
    weights_to_radii()
    
    print(f"[Capacity] Solving for target volumes...")
    print(f"           Target: μ={target_volumes_np.mean():.6e}, σ={target_volumes_np.std():.6e}")
    
    # Inner iterations
    for it in range(C.CAPACITY_INNER_ITERS):
        # Build power diagram with current weights
        grid.build_grid()
        grid.gather_neighbors()
        planes.build_planes()
        clipper.init_cubes()
        clipper.clip_all()
        measure.measure_all()
        
        # Extract face data for Laplacian
        extract_face_data()
        
        # Compute volume error
        volume_np = volume.to_numpy()
        overflow_np = overflow.to_numpy()
        valid_mask = (overflow_np == 0)
        
        if np.sum(valid_mask) == 0:
            print(f"           [Iter {it}] All cells overflowed - aborting")
            return False, 1.0
        
        rel_error = np.abs(volume_np[valid_mask] - target_volumes_np[valid_mask]) / (target_volumes_np[valid_mask] + 1e-12)
        mean_error = np.mean(rel_error)
        max_error = np.max(rel_error)
        
        print(f"           [Iter {it}] Volume error: μ={mean_error:.6e}, max={max_error:.6e}")
        
        # Check convergence
        if mean_error < C.CAPACITY_TOL:
            print(f"           Converged in {it+1} iterations")
            return True, mean_error
        
        # Jacobi/Gauss-Seidel sweeps
        for sweep in range(C.CAPACITY_SOLVER_SWEEPS):
            jacobi_sweep()
        
        # Update weights with damping
        update_weights(C.CAPACITY_ALPHA)
        
        # Re-center weights
        recenter_weights()
        
        # Convert to radii for next iteration
        weights_to_radii()
    
    print(f"           Did not converge after {C.CAPACITY_INNER_ITERS} iterations (error={mean_error:.6e})")
    return False, mean_error

# =============================================================================
# Target Volume Generators
# =============================================================================

def uniform_targets():
    """
    Generate uniform target volumes: V_i* = V_box / N
    
    Returns:
        numpy array of target volumes (shape: N)
    """
    V_box = C.BOX_SIZE ** 3
    return np.full(C.N, V_box / C.N, dtype=np.float32)

def from_radii_targets(radii_np):
    """
    Generate target volumes from radii: V_i* = (4/3)π r_i³
    
    Rescales to sum to V_box.
    
    Args:
        radii_np: numpy array of radii (shape: N)
    
    Returns:
        numpy array of target volumes (shape: N)
    """
    V_i = (4.0/3.0) * np.pi * (radii_np ** 3)
    V_box = C.BOX_SIZE ** 3
    V_i *= V_box / np.sum(V_i)  # rescale to match box volume
    return V_i.astype(np.float32)

# =============================================================================
# Initialization Helper
# =============================================================================

def initialize_from_targets(target_volumes_np=None):
    """
    Initialize weights and radii from target volumes.
    
    If target_volumes_np is None, uses uniform targets.
    
    Args:
        target_volumes_np: numpy array of target volumes (shape: N), or None
    """
    if target_volumes_np is None:
        target_volumes_np = uniform_targets()
    
    # Solve for weights
    converged, error = solve_capacity(target_volumes_np)
    
    if not converged:
        print(f"[Capacity] Warning: did not converge (error={error:.6e})")
    
    # Radii are already set by weights_to_radii() in solve_capacity
    rad_np = rad.to_numpy()
    print(f"[Capacity] Final radii: μ={rad_np.mean():.6f}, σ={rad_np.std():.6f}")

