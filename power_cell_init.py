"""
Initialization: Seed Particles with Blue Noise / Poisson Disk Sampling
"""

import taichi as ti
import numpy as np
import power_cell_config as C
from power_cell_state import pos, rad, vel, iq_prev

# =============================================================================
# Random Initialization (Simple)
# =============================================================================

def init_random():
    """
    Initialize particles with random positions and radii.
    
    Positions: uniform random in [-L, L]³
    Radii: uniform random in [INIT_RAD_MIN, INIT_RAD_MAX] for heterogeneous foam
    Velocities: zero
    """
    # Positions
    pos_np = np.random.uniform(-C.L, C.L, (C.N, 3)).astype(np.float32)
    pos.from_numpy(pos_np)
    
    # Radii (uniform random for heterogeneous foam)
    rad_np = np.random.uniform(C.INIT_RAD_MIN, C.INIT_RAD_MAX, C.N).astype(np.float32)
    rad.from_numpy(rad_np)
    
    # Velocities (zero)
    vel_np = np.zeros((C.N, 3), dtype=np.float32)
    vel.from_numpy(vel_np)
    
    # IQ prev (initialize to 0.5 for EMA)
    iq_prev_np = np.full(C.N, 0.5, dtype=np.float32)
    iq_prev.from_numpy(iq_prev_np)
    
    print(f"[Init] Random initialization: N={C.N}")
    print(f"       Position range: [{-C.L:.4f}, {C.L:.4f}]³")
    print(f"       Radius: μ={rad_np.mean():.6f}, range=[{C.INIT_RAD_MIN:.6f}, {C.INIT_RAD_MAX:.6f}]")


# =============================================================================
# Blue Noise Initialization (Poisson Disk Sampling)
# =============================================================================

def init_blue_noise():
    """
    Initialize particles with TRUE blue noise (Bridson's algorithm).
    
    Fast Poisson disk sampling with minimum distance constraint.
    O(N) complexity using spatial grid acceleration.
    """
    print("[Init] Blue noise initialization (Bridson's algorithm)...")
    
    min_dist = C.POISSON_MIN_DIST
    cell_size = min_dist / np.sqrt(3)  # ensures one sample per cell
    grid_size = int(np.ceil(C.BOX_SIZE / cell_size))
    
    # Grid for fast neighbor lookup
    grid = {}
    
    # Active list for sampling
    active = []
    samples = []
    
    # Start with one random seed
    seed = np.random.uniform(-C.L, C.L, 3).astype(np.float32)
    samples.append(seed)
    active.append(0)
    
    # Add to grid
    gx = int((seed[0] + C.L) / cell_size)
    gy = int((seed[1] + C.L) / cell_size)
    gz = int((seed[2] + C.L) / cell_size)
    grid[(gx, gy, gz)] = 0
    
    k_attempts = C.POISSON_K_ATTEMPTS  # attempts per active sample
    
    while active and len(samples) < C.N:
        # Pick random active sample
        idx = np.random.randint(len(active))
        sample_idx = active[idx]
        sample = samples[sample_idx]
        
        found = False
        for _ in range(k_attempts):
            # Generate candidate between min_dist and 2*min_dist
            angle_theta = np.random.uniform(0, 2*np.pi)
            angle_phi = np.random.uniform(0, np.pi)
            r = np.random.uniform(min_dist, 2*min_dist)
            
            # Spherical to Cartesian
            candidate = sample + r * np.array([
                np.sin(angle_phi) * np.cos(angle_theta),
                np.sin(angle_phi) * np.sin(angle_theta),
                np.cos(angle_phi)
            ], dtype=np.float32)
            
            # Periodic wrap
            candidate = candidate - np.round(candidate / C.BOX_SIZE) * C.BOX_SIZE
            
            # Check if valid (not too close to existing samples)
            cx = int((candidate[0] + C.L) / cell_size)
            cy = int((candidate[1] + C.L) / cell_size)
            cz = int((candidate[2] + C.L) / cell_size)
            
            valid = True
            # Check 3x3x3 neighborhood
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        ncx = (cx + dx) % grid_size
                        ncy = (cy + dy) % grid_size
                        ncz = (cz + dz) % grid_size
                        
                        if (ncx, ncy, ncz) in grid:
                            neighbor_idx = grid[(ncx, ncy, ncz)]
                            neighbor = samples[neighbor_idx]
                            
                            # Periodic distance
                            delta = candidate - neighbor
                            delta = delta - np.round(delta / C.BOX_SIZE) * C.BOX_SIZE
                            dist = np.linalg.norm(delta)
                            
                            if dist < min_dist:
                                valid = False
                                break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Add new sample
                new_idx = len(samples)
                samples.append(candidate)
                active.append(new_idx)
                grid[(cx, cy, cz)] = new_idx
                found = True
                break
        
        if not found:
            # Remove from active list
            active.pop(idx)
    
    print(f"[Init] Blue noise: generated {len(samples)} samples (target {C.N})")
    
    # If not enough, fill with random (relaxed constraint)
    while len(samples) < C.N:
        samples.append(np.random.uniform(-C.L, C.L, 3).astype(np.float32))
    
    pos_np = np.array(samples[:C.N], dtype=np.float32)
    pos.from_numpy(pos_np)
    
    # Radii (uniform random for heterogeneous foam)
    rad_np = np.random.uniform(C.INIT_RAD_MIN, C.INIT_RAD_MAX, C.N).astype(np.float32)
    rad.from_numpy(rad_np)
    
    # Velocities (zero)
    vel_np = np.zeros((C.N, 3), dtype=np.float32)
    vel.from_numpy(vel_np)
    
    # IQ prev
    iq_prev_np = np.full(C.N, 0.5, dtype=np.float32)
    iq_prev.from_numpy(iq_prev_np)
    
    print(f"       Min separation: {min_dist:.6f}")
    print(f"       Radius: μ={rad_np.mean():.6f}, range=[{C.INIT_RAD_MIN:.6f}, {C.INIT_RAD_MAX:.6f}]")


# =============================================================================
# Initialization Entry Point
# =============================================================================

def initialize():
    """
    Initialize particle state.
    
    Uses blue noise if enabled, otherwise random.
    Then solves for weights if capacity solver is enabled.
    """
    if C.USE_BLUE_NOISE:
        init_blue_noise()
    else:
        init_random()
    
    # Solve for weights from target volumes (if enabled)
    if C.USE_CAPACITY_SOLVER:
        import power_cell_capacity as capacity
        
        # Get current radii (from initialization)
        rad_np = rad.to_numpy()
        
        # Generate target volumes from initial radii
        # This preserves the initial size distribution
        target_volumes = capacity.from_radii_targets(rad_np)
        
        # Solve for weights
        capacity.initialize_from_targets(target_volumes)
        
        print(f"[Init] Capacity solver applied")

