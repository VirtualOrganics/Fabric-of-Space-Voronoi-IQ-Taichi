"""
Main Loop: Freeze → Measure → Adjust → Relax
"""

import time
import numpy as np
import power_cell_config as C
import power_cell_state as state
import power_cell_grid as grid
import power_cell_planes as planes
import power_cell_clipper as clipper
import power_cell_measure as measure
import power_cell_controller as controller
import power_cell_relax as relax
import power_cell_viz as viz

# =============================================================================
# Single Cycle: Freeze → Measure → Adjust → Relax
# =============================================================================

def run_cycle(cycle_num):
    """
    Run one freeze-measure-adjust-relax cycle.
    
    Args:
        cycle_num: cycle number (for logging)
    
    Returns:
        iq_stats: dict with IQ statistics
        converged: True if converged
    """
    print(f"\n{'='*60}")
    print(f"Cycle {cycle_num}")
    print(f"{'='*60}")
    
    # =========================================================================
    # FREEZE (implicit - positions are frozen during measurement)
    # =========================================================================
    
    # =========================================================================
    # MEASURE: Exact geometry via analytic clipping
    # =========================================================================
    print("[MEASURE] Building spatial hash...")
    t0 = time.time()
    grid.build_grid()
    grid.gather_neighbors()
    t_grid = time.time() - t0
    
    # Diagnostic: check neighbor counts
    nbr_ct_np = state.nbr_ct.to_numpy()
    print(f"[MEASURE] Neighbors: μ={nbr_ct_np.mean():.1f} min={nbr_ct_np.min()} max={nbr_ct_np.max()}")
    
    print("[MEASURE] Building power planes...")
    t0 = time.time()
    planes.build_planes()
    t_planes = time.time() - t0
    
    print("[MEASURE] Initializing cubes...")
    t0 = time.time()
    clipper.init_cubes()
    t_init = time.time() - t0
    
    print("[MEASURE] Clipping polyhedra...")
    t0 = time.time()
    clipper.clip_all()
    t_clip = time.time() - t0
    
    print("[MEASURE] Computing volume/area/IQ/FSC...")
    t0 = time.time()
    measure.measure_all()
    t_measure = time.time() - t0
    
    t_total_measure = t_grid + t_planes + t_init + t_clip + t_measure
    
    print(f"[MEASURE] Timing breakdown:")
    print(f"          Grid:    {t_grid*1000:.2f} ms")
    print(f"          Planes:  {t_planes*1000:.2f} ms")
    print(f"          Init:    {t_init*1000:.2f} ms")
    print(f"          Clip:    {t_clip*1000:.2f} ms")
    print(f"          Measure: {t_measure*1000:.2f} ms")
    print(f"          TOTAL:   {t_total_measure*1000:.2f} ms")
    
    # Get statistics
    iq_stats = measure.compute_iq_stats()
    overflow_stats = measure.compute_overflow_stats()
    
    print(f"[MEASURE] IQ: μ={iq_stats['mean']:.3f} σ={iq_stats['std']:.3f} "
          f"range=[{iq_stats['min']:.3f}, {iq_stats['max']:.3f}]")
    print(f"          Percentiles: p10={iq_stats['p10']:.3f} "
          f"p50={iq_stats['p50']:.3f} p90={iq_stats['p90']:.3f}")
    
    # DIAGNOSTIC: Check volume and area values
    vol_np = state.volume.to_numpy()
    area_np = state.area.to_numpy()
    valid = (state.overflow.to_numpy() == 0)
    print(f"          Volume: μ={vol_np[valid].mean():.6f} range=[{vol_np[valid].min():.6f}, {vol_np[valid].max():.6f}]")
    print(f"          Area: μ={area_np[valid].mean():.6f} range=[{area_np[valid].min():.6f}, {area_np[valid].max():.6f}]")
    print(f"          Overflow: {overflow_stats['count']} cells "
          f"({overflow_stats['fraction']*100:.1f}%)")
    
    # Warn if too many overflows
    if overflow_stats['fraction'] > C.OVERFLOW_WARN_THRESH:
        print(f"⚠️  WARNING: High overflow rate! Consider increasing V_MAX/F_MAX/I_MAX")
    
    # =========================================================================
    # ADJUST: IQ-driven radius changes
    # =========================================================================
    print("[ADJUST] Applying IQ-driven radius changes...")
    t0 = time.time()
    deficit, expand_count, shrink_count, mean_skew = controller.adjust_radii()
    t_adjust = time.time() - t0
    
    print(f"[ADJUST] IQ Threshold: {C.IQ_THRESHOLD:.3f}")
    print(f"         Logic: IQ < {C.IQ_THRESHOLD:.3f} → EXPAND (2× radius)")
    print(f"                IQ >= {C.IQ_THRESHOLD:.3f} → SHRINK (0.9× radius)")
    print(f"         Skewness: μ={mean_skew:.3f}")
    print(f"         Actions: expand={expand_count} shrink={shrink_count}")
    print(f"         Volume change: {deficit:.2e}")
    print(f"         Time: {t_adjust*1000:.2f} ms")
    
    # Diagnostic: Show IQ distribution for expand/shrink groups
    import numpy as np
    iq_np = state.iq.to_numpy()
    rad_np = state.rad.to_numpy()
    overflow_np = state.overflow.to_numpy()
    valid_mask = (overflow_np == 0)
    
    # Use same logic as controller
    thresh = C.IQ_THRESHOLD
    expand_mask = (iq_np < thresh) & valid_mask
    shrink_mask = (iq_np >= thresh) & valid_mask
    
    if expand_count > 0:
        print(f"         Expand group: IQ range=[{iq_np[expand_mask].min():.3f}, {iq_np[expand_mask].max():.3f}], "
              f"rad range=[{rad_np[expand_mask].min():.6f}, {rad_np[expand_mask].max():.6f}]")
    if shrink_count > 0:
        print(f"         Shrink group: IQ range=[{iq_np[shrink_mask].min():.3f}, {iq_np[shrink_mask].max():.3f}], "
              f"rad range=[{rad_np[shrink_mask].min():.6f}, {rad_np[shrink_mask].max():.6f}]")
    
    if abs(deficit) > 1e-4:
        print(f"⚠️  WARNING: Zero-sum violated! |deficit| = {abs(deficit):.2e}")
    
    # =========================================================================
    # RELAX: Soft forces + PBD
    # =========================================================================
    print(f"[RELAX] Running {C.RELAX_FRAMES} frames...")
    t0 = time.time()
    
    # Track particle movement during relax
    pos_before = state.pos.to_numpy().copy()
    
    for frame in range(C.RELAX_FRAMES):
        # Rebuild grid (particles have moved)
        grid.build_grid()
        grid.gather_neighbors()
        
        # Compute forces
        relax.zero_forces()
        relax.compute_forces()
        
        # Integrate
        relax.integrate()
        
        # PBD iterations
        for _ in range(C.PBD_ITERS):
            relax.pbd_project()
    
    t_relax = time.time() - t0
    fps_relax = C.RELAX_FRAMES / t_relax
    
    # Measure how much particles moved
    pos_after = state.pos.to_numpy()
    movement = np.linalg.norm(pos_after - pos_before, axis=1)
    print(f"[RELAX] Time: {t_relax*1000:.2f} ms ({fps_relax:.1f} FPS)")
    print(f"[RELAX] Movement: μ={movement.mean():.6f} max={movement.max():.6f} (mean spacing={C.MEAN_SPACING:.6f})")
    
    # Check overlaps after relax
    overlap_frac, max_pen = relax.get_overlap_stats()
    print(f"[RELAX] Overlaps: {overlap_frac*100:.1f}% max_pen={max_pen:.6f}")
    
    # =========================================================================
    # Convergence Check
    # =========================================================================
    # TODO: Track ΔIQ between cycles
    converged = False  # placeholder
    
    return iq_stats, converged


# =============================================================================
# Main Loop: Run Multiple Cycles
# =============================================================================

def run_simulation(max_cycles=None):
    """
    Run the freeze-measure-adjust-relax loop until convergence.
    
    Args:
        max_cycles: maximum number of cycles (None = use config default)
    """
    if max_cycles is None:
        max_cycles = C.MAX_CYCLES
    
    # Track IQ evolution
    iq_history = []
    
    print(f"\n{'#'*60}")
    print(f"# Analytic Power-Cell Foam Simulation (Option A)")
    print(f"# Freeze → Measure → Adjust → Relax")
    print(f"{'#'*60}")
    print(f"N = {C.N} particles")
    print(f"Domain: [{-C.L:.2f}, {C.L:.2f}]³")
    print(f"Mean spacing: {C.MEAN_SPACING:.4f}")
    print(f"Security radius: {C.R_SEC:.4f} ({C.R_SEC_ALPHA:.1f}× spacing)")
    print(f"Caps: K_MAX={C.K_MAX}, V_MAX={C.V_MAX}, F_MAX={C.F_MAX}, I_MAX={C.I_MAX}")
    print(f"Controller: β_s={C.BETA_S:.3f}, β_e={C.BETA_E:.3f}, dr_max={C.DR_MAX_FRAC:.3f}")
    print(f"Relax: {C.RELAX_FRAMES} frames, ka={C.K_ATTR:.2f}, kr={C.K_REP:.2f}, PBD={C.PBD_ITERS} iters")
    print(f"Max cycles: {max_cycles}")
    print(f"{'#'*60}\n")
    
    # Run cycles
    cycle = 0
    while cycle < max_cycles:
        # Check if window should close
        if viz.should_close():
            print(f"\n⚠️  User closed window at cycle {cycle}")
            break
        
        # Render (always update display, even when paused)
        viz.render_frame()
        
        # Skip simulation update if paused
        if viz.is_paused():
            continue
        
        # Run one cycle
        iq_stats, converged = run_cycle(cycle)
        
        # Track IQ evolution
        iq_history.append(iq_stats['mean'])
        if len(iq_history) >= 2:
            iq_change = iq_history[-1] - iq_history[-2]
            print(f"[EVOLUTION] IQ change: {iq_change:+.6f} (cycle {cycle-1}→{cycle})")
        
        if converged:
            print(f"\n✓ Converged at cycle {cycle}")
            break
        
        cycle += 1
    
    # Keep window open after simulation
    print(f"\n{'#'*60}")
    print(f"# Simulation Complete - Press ESC to close")
    print(f"{'#'*60}\n")
    
    while not viz.should_close():
        viz.render_frame()

