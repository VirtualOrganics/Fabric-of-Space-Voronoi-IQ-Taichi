"""
Visualization: Taichi GUI Window
Simple 3D particle rendering with interactive controls
"""

import taichi as ti
import power_cell_config as C
from power_cell_state import pos, rad, iq

# =============================================================================
# GUI Setup
# =============================================================================

# Create window
window = ti.ui.Window("Power Cell Foam", (1024, 768), vsync=True)
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.Camera()

# Camera setup
camera.position(0.0, 0.0, 3.0)
camera.lookat(0.0, 0.0, 0.0)
camera.up(0.0, 1.0, 0.0)

# Control state
paused = False
last_space_state = False  # for detecting space press (not hold)

# UI state
iq_threshold_slider = C.IQ_THRESHOLD  # current slider value

# =============================================================================
# Rendering
# =============================================================================

def render_frame(show_cells=False):
    """
    Render one frame.
    
    Shows particles as spheres, optionally with Voronoi cell wireframes.
    
    Args:
        show_cells: if True, render cell edges (expensive)
    """
    global paused, last_space_state, iq_threshold_slider
    
    # Handle pause toggle (space bar - detect press, not hold)
    current_space_state = window.is_pressed(ti.ui.SPACE)
    if current_space_state and not last_space_state:
        paused = not paused
        print(f"[VIZ] {'PAUSED' if paused else 'RESUMED'}")
    last_space_state = current_space_state
    
    # UI Panel
    gui = window.get_gui()
    
    # Draw UI panel
    with gui.sub_window("Controls", 0.02, 0.02, 0.35, 0.30):
        # Average IQ display and statistics
        import numpy as np
        iq_np = iq.to_numpy()
        avg_iq = float(np.mean(iq_np))
        gui.text(f"Average IQ: {avg_iq:.3f}")
        
        # Show how many cells will expand/shrink with current threshold
        from power_cell_state import overflow
        overflow_np = overflow.to_numpy()
        valid_mask = (overflow_np == 0)
        expand_count = int(np.sum((iq_np < iq_threshold_slider) & valid_mask))
        shrink_count = int(np.sum((iq_np >= iq_threshold_slider) & valid_mask))
        total_valid = int(np.sum(valid_mask))
        expand_pct = 100.0 * expand_count / total_valid if total_valid > 0 else 0.0
        gui.text(f"Expand: {expand_count} ({expand_pct:.1f}%)")
        gui.text(f"Shrink: {shrink_count} ({100-expand_pct:.1f}%)")
        
        # IQ Threshold slider with fine control at low values
        old_threshold = iq_threshold_slider
        
        # Ultra-fine control for very low IQ values (0.0000 - 0.0100)
        if iq_threshold_slider <= 0.01:
            gui.text(f"Threshold: {iq_threshold_slider:.6f}")
            iq_threshold_slider = gui.slider_float(
                "IQ (ultra-fine)", 
                iq_threshold_slider, 
                minimum=0.0, 
                maximum=0.01
            )
        # Fine control slider (0.00 - 0.10)
        elif iq_threshold_slider <= 0.1:
            gui.text(f"Threshold: {iq_threshold_slider:.4f}")
            iq_threshold_slider = gui.slider_float(
                "IQ (fine)", 
                iq_threshold_slider, 
                minimum=0.0, 
                maximum=0.1
            )
        else:
            # Coarse control slider (0.0 - 1.0)
            gui.text(f"Threshold: {iq_threshold_slider:.3f}")
            iq_threshold_slider = gui.slider_float(
                "IQ Threshold", 
                iq_threshold_slider, 
                minimum=0.0, 
                maximum=1.0
            )
        
        # Buttons
        if iq_threshold_slider <= 0.1:
            if gui.button("Switch to coarse (0-1.0)"):
                iq_threshold_slider = 0.5
        
        # Auto-set to 10th percentile
        if gui.button("Set to 10th percentile"):
            p10 = float(np.percentile(iq_np[valid_mask], 10))
            iq_threshold_slider = p10
            C.IQ_THRESHOLD = p10
            print(f"[VIZ] Set threshold to p10 = {p10:.6f}")
        
        # Auto-set to 5th percentile
        if gui.button("Set to 5th percentile"):
            p5 = float(np.percentile(iq_np[valid_mask], 5))
            iq_threshold_slider = p5
            C.IQ_THRESHOLD = p5
            print(f"[VIZ] Set threshold to p5 = {p5:.6f}")
        
        # Update config if slider changed
        if abs(iq_threshold_slider - old_threshold) > 0.001:
            C.IQ_THRESHOLD = iq_threshold_slider
            print(f"[VIZ] IQ Threshold = {iq_threshold_slider:.3f}")
            print(f"      IQ < {iq_threshold_slider:.3f} → EXPAND (irregular)")
            print(f"      IQ >= {iq_threshold_slider:.3f} → SHRINK (regular)")
        
        # Instructions
        gui.text("")
        gui.text("IQ Logic:")
        gui.text("  Low IQ = irregular")
        gui.text("  High IQ = regular")
        gui.text("  IQ=1.0 = perfect sphere")
        gui.text("")
        gui.text("Controls:")
        gui.text("SPACE - Pause/Resume")
        gui.text("SHIFT+DRAG - Rotate")
        gui.text("RIGHT DRAG - Pan")
        gui.text("SCROLL - Zoom")
        gui.text("WASD/Arrows - Move")
        gui.text("ESC - Exit")
    
    # Camera controls (always active, but rotation requires SHIFT)
    # This enables: WASD for movement, mouse wheel for zoom, arrow keys, etc.
    if window.is_pressed(ti.ui.SHIFT):
        # SHIFT held: enable rotation with left mouse
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.LMB)
    else:
        # No SHIFT: still allow zoom/pan but no rotation
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)
    
    # Lighting
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.point_light(pos=(0.5, 1.5, 0.5), color=(1, 1, 1))
    
    # Render particles
    # Color by IQ: red=irregular (low IQ), blue=regular (high IQ)
    scene.particles(
        pos,
        radius=0.001,  # base radius (required)
        per_vertex_radius=rad,
        color=(0.7, 0.8, 1.0)
    )
    
    # TODO: Render cell wireframes
    # This requires extracting edges from the polyhedra and drawing lines
    # For now, just particles
    
    # Draw to canvas
    canvas.scene(scene)
    
    # Show window
    window.show()


def should_close():
    """Check if window should close."""
    return window.is_pressed(ti.ui.ESCAPE) or not window.running


def is_paused():
    """Check if simulation is paused."""
    return paused

