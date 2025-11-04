"""
Main Entry Point: Analytic Power-Cell Foam Simulation
Freeze → Measure → Adjust → Relax architecture
"""

import power_cell_config as C
import power_cell_state  # Initialize Taichi and allocate fields
import power_cell_init as init
import power_cell_loop as loop

def main():
    """
    Main entry point for the simulation.
    
    Steps:
        1. Initialize particles (random or blue noise)
        2. Run freeze-measure-adjust-relax cycles
        3. Report final statistics
    """
    print("\n" + "="*60)
    print("Analytic Power-Cell Foam Simulation")
    print("Option A: Freeze → Measure → Adjust → Relax")
    print("="*60 + "\n")
    
    # Initialize
    print("[INIT] Initializing particles...")
    init.initialize()
    
    # Run simulation
    loop.run_simulation(max_cycles=C.MAX_CYCLES)
    
    print("\nDone.")


if __name__ == "__main__":
    main()

