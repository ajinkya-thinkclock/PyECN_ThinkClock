"""
Temperature Visualization Module for PyECN
==========================================
Runs PyECN simulation and generates temperature visualizations.
All parameters are read from the TOML config file.

Functions:
    - plot_temp_avg_vs_time: Average temperature evolution
    - plot_temp_minmax_vs_time: Temperature bounds (min/max) over time
    - plot_temp_delta_vs_time: Temperature gradient (max - min) over time
    - plot_temp_std_vs_time: Temperature standard deviation over time
    - plot_all_temperature: Combined view of all temperature metrics
    - plot_unrolled_jellyroll: Unrolled jellyroll spatial temperature map (Cylindrical cells)
    - run_simulation_and_visualize: Main function to run PyECN and generate all plots
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import sys
import toml


def plot_temp_avg_vs_time(time, temp_avg, cell_name, c_rate, T_cooling, T_initial,
                          save_path=None, show=False, ax=None):
    """Plot average cell temperature vs time."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()
    
    # Convert time to hours if in seconds
    time_hours = time / 3600 if np.max(time) > 100 else time
    time_label = "Time (h)" if np.max(time) > 100 else "Time (s)"
    
    # Convert temperature to °C
    temp_celsius = temp_avg - 273.15
    T_cooling_c = T_cooling - 273.15
    T_initial_c = T_initial - 273.15
    
    ax.plot(time_hours, temp_celsius, 'r-', linewidth=2, label='Average Temperature')
    ax.set_xlabel(time_label, fontsize=12, fontweight='bold')
    ax.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
    ax.set_title(f'Average Cell Temperature - {cell_name} ({c_rate}C)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    ax.axhline(y=T_cooling_c, color='b', linestyle='--', alpha=0.5, linewidth=1, 
               label=f'Cooling Temp = {T_cooling_c:.1f}°C')
    ax.axhline(y=T_initial_c, color='g', linestyle='--', alpha=0.5, linewidth=1,
               label=f'Initial Temp = {T_initial_c:.1f}°C')
    ax.legend(fontsize=10)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig, ax


def plot_temp_minmax_vs_time(time, temp_min, temp_max, cell_name, c_rate,
                             save_path=None, show=False, ax=None):
    """Plot minimum and maximum cell temperature vs time."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()
    
    time_hours = time / 3600 if np.max(time) > 100 else time
    time_label = "Time (h)" if np.max(time) > 100 else "Time (s)"
    
    temp_min_celsius = temp_min - 273.15
    temp_max_celsius = temp_max - 273.15
    
    ax.plot(time_hours, temp_max_celsius, 'r-', linewidth=2, label='Max Temperature')
    ax.plot(time_hours, temp_min_celsius, 'b-', linewidth=2, label='Min Temperature')
    ax.fill_between(time_hours, temp_min_celsius, temp_max_celsius, 
                     alpha=0.2, color='orange', label='Temperature Range')
    
    ax.set_xlabel(time_label, fontsize=12, fontweight='bold')
    ax.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
    ax.set_title(f'Temperature Bounds - {cell_name} ({c_rate}C)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig, ax


def plot_temp_delta_vs_time(time, temp_delta, cell_name, c_rate,
                            save_path=None, show=False, ax=None):
    """Plot temperature gradient (max - min) vs time."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()
    
    time_hours = time / 3600 if np.max(time) > 100 else time
    time_label = "Time (h)" if np.max(time) > 100 else "Time (s)"
    
    ax.plot(time_hours, temp_delta, 'purple', linewidth=2, label='ΔT (Max - Min)')
    ax.fill_between(time_hours, 0, temp_delta, alpha=0.2, color='purple')
    
    ax.set_xlabel(time_label, fontsize=12, fontweight='bold')
    ax.set_ylabel('Temperature Gradient (°C)', fontsize=12, fontweight='bold')
    ax.set_title(f'Temperature Uniformity - {cell_name} ({c_rate}C)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=5.0, color='r', linestyle='--', alpha=0.5, linewidth=1, label='5°C threshold')
    ax.legend(fontsize=10)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig, ax


def plot_temp_std_vs_time(time, temp_std, cell_name, c_rate,
                         save_path=None, show=False, ax=None):
    """Plot temperature standard deviation vs time."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()
    
    time_hours = time / 3600 if np.max(time) > 100 else time
    time_label = "Time (h)" if np.max(time) > 100 else "Time (s)"
    
    ax.plot(time_hours, temp_std, 'orange', linewidth=2, label='Temperature Std Dev')
    ax.fill_between(time_hours, 0, temp_std, alpha=0.2, color='orange')
    
    ax.set_xlabel(time_label, fontsize=12, fontweight='bold')
    ax.set_ylabel('Temperature Std Dev (°C)', fontsize=12, fontweight='bold')
    ax.set_title(f'Temperature Variation - {cell_name} ({c_rate}C)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig, ax


def plot_all_temperature(time, temp_avg, temp_min, temp_max, temp_delta, temp_std,
                        cell_name, form_factor, c_rate, model, coupling, 
                        T_cooling, T_initial, save_path=None, show=False):
    """Create a combined plot with all temperature metrics."""
    fig = plt.figure(figsize=(14, 12))
    gs = GridSpec(4, 1, figure=fig, hspace=0.35)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[3, 0])
    
    plot_temp_avg_vs_time(time, temp_avg, cell_name, c_rate, T_cooling, T_initial, show=False, ax=ax1)
    plot_temp_minmax_vs_time(time, temp_min, temp_max, cell_name, c_rate, show=False, ax=ax2)
    plot_temp_delta_vs_time(time, temp_delta, cell_name, c_rate, show=False, ax=ax3)
    plot_temp_std_vs_time(time, temp_std, cell_name, c_rate, show=False, ax=ax4)
    
    title = f'Thermal Analysis - {cell_name} ({form_factor}, {c_rate}C)\n'
    title += f'Model: {model}, Coupling: {coupling}'
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig, [ax1, ax2, ax3, ax4]


def plot_unrolled_jellyroll(cell_obj, temp_min, temp_max, temp_levels, time_step=-1,
                           save_path=None, show=False):
    """Plot unrolled jellyroll temperature distribution for cylindrical cells."""
    
    if cell_obj.status_FormFactor != 'Cylindrical':
        print(f"  ⚠ Skipped: Unrolled jellyroll is for Cylindrical cells only (current: {cell_obj.status_FormFactor})")
        return None, None
    
    thermal_bc = cell_obj.status_ThermalBC_Core if hasattr(cell_obj, 'status_ThermalBC_Core') else 'SepFill'
    
    if thermal_bc not in ['SepFill', 'SepAir']:
        print(f"  ⚠ Skipped: Unrolled jellyroll requires SepFill or SepAir core (current: {thermal_bc})")
        return None, None
    
    if time_step == -1:
        time_step = cell_obj.nt
    time_val = time_step * cell_obj.dt if hasattr(cell_obj, 'dt') else time_step
    
    climit_vector = np.linspace(temp_min, temp_max, temp_levels)
    
    # Create figure
    if cell_obj.nstack > 1:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Extract geometry
    n_v = cell_obj.ny
    n_h_Al = int(np.size(cell_obj.Al_4T) / n_v)
    
    ind0_Al_4T = cell_obj.Al_4T.reshape(n_v, n_h_Al)
    ind0_Cu_4T = cell_obj.Cu_4T.reshape(n_v, n_h_Al)
    ind0_Elb_4T = cell_obj.Elb_4T.reshape(n_v, int(np.size(cell_obj.Elb_4T) / n_v))
    
    if cell_obj.nstack > 1:
        ind0_Elr_4T = cell_obj.Elr_4T.reshape(n_v, int(np.size(cell_obj.Elr_4T) / n_v))
    
    # Extract coordinate and temperature arrays
    array_h_Al_4T = cell_obj.xi_4T[ind0_Al_4T]
    array_v_Al_4T = (cell_obj.LG_Jellyroll - cell_obj.yi_4T[ind0_Al_4T])
    array_c_T_Al_4T = cell_obj.T_record[:, time_step][ind0_Al_4T] - 273.15
    
    array_h_Cu_4T = cell_obj.xi_4T[ind0_Cu_4T]
    array_v_Cu_4T = (cell_obj.LG_Jellyroll - cell_obj.yi_4T[ind0_Cu_4T])
    array_c_T_Cu_4T = cell_obj.T_record[:, time_step][ind0_Cu_4T] - 273.15
    
    array_h_Elb_4T = cell_obj.xi_4T[ind0_Elb_4T]
    array_v_Elb_4T = (cell_obj.LG_Jellyroll - cell_obj.yi_4T[ind0_Elb_4T])
    array_c_T_Elb_4T = cell_obj.T_record[:, time_step][ind0_Elb_4T] - 273.15
    
    if cell_obj.nstack > 1:
        array_h_Elr_4T = cell_obj.xi_4T[ind0_Elr_4T]
        array_v_Elr_4T = (cell_obj.LG_Jellyroll - cell_obj.yi_4T[ind0_Elr_4T])
        array_c_T_Elr_4T = cell_obj.T_record[:, time_step][ind0_Elr_4T] - 273.15
    
    # Add separator column
    array_h_Sep_4T = (array_h_Al_4T[:, 0] - cell_obj.b01).reshape(-1, 1)
    array_v_Sep = array_v_Al_4T[:, 0].reshape(-1, 1)
    array_h_SepAl_4T = np.append(array_h_Sep_4T, array_h_Al_4T, axis=1)
    array_v_SepAl_4T = np.append(array_v_Sep, array_v_Al_4T, axis=1)
    array_c_SepAl_4T = np.append(
        cell_obj.T_record[:, time_step].reshape(-1, 1)[cell_obj.ind0_Geo_core_AddSep_4T_4SepFill] - 273.15,
        array_c_T_Al_4T, axis=1
    )
    
    # Calculate spiral scaling factor to convert computational coordinates to physical length
    if hasattr(cell_obj, 'Spiral_Sep_s_real') and hasattr(cell_obj, 'Spiral_Sep_s'):
        spiral_scale = cell_obj.Spiral_Sep_s_real / cell_obj.Spiral_Sep_s
        print(f"  Spiral scaling: {spiral_scale:.4f} (real={cell_obj.Spiral_Sep_s_real:.4f}, comp={cell_obj.Spiral_Sep_s:.4f})")
    else:
        spiral_scale = 1.0
        print(f"  ⚠ Warning: Spiral scaling attributes not found, using scale=1.0")
    
    print(f"  Unrolled length: {np.max(array_h_Al_4T * spiral_scale):.4f} m")
    
    # Plot Al current collector
    ax1 = axes[0]
    ax1.set_title('Al Current Collector', fontsize=13, fontweight='bold')
    ax1.contourf(array_h_SepAl_4T * spiral_scale, array_v_SepAl_4T, 
                array_c_SepAl_4T, climit_vector, cmap="RdBu_r")
    ax1.scatter(array_h_Sep_4T, array_v_Sep, facecolors='w', edgecolors='k', s=10)
    ax1.set_xlabel('Unrolled Distance (m)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Axial Position (m)', fontsize=11, fontweight='bold')
    
    # Plot Cu current collector
    ax2 = axes[1]
    ax2.set_title('Cu Current Collector', fontsize=13, fontweight='bold')
    surf2 = ax2.contourf(array_h_Cu_4T * spiral_scale, array_v_Cu_4T, 
                        array_c_T_Cu_4T, climit_vector, cmap="RdBu_r")
    surf2.cmap.set_under('cyan')
    surf2.cmap.set_over('yellow')
    ax2.set_xlabel('Unrolled Distance (m)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Axial Position (m)', fontsize=11, fontweight='bold')
    
    # Add colorbar
    cb_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(surf2, cax=cb_ax)
    cbar.set_label('Temperature (°C)', fontsize=11, fontweight='bold')
    fig.subplots_adjust(right=0.9, hspace=0.3, wspace=0.3)
    
    # Plot electrodes if nstack > 1
    if cell_obj.nstack > 1:
        cmin = np.min(np.append(array_c_T_Elb_4T, array_c_T_Elr_4T))
        cmax = np.max(np.append(array_c_T_Elb_4T, array_c_T_Elr_4T))
        climit_elec = np.linspace(cmin, cmax, temp_levels)
        
        ax3 = axes[2]
        ax3.set_title('Long Layer (Cathode)', fontsize=13, fontweight='bold')
        ax3.contourf(array_h_Elb_4T * spiral_scale, array_v_Elb_4T, 
                    array_c_T_Elb_4T, climit_elec, cmap="RdBu_r")
        ax3.set_xlabel('Unrolled Distance (m)', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Axial Position (m)', fontsize=11, fontweight='bold')
        
        ax4 = axes[3]
        ax4.set_title('Short Layer (Anode)', fontsize=13, fontweight='bold')
        ax4.contourf(array_h_Elr_4T * spiral_scale, array_v_Elr_4T, 
                    array_c_T_Elr_4T, climit_elec, cmap="RdBu_r")
        ax4.set_xlabel('Unrolled Distance (m)', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Axial Position (m)', fontsize=11, fontweight='bold')
    
    cell_name = cell_obj.status_Cells_name[0] if hasattr(cell_obj, 'status_Cells_name') else "Cell"
    fig.suptitle(f'Unrolled Jellyroll Temperature - {cell_name} (t={time_val:.1f}s)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig, axes


def run_simulation_and_visualize(config_path=None):
    """
    Run PyECN simulation and generate all temperature visualizations.
    
    Parameters:
    -----------
    config_path : str or Path, optional
        Path to TOML config file. If None, uses cylindrical_tabless_Fig_3.toml
    """
    print("="*70)
    print("PyECN Temperature Visualization")
    print("="*70)
    
    # Set up paths
    if config_path is None:
        config_path = Path(__file__).parent.parent / "Examples" / "cylindrical_tabless_Fig_3.toml"
    config_path = Path(config_path)
    
    print(f"\nConfig file: {config_path}")
    
    # Read TOML config directly
    with open(config_path, 'r') as f:
        config = toml.load(f)
    
    # Change to PROJECT ROOT (parent of pyecn) so relative paths work
    # parse_inputs expects paths like "pyecn\Input_LUTs\..." to be relative to project root
    pyecn_root = Path(__file__).parent.parent
    project_root = pyecn_root.parent
    import os
    original_dir = os.getcwd()
    original_argv = sys.argv.copy()
    os.chdir(project_root)  # Changed from pyecn_root to project_root
    
    # Add to path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # CRITICAL: Set sys.argv[1] BEFORE importing Core
    # Core imports parse_inputs, which reads sys.argv[1] during module initialization
    if len(sys.argv) == 1:
        sys.argv.append(str(config_path))
    else:
        sys.argv[1] = str(config_path)
    
    try:
        # Import PyECN run function and parse_inputs
        from pyecn import run
        import pyecn.parse_inputs as ip
        
        # Disable matplotlib figures but keep Mayavi visualization
        ip.status_fig1to9 = 'No'
        ip.status_PopFig_or_SaveGIF_instant = 'No'
        ip.status_PopFig_or_SaveGIF_replay = 'No'
        
        print("\n" + "-"*70)
        print("Running PyECN Simulation...")
        print("-"*70)
        
        # Run the full PyECN simulation
        run()
        
        # Get the cell object that was created during run()
        # PyECN stores cells as global variables with the name from the config
        cell_name = ip.status_Cells_name[0]
        # Import the cell from the pyecn module's globals
        from pyecn import __dict__ as pyecn_globals
        if cell_name not in pyecn_globals:
            raise RuntimeError(f"Cell '{cell_name}' was not created during simulation")
        cell = pyecn_globals[cell_name]
        
        # Extract config values from parse_inputs (which already read the TOML)
        cell_name = ip.status_Cells_name[0]
        c_rate = ip.C_rate
        form_factor = ip.status_FormFactor
        model = ip.status_Model
        coupling = ip.status_EandT_coupling
        T_cooling = ip.T_cooling
        T_initial = ip.T_initial
        temp_min_limit = ip.min_temp_limit
        temp_max_limit = ip.max_temp_limit
        temp_levels = ip.status_levels
        
        print(f"\n✓ Simulation complete!")
        print(f"  Cell: {cell_name}")
        print(f"  Form factor: {form_factor}")
        print(f"  Discretization: {cell.nx}×{cell.ny}×{cell.nstack}")
        print(f"  Time steps: {cell.nt}")
        if cell.nt > 0:
            print(f"  Final time: {cell.t_record[cell.nt]:.1f}s")
        else:
            print(f"  WARNING: No time steps recorded (nt={cell.nt})")
        
        # Debug: Check if we have data
        print(f"\n  Data check:")
        print(f"    t_record length: {len(cell.t_record)}")
        print(f"    T_avg_record length: {len(cell.T_avg_record)}")
        print(f"    nt value: {cell.nt}")
        
        # Extract data
        time = cell.t_record[:cell.nt + 1]
        temp_avg = cell.T_avg_record[:cell.nt + 1]
        temp_std = cell.T_SD_record[:cell.nt + 1]
        temp_delta = cell.T_Delta_record[:cell.nt + 1]
        temp_min = temp_avg - temp_delta / 2
        temp_max = temp_avg + temp_delta / 2
        
        print(f"    time array length: {len(time)}")
        print(f"    temp_avg array length: {len(temp_avg)}")
        if len(time) > 0:
            print(f"    time range: {time[0]:.2f} to {time[-1]:.2f}")
            if np.all(np.isnan(temp_avg)):
                print(f"    ⚠ WARNING: All temperature values are NaN - simulation may not have run properly")
            else:
                print(f"    temp_avg range: {np.nanmin(temp_avg):.2f} to {np.nanmax(temp_avg):.2f} K")

        
        print("\n" + "-"*70)
        print("Generating Visualizations...")
        print("-"*70)
        
        # Generate plots (no saving or showing)
        print("\n1. Average Temperature vs Time")
        plot_temp_avg_vs_time(time, temp_avg, cell_name, c_rate, T_cooling, T_initial,
                             save_path="temp_avg.png")
        print("  ✓ Saved: temp_avg.png")
        
        print("\n2. Min/Max Temperature vs Time")
        plot_temp_minmax_vs_time(time, temp_min, temp_max, cell_name, c_rate,
                                save_path="temp_minmax.png")
        print("  ✓ Saved: temp_minmax.png")
        
        print("\n3. Temperature Delta vs Time")
        plot_temp_delta_vs_time(time, temp_delta, cell_name, c_rate,
                               save_path="temp_delta.png")
        print("  ✓ Saved: temp_delta.png")
        
        print("\n4. Temperature Std Dev vs Time")
        plot_temp_std_vs_time(time, temp_std, cell_name, c_rate,
                             save_path="temp_std.png")
        print("  ✓ Saved: temp_std.png")
        
        print("\n5. Combined Temperature Analysis")
        plot_all_temperature(time, temp_avg, temp_min, temp_max, temp_delta, temp_std,
                            cell_name, form_factor, c_rate, model, coupling,
                            T_cooling, T_initial, save_path="temp_all.png")
        print("  ✓ Saved: temp_all.png")
        
        print("\n6. Unrolled Jellyroll Temperature Distribution")
        plot_unrolled_jellyroll(cell, temp_min_limit, temp_max_limit, temp_levels,
                               time_step=-1, save_path="temp_unrolled_jellyroll.png")
        print("  ✓ Saved: temp_unrolled_jellyroll.png")
        
        print("\n" + "="*70)
        print("✓ All visualizations complete!")
        print("="*70)
        print("\nGenerated files:")
        print("  - temp_avg.png")
        print("  - temp_minmax.png")
        print("  - temp_delta.png")
        print("  - temp_std.png")
        print("  - temp_all.png")
        if form_factor == 'Cylindrical':
            print("  - temp_unrolled_jellyroll.png")
        print("="*70)
        
        return cell
        
    finally:
        # Restore original directory and sys.argv
        os.chdir(original_dir)
        sys.argv[:] = original_argv


if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        run_simulation_and_visualize(config_path)
    else:
        print("Usage: python viz_temperature.py [config_file.toml]")
        print("Running with default config...\n")
        run_simulation_and_visualize()
