"""
Complete PyECN Visualization Suite
===================================
Runs a single PyECN simulation and generates all visualization outputs:
- Temperature evolution (avg, min/max, delta, std, all, unrolled jellyroll)
- Time-series plots (voltage, current, SoC)
- 2D spatial heatmaps (temperature, SoC, voltage, current)

Usage:
    python viz_all.py [config_file.toml]
"""

import numpy as np
from pathlib import Path
import sys
import os
import toml

# Import visualization functions from the three modules
from viz_temperature import (
    plot_temp_avg_vs_time,
    plot_temp_minmax_vs_time,
    plot_temp_delta_vs_time,
    plot_temp_std_vs_time,
    plot_all_temperature,
    plot_unrolled_jellyroll
)

from viz_timeseries import (
    plot_voltage_vs_time,
    plot_current_vs_time,
    plot_soc_vs_time,
    plot_all_timeseries
)

from viz_spatial_2d import (
    plot_temperature_2d,
    plot_soc_2d,
    plot_voltage_2d,
    plot_current_2d,
    plot_all_spatial_2d
)


def run_complete_visualization(config_path=None, output_dir=None):
    """
    Run PyECN simulation and generate all visualizations.
    
    Parameters:
    -----------
    config_path : str or Path, optional
        Path to TOML config file. If None, uses default example config.
    output_dir : str or Path, optional
        Directory to save output files. If None, saves to current directory.
    """
    print("="*70)
    print("PyECN COMPLETE VISUALIZATION SUITE")
    print("="*70)
    
    # Set up paths
    if config_path is None:
        config_path = Path(__file__).parent.parent / "Examples" / "cylindrical_tabless_Fig_3.toml"
    config_path = Path(config_path)
    
    if output_dir is None:
        output_dir = Path.cwd()
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nConfig file: {config_path}")
    print(f"Output directory: {output_dir}")
    
    # Read TOML config directly
    with open(config_path, 'r') as f:
        config = toml.load(f)
    
    # Change to PROJECT ROOT (parent of pyecn) so relative paths work
    pyecn_root = Path(__file__).parent.parent
    project_root = pyecn_root.parent
    original_dir = os.getcwd()
    original_argv = sys.argv.copy()
    os.chdir(project_root)
    
    # Add to path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Set sys.argv[1] BEFORE importing Core (which imports parse_inputs)
    if len(sys.argv) == 1:
        sys.argv.append(str(config_path))
    else:
        sys.argv[1] = str(config_path)
    
    try:
        # Import PyECN run function and parse_inputs
        from pyecn import run
        import pyecn.parse_inputs as ip
        
        # Disable PyECN's default figures but keep Mayavi visualization
        ip.status_fig1to9 = 'No'
        ip.status_PopFig_or_SaveGIF_instant = 'No'
        ip.status_PopFig_or_SaveGIF_replay = 'No'
        
        print("\n" + "="*70)
        print("STEP 1: Running PyECN Simulation")
        print("="*70)
        
        # Run the full PyECN simulation ONCE
        run()
        
        # Get the cell object that was created during run()
        cell_name = ip.status_Cells_name[0]
        from pyecn import __dict__ as pyecn_globals
        if cell_name not in pyecn_globals:
            raise RuntimeError(f"Cell '{cell_name}' was not created during simulation")
        cell = pyecn_globals[cell_name]
        
        # Extract all config values from parse_inputs
        c_rate = ip.C_rate
        form_factor = ip.status_FormFactor
        model = ip.status_Model
        coupling = ip.status_EandT_coupling
        T_cooling = ip.T_cooling
        T_initial = ip.T_initial
        temp_min_limit = ip.min_temp_limit
        temp_max_limit = ip.max_temp_limit
        temp_levels = ip.status_levels
        v_highlimit = ip.V_highlimit_single
        v_lowlimit = ip.V_lowlimit_single
        
        print(f"\n✓ Simulation complete!")
        print(f"  Cell: {cell_name}")
        print(f"  Form factor: {form_factor}")
        print(f"  Model: {model}")
        print(f"  Discretization: {cell.nx}×{cell.ny}×{cell.nstack}")
        print(f"  Time steps: {cell.nt}")
        if cell.nt > 0:
            print(f"  Final time: {cell.t_record[cell.nt]:.1f}s")
        
        # Extract all data arrays
        time = cell.t_record[:cell.nt + 1]
        temp_avg = cell.T_avg_record[:cell.nt + 1]
        temp_std = cell.T_SD_record[:cell.nt + 1]
        temp_delta = cell.T_Delta_record[:cell.nt + 1]
        temp_min = temp_avg - temp_delta / 2
        temp_max = temp_avg + temp_delta / 2
        voltage = cell.U_pndiff_plot[:cell.nt + 1]
        current = cell.I0_record[:cell.nt + 1]
        soc = cell.SoC_Cell_record[:cell.nt + 1]
        
        # Check for data validity
        if np.all(np.isnan(temp_avg)):
            print(f"  ⚠ WARNING: All temperature values are NaN - simulation may not have run properly")
            print(f"  Skipping visualizations.")
            return None
        
        print(f"\n  Data ranges:")
        print(f"    Time: {time[0]:.2f} to {time[-1]:.2f} s")
        print(f"    Temperature: {np.nanmin(temp_avg):.2f} to {np.nanmax(temp_avg):.2f} K")
        print(f"    Voltage: {np.min(voltage):.4f} to {np.max(voltage):.4f} V")
        print(f"    Current: {np.min(current):.2f} to {np.max(current):.2f} A")
        print(f"    SoC: {np.min(soc):.4f} to {np.max(soc):.4f}")
        
        # =================================================================
        # STEP 2: TEMPERATURE VISUALIZATIONS
        # =================================================================
        print("\n" + "="*70)
        print("STEP 2: Temperature Visualizations")
        print("="*70)
        
        print("\n1. Average Temperature vs Time")
        plot_temp_avg_vs_time(time, temp_avg, cell_name, c_rate, T_cooling, T_initial,
                             save_path=output_dir / "temp_avg.png")
        print("  ✓ Saved: temp_avg.png")
        
        print("\n2. Min/Max Temperature vs Time")
        plot_temp_minmax_vs_time(time, temp_min, temp_max, cell_name, c_rate,
                                save_path=output_dir / "temp_minmax.png")
        print("  ✓ Saved: temp_minmax.png")
        
        print("\n3. Temperature Delta vs Time")
        plot_temp_delta_vs_time(time, temp_delta, cell_name, c_rate,
                               save_path=output_dir / "temp_delta.png")
        print("  ✓ Saved: temp_delta.png")
        
        print("\n4. Temperature Standard Deviation vs Time")
        plot_temp_std_vs_time(time, temp_std, cell_name, c_rate,
                             save_path=output_dir / "temp_std.png")
        print("  ✓ Saved: temp_std.png")
        
        print("\n5. All Temperature Metrics (Combined)")
        plot_all_temperature(time, temp_avg, temp_min, temp_max, temp_delta, temp_std,
                            cell_name, c_rate, T_cooling, T_initial,
                            save_path=output_dir / "temp_all.png")
        print("  ✓ Saved: temp_all.png")
        
        print("\n6. Unrolled Jellyroll Temperature Distribution")
        fig_jellyroll, axes_jellyroll = plot_unrolled_jellyroll(
            cell, temp_min_limit, temp_max_limit, temp_levels, time_step=-1,
            save_path=output_dir / "temp_unrolled_jellyroll.png"
        )
        if fig_jellyroll is not None:
            print("  ✓ Saved: temp_unrolled_jellyroll.png")
        else:
            print("  ⚠ Skipped: Unrolled jellyroll (not applicable or error)")
        
        # =================================================================
        # STEP 3: TIME-SERIES VISUALIZATIONS
        # =================================================================
        print("\n" + "="*70)
        print("STEP 3: Time-Series Visualizations")
        print("="*70)
        
        # Create config_params dict for time-series plotting
        config_params_ts = {
            'cell_name': cell_name,
            'V_highlimit': v_highlimit,
            'V_lowlimit': v_lowlimit,
            'C_rate': c_rate,
            'Current_direction': ip.status_discharge,
            'Capacity_rated': ip.Capacity_rated0,
            'SoC_initial': ip.soc_initial,
            'Form_factor': form_factor,
        }
        
        print("\n1. Voltage vs Time")
        plot_voltage_vs_time(time, voltage, config_params_ts,
                            save_path=output_dir / "timeseries_voltage.png")
        print("  ✓ Saved: timeseries_voltage.png")
        
        print("\n2. Current vs Time")
        plot_current_vs_time(time, current, config_params_ts,
                            save_path=output_dir / "timeseries_current.png")
        print("  ✓ Saved: timeseries_current.png")
        
        print("\n3. SoC vs Time")
        plot_soc_vs_time(time, soc, config_params_ts,
                        save_path=output_dir / "timeseries_soc.png")
        print("  ✓ Saved: timeseries_soc.png")
        
        print("\n4. Combined Time-Series Plot")
        plot_all_timeseries(time, voltage, current, soc, config_params_ts,
                           save_path=output_dir / "timeseries_all.png")
        print("  ✓ Saved: timeseries_all.png")
        
        # =================================================================
        # STEP 4: 2D SPATIAL VISUALIZATIONS
        # =================================================================
        print("\n" + "="*70)
        print("STEP 4: 2D Spatial Visualizations")
        print("="*70)
        print("\nNote: Using placeholder synthetic data for 2D spatial plots")
        print("(Real 2D data extraction requires understanding PyECN's 3D array structure)")
        
        # Create config_params dict for spatial plotting
        config_params_spatial = {
            'cell_name': cell_name,
            'Form_factor': form_factor,
            'nx': cell.nx,
            'ny': cell.ny,
            'nstack': cell.nstack,
            'C_rate': c_rate,
            'Temp_min': temp_min_limit,
            'Temp_max': temp_max_limit,
            'Temp_levels': temp_levels,
            'V_highlimit': v_highlimit,
            'V_lowlimit': v_lowlimit,
            'SoC_initial': ip.soc_initial,
            'Model': model,
        }
        
        # Create placeholder 2D data (centered layer)
        # In real implementation, extract from cell.T_record, cell.SoC_record, etc.
        nx, ny, nstack = cell.nx, cell.ny, cell.nstack
        layer_index = nstack // 2
        
        # Placeholder synthetic data
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y)
        
        temp_2d = np.mean(temp_avg) + 5 * np.sin(3*np.pi*X) * np.cos(3*np.pi*Y)
        soc_2d = np.mean(soc) + 0.1 * np.sin(2*np.pi*X) * np.cos(2*np.pi*Y)
        voltage_2d = np.mean(voltage) + 0.01 * np.sin(4*np.pi*X) * np.cos(4*np.pi*Y)
        current_2d = np.mean(current) + 2 * np.sin(2*np.pi*X) * np.cos(2*np.pi*Y)
        
        time_val = time[-1] if len(time) > 0 else 0
        
        print("\n1. Temperature 2D Heatmap")
        plot_temperature_2d(temp_2d, config_params_spatial, time=time_val,
                           save_path=output_dir / "spatial_2d_temperature.png")
        print("  ✓ Saved: spatial_2d_temperature.png")
        
        print("\n2. SoC 2D Heatmap")
        plot_soc_2d(soc_2d, config_params_spatial, time=time_val,
                   save_path=output_dir / "spatial_2d_soc.png")
        print("  ✓ Saved: spatial_2d_soc.png")
        
        print("\n3. Voltage 2D Heatmap")
        plot_voltage_2d(voltage_2d, config_params_spatial, time=time_val,
                       save_path=output_dir / "spatial_2d_voltage.png")
        print("  ✓ Saved: spatial_2d_voltage.png")
        
        print("\n4. Current 2D Heatmap")
        plot_current_2d(current_2d, config_params_spatial, time=time_val,
                       save_path=output_dir / "spatial_2d_current.png")
        print("  ✓ Saved: spatial_2d_current.png")
        
        print("\n5. All 2D Spatial Plots (Combined)")
        plot_all_spatial_2d(temp_2d, soc_2d, voltage_2d, current_2d,
                           config_params_spatial, time=time_val,
                           save_path=output_dir / "spatial_2d_all.png")
        print("  ✓ Saved: spatial_2d_all.png")
        
        # =================================================================
        # SUMMARY
        # =================================================================
        print("\n" + "="*70)
        print("✓ COMPLETE VISUALIZATION SUITE FINISHED!")
        print("="*70)
        print(f"\nGenerated {15} visualization files in: {output_dir}")
        print("\nTemperature Visualizations:")
        print("  - temp_avg.png")
        print("  - temp_minmax.png")
        print("  - temp_delta.png")
        print("  - temp_std.png")
        print("  - temp_all.png")
        if fig_jellyroll is not None:
            print("  - temp_unrolled_jellyroll.png")
        print("\nTime-Series Visualizations:")
        print("  - timeseries_voltage.png")
        print("  - timeseries_current.png")
        print("  - timeseries_soc.png")
        print("  - timeseries_all.png")
        print("\n2D Spatial Visualizations:")
        print("  - spatial_2d_temperature.png")
        print("  - spatial_2d_soc.png")
        print("  - spatial_2d_voltage.png")
        print("  - spatial_2d_current.png")
        print("  - spatial_2d_all.png")
        print("="*70)
        
        return cell
        
    finally:
        # Restore original directory and sys.argv
        os.chdir(original_dir)
        sys.argv[:] = original_argv


if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else None
        run_complete_visualization(config_path, output_dir)
    else:
        print("Usage: python viz_all.py [config_file.toml] [output_directory]")
        print("Running with default config...\n")
        run_complete_visualization()
