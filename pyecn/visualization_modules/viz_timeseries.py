"""
Time-Series Visualization Module for PyECN
==========================================
Handles voltage, current, and SoC (State of Charge) plots over time.
All parameters are read from the TOML config file - no hardcoded values.

Functions:
    - plot_voltage_vs_time: Cell terminal voltage evolution
    - plot_current_vs_time: Current profile during operation
    - plot_soc_vs_time: State of charge depletion/charging
    - plot_all_timeseries: Combined view of all three
    - load_config: Load TOML configuration file
    - extract_config_params: Extract visualization parameters from config
    - extract_from_pyecn_cell: Extract data from PyECN cell object
    - test_module: Standalone test with config file
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import sys
import toml 


def plot_voltage_vs_time(time, voltage, config_params, 
                         save_path=None, show=False, ax=None):
    """
    Plot cell terminal voltage vs time.
    
    Parameters:
    -----------
    time : array-like
        Time points (seconds or hours)
    voltage : array-like
        Terminal voltage (V)
    config_params : dict
        Configuration parameters from TOML file
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display the plot
    ax : matplotlib axis, optional
        Axis to plot on (for subplots)
        
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()
    
    # Extract parameters from config
    cell_name = config_params['cell_name']
    v_high = config_params['V_highlimit']
    v_low = config_params['V_lowlimit']
    
    # Convert time to hours if in seconds
    time_hours = time / 3600 if np.max(time) > 100 else time
    time_label = "Time (h)" if np.max(time) > 100 else "Time (s)"
    
    ax.plot(time_hours, voltage, 'b-', linewidth=2, label='Terminal Voltage')
    ax.set_xlabel(time_label, fontsize=12, fontweight='bold')
    ax.set_ylabel('Voltage (V)', fontsize=12, fontweight='bold')
    ax.set_title(f'Cell Voltage vs Time - {cell_name}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add voltage limits from config as horizontal lines
    ax.axhline(y=v_high, color='g', linestyle='--', alpha=0.5, linewidth=1, 
               label=f'V_high = {v_high}V')
    ax.axhline(y=v_low, color='r', linestyle='--', alpha=0.5, linewidth=1,
               label=f'V_low = {v_low}V')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Voltage plot saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig, ax


def plot_current_vs_time(time, current, config_params, 
                         save_path=None, show=False, ax=None):
    """
    Plot cell current vs time.
    
    Parameters:
    -----------
    time : array-like
        Time points (seconds or hours)
    current : array-like
        Current (A), positive for discharge, negative for charge
    config_params : dict
        Configuration parameters from TOML file
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display the plot
    ax : matplotlib axis, optional
        Axis to plot on (for subplots)
        
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()
    
    # Extract parameters from config
    cell_name = config_params['cell_name']
    current_direction = config_params['Current_direction']
    c_rate = config_params['C_rate']
    capacity = config_params['Capacity_rated']
    
    # Convert time to hours if in seconds
    time_hours = time / 3600 if np.max(time) > 100 else time
    time_label = "Time (h)" if np.max(time) > 100 else "Time (s)"
    
    # Color based on current direction from config
    color = 'r' if current_direction == 1 else 'b'
    label = f'{c_rate}C Discharge' if current_direction == 1 else f'{c_rate}C Charge'
    
    ax.plot(time_hours, current, color=color, linewidth=2, label=label)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=1)
    ax.set_xlabel(time_label, fontsize=12, fontweight='bold')
    ax.set_ylabel('Current (A)', fontsize=12, fontweight='bold')
    ax.set_title(f'Cell Current vs Time - {cell_name}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Current plot saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig, ax


def plot_soc_vs_time(time, soc, config_params, 
                     save_path=None, show=False, ax=None):
    """
    Plot State of Charge (SoC) vs time.
    
    Parameters:
    -----------
    time : array-like
        Time points (seconds or hours)
    soc : array-like
        State of charge (0-1 or 0-100%)
    config_params : dict
        Configuration parameters from TOML file
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display the plot
    ax : matplotlib axis, optional
        Axis to plot on (for subplots)
        
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()
    
    # Extract parameters from config
    cell_name = config_params['cell_name']
    soc_initial = config_params['SoC_initial']
    
    # Convert time to hours if in seconds
    time_hours = time / 3600 if np.max(time) > 100 else time
    time_label = "Time (h)" if np.max(time) > 100 else "Time (s)"
    
    # Convert SoC to percentage if needed
    soc_percent = soc * 100 if np.max(soc) <= 1 else soc
    
    # Color gradient based on SoC level
    ax.plot(time_hours, soc_percent, 'g-', linewidth=2, label='State of Charge')
    ax.fill_between(time_hours, 0, soc_percent, alpha=0.2, color='g')
    
    ax.set_xlabel(time_label, fontsize=12, fontweight='bold')
    ax.set_ylabel('SoC (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'State of Charge vs Time - {cell_name}', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add reference lines at 20%, 50%, 80% and initial SoC
    for level in [20, 50, 80]:
        ax.axhline(y=level, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    # Mark initial SoC
    ax.axhline(y=soc_initial * 100, color='k', linestyle='--', alpha=0.3, linewidth=1,
               label=f'Initial SoC = {soc_initial*100:.0f}%')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"SoC plot saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig, ax


def plot_all_timeseries(time, voltage, current, soc, config_params,
                        save_path=None, show=False):
    """
    Create a combined plot with all three time-series in subplots.
    
    Parameters:
    -----------
    time : array-like
        Time points (seconds or hours)
    voltage : array-like
        Terminal voltage (V)
    current : array-like
        Current (A)
    soc : array-like
        State of charge (0-1 or 0-100%)
    config_params : dict
        Configuration parameters from TOML file
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display the plot
        
    Returns:
    --------
    fig : matplotlib figure object
    axes : array of matplotlib axis objects
    """
    cell_name = config_params['cell_name']
    form_factor = config_params['Form_factor']
    c_rate = config_params['C_rate']
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 1, figure=fig, hspace=0.3)
    
    # Create subplots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])
    
    # Plot each time series
    plot_voltage_vs_time(time, voltage, config_params, show=False, ax=ax1)
    plot_current_vs_time(time, current, config_params, show=False, ax=ax2)
    plot_soc_vs_time(time, soc, config_params, show=False, ax=ax3)
    
    # Add overall title with form factor and C-rate info
    fig.suptitle(f'Time-Series Analysis - {cell_name} ({form_factor}, {c_rate}C)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Combined time-series plot saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig, [ax1, ax2, ax3]


def extract_from_pyecn_cell(cell_obj):
    """
    Extract time-series data from PyECN cell object.
    
    Parameters:
    -----------
    cell_obj : PyECN Core object
        Cell object from PyECN simulation
        
    Returns:
    --------
    dict : Dictionary with time, voltage, current, soc arrays
    """
    data = {}
    
    # Time array
    data['time'] = cell_obj.t_record[:cell_obj.nt + 1]
    
    # Voltage (positive terminal - negative terminal)
    data['voltage'] = cell_obj.U_pndiff_plot[:cell_obj.nt + 1]
    
    # Current (stored in I0_record)
    data['current'] = cell_obj.I0_record[:cell_obj.nt + 1]
    
    # SoC (State of Charge)
    data['soc'] = cell_obj.SoC[:cell_obj.nt + 1]
    
    # Cell name
    data['cell_name'] = cell_obj.status_Cells_name[0] if hasattr(cell_obj, 'status_Cells_name') else "cell_1"
    
    return data


def test_module(config_path=None):
    """
    Run PyECN simulation and generate time-series visualizations.
    
    Parameters:
    -----------
    config_path : str or Path, optional
        Path to TOML config file. If None, uses default example config.
    """
    print("="*70)
    print("PyECN Time-Series Visualization")
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
    pyecn_root = Path(__file__).parent.parent
    project_root = pyecn_root.parent
    import os
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
        cell_name = ip.status_Cells_name[0]
        from pyecn import __dict__ as pyecn_globals
        if cell_name not in pyecn_globals:
            raise RuntimeError(f"Cell '{cell_name}' was not created during simulation")
        cell = pyecn_globals[cell_name]
        
        # Extract config values from parse_inputs
        c_rate = ip.C_rate
        v_highlimit = ip.V_highlimit_single
        v_lowlimit = ip.V_lowlimit_single
        form_factor = ip.status_FormFactor
        
        print(f"\n✓ Simulation complete!")
        print(f"  Cell: {cell_name}")
        print(f"  Time steps: {cell.nt}")
        if cell.nt > 0:
            print(f"  Final time: {cell.t_record[cell.nt]:.1f}s")
        
        # Extract data
        time = cell.t_record[:cell.nt + 1]
        voltage = cell.U_pndiff_plot[:cell.nt + 1]
        current = cell.I0_record[:cell.nt + 1]
        soc = cell.SoC_Cell_record[:cell.nt + 1]
        
        # Create config_params dict for plotting functions
        config_params = {
            'cell_name': cell_name,
            'V_highlimit': v_highlimit,
            'V_lowlimit': v_lowlimit,
            'C_rate': c_rate,
            'Current_direction': ip.status_discharge,
            'Capacity_rated': ip.Capacity_rated0,
            'SoC_initial': ip.soc_initial,
            'Form_factor': form_factor,
        }
        
        print("\n" + "-"*70)
        print("Generating Visualizations...")
        print("-"*70)
        
        print("\n1. Voltage vs Time")
        plot_voltage_vs_time(time, voltage, config_params,
                            save_path="timeseries_voltage.png")
        print("  ✓ Saved: timeseries_voltage.png")
        
        print("\n2. Current vs Time")
        plot_current_vs_time(time, current, config_params,
                            save_path="timeseries_current.png")
        print("  ✓ Saved: timeseries_current.png")
        
        print("\n3. SoC vs Time")
        plot_soc_vs_time(time, soc, config_params,
                        save_path="timeseries_soc.png")
        print("  ✓ Saved: timeseries_soc.png")
        
        print("\n4. Combined Time-Series Plot")
        plot_all_timeseries(time, voltage, current, soc, config_params,
                           save_path="timeseries_all.png")
        print("  ✓ Saved: timeseries_all.png")
        
        print("\n" + "="*70)
        print("✓ All visualizations complete!")
        print("="*70)
        print("\nGenerated files:")
        print("  - timeseries_voltage.png")
        print("  - timeseries_current.png")
        print("  - timeseries_soc.png")
        print("  - timeseries_all.png")
        print("="*70)
        
        return cell
        
    finally:
        # Restore original directory and sys.argv
        os.chdir(original_dir)
        sys.argv[:] = original_argv


if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        test_module(config_path)
    else:
        print("Usage: python viz_timeseries.py [config_file.toml]")
        print("Running with default config...\n")
        test_module()
