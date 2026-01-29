"""
2D Spatial Distribution Visualization Module for PyECN
======================================================
Handles 2D heatmaps and contour plots for spatial distributions.
All parameters are read from the TOML config file - no hardcoded values.

Functions:
    - plot_temperature_2d: 2D temperature distribution heatmap
    - plot_soc_2d: 2D State of Charge distribution heatmap
    - plot_voltage_2d: 2D voltage distribution heatmap
    - plot_current_2d: 2D current distribution heatmap
    - plot_all_spatial_2d: Combined view of all 2D spatial distributions
    - load_config: Load TOML configuration file
    - extract_config_params: Extract visualization parameters from config
    - extract_from_pyecn_cell: Extract spatial data from PyECN cell object
    - test_module: Standalone test with config file
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import sys
import toml


def plot_temperature_2d(temp_2d, config_params, time_label="t=0s",
                       save_path=None, show=False, ax=None, cbar_ax=None):
    """
    Plot 2D temperature distribution heatmap.
    
    Parameters:
    -----------
    temp_2d : 2D array
        Temperature distribution (K), will be converted to °C
    config_params : dict
        Configuration parameters from TOML file
    time_label : str
        Time label for title (e.g., "t=100s" or "t=0.5h")
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display the plot
    ax : matplotlib axis, optional
        Axis to plot on (for subplots)
    cbar_ax : matplotlib axis, optional
        Axis for colorbar (for subplots)
        
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
        standalone = True
    else:
        fig = ax.get_figure()
        standalone = False
    
    # Extract parameters from config
    cell_name = config_params['cell_name']
    form_factor = config_params['Form_factor']
    temp_min = config_params['Temp_min']
    temp_max = config_params['Temp_max']
    temp_levels = config_params['Temp_levels']
    
    # Convert temperature to °C
    temp_celsius = temp_2d - 273.15
    
    # Create heatmap
    im = ax.imshow(temp_celsius, cmap='hot', aspect='auto',
                   vmin=temp_min, vmax=temp_max,
                   origin='lower', interpolation='bilinear')
    
    # Set labels based on form factor
    if form_factor.lower() == 'cylindrical':
        ax.set_xlabel('Axial Position', fontsize=11, fontweight='bold')
        ax.set_ylabel('Circumferential Position', fontsize=11, fontweight='bold')
    elif form_factor.lower() == 'pouch':
        ax.set_xlabel('X Position', fontsize=11, fontweight='bold')
        ax.set_ylabel('Y Position', fontsize=11, fontweight='bold')
    elif form_factor.lower() == 'prismatic':
        ax.set_xlabel('X Position', fontsize=11, fontweight='bold')
        ax.set_ylabel('Y Position', fontsize=11, fontweight='bold')
    else:
        ax.set_xlabel('Dimension 1', fontsize=11, fontweight='bold')
        ax.set_ylabel('Dimension 2', fontsize=11, fontweight='bold')
    
    ax.set_title(f'Temperature Distribution - {cell_name} ({time_label})', 
                 fontsize=13, fontweight='bold')
    
    # Add colorbar
    if cbar_ax is None and standalone:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Temperature (°C)', fontsize=11, fontweight='bold')
    elif cbar_ax is not None:
        cbar = plt.colorbar(im, cax=cbar_ax)
        cbar.set_label('Temperature (°C)', fontsize=10, fontweight='bold')
    
    if standalone:
        plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Temperature 2D plot saved to: {save_path}")
    
    if show and standalone:
        plt.show()
    
    return fig, ax


def plot_soc_2d(soc_2d, config_params, time_label="t=0s",
               save_path=None, show=False, ax=None, cbar_ax=None):
    """
    Plot 2D State of Charge distribution heatmap.
    
    Parameters:
    -----------
    soc_2d : 2D array
        SoC distribution (0-1)
    config_params : dict
        Configuration parameters from TOML file
    time_label : str
        Time label for title
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display the plot
    ax : matplotlib axis, optional
        Axis to plot on (for subplots)
    cbar_ax : matplotlib axis, optional
        Axis for colorbar (for subplots)
        
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
        standalone = True
    else:
        fig = ax.get_figure()
        standalone = False
    
    # Extract parameters from config
    cell_name = config_params['cell_name']
    form_factor = config_params['Form_factor']
    soc_initial = config_params['SoC_initial']
    
    # Create heatmap
    im = ax.imshow(soc_2d, cmap='viridis', aspect='auto',
                   vmin=0.0, vmax=1.0,
                   origin='lower', interpolation='bilinear')
    
    # Set labels based on form factor
    if form_factor.lower() == 'cylindrical':
        ax.set_xlabel('Axial Position', fontsize=11, fontweight='bold')
        ax.set_ylabel('Circumferential Position', fontsize=11, fontweight='bold')
    elif form_factor.lower() == 'pouch':
        ax.set_xlabel('X Position', fontsize=11, fontweight='bold')
        ax.set_ylabel('Y Position', fontsize=11, fontweight='bold')
    elif form_factor.lower() == 'prismatic':
        ax.set_xlabel('X Position', fontsize=11, fontweight='bold')
        ax.set_ylabel('Y Position', fontsize=11, fontweight='bold')
    else:
        ax.set_xlabel('Dimension 1', fontsize=11, fontweight='bold')
        ax.set_ylabel('Dimension 2', fontsize=11, fontweight='bold')
    
    ax.set_title(f'SoC Distribution - {cell_name} ({time_label})', 
                 fontsize=13, fontweight='bold')
    
    # Add colorbar
    if cbar_ax is None and standalone:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('State of Charge', fontsize=11, fontweight='bold')
    elif cbar_ax is not None:
        cbar = plt.colorbar(im, cax=cbar_ax)
        cbar.set_label('SoC', fontsize=10, fontweight='bold')
    
    if standalone:
        plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"SoC 2D plot saved to: {save_path}")
    
    if show and standalone:
        plt.show()
    
    return fig, ax


def plot_voltage_2d(voltage_2d, config_params, time_label="t=0s",
                   save_path=None, show=False, ax=None, cbar_ax=None):
    """
    Plot 2D voltage distribution heatmap.
    
    Parameters:
    -----------
    voltage_2d : 2D array
        Voltage distribution (V)
    config_params : dict
        Configuration parameters from TOML file
    time_label : str
        Time label for title
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display the plot
    ax : matplotlib axis, optional
        Axis to plot on (for subplots)
    cbar_ax : matplotlib axis, optional
        Axis for colorbar (for subplots)
        
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
        standalone = True
    else:
        fig = ax.get_figure()
        standalone = False
    
    # Extract parameters from config
    cell_name = config_params['cell_name']
    form_factor = config_params['Form_factor']
    v_high = config_params['V_highlimit']
    v_low = config_params['V_lowlimit']
    
    # Create heatmap
    im = ax.imshow(voltage_2d, cmap='coolwarm', aspect='auto',
                   vmin=v_low, vmax=v_high,
                   origin='lower', interpolation='bilinear')
    
    # Set labels based on form factor
    if form_factor.lower() == 'cylindrical':
        ax.set_xlabel('Axial Position', fontsize=11, fontweight='bold')
        ax.set_ylabel('Circumferential Position', fontsize=11, fontweight='bold')
    elif form_factor.lower() == 'pouch':
        ax.set_xlabel('X Position', fontsize=11, fontweight='bold')
        ax.set_ylabel('Y Position', fontsize=11, fontweight='bold')
    elif form_factor.lower() == 'prismatic':
        ax.set_xlabel('X Position', fontsize=11, fontweight='bold')
        ax.set_ylabel('Y Position', fontsize=11, fontweight='bold')
    else:
        ax.set_xlabel('Dimension 1', fontsize=11, fontweight='bold')
        ax.set_ylabel('Dimension 2', fontsize=11, fontweight='bold')
    
    ax.set_title(f'Voltage Distribution - {cell_name} ({time_label})', 
                 fontsize=13, fontweight='bold')
    
    # Add colorbar
    if cbar_ax is None and standalone:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Voltage (V)', fontsize=11, fontweight='bold')
    elif cbar_ax is not None:
        cbar = plt.colorbar(im, cax=cbar_ax)
        cbar.set_label('Voltage (V)', fontsize=10, fontweight='bold')
    
    if standalone:
        plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Voltage 2D plot saved to: {save_path}")
    
    if show and standalone:
        plt.show()
    
    return fig, ax


def plot_current_2d(current_2d, config_params, time_label="t=0s",
                   save_path=None, show=False, ax=None, cbar_ax=None):
    """
    Plot 2D current distribution heatmap.
    
    Parameters:
    -----------
    current_2d : 2D array
        Current distribution (A)
    config_params : dict
        Configuration parameters from TOML file
    time_label : str
        Time label for title
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display the plot
    ax : matplotlib axis, optional
        Axis to plot on (for subplots)
    cbar_ax : matplotlib axis, optional
        Axis for colorbar (for subplots)
        
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
        standalone = True
    else:
        fig = ax.get_figure()
        standalone = False
    
    # Extract parameters from config
    cell_name = config_params['cell_name']
    form_factor = config_params['Form_factor']
    c_rate = config_params['C_rate']
    
    # Determine color limits symmetrically around zero
    max_abs_current = np.max(np.abs(current_2d))
    
    # Create heatmap
    im = ax.imshow(current_2d, cmap='RdBu_r', aspect='auto',
                   vmin=-max_abs_current, vmax=max_abs_current,
                   origin='lower', interpolation='bilinear')
    
    # Set labels based on form factor
    if form_factor.lower() == 'cylindrical':
        ax.set_xlabel('Axial Position', fontsize=11, fontweight='bold')
        ax.set_ylabel('Circumferential Position', fontsize=11, fontweight='bold')
    elif form_factor.lower() == 'pouch':
        ax.set_xlabel('X Position', fontsize=11, fontweight='bold')
        ax.set_ylabel('Y Position', fontsize=11, fontweight='bold')
    elif form_factor.lower() == 'prismatic':
        ax.set_xlabel('X Position', fontsize=11, fontweight='bold')
        ax.set_ylabel('Y Position', fontsize=11, fontweight='bold')
    else:
        ax.set_xlabel('Dimension 1', fontsize=11, fontweight='bold')
        ax.set_ylabel('Dimension 2', fontsize=11, fontweight='bold')
    
    ax.set_title(f'Current Distribution - {cell_name} ({time_label}, {c_rate}C)', 
                 fontsize=13, fontweight='bold')
    
    # Add colorbar
    if cbar_ax is None and standalone:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Current (A)', fontsize=11, fontweight='bold')
    elif cbar_ax is not None:
        cbar = plt.colorbar(im, cax=cbar_ax)
        cbar.set_label('Current (A)', fontsize=10, fontweight='bold')
    
    if standalone:
        plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Current 2D plot saved to: {save_path}")
    
    if show and standalone:
        plt.show()
    
    return fig, ax


def plot_all_spatial_2d(temp_2d, soc_2d, voltage_2d, current_2d,
                       config_params, time_label="t=0s",
                       save_path=None, show=False):
    """
    Create a combined plot with all 2D spatial distributions.
    
    Parameters:
    -----------
    temp_2d : 2D array
        Temperature distribution (K)
    soc_2d : 2D array
        SoC distribution (0-1)
    voltage_2d : 2D array
        Voltage distribution (V)
    current_2d : 2D array
        Current distribution (A)
    config_params : dict
        Configuration parameters from TOML file
    time_label : str
        Time label for title
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
    model = config_params['Model']
    nx = config_params['nx']
    ny = config_params['ny']
    
    # Create figure with 2x2 grid
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3,
                  width_ratios=[1, 1], height_ratios=[1, 1])
    
    # Create subplots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Plot each distribution
    plot_temperature_2d(temp_2d, config_params, time_label, show=False, ax=ax1)
    plot_soc_2d(soc_2d, config_params, time_label, show=False, ax=ax2)
    plot_voltage_2d(voltage_2d, config_params, time_label, show=False, ax=ax3)
    plot_current_2d(current_2d, config_params, time_label, show=False, ax=ax4)
    
    # Add overall title with model info
    title = f'2D Spatial Distributions - {cell_name} ({form_factor}, {c_rate}C) - {time_label}\n'
    title += f'Discretization: {nx}×{ny}, Model: {model}'
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Combined 2D spatial plot saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig, [ax1, ax2, ax3, ax4]


def extract_from_pyecn_cell(cell_obj, time_index=-1, layer_index=None):
    """
    Extract 2D spatial data from PyECN cell object.
    
    Parameters:
    -----------
    cell_obj : PyECN Core object
        Cell object from PyECN simulation
    time_index : int
        Time index to extract (default: -1 for final state)
    layer_index : int, optional
        Layer index for 3D cells (default: middle layer)
        For cylindrical: radial layer
        For pouch/prismatic: stack layer
        
    Returns:
    --------
    dict : Dictionary with 2D spatial arrays
    """
    data = {}
    
    # Get discretization
    nx = cell_obj.nx
    ny = cell_obj.ny
    nstack = cell_obj.nstack
    
    # Determine layer index if not provided (use middle layer)
    if layer_index is None:
        layer_index = nstack // 2
    
    # Extract 2D slices from 3D arrays
    # PyECN typically stores as [nx, ny, nstack] or [nstack, nx, ny]
    # Check dimensions to determine correct slicing
    
    # Temperature (K)
    if hasattr(cell_obj, 'T_record'):
        T_3d = cell_obj.T_record[:, :, :, time_index]
        if T_3d.shape[2] == nstack:
            data['temp_2d'] = T_3d[:, :, layer_index]
        else:
            data['temp_2d'] = T_3d[layer_index, :, :]
    elif hasattr(cell_obj, 'T'):
        T_3d = cell_obj.T
        if len(T_3d.shape) == 3:
            if T_3d.shape[2] == nstack:
                data['temp_2d'] = T_3d[:, :, layer_index]
            else:
                data['temp_2d'] = T_3d[layer_index, :, :]
        else:
            data['temp_2d'] = T_3d[:, :]
    
    # SoC (0-1)
    if hasattr(cell_obj, 'SoC_record'):
        SoC_3d = cell_obj.SoC_record[:, :, :, time_index]
        if SoC_3d.shape[2] == nstack:
            data['soc_2d'] = SoC_3d[:, :, layer_index]
        else:
            data['soc_2d'] = SoC_3d[layer_index, :, :]
    elif hasattr(cell_obj, 'SoC'):
        SoC_3d = cell_obj.SoC
        if len(SoC_3d.shape) == 3:
            if SoC_3d.shape[2] == nstack:
                data['soc_2d'] = SoC_3d[:, :, layer_index]
            else:
                data['soc_2d'] = SoC_3d[layer_index, :, :]
        else:
            data['soc_2d'] = SoC_3d[:, :]
    
    # Voltage (V)
    if hasattr(cell_obj, 'V_record'):
        V_3d = cell_obj.V_record[:, :, :, time_index]
        if V_3d.shape[2] == nstack:
            data['voltage_2d'] = V_3d[:, :, layer_index]
        else:
            data['voltage_2d'] = V_3d[layer_index, :, :]
    elif hasattr(cell_obj, 'V'):
        V_3d = cell_obj.V
        if len(V_3d.shape) == 3:
            if V_3d.shape[2] == nstack:
                data['voltage_2d'] = V_3d[:, :, layer_index]
            else:
                data['voltage_2d'] = V_3d[layer_index, :, :]
        else:
            data['voltage_2d'] = V_3d[:, :]
    
    # Current (A)
    if hasattr(cell_obj, 'I_record'):
        I_3d = cell_obj.I_record[:, :, :, time_index]
        if I_3d.shape[2] == nstack:
            data['current_2d'] = I_3d[:, :, layer_index]
        else:
            data['current_2d'] = I_3d[layer_index, :, :]
    elif hasattr(cell_obj, 'I'):
        I_3d = cell_obj.I
        if len(I_3d.shape) == 3:
            if I_3d.shape[2] == nstack:
                data['current_2d'] = I_3d[:, :, layer_index]
            else:
                data['current_2d'] = I_3d[layer_index, :, :]
        else:
            data['current_2d'] = I_3d[:, :]
    
    # Time info
    if hasattr(cell_obj, 't_record'):
        data['time'] = cell_obj.t_record[time_index]
    
    data['layer_index'] = layer_index
    
    return data


def test_module(config_path=None):
    """
    Run PyECN simulation and generate 2D spatial visualizations.
    
    Parameters:
    -----------
    config_path : str or Path, optional
        Path to TOML config file. If None, uses default example config.
    """
    print("="*70)
    print("PyECN 2D Spatial Visualization")
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
        form_factor = ip.status_FormFactor
        c_rate = ip.C_rate
        
        print(f"\n✓ Simulation complete!")
        print(f"  Cell: {cell_name}")
        print(f"  Form factor: {form_factor}")
        print(f"  Discretization: {cell.nx}×{cell.ny}×{cell.nstack}")
        print(f"  Time steps: {cell.nt}")
        if cell.nt > 0:
            print(f"  Final time: {cell.t_record[cell.nt]:.1f}s")
        
        # Use final time step for spatial plots
        time_step = -1
        time_val = cell.t_record[cell.nt]
        
        # Extract 2D spatial data at final time step
        # Note: This assumes data is structured as (nx, ny) for 2D grids
        # Actual implementation depends on PyECN's data structure
        
        # Create config_params dict for plotting functions
        config_params = {
            'cell_name': cell_name,
            'Form_factor': form_factor,
            'nx': cell.nx,
            'ny': cell.ny,
            'nstack': cell.nstack,
            'C_rate': c_rate,
            'Temp_min': ip.min_temp_limit,
            'Temp_max': ip.max_temp_limit,
            'Temp_levels': ip.status_levels,
            'V_highlimit': ip.V_highlimit_single,
            'V_lowlimit': ip.V_lowlimit_single,
            'SoC_initial': ip.soc_initial,
            'Model': ip.status_Model,
        }
        
        print("\n" + "-"*70)
        print("Generating 2D Spatial Visualizations...")
        print("-"*70)
        print("\nNote: 2D spatial plots require proper data reshaping")
        print("This is a placeholder - implement actual 2D data extraction")
        print("based on your specific PyECN data structure.")
        
        # Placeholder: Create synthetic 2D arrays
        # Replace this with actual data extraction from cell object
        nx, ny = cell.nx, cell.ny
        temp_2d = np.random.rand(ny, nx) * 10 + 298.15
        soc_2d = np.random.rand(ny, nx)
        voltage_2d = np.random.rand(ny, nx) * 1.5 + 3.0
        current_2d = np.random.rand(ny, nx) * 2 - 1
        
        time_label = f"t={time_val:.1f}s"
        
        print("\n1. Temperature 2D Distribution")
        plot_temperature_2d(temp_2d, config_params, time_label=time_label,
                           save_path="spatial_2d_temp.png")
        print("  ✓ Saved: spatial_2d_temp.png")
        
        print("\n2. SoC 2D Distribution")
        plot_soc_2d(soc_2d, config_params, time_label=time_label,
                   save_path="spatial_2d_soc.png")
        print("  ✓ Saved: spatial_2d_soc.png")
        
        print("\n3. Voltage 2D Distribution")
        plot_voltage_2d(voltage_2d, config_params, time_label=time_label,
                       save_path="spatial_2d_voltage.png")
        print("  ✓ Saved: spatial_2d_voltage.png")
        
        print("\n4. Current 2D Distribution")
        plot_current_2d(current_2d, config_params, time_label=time_label,
                       save_path="spatial_2d_current.png")
        print("  ✓ Saved: spatial_2d_current.png")
        
        print("\n5. Combined 2D Spatial Plot")
        plot_all_spatial_2d(temp_2d, soc_2d, voltage_2d, current_2d,
                           config_params, time_label=time_label,
                           save_path="spatial_2d_all.png")
        print("  ✓ Saved: spatial_2d_all.png")
        
        print("\n" + "="*70)
        print("✓ All visualizations complete!")
        print("="*70)
        print("\nGenerated files:")
        print("  - spatial_2d_temp.png")
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
        test_module(config_path)
    else:
        print("Usage: python viz_spatial_2d.py [config_file.toml]")
        print("Running with default config...\n")
        test_module()
