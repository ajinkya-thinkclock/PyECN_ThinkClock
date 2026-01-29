"""
Time-Series Animation Module for PyECN
======================================
Creates animated visualizations showing simulation evolution over time.

Functions:
    - animate_voltage_vs_time: Animated voltage curve building up
    - animate_current_vs_time: Animated current curve building up
    - animate_soc_vs_time: Animated SoC curve building up
    - animate_temperature_vs_time: Animated temperature curves building up
    - animate_all_timeseries: Combined 4-panel animation
    - run_simulation_and_animate: Main function to run PyECN and generate animations
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec
from pathlib import Path
import sys
import toml


def animate_voltage_vs_time(time, voltage, cell_name, c_rate, v_highlimit, v_lowlimit,
                            save_path="voltage_animation.gif", fps=30, downsample=10):
    """
    Create animated plot of voltage vs time (curve building up).
    
    Parameters:
    -----------
    time : array-like
        Time points (seconds)
    voltage : array-like
        Terminal voltage (V)
    cell_name : str
        Cell name for title
    c_rate : float
        C-rate for title
    v_highlimit : float
        Voltage upper limit for reference line
    v_lowlimit : float
        Voltage lower limit for reference line
    save_path : str
        Output file path (GIF or MP4)
    fps : int
        Frames per second
    downsample : int
        Use every Nth data point for animation (speeds up rendering)
    """
    print(f"Creating voltage animation...")
    
    # Downsample data
    indices = np.arange(0, len(time), downsample)
    time_ds = time[indices]
    voltage_ds = voltage[indices]
    
    # Convert time to hours if needed
    time_hours = time_ds / 3600 if np.max(time_ds) > 100 else time_ds
    time_label = "Time (h)" if np.max(time_ds) > 100 else "Time (s)"
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    line, = ax.plot([], [], 'b-', linewidth=2, label='Terminal Voltage')
    
    # Reference lines
    ax.axhline(y=v_highlimit, color='r', linestyle='--', alpha=0.5, linewidth=1,
               label=f'V_high = {v_highlimit:.2f}V')
    ax.axhline(y=v_lowlimit, color='g', linestyle='--', alpha=0.5, linewidth=1,
               label=f'V_low = {v_lowlimit:.2f}V')
    
    # Set fixed axis limits
    ax.set_xlim(0, time_hours[-1] * 1.05)
    ax.set_ylim(min(voltage_ds) * 0.95, max(voltage_ds) * 1.05)
    ax.set_xlabel(time_label, fontsize=12, fontweight='bold')
    ax.set_ylabel('Voltage (V)', fontsize=12, fontweight='bold')
    ax.set_title(f'Cell Terminal Voltage - {cell_name} ({c_rate}C)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10)
    
    # Time text
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                       fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text
    
    def animate(frame):
        line.set_data(time_hours[:frame+1], voltage_ds[:frame+1])
        time_text.set_text(f'{time_label.split()[0]}: {time_hours[frame]:.2f}')
        return line, time_text
    
    anim = FuncAnimation(fig, animate, init_func=init, frames=len(time_ds),
                        interval=1000/fps, blit=True, repeat=True)
    
    # Save animation
    writer = PillowWriter(fps=fps)
    anim.save(save_path, writer=writer, dpi=100)
    plt.close(fig)
    
    print(f"  ✓ Saved: {save_path} ({len(indices)} frames)")


def animate_current_vs_time(time, current, cell_name, c_rate,
                           save_path="current_animation.gif", fps=30, downsample=10):
    """Create animated plot of current vs time."""
    print(f"Creating current animation...")
    
    # Downsample
    indices = np.arange(0, len(time), downsample)
    time_ds = time[indices]
    current_ds = current[indices]
    
    time_hours = time_ds / 3600 if np.max(time_ds) > 100 else time_ds
    time_label = "Time (h)" if np.max(time_ds) > 100 else "Time (s)"
    
    fig, ax = plt.subplots(figsize=(10, 6))
    line, = ax.plot([], [], 'r-', linewidth=2, label='Cell Current')
    
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=1)
    
    ax.set_xlim(0, time_hours[-1] * 1.05)
    ax.set_ylim(min(current_ds) * 1.1, max(current_ds) * 1.1)
    ax.set_xlabel(time_label, fontsize=12, fontweight='bold')
    ax.set_ylabel('Current (A)', fontsize=12, fontweight='bold')
    ax.set_title(f'Cell Current - {cell_name} ({c_rate}C)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10)
    
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                       fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text
    
    def animate(frame):
        line.set_data(time_hours[:frame+1], current_ds[:frame+1])
        time_text.set_text(f'{time_label.split()[0]}: {time_hours[frame]:.2f}')
        return line, time_text
    
    anim = FuncAnimation(fig, animate, init_func=init, frames=len(time_ds),
                        interval=1000/fps, blit=True, repeat=True)
    
    writer = PillowWriter(fps=fps)
    anim.save(save_path, writer=writer, dpi=100)
    plt.close(fig)
    
    print(f"  ✓ Saved: {save_path} ({len(indices)} frames)")


def animate_soc_vs_time(time, soc, cell_name, c_rate,
                       save_path="soc_animation.gif", fps=30, downsample=10):
    """Create animated plot of SoC vs time."""
    print(f"Creating SoC animation...")
    
    # Downsample
    indices = np.arange(0, len(time), downsample)
    time_ds = time[indices]
    soc_ds = soc[indices] * 100  # Convert to percentage
    
    time_hours = time_ds / 3600 if np.max(time_ds) > 100 else time_ds
    time_label = "Time (h)" if np.max(time_ds) > 100 else "Time (s)"
    
    fig, ax = plt.subplots(figsize=(10, 6))
    line, = ax.plot([], [], 'g-', linewidth=2, label='State of Charge')
    
    ax.axhline(y=100, color='b', linestyle='--', alpha=0.5, linewidth=1,
               label='Full Charge (100%)')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=1,
               label='Empty (0%)')
    
    ax.set_xlim(0, time_hours[-1] * 1.05)
    ax.set_ylim(-5, 105)
    ax.set_xlabel(time_label, fontsize=12, fontweight='bold')
    ax.set_ylabel('State of Charge (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'State of Charge - {cell_name} ({c_rate}C)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10)
    
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                       fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text
    
    def animate(frame):
        line.set_data(time_hours[:frame+1], soc_ds[:frame+1])
        time_text.set_text(f'{time_label.split()[0]}: {time_hours[frame]:.2f} | SoC: {soc_ds[frame]:.1f}%')
        return line, time_text
    
    anim = FuncAnimation(fig, animate, init_func=init, frames=len(time_ds),
                        interval=1000/fps, blit=True, repeat=True)
    
    writer = PillowWriter(fps=fps)
    anim.save(save_path, writer=writer, dpi=100)
    plt.close(fig)
    
    print(f"  ✓ Saved: {save_path} ({len(indices)} frames)")


def animate_temperature_vs_time(time, temp_avg, temp_min, temp_max, temp_delta, temp_std,
                                cell_name, c_rate, T_cooling, T_initial,
                                save_path="temperature_animation.gif", fps=30, downsample=10):
    """Create animated plot of temperature metrics vs time."""
    print(f"Creating temperature animation...")
    
    # Downsample
    indices = np.arange(0, len(time), downsample)
    time_ds = time[indices]
    temp_avg_ds = temp_avg[indices] - 273.15
    temp_min_ds = temp_min[indices] - 273.15
    temp_max_ds = temp_max[indices] - 273.15
    temp_delta_ds = temp_delta[indices]
    temp_std_ds = temp_std[indices]
    
    time_hours = time_ds / 3600 if np.max(time_ds) > 100 else time_ds
    time_label = "Time (h)" if np.max(time_ds) > 100 else "Time (s)"
    
    T_cooling_c = T_cooling - 273.15
    T_initial_c = T_initial - 273.15
    
    # Create 4-panel figure
    fig = plt.figure(figsize=(14, 12))
    gs = GridSpec(4, 1, figure=fig, hspace=0.35)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[3, 0])
    
    # Panel 1: Average Temperature
    line1, = ax1.plot([], [], 'r-', linewidth=2, label='Average Temperature')
    ax1.axhline(y=T_cooling_c, color='b', linestyle='--', alpha=0.5, linewidth=1,
                label=f'Cooling Temp = {T_cooling_c:.1f}°C')
    ax1.axhline(y=T_initial_c, color='g', linestyle='--', alpha=0.5, linewidth=1,
                label=f'Initial Temp = {T_initial_c:.1f}°C')
    ax1.set_xlim(0, time_hours[-1] * 1.05)
    ax1.set_ylim(min(temp_avg_ds) * 0.99, max(temp_avg_ds) * 1.01)
    ax1.set_ylabel('Temperature (°C)', fontsize=11, fontweight='bold')
    ax1.set_title(f'Average Temperature - {cell_name} ({c_rate}C)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=9)
    
    # Panel 2: Min/Max Temperature
    line2_min, = ax2.plot([], [], 'b-', linewidth=2, label='Min Temperature')
    line2_max, = ax2.plot([], [], 'r-', linewidth=2, label='Max Temperature')
    ax2.set_xlim(0, time_hours[-1] * 1.05)
    ax2.set_ylim(min(temp_min_ds) * 0.99, max(temp_max_ds) * 1.01)
    ax2.set_ylabel('Temperature (°C)', fontsize=11, fontweight='bold')
    ax2.set_title(f'Min/Max Temperature - {cell_name} ({c_rate}C)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=9)
    
    # Panel 3: Temperature Delta
    line3, = ax3.plot([], [], 'orange', linewidth=2, label='Temperature Gradient')
    ax3.set_xlim(0, time_hours[-1] * 1.05)
    ax3.set_ylim(0, max(temp_delta_ds) * 1.1)
    ax3.set_ylabel('Temp Delta (°C)', fontsize=11, fontweight='bold')
    ax3.set_title(f'Temperature Gradient - {cell_name} ({c_rate}C)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(fontsize=9)
    
    # Panel 4: Temperature Std Dev
    line4, = ax4.plot([], [], 'purple', linewidth=2, label='Temperature Std Dev')
    ax4.set_xlim(0, time_hours[-1] * 1.05)
    ax4.set_ylim(0, max(temp_std_ds) * 1.1)
    ax4.set_xlabel(time_label, fontsize=11, fontweight='bold')
    ax4.set_ylabel('Temp Std Dev (°C)', fontsize=11, fontweight='bold')
    ax4.set_title(f'Temperature Variation - {cell_name} ({c_rate}C)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.legend(fontsize=9)
    
    fig.suptitle(f'Thermal Analysis Animation - {cell_name}', fontsize=16, fontweight='bold')
    
    # Time text
    time_text = fig.text(0.5, 0.02, '', ha='center', fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def init():
        line1.set_data([], [])
        line2_min.set_data([], [])
        line2_max.set_data([], [])
        line3.set_data([], [])
        line4.set_data([], [])
        time_text.set_text('')
        return line1, line2_min, line2_max, line3, line4, time_text
    
    def animate(frame):
        line1.set_data(time_hours[:frame+1], temp_avg_ds[:frame+1])
        line2_min.set_data(time_hours[:frame+1], temp_min_ds[:frame+1])
        line2_max.set_data(time_hours[:frame+1], temp_max_ds[:frame+1])
        line3.set_data(time_hours[:frame+1], temp_delta_ds[:frame+1])
        line4.set_data(time_hours[:frame+1], temp_std_ds[:frame+1])
        time_text.set_text(f'{time_label}: {time_hours[frame]:.2f} | Avg Temp: {temp_avg_ds[frame]:.2f}°C')
        return line1, line2_min, line2_max, line3, line4, time_text
    
    anim = FuncAnimation(fig, animate, init_func=init, frames=len(time_ds),
                        interval=1000/fps, blit=True, repeat=True)
    
    writer = PillowWriter(fps=fps)
    anim.save(save_path, writer=writer, dpi=100)
    plt.close(fig)
    
    print(f"  ✓ Saved: {save_path} ({len(indices)} frames)")


def animate_all_timeseries(time, voltage, current, soc, temp_avg,
                           cell_name, c_rate, v_highlimit, v_lowlimit, T_cooling,
                           save_path="all_timeseries_animation.gif", fps=30, downsample=10):
    """Create combined 4-panel animated plot of all time-series data."""
    print(f"Creating combined time-series animation...")
    
    # Downsample
    indices = np.arange(0, len(time), downsample)
    time_ds = time[indices]
    voltage_ds = voltage[indices]
    current_ds = current[indices]
    soc_ds = soc[indices] * 100
    temp_ds = temp_avg[indices] - 273.15
    
    time_hours = time_ds / 3600 if np.max(time_ds) > 100 else time_ds
    time_label = "Time (h)" if np.max(time_ds) > 100 else "Time (s)"
    T_cooling_c = T_cooling - 273.15
    
    # Create 4-panel figure
    fig = plt.figure(figsize=(14, 12))
    gs = GridSpec(4, 1, figure=fig, hspace=0.35)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[3, 0])
    
    # Panel 1: Voltage
    line1, = ax1.plot([], [], 'b-', linewidth=2, label='Terminal Voltage')
    ax1.axhline(y=v_highlimit, color='r', linestyle='--', alpha=0.5, linewidth=1,
                label=f'V_high = {v_highlimit:.2f}V')
    ax1.axhline(y=v_lowlimit, color='g', linestyle='--', alpha=0.5, linewidth=1,
                label=f'V_low = {v_lowlimit:.2f}V')
    ax1.set_xlim(0, time_hours[-1] * 1.05)
    ax1.set_ylim(min(voltage_ds) * 0.95, max(voltage_ds) * 1.05)
    ax1.set_ylabel('Voltage (V)', fontsize=11, fontweight='bold')
    ax1.set_title('Terminal Voltage', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=9)
    
    # Panel 2: Current
    line2, = ax2.plot([], [], 'r-', linewidth=2, label='Cell Current')
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=1)
    ax2.set_xlim(0, time_hours[-1] * 1.05)
    ax2.set_ylim(min(current_ds) * 1.1, max(current_ds) * 1.1)
    ax2.set_ylabel('Current (A)', fontsize=11, fontweight='bold')
    ax2.set_title('Cell Current', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=9)
    
    # Panel 3: SoC
    line3, = ax3.plot([], [], 'g-', linewidth=2, label='State of Charge')
    ax3.axhline(y=100, color='b', linestyle='--', alpha=0.5, linewidth=1)
    ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=1)
    ax3.set_xlim(0, time_hours[-1] * 1.05)
    ax3.set_ylim(-5, 105)
    ax3.set_ylabel('SoC (%)', fontsize=11, fontweight='bold')
    ax3.set_title('State of Charge', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(fontsize=9)
    
    # Panel 4: Temperature
    line4, = ax4.plot([], [], 'orange', linewidth=2, label='Average Temperature')
    ax4.axhline(y=T_cooling_c, color='b', linestyle='--', alpha=0.5, linewidth=1,
                label=f'Cooling = {T_cooling_c:.1f}°C')
    ax4.set_xlim(0, time_hours[-1] * 1.05)
    ax4.set_ylim(min(temp_ds) * 0.99, max(temp_ds) * 1.01)
    ax4.set_xlabel(time_label, fontsize=11, fontweight='bold')
    ax4.set_ylabel('Temperature (°C)', fontsize=11, fontweight='bold')
    ax4.set_title('Average Temperature', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.legend(fontsize=9)
    
    fig.suptitle(f'Complete Time-Series Animation - {cell_name} ({c_rate}C)', 
                 fontsize=16, fontweight='bold')
    
    # Time text
    time_text = fig.text(0.5, 0.02, '', ha='center', fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        line3.set_data([], [])
        line4.set_data([], [])
        time_text.set_text('')
        return line1, line2, line3, line4, time_text
    
    def animate(frame):
        line1.set_data(time_hours[:frame+1], voltage_ds[:frame+1])
        line2.set_data(time_hours[:frame+1], current_ds[:frame+1])
        line3.set_data(time_hours[:frame+1], soc_ds[:frame+1])
        line4.set_data(time_hours[:frame+1], temp_ds[:frame+1])
        time_text.set_text(f'{time_label}: {time_hours[frame]:.2f} | V: {voltage_ds[frame]:.3f}V | SoC: {soc_ds[frame]:.1f}% | T: {temp_ds[frame]:.2f}°C')
        return line1, line2, line3, line4, time_text
    
    anim = FuncAnimation(fig, animate, init_func=init, frames=len(time_ds),
                        interval=1000/fps, blit=True, repeat=True)
    
    writer = PillowWriter(fps=fps)
    anim.save(save_path, writer=writer, dpi=100)
    plt.close(fig)
    
    print(f"  ✓ Saved: {save_path} ({len(indices)} frames)")


def run_simulation_and_animate(config_path=None, fps=30, downsample=10):
    """
    Run PyECN simulation and generate all time-series animations.
    
    Parameters:
    -----------
    config_path : str or Path, optional
        Path to TOML config file. If None, uses cylindrical_tabless_Fig_3.toml
    fps : int
        Frames per second for animations
    downsample : int
        Use every Nth data point (reduces file size and rendering time)
    """
    print("="*70)
    print("PyECN Time-Series Animation Generator")
    print("="*70)
    
    # Set up paths
    if config_path is None:
        config_path = Path(__file__).parent.parent / "Examples" / "cylindrical_tabless_Fig_3.toml"
    config_path = Path(config_path)
    
    print(f"\nConfig file: {config_path}")
    print(f"Animation settings: {fps} FPS, downsample={downsample}")
    
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
        T_cooling = ip.T_cooling
        T_initial = ip.T_initial
        
        print(f"\n✓ Simulation complete!")
        print(f"  Cell: {cell_name}")
        print(f"  Time steps: {cell.nt}")
        if cell.nt > 0:
            print(f"  Final time: {cell.t_record[cell.nt]:.1f}s")
        
        # Extract data
        time = cell.t_record[:cell.nt + 1]
        voltage = cell.V_Cell_record[:cell.nt + 1]
        current = cell.I_Cell_record[:cell.nt + 1]
        soc = cell.SoC_Cell_record[:cell.nt + 1]
        temp_avg = cell.T_avg_record[:cell.nt + 1]
        temp_std = cell.T_SD_record[:cell.nt + 1]
        temp_delta = cell.T_Delta_record[:cell.nt + 1]
        temp_min = temp_avg - temp_delta / 2
        temp_max = temp_avg + temp_delta / 2
        
        print("\n" + "-"*70)
        print("Generating Animations...")
        print("-"*70)
        
        print("\n1. Voltage Animation")
        animate_voltage_vs_time(time, voltage, cell_name, c_rate, v_highlimit, v_lowlimit,
                               save_path="anim_voltage.gif", fps=fps, downsample=downsample)
        
        print("\n2. Current Animation")
        animate_current_vs_time(time, current, cell_name, c_rate,
                               save_path="anim_current.gif", fps=fps, downsample=downsample)
        
        print("\n3. SoC Animation")
        animate_soc_vs_time(time, soc, cell_name, c_rate,
                           save_path="anim_soc.gif", fps=fps, downsample=downsample)
        
        print("\n4. Temperature Animation (4-panel)")
        animate_temperature_vs_time(time, temp_avg, temp_min, temp_max, temp_delta, temp_std,
                                   cell_name, c_rate, T_cooling, T_initial,
                                   save_path="anim_temperature.gif", fps=fps, downsample=downsample)
        
        print("\n5. Combined Time-Series Animation (4-panel)")
        animate_all_timeseries(time, voltage, current, soc, temp_avg,
                              cell_name, c_rate, v_highlimit, v_lowlimit, T_cooling,
                              save_path="anim_all_timeseries.gif", fps=fps, downsample=downsample)
        
        print("\n" + "="*70)
        print("✓ All animations complete!")
        print("="*70)
        print("\nGenerated files:")
        print("  - anim_voltage.gif")
        print("  - anim_current.gif")
        print("  - anim_soc.gif")
        print("  - anim_temperature.gif")
        print("  - anim_all_timeseries.gif")
        print("="*70)
        
        return cell
        
    finally:
        # Restore original directory and sys.argv
        os.chdir(original_dir)
        sys.argv[:] = original_argv


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate PyECN time-series animations')
    parser.add_argument('config', nargs='?', default=None, 
                       help='Path to TOML config file')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second (default: 30)')
    parser.add_argument('--downsample', type=int, default=10,
                       help='Use every Nth data point (default: 10)')
    
    args = parser.parse_args()
    
    if args.config:
        run_simulation_and_animate(args.config, fps=args.fps, downsample=args.downsample)
    else:
        print("Usage: python viz_animations.py [config_file.toml] [--fps 30] [--downsample 10]")
        print("Running with default config...\n")
        run_simulation_and_animate(fps=args.fps, downsample=args.downsample)
