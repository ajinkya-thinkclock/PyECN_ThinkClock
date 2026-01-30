"""
Interactive time-series visualization using Plotly.
Creates interactive HTML plots for PyECN simulation results.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os


def create_interactive_voltage_plot(time, voltage, cell_name, c_rate, v_highlimit, v_lowlimit,
                                     output_dir="interactive_plots"):
    """Create interactive voltage vs time plot."""
    print(f"Creating interactive voltage plot...")
    save_path = os.path.join(output_dir, "voltage_interactive.html")
    
    time_hours = time / 3600 if np.max(time) > 100 else time
    time_label = "Time (h)" if np.max(time) > 100 else "Time (s)"
    
    fig = go.Figure()
    
    # Voltage trace
    fig.add_trace(go.Scatter(
        x=time_hours,
        y=voltage,
        mode='lines',
        name='Cell Voltage',
        line=dict(color='blue', width=2),
        hovertemplate=f'{time_label}: %{{x:.2f}}<br>Voltage: %{{y:.3f}} V<extra></extra>'
    ))
    
    # Reference lines
    fig.add_hline(y=v_highlimit, line_dash="dash", line_color="green", 
                  annotation_text=f"V_high = {v_highlimit:.2f}V",
                  annotation_position="right")
    fig.add_hline(y=v_lowlimit, line_dash="dash", line_color="red",
                  annotation_text=f"V_low = {v_lowlimit:.2f}V",
                  annotation_position="right")
    
    fig.update_layout(
        title=f'Cell Voltage - {cell_name} ({c_rate}C)',
        xaxis_title=time_label,
        yaxis_title='Voltage (V)',
        hovermode='x unified',
        template='plotly_white',
        width=1000,
        height=600
    )
    
    fig.write_html(save_path)
    print(f"  ✓ Saved: {save_path}")
    return fig


def create_interactive_current_plot(time, current, cell_name, c_rate,
                                     output_dir="interactive_plots"):
    """Create interactive current vs time plot."""
    print(f"Creating interactive current plot...")
    save_path = os.path.join(output_dir, "current_interactive.html")
    
    time_hours = time / 3600 if np.max(time) > 100 else time
    time_label = "Time (h)" if np.max(time) > 100 else "Time (s)"
    
    fig = go.Figure()
    
    # Current trace
    fig.add_trace(go.Scatter(
        x=time_hours,
        y=current,
        mode='lines',
        name='Cell Current',
        line=dict(color='red', width=2),
        hovertemplate=f'{time_label}: %{{x:.2f}}<br>Current: %{{y:.3f}} A<extra></extra>'
    ))
    
    # Zero line
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1, opacity=0.3)
    
    fig.update_layout(
        title=f'Cell Current - {cell_name} ({c_rate}C)',
        xaxis_title=time_label,
        yaxis_title='Current (A)',
        hovermode='x unified',
        template='plotly_white',
        width=1000,
        height=600
    )
    
    fig.write_html(save_path)
    print(f"  ✓ Saved: {save_path}")
    return fig


def create_interactive_soc_plot(time, soc, cell_name, c_rate,
                                 output_dir="interactive_plots"):
    """Create interactive SoC vs time plot."""
    print(f"Creating interactive SoC plot...")
    save_path = os.path.join(output_dir, "soc_interactive.html")
    
    time_hours = time / 3600 if np.max(time) > 100 else time
    time_label = "Time (h)" if np.max(time) > 100 else "Time (s)"
    
    fig = go.Figure()
    
    # SoC trace
    fig.add_trace(go.Scatter(
        x=time_hours,
        y=soc * 100,  # Convert to percentage
        mode='lines',
        name='State of Charge',
        line=dict(color='green', width=2),
        fill='tozeroy',
        fillcolor='rgba(0,255,0,0.1)',
        hovertemplate=f'{time_label}: %{{x:.2f}}<br>SoC: %{{y:.1f}}%<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'State of Charge - {cell_name} ({c_rate}C)',
        xaxis_title=time_label,
        yaxis_title='SoC (%)',
        yaxis_range=[0, 105],
        hovermode='x unified',
        template='plotly_white',
        width=1000,
        height=600
    )
    
    fig.write_html(save_path)
    print(f"  ✓ Saved: {save_path}")
    return fig


def create_interactive_temperature_plot(time, temp_avg, temp_min, temp_max, temp_delta, temp_std,
                                        cell_name, c_rate, T_cooling, T_initial,
                                        output_dir="interactive_plots"):
    """Create interactive temperature 4-panel plot."""
    print(f"Creating interactive temperature plot...")
    save_path = os.path.join(output_dir, "temperature_interactive.html")
    
    # Check for valid temperature data
    has_valid_temp = not (np.all(np.isnan(temp_avg)) or np.all(np.isnan(temp_min)) or 
                          np.all(np.isnan(temp_max)) or np.all(np.isnan(temp_delta)))
    
    if not has_valid_temp:
        print("  ⚠ Warning: Temperature data contains only NaN values (E-only model?)")
        print("  Creating placeholder plot...")
    
    time_hours = time / 3600 if np.max(time) > 100 else time
    time_label = "Time (h)" if np.max(time) > 100 else "Time (s)"
    
    # Create 2x2 subplot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Average Temperature', 'Min/Max Temperature',
                       'Temperature Delta', 'Temperature Std Dev'),
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )
    
    if has_valid_temp:
        # Panel 1: Average temperature
        fig.add_trace(go.Scatter(
            x=time_hours, y=temp_avg,
            mode='lines', name='T_avg',
            line=dict(color='blue', width=2),
            hovertemplate=f'{time_label}: %{{x:.2f}}<br>T_avg: %{{y:.2f}}°C<extra></extra>'
        ), row=1, col=1)
        fig.add_hline(y=T_cooling, line_dash="dash", line_color="cyan",
                     annotation_text=f"T_cooling = {T_cooling}°C",
                     row=1, col=1)
        
        # Panel 2: Min/Max temperature
        fig.add_trace(go.Scatter(
            x=time_hours, y=temp_max,
            mode='lines', name='T_max',
            line=dict(color='red', width=2),
            hovertemplate=f'{time_label}: %{{x:.2f}}<br>T_max: %{{y:.2f}}°C<extra></extra>'
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=time_hours, y=temp_min,
            mode='lines', name='T_min',
            line=dict(color='purple', width=2),
            hovertemplate=f'{time_label}: %{{x:.2f}}<br>T_min: %{{y:.2f}}°C<extra></extra>'
        ), row=1, col=2)
        
        # Panel 3: Temperature delta
        fig.add_trace(go.Scatter(
            x=time_hours, y=temp_delta,
            mode='lines', name='ΔT',
            line=dict(color='orange', width=2),
            hovertemplate=f'{time_label}: %{{x:.2f}}<br>ΔT: %{{y:.2f}}°C<extra></extra>'
        ), row=2, col=1)
        
        # Panel 4: Temperature std dev
        fig.add_trace(go.Scatter(
            x=time_hours, y=temp_std,
            mode='lines', name='σ_T',
            line=dict(color='green', width=2),
            hovertemplate=f'{time_label}: %{{x:.2f}}<br>σ_T: %{{y:.2f}}°C<extra></extra>'
        ), row=2, col=2)
    
    # Update axes
    for row in [1, 2]:
        for col in [1, 2]:
            fig.update_xaxes(title_text=time_label, row=row, col=col)
    fig.update_yaxes(title_text='Temperature (°C)', row=1, col=1)
    fig.update_yaxes(title_text='Temperature (°C)', row=1, col=2)
    fig.update_yaxes(title_text='ΔT (°C)', row=2, col=1)
    fig.update_yaxes(title_text='σ_T (°C)', row=2, col=2)
    
    fig.update_layout(
        title_text=f'Temperature Analysis - {cell_name} ({c_rate}C)',
        showlegend=True,
        hovermode='x unified',
        template='plotly_white',
        width=1200,
        height=800
    )
    
    fig.write_html(save_path)
    print(f"  ✓ Saved: {save_path}")
    return fig


def create_interactive_combined_plot(time, voltage, current, soc, temp_avg,
                                     cell_name, c_rate, v_highlimit, v_lowlimit, T_cooling,
                                     output_dir="interactive_plots"):
    """Create interactive combined 4-panel time-series plot."""
    print(f"Creating interactive combined plot...")
    save_path = os.path.join(output_dir, "timeseries_interactive.html")
    
    time_hours = time / 3600 if np.max(time) > 100 else time
    time_label = "Time (h)" if np.max(time) > 100 else "Time (s)"
    
    # Check for valid temperature
    has_valid_temp = not np.all(np.isnan(temp_avg))
    
    # Create 2x2 subplot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Voltage', 'Current', 'State of Charge', 
                       'Temperature' if has_valid_temp else 'Temperature (No data)'),
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )
    
    # Panel 1: Voltage
    fig.add_trace(go.Scatter(
        x=time_hours, y=voltage,
        mode='lines', name='Voltage',
        line=dict(color='blue', width=2),
        hovertemplate=f'{time_label}: %{{x:.2f}}<br>Voltage: %{{y:.3f}} V<extra></extra>'
    ), row=1, col=1)
    fig.add_hline(y=v_highlimit, line_dash="dash", line_color="green", row=1, col=1,
                 annotation_text=f"V_high={v_highlimit:.2f}V", annotation_position="right")
    fig.add_hline(y=v_lowlimit, line_dash="dash", line_color="red", row=1, col=1,
                 annotation_text=f"V_low={v_lowlimit:.2f}V", annotation_position="right")
    
    # Panel 2: Current
    fig.add_trace(go.Scatter(
        x=time_hours, y=current,
        mode='lines', name='Current',
        line=dict(color='red', width=2),
        hovertemplate=f'{time_label}: %{{x:.2f}}<br>Current: %{{y:.3f}} A<extra></extra>'
    ), row=1, col=2)
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1, opacity=0.3, row=1, col=2)
    
    # Panel 3: SoC
    fig.add_trace(go.Scatter(
        x=time_hours, y=soc * 100,
        mode='lines', name='SoC',
        line=dict(color='green', width=2),
        fill='tozeroy',
        fillcolor='rgba(0,255,0,0.1)',
        hovertemplate=f'{time_label}: %{{x:.2f}}<br>SoC: %{{y:.1f}}%<extra></extra>'
    ), row=2, col=1)
    
    # Panel 4: Temperature
    if has_valid_temp:
        fig.add_trace(go.Scatter(
            x=time_hours, y=temp_avg,
            mode='lines', name='T_avg',
            line=dict(color='orange', width=2),
            hovertemplate=f'{time_label}: %{{x:.2f}}<br>T_avg: %{{y:.2f}}°C<extra></extra>'
        ), row=2, col=2)
        fig.add_hline(y=T_cooling, line_dash="dash", line_color="cyan", row=2, col=2,
                     annotation_text=f"T_cooling={T_cooling}°C", annotation_position="right")
    
    # Update axes
    for row in [1, 2]:
        for col in [1, 2]:
            fig.update_xaxes(title_text=time_label, row=row, col=col)
    fig.update_yaxes(title_text='Voltage (V)', row=1, col=1)
    fig.update_yaxes(title_text='Current (A)', row=1, col=2)
    fig.update_yaxes(title_text='SoC (%)', range=[0, 105], row=2, col=1)
    fig.update_yaxes(title_text='Temperature (°C)', row=2, col=2)
    
    fig.update_layout(
        title_text=f'Battery Performance - {cell_name} ({c_rate}C)',
        showlegend=False,
        hovermode='x unified',
        template='plotly_white',
        width=1400,
        height=900
    )
    
    fig.write_html(save_path)
    print(f"  ✓ Saved: {save_path}")
    return fig


def run_simulation_and_visualize(config_file, skip_individual=False):
    """Run PyECN simulation and create interactive visualizations."""
    
    print("="*70)
    print("PyECN INTERACTIVE VISUALIZATION")
    print("="*70)
    
    # Add parent directory to path to import pyecn
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pyecn_parent = os.path.dirname(os.path.dirname(script_dir))
    if pyecn_parent not in sys.path:
        sys.path.insert(0, pyecn_parent)
    
    # Import PyECN
    import pyecn
    
    # Store original state
    original_dir = os.getcwd()
    original_argv = sys.argv[:]
    
    try:
        # Set up for PyECN run
        # PyECN expects to be run from the parent directory of pyecn/
        # and it looks for config files in pyecn/ directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        pyecn_parent_dir = os.path.dirname(os.path.dirname(script_dir))  # Go up to PyECN/
        
        os.chdir(pyecn_parent_dir)
        
        # PyECN's parse_inputs.py prepends "pyecn/" to the config filename
        # So we just pass the filename (e.g., "cylindrical.toml")
        sys.argv = ['pyecn', config_file]
        
        print(f"\nRunning PyECN simulation...")
        print(f"  Config: {config_file}")
        print(f"  Working directory: {os.getcwd()}")
        
        # Run simulation
        pyecn.run()
        
        # Get the cell object from PyECN module's global namespace
        import pyecn as pyecn_module
        cell_name = None
        cell = None
        for name, obj in vars(pyecn_module).items():
            if hasattr(obj, 'U_pndiff_plot') and hasattr(obj, 't_record'):
                cell = obj
                cell_name = name
                break
        
        if cell is None:
            raise RuntimeError("Could not find cell object after simulation")
        
        # Get config parameters
        c_rate = cell.C_rate if hasattr(cell, 'C_rate') else 1.0
        v_highlimit = cell.V_highlimit_single if hasattr(cell, 'V_highlimit_single') else 4.2
        v_lowlimit = cell.V_lowlimit_single if hasattr(cell, 'V_lowlimit_single') else 2.7
        T_cooling = cell.T_Cooling if hasattr(cell, 'T_Cooling') else 25.0
        T_initial = cell.T0_single if hasattr(cell, 'T0_single') else 25.0
        cell_name = "Cell" if cell_name is None else cell_name
        
        print(f"\n✓ Simulation complete!")
        print(f"  Cell: {cell_name}")
        print(f"  Time steps: {cell.nt}")
        if cell.nt > 0:
            print(f"  Final time: {cell.t_record[cell.nt]:.1f}s")
        
        # Extract data
        time = cell.t_record[:cell.nt + 1]
        voltage = cell.U_pndiff_plot[:cell.nt + 1]
        
        # Current: Check if I0_record has actual data, otherwise use applied current
        current_from_record = cell.I0_record[:cell.nt + 1]
        if len(current_from_record) > 1 and np.all(current_from_record[1:] == 0):
            current = np.full_like(time, current_from_record[0])
            print(f"  Note: Using constant current ({current_from_record[0]:.3f} A) from I0_record[0]")
        else:
            current = current_from_record
        
        soc = cell.SoC_Cell_record[:cell.nt + 1]
        
        # Temperature data - PyECN stores in Kelvin, convert to Celsius
        temp_avg = cell.T_avg_record[:cell.nt + 1] - 273.15
        temp_std = cell.T_SD_record[:cell.nt + 1]  # Temperature difference, no conversion needed
        temp_delta = cell.T_Delta_record[:cell.nt + 1]  # Temperature difference, no conversion needed
        temp_min = temp_avg - temp_delta / 2
        temp_max = temp_avg + temp_delta / 2
        
        # Convert cooling and initial temps from Kelvin to Celsius
        T_cooling = T_cooling - 273.15 if T_cooling > 100 else T_cooling
        T_initial = T_initial - 273.15 if T_initial > 100 else T_initial
        
        # Create output directory
        output_dir = "interactive_plots"
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "-"*70)
        print("Generating Interactive Visualizations...")
        print("-"*70)
        print(f"Output directory: {os.path.abspath(output_dir)}")
        
        if not skip_individual:
            print("\n1. Voltage Plot")
            create_interactive_voltage_plot(time, voltage, cell_name, c_rate, v_highlimit, v_lowlimit,
                                          output_dir=output_dir)
            
            print("\n2. Current Plot")
            create_interactive_current_plot(time, current, cell_name, c_rate,
                                          output_dir=output_dir)
            
            print("\n3. SoC Plot")
            create_interactive_soc_plot(time, soc, cell_name, c_rate,
                                      output_dir=output_dir)
            
            print("\n4. Temperature Plot (4-panel)")
            create_interactive_temperature_plot(time, temp_avg, temp_min, temp_max, temp_delta, temp_std,
                                               cell_name, c_rate, T_cooling, T_initial,
                                               output_dir=output_dir)
        else:
            print("\nSkipping individual plots (--skip-individual flag)")
        
        print("\n5. Combined Time-Series Plot (4-panel)")
        create_interactive_combined_plot(time, voltage, current, soc, temp_avg,
                                        cell_name, c_rate, v_highlimit, v_lowlimit, T_cooling,
                                        output_dir=output_dir)
        
        print("\n" + "="*70)
        print("✓ All visualizations complete!")
        print("="*70)
        print(f"\nGenerated files in {os.path.abspath(output_dir)}/:")
        if not skip_individual:
            print("  - voltage_interactive.html")
            print("  - current_interactive.html")
            print("  - soc_interactive.html")
            print("  - temperature_interactive.html")
            print("  - timeseries_interactive.html")
        else:
            print("  - timeseries_interactive.html")
        print("\nOpen any HTML file in your browser to explore the data!")
        print("="*70)
        
        return cell
        
    finally:
        # Restore original directory and sys.argv
        os.chdir(original_dir)
        sys.argv[:] = original_argv


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create interactive visualizations for PyECN simulation')
    parser.add_argument('config', nargs='?', default='cylindrical.toml',
                       help='Path to PyECN config file (default: cylindrical.toml)')
    parser.add_argument('--skip-individual', action='store_true',
                       help='Skip individual plots, only create combined plot')
    
    args = parser.parse_args()
    
    run_simulation_and_visualize(args.config, skip_individual=args.skip_individual)
