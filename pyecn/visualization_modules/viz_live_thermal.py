"""
Live Thermal Visualization Module for PyECN
==========================================
Real-time visualization of thermal distribution in unrolled electrode structure.
Shows temperature evolution during simulation with interactive controls.

Features:
- Real-time unrolled jellyroll temperature heatmap
- Pause/resume/step controls
- Frame skip for faster visualization
- Temperature statistics overlay
- Works with cylindrical and pouch cells
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button, Slider
import sys
from pathlib import Path


class LiveThermalVisualizer:
    """Interactive real-time thermal visualization for electrode structures."""
    
    def __init__(self, cell_obj, temp_min=None, temp_max=None, temp_levels=50, 
                 update_interval=16, frame_skip=1):
        """
        Initialize live thermal visualizer.
        
        Parameters:
        -----------
        cell_obj : Cell object
            PyECN cell object with thermal data
        temp_min : float, optional
            Minimum temperature for colormap (°C). Auto-detected if None.
        temp_max : float, optional
            Maximum temperature for colormap (°C). Auto-detected if None.
        temp_levels : int
            Number of temperature contour levels
        update_interval : int
            Update interval in milliseconds
        frame_skip : int
            Number of timesteps to skip between frames (1 = show all)
        """
        self.cell = cell_obj
        self.temp_levels = temp_levels
        self.update_interval = update_interval
        self.frame_skip = frame_skip
        
        # Animation state
        self.current_frame = 0
        self.is_paused = False
        self.is_running = True
        
        # Validate cell type
        if not hasattr(cell_obj, 'status_FormFactor'):
            raise ValueError("Cell object must have status_FormFactor attribute")
        
        # Auto-detect temperature range if not provided
        if hasattr(cell_obj, 'T_record') and cell_obj.T_record is not None:
            if temp_min is None:
                temp_min = np.min(cell_obj.T_record) - 273.15
            if temp_max is None:
                temp_max = np.max(cell_obj.T_record) - 273.15
        else:
            temp_min = temp_min or 20.0
            temp_max = temp_max or 60.0
        
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.climit_vector = np.linspace(temp_min, temp_max, temp_levels)
        
        # Setup visualization based on form factor
        self._setup_geometry()
        self._create_figure()
        
    def _setup_geometry(self):
        """Extract electrode geometry for visualization."""
        self.form_factor = self.cell.status_FormFactor
        
        if self.form_factor == 'Cylindrical':
            self._setup_cylindrical_geometry()
        elif self.form_factor == 'Pouch':
            self._setup_pouch_geometry()
        elif self.form_factor == 'Prismatic':
            self._setup_prismatic_geometry()
        else:
            raise ValueError(f"Unsupported form factor: {self.form_factor}")
    
    def _setup_cylindrical_geometry(self):
        """Setup geometry for cylindrical cells (unrolled jellyroll)."""
        n_v = self.cell.ny
        n_h_Al = int(np.size(self.cell.Al_4T) / n_v)
        
        # Extract node indices
        self.ind0_Al_4T = self.cell.Al_4T.reshape(n_v, n_h_Al)
        self.ind0_Cu_4T = self.cell.Cu_4T.reshape(n_v, n_h_Al)
        self.ind0_Elb_4T = self.cell.Elb_4T.reshape(n_v, int(np.size(self.cell.Elb_4T) / n_v))
        
        if self.cell.nstack > 1:
            self.ind0_Elr_4T = self.cell.Elr_4T.reshape(n_v, int(np.size(self.cell.Elr_4T) / n_v))
        
        # Extract coordinates
        self.array_h_Al_4T = self.cell.xi_4T[self.ind0_Al_4T]
        self.array_v_Al_4T = (self.cell.LG_Jellyroll - self.cell.yi_4T[self.ind0_Al_4T])
        
        self.array_h_Cu_4T = self.cell.xi_4T[self.ind0_Cu_4T]
        self.array_v_Cu_4T = (self.cell.LG_Jellyroll - self.cell.yi_4T[self.ind0_Cu_4T])
        
        self.array_h_Elb_4T = self.cell.xi_4T[self.ind0_Elb_4T]
        self.array_v_Elb_4T = (self.cell.LG_Jellyroll - self.cell.yi_4T[self.ind0_Elb_4T])
        
        if self.cell.nstack > 1:
            self.array_h_Elr_4T = self.cell.xi_4T[self.ind0_Elr_4T]
            self.array_v_Elr_4T = (self.cell.LG_Jellyroll - self.cell.yi_4T[self.ind0_Elr_4T])
        
        # Add separator column for visualization
        array_h_Sep_4T = (self.array_h_Al_4T[:, 0] - self.cell.b01).reshape(-1, 1)
        array_v_Sep = self.array_v_Al_4T[:, 0].reshape(-1, 1)
        self.array_h_SepAl_4T = np.append(array_h_Sep_4T, self.array_h_Al_4T, axis=1)
        self.array_v_SepAl_4T = np.append(array_v_Sep, self.array_v_Al_4T, axis=1)
        
        # Spiral scaling factor
        if hasattr(self.cell, 'Spiral_Sep_s_real') and hasattr(self.cell, 'Spiral_Sep_s'):
            self.spiral_scale = self.cell.Spiral_Sep_s_real / self.cell.Spiral_Sep_s
        else:
            self.spiral_scale = 1.0
        
        self.geometry_type = 'cylindrical'
        self.num_panels = 4 if self.cell.nstack > 1 else 2
        
    def _setup_pouch_geometry(self):
        """Setup geometry for pouch cells."""
        # For pouch cells, create a pseudo-unrolled view (thickness vs width)
        n_v = self.cell.ny  # Width direction
        n_h = self.cell.nz  # Thickness direction
        
        # Extract node indices for different materials
        self.ind0_Al_4T = self.cell.Al.reshape(-1, self.cell.nx)[:, 0].reshape(n_v, -1)
        self.ind0_Cu_4T = self.cell.Cu.reshape(-1, self.cell.nx)[:, 0].reshape(n_v, -1)
        self.ind0_Elb_4T = self.cell.Elb.reshape(-1, self.cell.nx)[:, 0].reshape(n_v, -1)
        
        if self.cell.nstack > 1:
            self.ind0_Elr_4T = self.cell.Elr.reshape(-1, self.cell.nx)[:, 0].reshape(n_v, -1)
        
        # Extract coordinates (thickness vs width cross-section at x=0)
        self.array_h_Al_4T = self.cell.zi[self.ind0_Al_4T]  # Thickness
        self.array_v_Al_4T = self.cell.yi[self.ind0_Al_4T]  # Width
        
        self.array_h_Cu_4T = self.cell.zi[self.ind0_Cu_4T]
        self.array_v_Cu_4T = self.cell.yi[self.ind0_Cu_4T]
        
        self.array_h_Elb_4T = self.cell.zi[self.ind0_Elb_4T]
        self.array_v_Elb_4T = self.cell.yi[self.ind0_Elb_4T]
        
        if self.cell.nstack > 1:
            self.array_h_Elr_4T = self.cell.zi[self.ind0_Elr_4T]
            self.array_v_Elr_4T = self.cell.yi[self.ind0_Elr_4T]
        
        self.spiral_scale = 1.0  # No spiral scaling for pouch
        self.geometry_type = 'pouch'
        self.num_panels = 4 if self.cell.nstack > 1 else 2
        
    def _setup_prismatic_geometry(self):
        """Setup geometry for prismatic cells (hybrid spiral-stripe)."""
        # Similar to cylindrical but with stripe sections
        self._setup_cylindrical_geometry()
        self.geometry_type = 'prismatic'
    
    def _create_figure(self):
        """Create matplotlib figure with subplots and controls."""
        # Create figure with extra space for controls
        self.fig = plt.figure(figsize=(18, 10))
        gs = GridSpec(4, 3, figure=self.fig, 
                     height_ratios=[0.4, 3, 3, 0.4],
                     width_ratios=[1, 1, 0.1],
                     hspace=0.3, wspace=0.3)
        
        # Title area
        self.ax_title = self.fig.add_subplot(gs[0, :2])
        self.ax_title.axis('off')
        self.title_text = self.ax_title.text(0.5, 0.5, '', 
                                              ha='center', va='center',
                                              fontsize=16, fontweight='bold')
        
        # Main plots
        if self.num_panels == 2:
            self.ax_al = self.fig.add_subplot(gs[1, 0])
            self.ax_cu = self.fig.add_subplot(gs[1, 1])
            self.axes = [self.ax_al, self.ax_cu]
        else:
            self.ax_al = self.fig.add_subplot(gs[1, 0])
            self.ax_cu = self.fig.add_subplot(gs[1, 1])
            self.ax_elb = self.fig.add_subplot(gs[2, 0])
            self.ax_elr = self.fig.add_subplot(gs[2, 1])
            self.axes = [self.ax_al, self.ax_cu, self.ax_elb, self.ax_elr]
        
        # Colorbar
        self.cbar_ax = self.fig.add_subplot(gs[1:3, 2])
        
        # Statistics panel
        self.ax_stats = self.fig.add_subplot(gs[3, :2])
        self.ax_stats.axis('off')
        self.stats_text = self.ax_stats.text(0.05, 0.5, '', 
                                             ha='left', va='center',
                                             fontsize=10, family='monospace')
        
        # Control buttons
        button_width = 0.08
        button_height = 0.04
        button_y = 0.02
        
        ax_play = plt.axes([0.35, button_y, button_width, button_height])
        ax_pause = plt.axes([0.44, button_y, button_width, button_height])
        ax_step = plt.axes([0.53, button_y, button_width, button_height])
        ax_reset = plt.axes([0.62, button_y, button_width, button_height])
        
        self.btn_play = Button(ax_play, 'Play')
        self.btn_pause = Button(ax_pause, 'Pause')
        self.btn_step = Button(ax_step, 'Step')
        self.btn_reset = Button(ax_reset, 'Reset')
        
        self.btn_play.on_clicked(self._on_play)
        self.btn_pause.on_clicked(self._on_pause)
        self.btn_step.on_clicked(self._on_step)
        self.btn_reset.on_clicked(self._on_reset)

        # Frame speed slider (controls frame skip)
        ax_speed = plt.axes([0.10, button_y, 0.20, button_height])
        self.speed_steps = [1, 2, 5, 10, 20, 60, 100]
        initial_speed = self.frame_skip if self.frame_skip in self.speed_steps else 1
        self.slider_speed = Slider(
            ax=ax_speed,
            label='Speed (x)',
            valmin=self.speed_steps[0],
            valmax=self.speed_steps[-1],
            valinit=initial_speed,
            valstep=self.speed_steps
        )
        self.slider_speed.on_changed(self._on_speed_change)
        
        # Initialize contour plots
        self._init_plots()
    
    def _init_plots(self):
        """Initialize contour plots with first frame."""
        time_step = 0
        
        # Extract temperature data
        if self.geometry_type in ['cylindrical', 'prismatic']:
            # Al current collector
            array_c_T_Al = self.cell.T_record[:, time_step][self.ind0_Al_4T] - 273.15
            
            # Add separator for Al plot
            if hasattr(self.cell, 'ind0_Geo_core_AddSep_4T_4SepFill'):
                array_c_Sep = self.cell.T_record[:, time_step].reshape(-1, 1)[
                    self.cell.ind0_Geo_core_AddSep_4T_4SepFill] - 273.15
                array_c_SepAl = np.append(array_c_Sep, array_c_T_Al, axis=1)
            else:
                array_c_SepAl = array_c_T_Al
                
            X_al = (self.array_h_SepAl_4T * self.spiral_scale) if hasattr(self, 'array_h_SepAl_4T') else (self.array_h_Al_4T * self.spiral_scale)
            Y_al = self.array_v_SepAl_4T if hasattr(self, 'array_v_SepAl_4T') else self.array_v_Al_4T
            C_al = array_c_SepAl if hasattr(self, 'array_h_SepAl_4T') else array_c_T_Al
            self.mesh_al = self.ax_al.pcolormesh(
                X_al, Y_al, C_al, cmap="RdBu_r", shading="gouraud",
                vmin=self.temp_min, vmax=self.temp_max
            )
            self.ax_al.set_title('Al Current Collector', fontsize=12, fontweight='bold')
            self.ax_al.set_xlabel('Unrolled Distance (m)', fontsize=10)
            self.ax_al.set_ylabel('Axial Position (m)', fontsize=10)
            
            # Cu current collector
            array_c_T_Cu = self.cell.T_record[:, time_step][self.ind0_Cu_4T] - 273.15
            self.mesh_cu = self.ax_cu.pcolormesh(
                self.array_h_Cu_4T * self.spiral_scale, self.array_v_Cu_4T,
                array_c_T_Cu, cmap="RdBu_r", shading="gouraud",
                vmin=self.temp_min, vmax=self.temp_max
            )
            self.ax_cu.set_title('Cu Current Collector', fontsize=12, fontweight='bold')
            self.ax_cu.set_xlabel('Unrolled Distance (m)', fontsize=10)
            self.ax_cu.set_ylabel('Axial Position (m)', fontsize=10)
            
            # Electrodes if nstack > 1
            if self.num_panels == 4:
                array_c_T_Elb = self.cell.T_record[:, time_step][self.ind0_Elb_4T] - 273.15
                self.mesh_elb = self.ax_elb.pcolormesh(
                    self.array_h_Elb_4T * self.spiral_scale, self.array_v_Elb_4T,
                    array_c_T_Elb, cmap="RdBu_r", shading="gouraud",
                    vmin=self.temp_min, vmax=self.temp_max
                )
                self.ax_elb.set_title('Cathode Layer', fontsize=12, fontweight='bold')
                self.ax_elb.set_xlabel('Unrolled Distance (m)', fontsize=10)
                self.ax_elb.set_ylabel('Axial Position (m)', fontsize=10)
                
                array_c_T_Elr = self.cell.T_record[:, time_step][self.ind0_Elr_4T] - 273.15
                self.mesh_elr = self.ax_elr.pcolormesh(
                    self.array_h_Elr_4T * self.spiral_scale, self.array_v_Elr_4T,
                    array_c_T_Elr, cmap="RdBu_r", shading="gouraud",
                    vmin=self.temp_min, vmax=self.temp_max
                )
                self.ax_elr.set_title('Anode Layer', fontsize=12, fontweight='bold')
                self.ax_elr.set_xlabel('Unrolled Distance (m)', fontsize=10)
                self.ax_elr.set_ylabel('Axial Position (m)', fontsize=10)
        
        elif self.geometry_type == 'pouch':
            # Similar structure for pouch cells
            array_c_T_Al = self.cell.T_record[:, time_step][self.ind0_Al_4T] - 273.15
            self.mesh_al = self.ax_al.pcolormesh(
                self.array_h_Al_4T, self.array_v_Al_4T,
                array_c_T_Al, cmap="RdBu_r", shading="gouraud",
                vmin=self.temp_min, vmax=self.temp_max
            )
            self.ax_al.set_title('Al Current Collector', fontsize=12, fontweight='bold')
            self.ax_al.set_xlabel('Thickness (m)', fontsize=10)
            self.ax_al.set_ylabel('Width (m)', fontsize=10)
            
            array_c_T_Cu = self.cell.T_record[:, time_step][self.ind0_Cu_4T] - 273.15
            self.mesh_cu = self.ax_cu.pcolormesh(
                self.array_h_Cu_4T, self.array_v_Cu_4T,
                array_c_T_Cu, cmap="RdBu_r", shading="gouraud",
                vmin=self.temp_min, vmax=self.temp_max
            )
            self.ax_cu.set_title('Cu Current Collector', fontsize=12, fontweight='bold')
            self.ax_cu.set_xlabel('Thickness (m)', fontsize=10)
            self.ax_cu.set_ylabel('Width (m)', fontsize=10)
            
            if self.num_panels == 4:
                array_c_T_Elb = self.cell.T_record[:, time_step][self.ind0_Elb_4T] - 273.15
                self.mesh_elb = self.ax_elb.pcolormesh(
                    self.array_h_Elb_4T, self.array_v_Elb_4T,
                    array_c_T_Elb, cmap="RdBu_r", shading="gouraud",
                    vmin=self.temp_min, vmax=self.temp_max
                )
                self.ax_elb.set_title('Cathode Layer', fontsize=12, fontweight='bold')
                self.ax_elb.set_xlabel('Thickness (m)', fontsize=10)
                self.ax_elb.set_ylabel('Width (m)', fontsize=10)
                
                array_c_T_Elr = self.cell.T_record[:, time_step][self.ind0_Elr_4T] - 273.15
                self.mesh_elr = self.ax_elr.pcolormesh(
                    self.array_h_Elr_4T, self.array_v_Elr_4T,
                    array_c_T_Elr, cmap="RdBu_r", shading="gouraud",
                    vmin=self.temp_min, vmax=self.temp_max
                )
                self.ax_elr.set_title('Anode Layer', fontsize=12, fontweight='bold')
                self.ax_elr.set_xlabel('Thickness (m)', fontsize=10)
                self.ax_elr.set_ylabel('Width (m)', fontsize=10)
        
        # Add colorbar
        plt.colorbar(self.mesh_cu, cax=self.cbar_ax, label='Temperature (°C)')
    
    def _update_frame(self, frame):
        """Update visualization for given frame."""
        if not self.is_running:
            return
        
        if self.is_paused:
            return
        
        # Calculate actual timestep
        time_step = min(self.current_frame * self.frame_skip, self.cell.nt - 1)
        
        # Extract temperature data for this timestep
        if self.geometry_type in ['cylindrical', 'prismatic']:
            array_c_T_Al = self.cell.T_record[:, time_step][self.ind0_Al_4T] - 273.15
            
            # Add separator for Al plot
            if hasattr(self.cell, 'ind0_Geo_core_AddSep_4T_4SepFill'):
                array_c_Sep = self.cell.T_record[:, time_step].reshape(-1, 1)[
                    self.cell.ind0_Geo_core_AddSep_4T_4SepFill] - 273.15
                array_c_SepAl = np.append(array_c_Sep, array_c_T_Al, axis=1)
            else:
                array_c_SepAl = array_c_T_Al
            
            C_al = array_c_SepAl if hasattr(self, 'array_h_SepAl_4T') else array_c_T_Al
            self.mesh_al.set_array(C_al.ravel())
            
            array_c_T_Cu = self.cell.T_record[:, time_step][self.ind0_Cu_4T] - 273.15
            self.mesh_cu.set_array(array_c_T_Cu.ravel())
            
            if self.num_panels == 4:
                array_c_T_Elb = self.cell.T_record[:, time_step][self.ind0_Elb_4T] - 273.15
                self.mesh_elb.set_array(array_c_T_Elb.ravel())
                
                array_c_T_Elr = self.cell.T_record[:, time_step][self.ind0_Elr_4T] - 273.15
                self.mesh_elr.set_array(array_c_T_Elr.ravel())
        
        elif self.geometry_type == 'pouch':
            array_c_T_Al = self.cell.T_record[:, time_step][self.ind0_Al_4T] - 273.15
            self.mesh_al.set_array(array_c_T_Al.ravel())
            
            array_c_T_Cu = self.cell.T_record[:, time_step][self.ind0_Cu_4T] - 273.15
            self.mesh_cu.set_array(array_c_T_Cu.ravel())
            
            if self.num_panels == 4:
                array_c_T_Elb = self.cell.T_record[:, time_step][self.ind0_Elb_4T] - 273.15
                self.mesh_elb.set_array(array_c_T_Elb.ravel())
                
                array_c_T_Elr = self.cell.T_record[:, time_step][self.ind0_Elr_4T] - 273.15
                self.mesh_elr.set_array(array_c_T_Elr.ravel())
        
        # Update title and statistics
        time_val = time_step * self.cell.dt if hasattr(self.cell, 'dt') else time_step
        cell_name = self.cell.status_Cells_name[0] if hasattr(self.cell, 'status_Cells_name') else "Cell"
        
        self.title_text.set_text(
            f'Live Thermal Visualization - {cell_name} ({self.form_factor})\n'
            f'Time: {time_val:.1f}s | Frame: {time_step}/{self.cell.nt-1}'
        )
        
        # Calculate and display statistics
        T_current = self.cell.T_record[:, time_step] - 273.15
        T_avg = np.mean(T_current)
        T_min = np.min(T_current)
        T_max = np.max(T_current)
        T_std = np.std(T_current)
        
        stats_str = (
            f'T_avg: {T_avg:6.2f}°C  |  '
            f'T_min: {T_min:6.2f}°C  |  '
            f'T_max: {T_max:6.2f}°C  |  '
            f'ΔT: {T_max-T_min:5.2f}°C  |  '
            f'σ: {T_std:5.2f}°C'
        )
        self.stats_text.set_text(stats_str)
        
        # Increment frame counter
        if not self.is_paused:
            self.current_frame += 1
            if self.current_frame * self.frame_skip >= self.cell.nt:
                self.is_paused = True
                print("\n✓ Animation complete!")
    
    def _on_play(self, event):
        """Resume animation."""
        self.is_paused = False
        print("▶ Playing...")
    
    def _on_pause(self, event):
        """Pause animation."""
        self.is_paused = True
        print("⏸ Paused")
    
    def _on_step(self, event):
        """Step forward one frame."""
        if self.current_frame * self.frame_skip < self.cell.nt - 1:
            self._update_frame(self.current_frame)
            print(f"⏭ Step to frame {self.current_frame * self.frame_skip}")
    
    def _on_reset(self, event):
        """Reset to first frame."""
        self.current_frame = 0
        self.is_paused = True
        self._update_frame(0)
        print("⏮ Reset to start")

    def _on_speed_change(self, value):
        """Update playback speed by changing frame skip."""
        new_skip = int(value)
        if new_skip < 1:
            new_skip = 1
        if new_skip != self.frame_skip:
            self.frame_skip = new_skip
            # Ensure current frame maps to valid timestep
            if self.current_frame * self.frame_skip >= self.cell.nt:
                self.current_frame = max(0, (self.cell.nt - 1) // self.frame_skip)
            print(f"⏩ Speed set to {self.frame_skip}x")
    
    def show(self):
        """Start live visualization."""
        print("\n" + "="*70)
        print("LIVE THERMAL VISUALIZATION")
        print("="*70)
        print(f"Form Factor: {self.form_factor}")
        print(f"Total Frames: {self.cell.nt}")
        print(f"Frame Skip: {self.frame_skip}")
        print(f"Update Interval: {self.update_interval}ms")
        print(f"Temperature Range: {self.temp_min:.1f}°C to {self.temp_max:.1f}°C")
        print("\nControls: Play | Pause | Step | Reset")
        print("="*70 + "\n")
        
        # Create animation
        self.anim = FuncAnimation(
            self.fig, self._update_frame,
            frames=None,
            interval=self.update_interval,
            blit=False,
            repeat=False
        )
        
        plt.show()
    
    def save_animation(self, filename='thermal_animation.mp4', fps=10, dpi=150):
        """Save animation to video file."""
        print(f"\n💾 Saving animation to {filename}...")
        
        # Create animation
        total_frames = int(np.ceil(self.cell.nt / self.frame_skip))
        
        def animate_frame(i):
            self.current_frame = i
            self._update_frame(i)
            return self.axes
        
        anim = FuncAnimation(
            self.fig, animate_frame,
            frames=total_frames,
            interval=1000/fps,
            blit=False
        )
        
        # Save with ffmpeg or pillow writer
        try:
            from matplotlib.animation import FFMpegWriter
            writer = FFMpegWriter(fps=fps, bitrate=5000)
            anim.save(filename, writer=writer, dpi=dpi)
            print(f"✓ Saved: {filename}")
        except Exception as e:
            print(f"❌ Error saving animation: {e}")
            print("   Install ffmpeg for video export")


def run_live_visualization(config_path=None, temp_min=None, temp_max=None, 
                           frame_skip=1, update_interval=16):
    """
    Run PyECN simulation with live thermal visualization.
    
    Parameters:
    -----------
    config_path : str or Path, optional
        Path to TOML config file
    temp_min : float, optional
        Minimum temperature for colormap (°C)
    temp_max : float, optional
        Maximum temperature for colormap (°C)
    frame_skip : int
        Number of timesteps to skip between frames
    update_interval : int
        Update interval in milliseconds
    """
    import os
    import sys
    from pathlib import Path
    
    # Set up paths
    if config_path is None:
        config_path = Path(__file__).parent.parent / "Examples" / "cylindrical_tabless_Fig_3.toml"
    config_path = Path(config_path)
    
    # Change to project root
    pyecn_root = Path(__file__).parent.parent
    project_root = pyecn_root.parent
    original_dir = os.getcwd()
    os.chdir(project_root)
    
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Set sys.argv[1] BEFORE importing
    if len(sys.argv) == 1:
        sys.argv.append(str(config_path))
    else:
        sys.argv[1] = str(config_path)
    
    try:
        # Import and run simulation
        import pyecn
        import pyecn.parse_inputs as ip
        
        print("Running PyECN simulation...")
        pyecn.run()
        
        # Get cell object from pyecn module's globals (where they were created)
        # Cell objects are created dynamically as globals in pyecn.__init__
        import pyecn as pyecn_module
        cell_name = ip.status_Cells_name[0]  # Get the first cell name (e.g., 'cell_1')
        
        # Access the cell from pyecn module's namespace
        if hasattr(pyecn_module, cell_name):
            cell_obj = getattr(pyecn_module, cell_name)
        else:
            # Fallback: try to get from pyecn.__init__ module directly
            import pyecn.__init__ as pyecn_init
            cell_obj = getattr(pyecn_init, cell_name, None)
            if cell_obj is None:
                raise RuntimeError(f"Could not find cell object '{cell_name}' after simulation")
        
        print(f"Found cell object: {cell_name}")
        
        # Create and show live visualization
        viz = LiveThermalVisualizer(
            cell_obj,
            temp_min=temp_min,
            temp_max=temp_max,
            update_interval=update_interval,
            frame_skip=frame_skip
        )
        viz.show()
        
    finally:
        os.chdir(original_dir)


if __name__ == '__main__':
    import sys
    
    # Parse command line arguments
    config_file = sys.argv[1] if len(sys.argv) > 1 else None
    frame_skip = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    run_live_visualization(config_file, frame_skip=frame_skip)
