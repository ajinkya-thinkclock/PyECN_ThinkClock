# Live Thermal Visualization Module

Real-time visualization of thermal distribution in unrolled battery electrode structures during PyECN simulation.

## Features

- **Real-time Updates**: Watch temperature evolution as simulation progresses
- **Unrolled Electrode View**: See thermal distribution across the full electrode structure
- **Interactive Controls**: Play, pause, step, and reset animation
- **Multi-Form Factor Support**: Works with cylindrical, pouch, and prismatic cells
- **Temperature Statistics**: Live display of avg, min, max, ΔT, and standard deviation
- **Video Export**: Save animations as MP4 files (requires ffmpeg)

## Quick Start

### Basic Usage

```bash
# Run with default config (cylindrical cell)
python pyecn/visualization_modules/viz_live_thermal.py

# Run with custom config
python pyecn/visualization_modules/viz_live_thermal.py pyecn/Examples/pouch_Fig_4a.toml

# Skip frames for faster playback (show every 5th frame)
python pyecn/visualization_modules/viz_live_thermal.py pyecn/Examples/cylindrical_tabless_Fig_3.toml 5
```

### Python API

```python
from pyecn.visualization_modules.viz_live_thermal import LiveThermalVisualizer

# After running a PyECN simulation...
cell_obj = cell_1  # Your cell object

# Create visualizer
viz = LiveThermalVisualizer(
    cell_obj,
    temp_min=20.0,          # Min temperature for colormap (°C)
    temp_max=60.0,          # Max temperature for colormap (°C)
    temp_levels=50,         # Number of contour levels
    update_interval=100,    # Update every 100ms
    frame_skip=1            # Show every frame (1 = all, 5 = every 5th)
)

# Show live visualization
viz.show()

# Or save as video
viz.save_animation('output.mp4', fps=20, dpi=150)
```

## Visualization Layouts

### Cylindrical Cells (nstack > 1)
Shows 4 panels in 2x2 grid:
- Top-left: Al current collector (unrolled spiral)
- Top-right: Cu current collector (unrolled spiral)
- Bottom-left: Cathode layer
- Bottom-right: Anode layer

### Cylindrical Cells (nstack = 1)
Shows 2 panels:
- Left: Al current collector
- Right: Cu current collector

### Pouch Cells
Shows thickness vs width cross-section:
- Al and Cu current collectors
- Cathode and anode layers (if multi-layer)

### Prismatic Cells
Similar to cylindrical with hybrid spiral-stripe structure

## Controls

- **Play**: Resume animation
- **Pause**: Pause at current frame
- **Step**: Advance one frame (when paused)
- **Reset**: Return to first frame

## Display Information

### Title Area
- Cell name and form factor
- Current simulation time (seconds)
- Current frame / total frames

### Statistics Panel
Live updating values:
- `T_avg`: Average temperature across all nodes
- `T_min`: Minimum temperature
- `T_max`: Maximum temperature  
- `ΔT`: Temperature gradient (max - min)
- `σ`: Standard deviation

### Colorbar
- Blue: Cooler temperatures
- White: Mid-range
- Red: Hotter temperatures

## Performance Tips

1. **Frame Skip**: Use `frame_skip=5` or higher for long simulations
2. **Update Interval**: Increase to 200-500ms if visualization lags
3. **Temperature Range**: Auto-detection works well, but custom ranges can improve contrast
4. **Resolution**: Lower `temp_levels` (e.g., 25) for faster rendering

## Video Export

Requires ffmpeg installed:

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/
```

Then in Python:

```python
viz.save_animation(
    filename='thermal_evolution.mp4',
    fps=20,        # Frames per second
    dpi=150        # Resolution (higher = larger file)
)
```

## Integration with PyECN Workflow

The visualizer works seamlessly with PyECN's module construction system:

```python
# Standard PyECN workflow
import pyecn
pyecn.run()  # Runs simulation

# Access dynamically created cell objects
from pyecn.visualization_modules.viz_live_thermal import LiveThermalVisualizer

# Cells are named cell_1, cell_2, etc. based on status_Cells_name
viz = LiveThermalVisualizer(cell_1, frame_skip=10)
viz.show()
```

## Architecture Notes

### Geometry Extraction
- Uses cell object's `xi_4T`, `yi_4T`, `zi_4T` coordinate arrays
- Extracts material-specific nodes: `Al_4T`, `Cu_4T`, `Elb_4T`, `Elr_4T`
- Applies spiral scaling for cylindrical cells: `Spiral_Sep_s_real / Spiral_Sep_s`

### Temperature Data Access
- Reads from `cell_obj.T_record[:, timestep]` array
- Converts Kelvin to Celsius: `T - 273.15`
- Reshapes data to match 2D electrode geometry

### Real-time Updates
- Uses matplotlib's `FuncAnimation` for smooth updates
- Clears and redraws contours each frame (no blit mode due to contourf limitations)
- Updates title and statistics text on each frame

## Troubleshooting

**"Cell object must have status_FormFactor attribute"**
- Make sure you're passing a valid PyECN cell object (Core instance)

**"Unrolled jellyroll requires SepFill or SepAir core"**
- Some visualizations only work with specific thermal BC configurations
- Check `status_ThermalBC_Core` in your config

**Animation is slow/choppy**
- Increase `frame_skip` parameter
- Increase `update_interval` (more ms between frames)
- Reduce `temp_levels` for simpler contours

**Video export fails**
- Install ffmpeg: `pip install ffmpeg-python` and system ffmpeg
- Try reducing `dpi` parameter for smaller file size

## Examples

See `pyecn/examples/live_thermal_demo.py` for complete working examples.

## Related Modules

- `viz_temperature.py` - Static temperature plots including unrolled jellyroll snapshots
- `viz_spatial_2d.py` - 2D heatmaps at specific timesteps
- `viz_all.py` - Complete visualization suite (generates all plots from one simulation)
