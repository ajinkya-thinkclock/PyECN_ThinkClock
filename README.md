# PyECN (ThinkClock fork)

PyECN is a Python-based equivalent circuit network (ECN) framework for modeling lithium-ion batteries. This repository (PyECN_ThinkClock) tracks the upstream solver and adds live visualization and browser tooling.

## What's in this fork

- Live browser visualization API (FastAPI) and React/Vite frontend (see `web/`)
- Dash-based live visualization UI (`pyecn.browser_visualizations.live_dash`)
- Live thermal visualizer for unrolled electrodes (`pyecn/visualization_modules/viz_live_thermal.py`)
- Extra example configs in `pyecn/Examples/`

## Installation

Python 3.10 is required.

<details>
  <summary>Linux/macOS</summary>

  1. Clone the repository and enter the directory:
  ```bash
  git clone https://github.com/ImperialCollegeLondon/PyECN.git
  cd PyECN
  ```

  2. Create and activate a virtual environment:
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  ```

  3. Install the dependencies:
  ```bash
  pip install -U pip
  pip install -r requirements.txt
  ```
</details>

<details>
  <summary>Windows</summary>

  1. Clone the repository and enter the directory:
  ```bat
  git clone https://github.com/ImperialCollegeLondon/PyECN.git
  cd PyECN
  ```

  2. Create and activate a virtual environment:
  ```bat
  python -m venv .venv
  .venv\Scripts\activate.bat
  ```

  3. Install the dependencies:
  ```bat
  pip install -U pip
  pip install -r requirements.txt
  ```
</details>

Optional developer tools:

```bash
pip install -r dev-requirements.txt
```

For the React/Vite frontend, install Node dependencies in `web/` (see Live browser visualization below).

## Running PyECN

Config paths are resolved relative to the `pyecn/` folder.

```bash
# Run with a default config
python -m pyecn pouch.toml

# Run with another built-in config
python -m pyecn cylindrical.toml

# Run with an example config
python -m pyecn Examples/pouch_Fig_4a.toml
```

PyECN can also be run in an interactive Python session:

```bash
python
>>> import pyecn
>>> pyecn.run()
Enter config file name:
pouch.toml
```

## Visualization modules

PyECN_ThinkClock includes comprehensive visualization modules for post-processing simulation results.

### Individual visualization modules

- **Temperature visualizations** (`pyecn/visualization_modules/viz_temperature.py`)
  - Average temperature vs time
  - Min/max temperature vs time
  - Temperature delta and standard deviation
  - Unrolled jellyroll temperature distribution (cylindrical cells)
  - Combined temperature plots

- **Time-series visualizations** (`pyecn/visualization_modules/viz_timeseries.py`)
  - Cell voltage vs time
  - Current vs time
  - State of Charge (SoC) vs time
  - Combined time-series plots

- **2D spatial visualizations** (`pyecn/visualization_modules/viz_spatial_2d.py`)
  - Temperature 2D heatmaps
  - SoC 2D spatial distribution
  - Voltage 2D heatmaps
  - Current density 2D heatmaps
  - Combined spatial plots

### Live thermal visualization

Real-time thermal visualization for unrolled electrodes:

```bash
python pyecn/visualization_modules/viz_live_thermal.py
python pyecn/visualization_modules/viz_live_thermal.py Examples/pouch_Fig_4a.toml
```

See `pyecn/visualization_modules/README_live_thermal.md` for full usage.

### Complete visualization suite

Run all visualizations from a single simulation using `viz_all.py`:

```bash
# Default configuration
python pyecn/visualization_modules/viz_all.py

# Custom configuration file
python pyecn/visualization_modules/viz_all.py config.toml

# Custom configuration and output directory
python pyecn/visualization_modules/viz_all.py config.toml output_folder
```

This generates 15+ visualization files including:
- 6 temperature plots
- 4 time-series plots
- 5 spatial 2D heatmaps

**Features:**
- Single simulation run for all visualizations
- Automatic extraction of parameters from TOML config files
- Customizable output directories
- Matplotlib-based plots (PNG export, 300 DPI)
- Mayavi 3D visualization support for spatial temperature distributions

## Live browser visualization (ThinkClock)

### Option A: FastAPI + React/Vite frontend (recommended)

1. Start the API server (from the repo root):

```bash
python -m pyecn.browser_visualizations.live_api
```

The API runs at http://127.0.0.1:8000.

2. Start the Vite dev server (from the repo root):

```bash
cd web
npm install
npm run dev
```

The frontend proxies `/api` to http://127.0.0.1:8000. See `web/README.md` for details.

### Option B: Dash-only UI

```bash
python -m pyecn.browser_visualizations.live_dash
```

Open http://127.0.0.1:8050/ in your browser.

## Troubleshooting

- Config file not found or prompt repeats: run from the repo root and pass a path relative to [pyecn/](pyecn/), for example `python -m pyecn Examples/pouch_Fig_4a.toml`.
- Mayavi or VTK import errors, or blank 3D figures: reinstall `mayavi`, `vtk`, and `pyqt5` in the same environment; for headless runs, disable Mayavi plots in the config.
- Live API shows no plots: ensure a simulation finished or results were loaded; check `http://127.0.0.1:8000/api/health` and the run log in the UI.
- Frontend cannot reach the API: start `python -m pyecn.browser_visualizations.live_api` and confirm port 8000; if you changed it, update the dev proxy in [web/vite.config.js](web/vite.config.js).
- Video export fails in live thermal: install ffmpeg and see [pyecn/visualization_modules/README_live_thermal.md](pyecn/visualization_modules/README_live_thermal.md).

## Citing PyECN

If you use PyECN in your work, please cite:

Li, S., Rawat, S. K., Zhu, T., Offer, G. J., & Marinescu, M. (2023). Python-based Equivalent Circuit Network (PyECN) Modelling Framework for Lithium-ion Batteries. Engineering Archive. DOI: 10.31224/2972.

You can also use the machine-readable metadata in `CITATION.cff`.

BibTeX:

```
@article{pyecn2023,
  title = {Python-based Equivalent Circuit Network (PyECN) Modelling Framework for Lithium-ion Batteries},
  author = {Li, Shen and Rawat, Sunil Kumar and Zhu, Tao and Offer, Gregory J and Marinescu, Monica},
  year = {2023},
  publisher = {Engineering Archive},
  doi = {10.31224/2972}
}
```

## Contributing to PyECN

Contributions are welcome. Please open an issue or pull request with a clear description of the change.

## License

PyECN is fully open source. For more information about its license, see `LICENSE.md`.

## Contributors

- Shen Li: Conceptualisation, methodology, creator and lead developer of PyECN, writing and review;
- Sunil Rawat: Contributor of PyECN, discussion, writing and review;
- Tao Zhu: Contributor of PyECN, discussion, writing and review;
- Gregory J Offer: Conceptualisation, funding acquisition, supervision, writing – review & editing;
- Monica Marinescu: Conceptualisation, funding acquisition, supervision, writing – review & editing;
