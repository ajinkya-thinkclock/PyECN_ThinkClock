"""
Dash-based live visualization for PyECN simulations.

Features:
- Upload TOML config
- Optional upload of external current profile CSV
- Live playback controls
- Plotly plots: electrode temperature map, SoC, current density, 3D temperature,
  and min/max/avg temperature time series
- Heartbeat monitor to confirm UI is updating
"""

from __future__ import annotations

import base64
import importlib
import os
import sys
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional, Tuple

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import numpy as np
import plotly.graph_objects as go

try:
    import toml
except ImportError as exc:  # pragma: no cover
    raise ImportError("toml is required for browser visualizations") from exc

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pyecn.visualization_modules.viz_spatial_2d import extract_from_pyecn_cell

UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

LOG_MAX_LINES = 200
LOG_MAX_CHARS = 20000
LOG_LINES = deque(maxlen=LOG_MAX_LINES)
LOG_LOCK = threading.Lock()

SIM_STATE = {
    "cell": None,
    "time": None,
    "nt": 0,
    "running": False,
    "last_message": "Idle",
    "last_error": None,
    "results_path": None,
    "results_meta": {},
}

MAX_LENGTH = 100
TIMES = deque(maxlen=MAX_LENGTH)
HEARTBEAT_VALUES = deque(maxlen=MAX_LENGTH)
START_TIME = time.time()

app = dash.Dash(__name__)
app.config.suppress_callback_exceptions = True


def _save_upload(contents: str, filename: str) -> Path:
    _, encoded = contents.split(",", 1)
    data = base64.b64decode(encoded)
    path = UPLOAD_DIR / filename
    path.write_bytes(data)
    return path


def _prepare_config(config_path: Path, current_csv_path: Optional[Path]) -> Path:
    config = toml.load(config_path)
    if current_csv_path is not None:
        config.setdefault("operating_conditions", {})["I_ext_fpath"] = str(current_csv_path)
    temp_config = UPLOAD_DIR / f"run_{config_path.stem}.toml"
    with open(temp_config, "w", encoding="utf-8") as f:
        toml.dump(config, f)
    return temp_config


def _load_results(results_path: Path) -> None:
    data = np.load(results_path, allow_pickle=True)

    class _Cell:
        pass

    cell = _Cell()
    for key in data.files:
        setattr(cell, key, data[key])

    SIM_STATE["cell"] = cell
    SIM_STATE["nt"] = int(getattr(cell, "nt", 0))
    dt = float(getattr(cell, "dt", 1.0))
    SIM_STATE["time"] = np.arange(SIM_STATE["nt"]) * dt
    SIM_STATE["results_meta"] = {k: getattr(data[k], "shape", None) for k in data.files}
    if hasattr(cell, "T_record"):
        tmin = np.nanmin(cell.T_record)
        tmax = np.nanmax(cell.T_record)
        SIM_STATE["results_meta"]["T_record_minmax_K"] = (float(tmin), float(tmax))
    if not data.files:
        SIM_STATE["last_message"] = "Simulation finished but results file is empty."


def _run_pyecn_sim_async(config_path: Path) -> None:
    SIM_STATE["running"] = True
    SIM_STATE["last_error"] = None
    SIM_STATE["last_message"] = "Running PyECN simulation..."
    try:
        results_path = UPLOAD_DIR / f"results_{int(time.time())}.npz"
        SIM_STATE["results_path"] = str(results_path)

        cmd = [
            sys.executable,
            str(Path(__file__).parent / "run_pyecn_job.py"),
            str(config_path),
            str(results_path),
        ]
        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )

        assert process.stdout is not None
        for line in process.stdout:
            with LOG_LOCK:
                LOG_LINES.append(line.rstrip())

        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"PyECN process failed with code {return_code}")

        _load_results(results_path)
        SIM_STATE["last_message"] = f"Simulation complete. nt={SIM_STATE['nt']}"
    except Exception as exc:
        SIM_STATE["last_error"] = str(exc)
        SIM_STATE["last_message"] = f"Simulation error: {exc}"
    finally:
        SIM_STATE["running"] = False


def _get_temp3d_trace(cell, time_index: int, max_points: int = 8000) -> go.Scatter3d:
    T = cell.T_record[:, time_index] - 273.15
    if hasattr(cell, "xi_4T") and len(cell.xi_4T) == len(T):
        x = cell.xi_4T
        y = cell.yi_4T
        z = cell.zi_4T
    else:
        x = cell.xi
        y = cell.yi
        z = cell.zi

    n = len(T)
    if n > max_points:
        idx = np.linspace(0, n - 1, max_points).astype(int)
        x = x[idx]
        y = y[idx]
        z = z[idx]
        T = T[idx]

    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        marker=dict(size=2, color=T, colorscale="RdBu_r", colorbar=dict(title="°C")),
    )


def _get_electrode_temp_map(cell, time_index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if hasattr(cell, "Al_4T") and hasattr(cell, "xi_4T"):
        n_v = cell.ny
        n_h = int(np.size(cell.Al_4T) / n_v)
        ind0_Al_4T = cell.Al_4T.reshape(n_v, n_h)
        x = np.asarray(cell.xi_4T[ind0_Al_4T], dtype=float)
        lg = float(cell.LG_Jellyroll) if np.ndim(cell.LG_Jellyroll) == 0 else float(np.asarray(cell.LG_Jellyroll).reshape(-1)[0])
        y = (lg - np.asarray(cell.yi_4T[ind0_Al_4T], dtype=float))
        z = np.asarray(cell.T_record[:, time_index][ind0_Al_4T], dtype=float) - 273.15
        return x, y, z

    data = extract_from_pyecn_cell(cell, time_index=time_index)
    z = data.get("temp_2d")
    if z is None:
        z = np.zeros((cell.nx, cell.ny))
    x = np.arange(z.shape[1])
    y = np.arange(z.shape[0])
    X, Y = np.meshgrid(x, y)
    return X, Y, z


def _make_temp_map_fig(cell, time_index: int) -> go.Figure:
    if not hasattr(cell, "T_record"):
        fig = _empty_fig("Electrode Temperature Map")
        fig.add_annotation(text="No T_record in results", x=0.5, y=0.5, showarrow=False)
        return fig

    x, y, z = _get_electrode_temp_map(cell, time_index)
    z = np.asarray(z, dtype=float)
    if z.size == 0 or not np.isfinite(z).any():
        fig = _empty_fig("Electrode Temperature Map")
        fig.add_annotation(text="Temperature map has no finite values", x=0.5, y=0.5, showarrow=False)
        return fig

    x1 = x[0] if x.ndim == 2 else x
    y1 = y[:, 0] if y.ndim == 2 else y
    if x1.size == 0 or y1.size == 0:
        fig = _empty_fig("Electrode Temperature Map")
        fig.add_annotation(text="Temperature map has empty axes", x=0.5, y=0.5, showarrow=False)
        return fig

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x1,
            y=y1,
            colorscale="RdBu_r",
            colorbar=dict(title="°C"),
        )
    )
    fig.update_layout(
        title="Electrode Temperature Map",
        xaxis_title="Unrolled/Width",
        yaxis_title="Axial/Height",
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def _make_current_density_fig(cell, time_index: int) -> go.Figure:
    required = ["I_ele_record", "Axy_ele", "List_node2ele_4T", "Elb_4T", "xi_4T", "yi_4T"]
    if all(hasattr(cell, k) for k in required):
        try:
            n_v = cell.ny
            n_h = int(np.size(cell.Elb_4T) / n_v)
            ind0_Elb_4T = cell.Elb_4T.reshape(n_v, n_h)
            ind0_ele_Elb_4T = cell.List_node2ele_4T[ind0_Elb_4T, 0]

            scalefactor_z = getattr(cell, "scalefactor_z", 1.0)
            rouI = cell.I_ele_record[:, time_index][ind0_ele_Elb_4T] / (
                cell.Axy_ele[ind0_ele_Elb_4T, 0] * scalefactor_z
            )
            rouI = np.asarray(rouI, dtype=float)
            if rouI.size == 0 or not np.isfinite(rouI).any():
                fig = _empty_fig("Current Density (Electrode)")
                fig.add_annotation(text="Current density has no finite values", x=0.5, y=0.5, showarrow=False)
                return fig

            array_h = np.asarray(cell.xi_4T[ind0_Elb_4T], dtype=float)
            if hasattr(cell, "LG_Jellyroll"):
                lg = float(cell.LG_Jellyroll) if np.ndim(cell.LG_Jellyroll) == 0 else float(np.asarray(cell.LG_Jellyroll).reshape(-1)[0])
                array_v = lg - np.asarray(cell.yi_4T[ind0_Elb_4T], dtype=float)
            else:
                array_v = np.asarray(cell.yi_4T[ind0_Elb_4T], dtype=float)

            if hasattr(cell, "Spiral_Sep_s_real") and hasattr(cell, "Spiral_Sep_s"):
                scale = cell.Spiral_Sep_s_real / cell.Spiral_Sep_s
                array_h = array_h * scale
            elif hasattr(cell, "SpiralandStripe_Sep_s_real"):
                array_h = array_h * cell.SpiralandStripe_Sep_s_real

            fig = go.Figure(
                data=go.Heatmap(
                    z=rouI,
                    x=array_h[0],
                    y=array_v[:, 0],
                    colorscale="Blues",
                    colorbar=dict(title="A/m²"),
                )
            )
            fig.update_layout(
                title="Current Density (Electrode)",
                xaxis_title="Unrolled Distance",
                yaxis_title="Axial Position",
                margin=dict(l=40, r=20, t=40, b=40),
            )
            return fig
        except Exception:
            pass

    if hasattr(cell, "I_record"):
        data = extract_from_pyecn_cell(cell, time_index=time_index)
        cur = data.get("current_2d")
        if cur is None:
            cur = np.zeros((cell.nx, cell.ny))
        fig = go.Figure(data=go.Heatmap(z=cur, colorscale="Viridis", colorbar=dict(title="A")))
        fig.update_layout(
            title="Current Density (2D Slice)",
            xaxis_title="X",
            yaxis_title="Y",
            margin=dict(l=40, r=20, t=40, b=40),
        )
        return fig

    fig = _empty_fig("Current Density (2D Slice)")
    fig.add_annotation(text="No current data in results", x=0.5, y=0.5, showarrow=False)
    return fig


def _make_voltage_fig(cell) -> go.Figure:
    if hasattr(cell, "U_pndiff_plot"):
        v = np.asarray(cell.U_pndiff_plot, dtype=float)
        time_vec = SIM_STATE["time"]
        v = v[: len(time_vec)]
        fig = go.Figure(data=go.Scatter(x=time_vec, y=v, mode="lines", name="Voltage"))
        fig.update_layout(title="Voltage vs Time", xaxis_title="Time (s)", yaxis_title="Voltage (V)")
        return fig

    if hasattr(cell, "V_record"):
        v = np.asarray(cell.V_record, dtype=float)
        time_vec = SIM_STATE["time"]
        if v.ndim == 2:
            v_mean = np.mean(v, axis=0)
            v_mean = v_mean[: len(time_vec)]
            fig = go.Figure(data=go.Scatter(x=time_vec, y=v_mean, mode="lines", name="Voltage"))
            fig.update_layout(title="Voltage vs Time (mean node voltage)", xaxis_title="Time (s)", yaxis_title="Voltage (V)")
            return fig

    fig = _empty_fig("Voltage vs Time")
    fig.add_annotation(text="No voltage data in results", x=0.5, y=0.5, showarrow=False)
    return fig


def _make_soc_heatmap_fig(cell, time_index: int) -> go.Figure:
    if not hasattr(cell, "SoC_ele_record"):
        fig = _empty_fig("SoC Heatmap (Electrode)")
        fig.add_annotation(text="No SoC_ele_record in results", x=0.5, y=0.5, showarrow=False)
        return fig

    required = ["SoC_ele_record", "List_node2ele_4T", "Elb_4T", "xi_4T", "yi_4T"]
    if not all(hasattr(cell, k) for k in required):
        fig = _empty_fig("SoC Heatmap (Electrode)")
        fig.add_annotation(text="Missing electrode mapping arrays", x=0.5, y=0.5, showarrow=False)
        return fig

    n_v = cell.ny
    n_h = int(np.size(cell.Elb_4T) / n_v)
    ind0_Elb_4T = cell.Elb_4T.reshape(n_v, n_h)
    ind0_ele_Elb_4T = cell.List_node2ele_4T[ind0_Elb_4T, 0]

    soc = cell.SoC_ele_record[:, time_index][ind0_ele_Elb_4T] * 100
    soc = np.asarray(soc, dtype=float)
    if soc.size == 0 or not np.isfinite(soc).any():
        fig = _empty_fig("SoC Heatmap (Electrode)")
        fig.add_annotation(text="SoC heatmap has no finite values", x=0.5, y=0.5, showarrow=False)
        return fig

    array_h = np.asarray(cell.xi_4T[ind0_Elb_4T], dtype=float)
    if hasattr(cell, "LG_Jellyroll"):
        lg = float(cell.LG_Jellyroll) if np.ndim(cell.LG_Jellyroll) == 0 else float(np.asarray(cell.LG_Jellyroll).reshape(-1)[0])
        array_v = lg - np.asarray(cell.yi_4T[ind0_Elb_4T], dtype=float)
    else:
        array_v = np.asarray(cell.yi_4T[ind0_Elb_4T], dtype=float)

    if hasattr(cell, "Spiral_Sep_s_real") and hasattr(cell, "Spiral_Sep_s"):
        array_h = array_h * (cell.Spiral_Sep_s_real / cell.Spiral_Sep_s)

    fig = go.Figure(
        data=go.Heatmap(
            z=soc,
            x=array_h[0],
            y=array_v[:, 0],
            colorscale="Viridis",
            colorbar=dict(title="SoC (%)"),
        )
    )
    fig.update_layout(title="SoC Heatmap (Electrode)", xaxis_title="Unrolled Distance", yaxis_title="Axial Position")
    return fig


def _make_heatgen_fig(cell) -> go.Figure:
    if not hasattr(cell, "q_4T_record") or not hasattr(cell, "V_stencil_4T_ALL"):
        fig = _empty_fig("Heat Generation vs Time")
        fig.add_annotation(text="No q_4T_record/V_stencil_4T_ALL in results", x=0.5, y=0.5, showarrow=False)
        return fig

    q = np.asarray(cell.q_4T_record, dtype=float)
    v = np.asarray(cell.V_stencil_4T_ALL, dtype=float).reshape(-1)
    ntotal_4T = int(getattr(cell, "ntotal_4T", len(v)))
    v = v[:ntotal_4T]
    q = q[:ntotal_4T, :]
    if q.size == 0 or not np.isfinite(q).any():
        fig = _empty_fig("Heat Generation vs Time")
        fig.add_annotation(text="Heat generation has no finite values", x=0.5, y=0.5, showarrow=False)
        return fig

    power = np.sum(q * v[:, None], axis=0)
    time_vec = SIM_STATE["time"]
    power = power[: len(time_vec)]
    fig = go.Figure(data=go.Scatter(x=time_vec, y=power, mode="lines", name="Heat Gen"))
    fig.update_layout(title="Heat Generation vs Time", xaxis_title="Time (s)", yaxis_title="W")
    return fig


def _make_soc_fig(cell) -> go.Figure:
    if hasattr(cell, "SoC_Cell_record"):
        soc = np.asarray(cell.SoC_Cell_record, dtype=float) * 100
        time_vec = SIM_STATE["time"]
    else:
        fig = _empty_fig("State of Charge")
        fig.add_annotation(text="No SoC_Cell_record in results", x=0.5, y=0.5, showarrow=False)
        return fig
    if soc.size == 0 or not np.isfinite(soc).any():
        fig = _empty_fig("State of Charge")
        fig.add_annotation(text="SoC series has no finite values", x=0.5, y=0.5, showarrow=False)
        return fig
    fig = go.Figure(data=go.Scatter(x=time_vec, y=soc, mode="lines", name="SoC"))
    fig.update_layout(title="State of Charge", xaxis_title="Time (s)", yaxis_title="SoC (%)")
    return fig


def _make_temp_stats_fig(cell) -> go.Figure:
    if not hasattr(cell, "T_record"):
        fig = _empty_fig("Temperature Min/Max/Avg")
        fig.add_annotation(text="No T_record in results", x=0.5, y=0.5, showarrow=False)
        return fig

    T = np.asarray(cell.T_record, dtype=float) - 273.15
    if T.size == 0 or not np.isfinite(T).any():
        fig = _empty_fig("Temperature Min/Max/Avg")
        fig.add_annotation(text="Temperature series has no finite values", x=0.5, y=0.5, showarrow=False)
        return fig

    t = SIM_STATE["time"]
    tmin = np.min(T, axis=0)
    tmax = np.max(T, axis=0)
    tavg = np.mean(T, axis=0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=tavg, name="Avg"))
    fig.add_trace(go.Scatter(x=t, y=tmin, name="Min"))
    fig.add_trace(go.Scatter(x=t, y=tmax, name="Max"))
    fig.update_layout(title="Temperature Min/Max/Avg", xaxis_title="Time (s)", yaxis_title="°C")
    return fig


def _make_temp3d_fig(cell, time_index: int) -> go.Figure:
    if not hasattr(cell, "T_record"):
        fig = _empty_fig("3D Temperature Map")
        fig.add_annotation(text="No T_record in results", x=0.5, y=0.5, showarrow=False)
        return fig

    T = np.asarray(cell.T_record, dtype=float)
    if T.size == 0 or not np.isfinite(T).any():
        fig = _empty_fig("3D Temperature Map")
        fig.add_annotation(text="Temperature data has no finite values", x=0.5, y=0.5, showarrow=False)
        return fig

    trace = _get_temp3d_trace(cell, time_index)
    fig = go.Figure(data=[trace])
    fig.update_layout(
        title="3D Temperature Map",
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
    )
    return fig


def _empty_fig(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis_title="",
        yaxis_title="",
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


app.layout = html.Div(
    [
        html.H1("PyECN Live Browser Visualization", style={"marginBottom": "6px"}),
        html.Div("Interactive battery simulation dashboard", style={"color": "#6b7280", "marginBottom": "16px"}),
        html.Div(
            [
                dcc.Upload(id="config-upload", children=html.Button("Upload Config TOML")),
                dcc.Upload(id="current-upload", children=html.Button("Upload Current CSV (optional)")),
                html.Button("Run Simulation", id="run-btn", n_clicks=0),
                dcc.Loading(
                    id="run-loading",
                    type="default",
                    children=html.Div(id="run-status", children="Idle", style={"marginTop": "10px"}),
                ),
            ],
            style={
                "display": "flex",
                "gap": "10px",
                "alignItems": "center",
                "background": "#ffffff",
                "padding": "12px",
                "borderRadius": "10px",
                "boxShadow": "0 2px 6px rgba(0,0,0,0.06)",
                "border": "1px solid #e5e7eb",
            },
        ),
        html.Div(
            [
                html.Div(id="heartbeat", children="UI heartbeat: 0", style={"color": "#6b7280"}),
                html.Div(id="run-log-title", children="Run Log", style={"fontWeight": "600", "marginTop": "8px"}),
                html.Pre(id="run-log", style={"whiteSpace": "pre-wrap", "background": "#f9fafb", "padding": "10px", "border": "1px solid #e5e7eb", "borderRadius": "8px"}),
                html.Div(id="results-meta-title", children="Results Keys", style={"fontWeight": "600", "marginTop": "8px"}),
                html.Pre(id="results-meta", style={"whiteSpace": "pre-wrap", "background": "#f9fafb", "padding": "10px", "border": "1px solid #e5e7eb", "borderRadius": "8px"}),
            ],
            style={"marginTop": "12px", "background": "#ffffff", "padding": "12px", "borderRadius": "10px", "border": "1px solid #e5e7eb"},
        ),
        dcc.Interval(id="heartbeat-interval", interval=1000, n_intervals=0),
        dcc.Interval(id="status-interval", interval=500, n_intervals=0),
        html.Hr(style={"margin": "20px 0"}),
        html.Div(
            [
                html.Button("Play", id="play-btn", n_clicks=0),
                html.Button("Pause", id="pause-btn", n_clicks=0),
                dcc.Slider(id="time-slider", min=0, max=0, step=1, value=0),
            ],
            style={
                "marginBottom": "20px",
                "background": "#ffffff",
                "padding": "12px",
                "borderRadius": "10px",
                "border": "1px solid #e5e7eb",
            },
        ),
        dcc.Interval(id="play-interval", interval=200, n_intervals=0, disabled=True),
        dcc.Store(id="play-state", data={"playing": False}),
        dcc.Store(id="sim-meta", data={"nt": 0}),
        html.Div(
            [
                dcc.Graph(id="temp-map"),
                dcc.Graph(id="soc-heatmap"),
                dcc.Graph(id="voltage-plot"),
                dcc.Graph(id="soc-plot"),
                dcc.Graph(id="current-density"),
                dcc.Graph(id="heatgen-plot"),
                dcc.Graph(id="temp-stats"),
                dcc.Graph(id="heartbeat-graph"),
            ],
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"},
        ),
    ],
    style={
        "padding": "24px",
        "background": "#f3f4f6",
        "fontFamily": "Segoe UI, Roboto, Arial, sans-serif",
        "color": "#111827",
    },
)


@app.callback(
    Output("run-status", "children"),
    Output("sim-meta", "data"),
    Output("time-slider", "max"),
    Input("run-btn", "n_clicks"),
    Input("status-interval", "n_intervals"),
    State("config-upload", "contents"),
    State("config-upload", "filename"),
    State("current-upload", "contents"),
    State("current-upload", "filename"),
)
def update_status(n_clicks, n_intervals, config_contents, config_filename, current_contents, current_filename):
    ctx = dash.callback_context
    triggered = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

    if triggered == "run-btn":
        if SIM_STATE["running"]:
            return "Simulation already running...", {"nt": SIM_STATE["nt"]}, max(0, SIM_STATE["nt"] - 1)
        if not config_contents or not config_filename:
            SIM_STATE["last_message"] = "Please upload a config TOML file."
            return SIM_STATE["last_message"], {"nt": 0}, 0

        config_path = _save_upload(config_contents, config_filename)
        current_path = None
        if current_contents and current_filename:
            current_path = _save_upload(current_contents, current_filename)

        try:
            temp_config = _prepare_config(config_path, current_path)
            thread = threading.Thread(target=_run_pyecn_sim_async, args=(temp_config,), daemon=True)
            thread.start()
            SIM_STATE["last_message"] = "Simulation started..."
        except Exception as exc:
            SIM_STATE["last_message"] = f"Simulation error: {exc}"

    nt = SIM_STATE.get("nt", 0)
    status = SIM_STATE.get("last_message", "Idle")
    if SIM_STATE.get("running"):
        status = "Running PyECN simulation..."

    return status, {"nt": nt}, max(0, nt - 1)


@app.callback(
    Output("results-meta", "children"),
    Input("status-interval", "n_intervals"),
)
def update_results_meta(_n_intervals):
    meta = SIM_STATE.get("results_meta", {})
    return "Results keys/shapes:\n" + "\n".join([f"- {k}: {v}" for k, v in meta.items()]) if meta else "Results keys/shapes: (none)"


@app.callback(
    Output("play-state", "data"),
    Output("play-interval", "disabled"),
    Input("play-btn", "n_clicks"),
    Input("pause-btn", "n_clicks"),
    State("play-state", "data"),
)
def toggle_play(play_clicks, pause_clicks, state):
    ctx = dash.callback_context
    if not ctx.triggered:
        return state, True
    btn_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if btn_id == "play-btn":
        return {"playing": True}, False
    if btn_id == "pause-btn":
        return {"playing": False}, True
    return state, True


@app.callback(
    Output("time-slider", "value"),
    Input("play-interval", "n_intervals"),
    State("play-state", "data"),
    State("time-slider", "value"),
    State("sim-meta", "data"),
)
def advance_time(n, play_state, current_value, sim_meta):
    if not play_state.get("playing"):
        return current_value
    nt = sim_meta.get("nt", 0)
    if nt == 0:
        return current_value
    next_value = current_value + 1
    if next_value >= nt:
        return nt - 1
    return next_value


@app.callback(
    Output("temp-map", "figure"),
    Output("soc-heatmap", "figure"),
    Output("voltage-plot", "figure"),
    Output("soc-plot", "figure"),
    Output("current-density", "figure"),
    Output("heatgen-plot", "figure"),
    Output("temp-stats", "figure"),
    Input("time-slider", "value"),
    Input("status-interval", "n_intervals"),
)
def update_plots(time_index, _n_intervals):
    cell = SIM_STATE.get("cell")
    if cell is None:
        return (
            _empty_fig("Electrode Temperature Map"),
            _empty_fig("SoC Heatmap (Electrode)"),
            _empty_fig("Voltage vs Time"),
            _empty_fig("State of Charge"),
            _empty_fig("Current Density (2D Slice)"),
            _empty_fig("Heat Generation vs Time"),
            _empty_fig("Temperature Min/Max/Avg"),
        )

    time_index = int(time_index or 0)
    if SIM_STATE.get("nt", 0) > 0:
        time_index = min(time_index, SIM_STATE["nt"] - 1)

    temp_map = _make_temp_map_fig(cell, time_index)
    soc_heatmap = _make_soc_heatmap_fig(cell, time_index)
    voltage_fig = _make_voltage_fig(cell)
    soc_fig = _make_soc_fig(cell)
    current_fig = _make_current_density_fig(cell, time_index)
    heatgen_fig = _make_heatgen_fig(cell)
    temp_stats = _make_temp_stats_fig(cell)

    return temp_map, soc_heatmap, voltage_fig, soc_fig, current_fig, heatgen_fig, temp_stats


@app.callback(
    Output("heartbeat", "children"),
    Output("heartbeat-graph", "figure"),
    Output("run-log", "children"),
    Input("heartbeat-interval", "n_intervals"),
)
def update_heartbeat(n):
    current_time = time.time() - START_TIME
    TIMES.append(current_time)
    HEARTBEAT_VALUES.append(n)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(TIMES),
            y=list(HEARTBEAT_VALUES),
            mode="lines+markers",
            name="Heartbeat",
            line=dict(color="#e74c3c", width=2),
            marker=dict(size=6),
        )
    )
    fig.update_layout(
        title="Heartbeat Over Time",
        xaxis_title="Time (seconds)",
        yaxis_title="Heartbeat Count",
        template="plotly_white",
        hovermode="x unified",
    )

    with LOG_LOCK:
        log_text = "\n".join(LOG_LINES)
        if len(log_text) > LOG_MAX_CHARS:
            log_text = "... (truncated) ...\n" + log_text[-LOG_MAX_CHARS:]

    return f"UI heartbeat: {n}", fig, log_text


def main() -> None:
    print("Starting Dash visualization on http://127.0.0.1:8050/")
    app.run(debug=False, use_reloader=False)


if __name__ == "__main__":
    main()