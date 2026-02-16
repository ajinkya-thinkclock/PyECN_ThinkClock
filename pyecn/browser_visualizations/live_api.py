"""
FastAPI backend for PyECN live browser visualization.
"""

from __future__ import annotations

import os
import sys
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

try:
    import toml
except ImportError as exc:  # pragma: no cover
    raise ImportError("toml is required for browser visualizations") from exc

PROJECT_ROOT = Path(__file__).parent.parent.parent

from pyecn.visualization_modules.viz_spatial_2d import extract_from_pyecn_cell

UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

LOG_MAX_LINES = 200
LOG_MAX_CHARS = 2000
LOG_LINES = deque(maxlen=LOG_MAX_LINES)
LOG_LOCK = threading.Lock()
STATE_LOCK = threading.Lock()

SIM_STATE: Dict[str, Any] = {
    "cell": None,
    "time": None,
    "nt": 0,
    "running": False,
    "last_message": "Idle",
    "last_error": None,
    "results_path": None,
    "results_meta": {},
    "cache": {},
    "series_cache": {},
}

app = FastAPI(title="PyECN Live API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"] ,
    allow_headers=["*"],
)


def _save_upload_file(upload: UploadFile) -> Path:
    path = UPLOAD_DIR / (upload.filename or f"upload_{int(time.time())}")
    with path.open("wb") as f:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    return path


def _prepare_config(config_path: Path, current_csv_path: Optional[Path]) -> Path:
    config = toml.load(config_path)
    if current_csv_path is not None:
        config.setdefault("operating_conditions", {})["I_ext_fpath"] = str(current_csv_path)
    post = config.setdefault("postprocessing", {})
    post["PostProcessor"] = "No"
    post["PostProcessor_module"] = "No"
    post["Visualisation_method"] = "none"
    post["Fig1to9"] = "No"
    post["PopFig_or_SaveGIF_instant"] = "No"
    post["PopFig_or_SaveGIF_replay"] = "No"
    temp_config = UPLOAD_DIR / f"run_{config_path.stem}.toml"
    with open(temp_config, "w", encoding="utf-8") as f:
        toml.dump(config, f)
    return temp_config


def _shape_info(value: Any) -> Optional[list]:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return list(shape)


def _load_results(results_path: Path) -> None:
    data = np.load(results_path, allow_pickle=True)

    class _Cell:
        pass

    cell = _Cell()
    for key in data.files:
        setattr(cell, key, data[key])

    with STATE_LOCK:
        SIM_STATE["cell"] = cell
        SIM_STATE["nt"] = int(getattr(cell, "nt", 0))
        dt = float(getattr(cell, "dt", 1.0))
        SIM_STATE["time"] = np.arange(SIM_STATE["nt"]) * dt
        SIM_STATE["cache"] = {}
        SIM_STATE["series_cache"] = {}
        SIM_STATE["results_meta"] = {k: _shape_info(data[k]) for k in data.files}

    if hasattr(cell, "T_record"):
        tmin = np.nanmin(cell.T_record)
        tmax = np.nanmax(cell.T_record)
        SIM_STATE["results_meta"]["T_record_minmax_K"] = [float(tmin), float(tmax)]

    if not data.files:
        SIM_STATE["last_message"] = "Simulation finished but results file is empty."
        return

    cache = SIM_STATE["cache"]
    if hasattr(cell, "Al_4T") and hasattr(cell, "xi_4T") and hasattr(cell, "yi_4T"):
        n_v = cell.ny
        n_h = int(np.size(cell.Al_4T) / n_v)
        ind0_Al_4T = cell.Al_4T.reshape(n_v, n_h)
        x = np.asarray(cell.xi_4T[ind0_Al_4T], dtype=float)
        lg = float(cell.LG_Jellyroll) if np.ndim(cell.LG_Jellyroll) == 0 else float(np.asarray(cell.LG_Jellyroll).reshape(-1)[0])
        y = (lg - np.asarray(cell.yi_4T[ind0_Al_4T], dtype=float))
        cache["temp_map_ind0"] = ind0_Al_4T
        cache["temp_map_x"] = x
        cache["temp_map_y"] = y

    if all(hasattr(cell, k) for k in ["Elb_4T", "xi_4T", "yi_4T", "List_node2ele_4T"]):
        n_v = cell.ny
        n_h = int(np.size(cell.Elb_4T) / n_v)
        ind0_Elb_4T = cell.Elb_4T.reshape(n_v, n_h)
        ind0_ele_Elb_4T = cell.List_node2ele_4T[ind0_Elb_4T, 0]
        array_h = np.asarray(cell.xi_4T[ind0_Elb_4T], dtype=float)
        if hasattr(cell, "LG_Jellyroll"):
            lg = float(cell.LG_Jellyroll) if np.ndim(cell.LG_Jellyroll) == 0 else float(np.asarray(cell.LG_Jellyroll).reshape(-1)[0])
            array_v = lg - np.asarray(cell.yi_4T[ind0_Elb_4T], dtype=float)
        else:
            array_v = np.asarray(cell.yi_4T[ind0_Elb_4T], dtype=float)
        if hasattr(cell, "Spiral_Sep_s_real") and hasattr(cell, "Spiral_Sep_s"):
            array_h = array_h * (cell.Spiral_Sep_s_real / cell.Spiral_Sep_s)
        elif hasattr(cell, "SpiralandStripe_Sep_s_real"):
            array_h = array_h * cell.SpiralandStripe_Sep_s_real
        cache["ind0_Elb_4T"] = ind0_Elb_4T
        cache["ind0_ele_Elb_4T"] = ind0_ele_Elb_4T
        cache["array_h"] = array_h
        cache["array_v"] = array_v

    series = SIM_STATE["series_cache"]
    time_vec = SIM_STATE["time"]
    if hasattr(cell, "U_pndiff_plot"):
        v = np.asarray(cell.U_pndiff_plot, dtype=float)
        series["voltage"] = v[: len(time_vec)]
    elif hasattr(cell, "V_record"):
        v = np.asarray(cell.V_record, dtype=float)
        if v.ndim == 2:
            series["voltage"] = np.mean(v, axis=0)[: len(time_vec)]

    if hasattr(cell, "SoC_Cell_record"):
        series["soc"] = np.asarray(cell.SoC_Cell_record, dtype=float) * 100

    if hasattr(cell, "I_record"):
        i = np.asarray(cell.I_record, dtype=float)
        if i.ndim == 2:
            i = np.mean(i, axis=0)
        series["current"] = i[: len(time_vec)]
    elif hasattr(cell, "I0_record"):
        i = np.asarray(cell.I0_record, dtype=float)
        if i.ndim == 2:
            i = np.mean(i, axis=0)
        series["current"] = i[: len(time_vec)]

    if hasattr(cell, "T_record"):
        T = np.asarray(cell.T_record, dtype=float) - 273.15
        if T.size:
            series["temp_min"] = np.min(T, axis=0)
            series["temp_max"] = np.max(T, axis=0)
            series["temp_avg"] = np.mean(T, axis=0)

    if hasattr(cell, "q_4T_record") and hasattr(cell, "V_stencil_4T_ALL"):
        q = np.asarray(cell.q_4T_record, dtype=float)
        v = np.asarray(cell.V_stencil_4T_ALL, dtype=float).reshape(-1)
        ntotal_4T = int(getattr(cell, "ntotal_4T", len(v)))
        v = v[:ntotal_4T]
        q = q[:ntotal_4T, :]
        if q.size:
            series["heatgen"] = np.sum(q * v[:, None], axis=0)


def _run_pyecn_sim_async(config_path: Path) -> None:
    with STATE_LOCK:
        SIM_STATE["running"] = True
        SIM_STATE["last_error"] = None
        SIM_STATE["last_message"] = "Running PyECN simulation..."

    try:
        results_path = UPLOAD_DIR / f"results_{int(time.time())}.npz"
        SIM_STATE["results_path"] = str(results_path)

        cmd = [
            os.fspath(Path(sys.executable)),
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


def _serialize_array(value: Any) -> Optional[Any]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        scalar = float(arr)
        return scalar if np.isfinite(scalar) else None
    mask = np.isfinite(arr)
    cleaned = arr.astype(object)
    cleaned[~mask] = None
    return cleaned.tolist()


def _get_electrode_temp_map(cell, time_index: int):
    cache = SIM_STATE.get("cache", {})
    if "temp_map_ind0" in cache and hasattr(cell, "T_record"):
        ind0_Al_4T = cache["temp_map_ind0"]
        x = cache["temp_map_x"]
        y = cache["temp_map_y"]
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


def _temp_map_data(cell, time_index: int) -> Optional[Dict[str, Any]]:
    if not hasattr(cell, "T_record"):
        return None
    x, y, z = _get_electrode_temp_map(cell, time_index)
    z = np.asarray(z, dtype=float)
    if z.size == 0 or not np.isfinite(z).any():
        return None
    x1 = x[0] if x.ndim == 2 else x
    y1 = y[:, 0] if y.ndim == 2 else y
    if x1.size == 0 or y1.size == 0:
        return None
    return {"x": _serialize_array(x1), "y": _serialize_array(y1), "z": _serialize_array(z)}


def _current_density_data(cell, time_index: int) -> Optional[Dict[str, Any]]:
    required = ["I_ele_record", "Axy_ele", "List_node2ele_4T", "Elb_4T", "xi_4T", "yi_4T"]
    cache = SIM_STATE.get("cache", {})
    if all(hasattr(cell, k) for k in required) and "ind0_Elb_4T" in cache:
        try:
            ind0_Elb_4T = cache["ind0_Elb_4T"]
            ind0_ele_Elb_4T = cache["ind0_ele_Elb_4T"]
            scalefactor_z = getattr(cell, "scalefactor_z", 1.0)
            rouI = cell.I_ele_record[:, time_index][ind0_ele_Elb_4T] / (
                cell.Axy_ele[ind0_ele_Elb_4T, 0] * scalefactor_z
            )
            rouI = np.asarray(rouI, dtype=float)
            if rouI.size == 0 or not np.isfinite(rouI).any():
                return None
            array_h = cache.get("array_h")
            array_v = cache.get("array_v")
            return {
                "x": _serialize_array(array_h[0]),
                "y": _serialize_array(array_v[:, 0]),
                "z": _serialize_array(rouI),
            }
        except Exception:
            return None

    if hasattr(cell, "I_record"):
        data = extract_from_pyecn_cell(cell, time_index=time_index)
        cur = data.get("current_2d")
        if cur is None:
            cur = np.zeros((cell.nx, cell.ny))
        return {"x": None, "y": None, "z": _serialize_array(cur)}

    return None


def _soc_heatmap_data(cell, time_index: int) -> Optional[Dict[str, Any]]:
    if not hasattr(cell, "SoC_ele_record"):
        return None
    required = ["SoC_ele_record", "List_node2ele_4T", "Elb_4T", "xi_4T", "yi_4T"]
    if not all(hasattr(cell, k) for k in required):
        return None

    n_v = cell.ny
    n_h = int(np.size(cell.Elb_4T) / n_v)
    ind0_Elb_4T = cell.Elb_4T.reshape(n_v, n_h)
    ind0_ele_Elb_4T = cell.List_node2ele_4T[ind0_Elb_4T, 0]

    soc = cell.SoC_ele_record[:, time_index][ind0_ele_Elb_4T] * 100
    soc = np.asarray(soc, dtype=float)
    if soc.size == 0 or not np.isfinite(soc).any():
        return None

    array_h = np.asarray(cell.xi_4T[ind0_Elb_4T], dtype=float)
    if hasattr(cell, "LG_Jellyroll"):
        lg = float(cell.LG_Jellyroll) if np.ndim(cell.LG_Jellyroll) == 0 else float(np.asarray(cell.LG_Jellyroll).reshape(-1)[0])
        array_v = lg - np.asarray(cell.yi_4T[ind0_Elb_4T], dtype=float)
    else:
        array_v = np.asarray(cell.yi_4T[ind0_Elb_4T], dtype=float)

    if hasattr(cell, "Spiral_Sep_s_real") and hasattr(cell, "Spiral_Sep_s"):
        array_h = array_h * (cell.Spiral_Sep_s_real / cell.Spiral_Sep_s)

    grad = None
    if soc.ndim == 2 and soc.shape[0] > 1 and soc.shape[1] > 1:
        grad_y, grad_x = np.gradient(soc)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        if np.isfinite(grad_mag).any():
            grad = grad_mag

    return {
        "x": _serialize_array(array_h[0]),
        "y": _serialize_array(array_v[:, 0]),
        "z": _serialize_array(soc),
        "grad": _serialize_array(grad),
    }


def _series_data(name: str, time_index: int) -> Optional[Dict[str, Any]]:
    series = SIM_STATE.get("series_cache", {})
    if name not in series:
        return None
    time_vec = SIM_STATE.get("time")
    if time_vec is None:
        return None
    values = np.asarray(series[name], dtype=float)
    end = min(len(time_vec), time_index + 1)
    return {"time": _serialize_array(time_vec[:end]), "values": _serialize_array(values[:end])}


def _temp_stats_data(time_index: int) -> Optional[Dict[str, Any]]:
    series = SIM_STATE.get("series_cache", {})
    if not ("temp_min" in series and "temp_max" in series and "temp_avg" in series):
        return None
    t = SIM_STATE.get("time")
    if t is None:
        return None
    end = min(len(t), time_index + 1)
    return {
        "time": _serialize_array(t[:end]),
        "min": _serialize_array(np.asarray(series["temp_min"], dtype=float)[:end]),
        "max": _serialize_array(np.asarray(series["temp_max"], dtype=float)[:end]),
        "avg": _serialize_array(np.asarray(series["temp_avg"], dtype=float)[:end]),
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/sim/status")
def sim_status():
    with STATE_LOCK:
        payload = {
            "running": SIM_STATE.get("running"),
            "last_message": SIM_STATE.get("last_message"),
            "last_error": SIM_STATE.get("last_error"),
            "nt": SIM_STATE.get("nt"),
            "results_meta": SIM_STATE.get("results_meta"),
            "results_path": SIM_STATE.get("results_path"),
        }
    return payload


@app.get("/api/sim/log")
def sim_log():
    with LOG_LOCK:
        log_text = "\n".join(LOG_LINES)
        if len(log_text) > LOG_MAX_CHARS:
            log_text = "... (truncated) ...\n" + log_text[-LOG_MAX_CHARS:]
    return {"log": log_text}


@app.post("/api/sim/run")
async def sim_run(config: UploadFile = File(...), current: Optional[UploadFile] = File(None)):
    if SIM_STATE.get("running"):
        return JSONResponse(status_code=409, content={"status": "running", "message": "Simulation already running."})

    config_path = _save_upload_file(config)
    current_path = None
    if current is not None:
        current_path = _save_upload_file(current)

    try:
        temp_config = _prepare_config(config_path, current_path)
        thread = threading.Thread(target=_run_pyecn_sim_async, args=(temp_config,), daemon=True)
        thread.start()
        SIM_STATE["last_message"] = "Simulation started..."
    except Exception as exc:
        SIM_STATE["last_message"] = f"Simulation error: {exc}"
        SIM_STATE["last_error"] = str(exc)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})

    return {"status": "started"}


@app.post("/api/results/load")
async def results_load(results: UploadFile = File(...)):
    try:
        results_path = _save_upload_file(results)
        SIM_STATE["results_path"] = str(results_path)
        _load_results(results_path)
        SIM_STATE["last_message"] = f"Results loaded: {results.filename}"
        return {"status": "loaded", "nt": SIM_STATE.get("nt"), "results_meta": SIM_STATE.get("results_meta")}
    except Exception as exc:
        SIM_STATE["last_message"] = f"Load error: {exc}"
        SIM_STATE["last_error"] = str(exc)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})


@app.get("/api/plots/frame")
def plots_frame(time_index: int = 0):
    cell = SIM_STATE.get("cell")
    if cell is None:
        return {"ready": False, "message": "No results loaded."}

    time_index = int(time_index or 0)
    if SIM_STATE.get("nt", 0) > 0:
        time_index = min(time_index, SIM_STATE["nt"] - 1)

    payload = {
        "ready": True,
        "time_index": time_index,
        "temp_map": _temp_map_data(cell, time_index),
        "soc_heatmap": _soc_heatmap_data(cell, time_index),
        "voltage": _series_data("voltage", time_index),
        "current": _series_data("current", time_index),
        "soc": _series_data("soc", time_index),
        "current_density": _current_density_data(cell, time_index),
        "heatgen": _series_data("heatgen", time_index),
        "temp_stats": _temp_stats_data(time_index),
    }
    return payload


def main() -> None:
    import uvicorn

    print("Starting PyECN API on http://127.0.0.1:8000")
    uvicorn.run("pyecn.browser_visualizations.live_api:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
