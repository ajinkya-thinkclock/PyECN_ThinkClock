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
import csv
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
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
    "current_csv_path": None,
    "measured_csv_path": None,
    "measured_voltage": None,
    "rct_percent": None,
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


def _scale_unrolled_h(cell: Any, array_h: np.ndarray) -> np.ndarray:
    if hasattr(cell, "Spiral_Sep_s_real") and hasattr(cell, "Spiral_Sep_s"):
        return array_h * (cell.Spiral_Sep_s_real / cell.Spiral_Sep_s)
    if hasattr(cell, "SpiralandStripe_Sep_s_real"):
        return array_h * cell.SpiralandStripe_Sep_s_real
    return array_h


def _detect_column(fieldnames, keywords):
    for name in fieldnames:
        lower = name.lower()
        if any(key in lower for key in keywords):
            return name
    return None


def _load_measured_voltage(csv_path: Path) -> Optional[Dict[str, Any]]:
    if not csv_path.exists():
        return None
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return None
        time_key = _detect_column(reader.fieldnames, ["time"])
        voltage_key = None
        for name in reader.fieldnames:
            lower = name.lower()
            if "current" in lower:
                continue
            if "voltage" in lower or lower.endswith("_v") or "_v" in lower or "cell" in lower:
                voltage_key = name
                break
        if time_key is None or voltage_key is None:
            return None
        times = []
        volts = []
        for row in reader:
            try:
                t = float(row.get(time_key, ""))
                v = float(row.get(voltage_key, ""))
            except (TypeError, ValueError):
                continue
            times.append(t)
            volts.append(v)
    if not times:
        return None
    return {"time": np.asarray(times, dtype=float), "values": np.asarray(volts, dtype=float)}


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
    lg = None
    if hasattr(cell, "LG_Jellyroll"):
        lg = float(cell.LG_Jellyroll) if np.ndim(cell.LG_Jellyroll) == 0 else float(np.asarray(cell.LG_Jellyroll).reshape(-1)[0])

    if all(hasattr(cell, k) for k in ["Elb_4T", "xi_4T", "yi_4T", "List_node2ele_4T"]):
        n_v = cell.ny
        n_h = int(np.size(cell.Elb_4T) / n_v)
        ind0_Elb_4T = cell.Elb_4T.reshape(n_v, n_h)
        ind0_ele_Elb_4T = cell.List_node2ele_4T[ind0_Elb_4T, 0]
        array_h_elb = np.asarray(cell.xi_4T[ind0_Elb_4T], dtype=float)
        array_h_elb = _scale_unrolled_h(cell, array_h_elb)
        if lg is not None:
            array_v_elb = lg - np.asarray(cell.yi_4T[ind0_Elb_4T], dtype=float)
        else:
            array_v_elb = np.asarray(cell.yi_4T[ind0_Elb_4T], dtype=float)
        cache["ind0_Elb_4T"] = ind0_Elb_4T
        cache["ind0_ele_Elb_4T"] = ind0_ele_Elb_4T
        cache["array_h_elb"] = array_h_elb
        cache["array_v_elb"] = array_v_elb

        if hasattr(cell, "Elr_4T"):
            n_h_elr = int(np.size(cell.Elr_4T) / n_v)
            ind0_Elr_4T = cell.Elr_4T.reshape(n_v, n_h_elr)
            ind0_ele_Elr_4T = cell.List_node2ele_4T[ind0_Elr_4T, 0]
            array_h_elr = np.asarray(cell.xi_4T[ind0_Elr_4T], dtype=float)
            array_h_elr = _scale_unrolled_h(cell, array_h_elr)
            if lg is not None:
                array_v_elr = lg - np.asarray(cell.yi_4T[ind0_Elr_4T], dtype=float)
            else:
                array_v_elr = np.asarray(cell.yi_4T[ind0_Elr_4T], dtype=float)
            cache["ind0_Elr_4T"] = ind0_Elr_4T
            cache["ind0_ele_Elr_4T"] = ind0_ele_Elr_4T
            cache["array_h_elr"] = array_h_elr
            cache["array_v_elr"] = array_v_elr

    if all(hasattr(cell, k) for k in ["Al_4T", "Cu_4T", "xi_4T", "yi_4T"]):
        n_v = cell.ny
        n_h = int(np.size(cell.Al_4T) / n_v)
        ind0_Al_4T = cell.Al_4T.reshape(n_v, n_h)
        ind0_Cu_4T = cell.Cu_4T.reshape(n_v, n_h)
        array_h_al = np.asarray(cell.xi_4T[ind0_Al_4T], dtype=float)
        array_h_cu = np.asarray(cell.xi_4T[ind0_Cu_4T], dtype=float)
        array_h_al = _scale_unrolled_h(cell, array_h_al)
        array_h_cu = _scale_unrolled_h(cell, array_h_cu)
        if lg is not None:
            array_v_al = lg - np.asarray(cell.yi_4T[ind0_Al_4T], dtype=float)
            array_v_cu = lg - np.asarray(cell.yi_4T[ind0_Cu_4T], dtype=float)
        else:
            array_v_al = np.asarray(cell.yi_4T[ind0_Al_4T], dtype=float)
            array_v_cu = np.asarray(cell.yi_4T[ind0_Cu_4T], dtype=float)
        cache["temp_map_al_ind0"] = ind0_Al_4T
        cache["temp_map_al_x"] = array_h_al
        cache["temp_map_al_y"] = array_v_al
        cache["temp_map_cu_ind0"] = ind0_Cu_4T
        cache["temp_map_cu_x"] = array_h_cu
        cache["temp_map_cu_y"] = array_v_cu

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
        if i.ndim > 1:
            axes = tuple(range(i.ndim - 1))
            i = np.mean(i, axis=axes)
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

    measured = None
    csv_path = SIM_STATE.get("measured_csv_path") or SIM_STATE.get("current_csv_path")
    if csv_path:
        measured = _load_measured_voltage(Path(csv_path))
    SIM_STATE["measured_voltage"] = measured


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
        rct_percent = SIM_STATE.get("rct_percent")
        if rct_percent is not None:
            env["PYECN_RCT_PCT"] = str(rct_percent)
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


def _build_map_payload(x: Any, y: Any, z: Any) -> Optional[Dict[str, Any]]:
    z = np.asarray(z, dtype=float)
    if z.size == 0 or not np.isfinite(z).any():
        return None
    x1 = x[0] if (x is not None and getattr(x, "ndim", 0) == 2) else x
    y1 = y[:, 0] if (y is not None and getattr(y, "ndim", 0) == 2) else y
    if x1 is not None and y1 is not None:
        if x1.size == 0 or y1.size == 0:
            return None
    return {
        "x": _serialize_array(x1) if x1 is not None else None,
        "y": _serialize_array(y1) if y1 is not None else None,
        "z": _serialize_array(z),
    }


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
    cache = SIM_STATE.get("cache", {})
    maps: Dict[str, Any] = {}

    if "ind0_Elb_4T" in cache:
        ind0 = cache["ind0_Elb_4T"]
        x = cache.get("array_h_elb")
        y = cache.get("array_v_elb")
        z = np.asarray(cell.T_record[:, time_index][ind0], dtype=float) - 273.15
        payload = _build_map_payload(x, y, z)
        if payload:
            maps["elb"] = payload

    if "ind0_Elr_4T" in cache:
        ind0 = cache["ind0_Elr_4T"]
        x = cache.get("array_h_elr")
        y = cache.get("array_v_elr")
        z = np.asarray(cell.T_record[:, time_index][ind0], dtype=float) - 273.15
        payload = _build_map_payload(x, y, z)
        if payload:
            maps["elr"] = payload

    if maps:
        return maps

    x, y, z = _get_electrode_temp_map(cell, time_index)
    payload = _build_map_payload(x, y, z)
    if payload:
        return {"combined": payload}
    return None


def _current_density_data(cell, time_index: int) -> Optional[Dict[str, Any]]:
    required = ["I_ele_record", "Axy_ele", "List_node2ele_4T", "Elb_4T", "xi_4T", "yi_4T"]
    cache = SIM_STATE.get("cache", {})
    if all(hasattr(cell, k) for k in required) and "ind0_Elb_4T" in cache:
        try:
            maps: Dict[str, Any] = {}
            scalefactor_z = getattr(cell, "scalefactor_z", 1.0)

            def build_map(label: str, ind0_key: str, ind0_ele_key: str, h_key: str, v_key: str) -> None:
                ind0_ele = cache.get(ind0_ele_key)
                if ind0_ele is None:
                    return
                rouI = cell.I_ele_record[:, time_index][ind0_ele] / (
                    cell.Axy_ele[ind0_ele, 0] * scalefactor_z
                )
                payload = _build_map_payload(cache.get(h_key), cache.get(v_key), rouI)
                if payload:
                    maps[label] = payload

            build_map("elb", "ind0_Elb_4T", "ind0_ele_Elb_4T", "array_h_elb", "array_v_elb")
            build_map("elr", "ind0_Elr_4T", "ind0_ele_Elr_4T", "array_h_elr", "array_v_elr")

            if maps:
                return maps
        except Exception:
            return None

    if hasattr(cell, "I_record"):
        data = extract_from_pyecn_cell(cell, time_index=time_index)
        cur = data.get("current_2d")
        if cur is None:
            cur = np.zeros((cell.nx, cell.ny))
        return {"combined": {"x": None, "y": None, "z": _serialize_array(cur)}}

    return None


def _rct_series_data(cell, rc_index: int) -> Optional[Dict[str, Any]]:
    cache = SIM_STATE.get("cache", {})
    rct = getattr(cell, "Rct_scale", None)
    if rct is None:
        return None
    rct = np.asarray(rct, dtype=float)
    if rct.ndim != 2 or rct.size == 0:
        return None
    rc_index = int(rc_index)
    rc_index = max(0, min(rc_index, rct.shape[1] - 1))
    values = rct[:, rc_index]
    return {
        "indices": _serialize_array(np.arange(values.size)),
        "values": _serialize_array(values),
        "rc_index": rc_index,
        "rc_count": int(rct.shape[1]),
    }


def _soc_heatmap_data(cell, time_index: int) -> Optional[Dict[str, Any]]:
    if not hasattr(cell, "SoC_ele_record"):
        return None
    required = ["SoC_ele_record", "List_node2ele_4T", "Elb_4T", "xi_4T", "yi_4T"]
    if not all(hasattr(cell, k) for k in required):
        return None
    cache = SIM_STATE.get("cache", {})
    maps: Dict[str, Any] = {}

    def build_map(label: str, ind0_ele_key: str, h_key: str, v_key: str) -> None:
        ind0_ele = cache.get(ind0_ele_key)
        if ind0_ele is None:
            return
        soc = cell.SoC_ele_record[:, time_index][ind0_ele] * 100
        payload = _build_map_payload(cache.get(h_key), cache.get(v_key), soc)
        if not payload:
            return
        grad = None
        if soc.ndim == 2 and soc.shape[0] > 1 and soc.shape[1] > 1:
            grad_y, grad_x = np.gradient(soc)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            if np.isfinite(grad_mag).any():
                grad = grad_mag
        payload["grad"] = _serialize_array(grad)
        maps[label] = payload

    build_map("elb", "ind0_ele_Elb_4T", "array_h_elb", "array_v_elb")
    build_map("elr", "ind0_ele_Elr_4T", "array_h_elr", "array_v_elr")

    if maps:
        return maps
    return None


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
            "rct_percent": SIM_STATE.get("rct_percent"),
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
async def sim_run(
    config: UploadFile = File(...),
    current: Optional[UploadFile] = File(None),
    measured: Optional[UploadFile] = File(None),
    rct_percent: Optional[str] = Form(None),
):
    if SIM_STATE.get("running"):
        return JSONResponse(status_code=409, content={"status": "running", "message": "Simulation already running."})

    config_path = _save_upload_file(config)
    current_path = None
    if current is not None:
        current_path = _save_upload_file(current)
        SIM_STATE["current_csv_path"] = str(current_path)

    if measured is not None:
        measured_path = _save_upload_file(measured)
        SIM_STATE["measured_csv_path"] = str(measured_path)
    elif current_path is not None:
        SIM_STATE["measured_csv_path"] = str(current_path)

    if rct_percent is not None:
        try:
            SIM_STATE["rct_percent"] = float(rct_percent)
        except ValueError:
            SIM_STATE["rct_percent"] = None
    with LOG_LOCK:
        LOG_LINES.append(f"Rct spread (%): {SIM_STATE.get('rct_percent')}")

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
async def results_load(results: UploadFile = File(...), measured: Optional[UploadFile] = File(None)):
    try:
        results_path = _save_upload_file(results)
        SIM_STATE["results_path"] = str(results_path)
        if measured is not None:
            measured_path = _save_upload_file(measured)
            SIM_STATE["measured_csv_path"] = str(measured_path)
        _load_results(results_path)
        SIM_STATE["last_message"] = f"Results loaded: {results.filename}"
        return {"status": "loaded", "nt": SIM_STATE.get("nt"), "results_meta": SIM_STATE.get("results_meta")}
    except Exception as exc:
        SIM_STATE["last_message"] = f"Load error: {exc}"
        SIM_STATE["last_error"] = str(exc)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})


@app.get("/api/plots/frame")
def plots_frame(time_index: int = 0, rct_index: int = 0):
    cell = SIM_STATE.get("cell")
    if cell is None:
        return {"ready": False, "message": "No results loaded."}

    time_index = int(time_index or 0)
    if SIM_STATE.get("nt", 0) > 0:
        time_index = min(time_index, SIM_STATE["nt"] - 1)

    temp_maps = _temp_map_data(cell, time_index)
    soc_maps = _soc_heatmap_data(cell, time_index)
    current_maps = _current_density_data(cell, time_index)

    def pick_map(maps: Optional[Dict[str, Any]], keys) -> Optional[Dict[str, Any]]:
        if not maps:
            return None
        for key in keys:
            if key in maps and maps[key]:
                return maps[key]
        if "combined" in maps:
            return maps["combined"]
        return None

    payload = {
        "ready": True,
        "time_index": time_index,
        "temp_maps": temp_maps,
        "soc_heatmaps": soc_maps,
        "voltage": _series_data("voltage", time_index),
        "voltage_measured": (
            None
            if SIM_STATE.get("measured_voltage") is None
            else {
                "time": _serialize_array(SIM_STATE["measured_voltage"]["time"]),
                "values": _serialize_array(SIM_STATE["measured_voltage"]["values"]),
            }
        ),
        "current": _series_data("current", time_index),
        "soc": _series_data("soc", time_index),
        "current_density_maps": current_maps,
        "temp_map": pick_map(temp_maps, ["elb", "elr"]),
        "soc_heatmap": pick_map(soc_maps, ["elb", "elr"]),
        "current_density": pick_map(current_maps, ["elb", "elr"]),
        "rct_series": _rct_series_data(cell, rct_index),
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
