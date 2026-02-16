"""
Helper script to run a PyECN simulation and save results to a .npz file.
Usage: python run_pyecn_job.py <config_path> <output_npz>
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
import numpy as np

MAX_LIVE_STEPS = int(os.environ.get("PYECN_LIVE_MAX_STEPS", "0"))


def _downsample_time_axis(array, stride: int):
    if array is None or stride <= 1:
        return array
    arr = np.asarray(array)
    if arr.ndim == 0:
        return arr
    if arr.ndim == 1:
        return arr[::stride]
    slicer = [slice(None)] * arr.ndim
    slicer[-1] = slice(None, None, stride)
    return arr[tuple(slicer)]


def main() -> int:
    if len(sys.argv) < 3:
        print("Usage: python run_pyecn_job.py <config_path> <output_npz>")
        return 1

    config_path = Path(sys.argv[1]).resolve()
    output_path = Path(sys.argv[2]).resolve()

    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    if len(sys.argv) == 1:
        sys.argv.append(str(config_path))
    else:
        sys.argv[1] = str(config_path)

    import pyecn
    import pyecn.parse_inputs as ip

    pyecn.run()

    cell_name = ip.status_Cells_name[0]
    if not hasattr(pyecn, cell_name):
        raise RuntimeError(f"Cell object '{cell_name}' not found after simulation")

    cell = getattr(pyecn, cell_name)

    debug_keys = [
        "T_record",
        "SoC_Cell_record",
        "SoC_ele_record",
        "I_record",
        "V_record",
        "xi",
        "yi",
        "zi",
        "xi_4T",
        "yi_4T",
        "zi_4T",
        "Al_4T",
        "Cu_4T",
        "Elb_4T",
        "Elr_4T",
    ]
    print("Cell attribute summary:")
    for key in debug_keys:
        val = getattr(cell, key, None)
        shape = getattr(val, "shape", None)
        print(f"  - {key}: {'present' if val is not None else 'missing'} | shape={shape}")

    orig_nt = int(getattr(cell, "nt", 0) or 0)
    orig_dt = float(getattr(cell, "dt", 1.0) or 1.0)
    if MAX_LIVE_STEPS > 0 and orig_nt > 0:
        stride = max(1, int(np.ceil(orig_nt / MAX_LIVE_STEPS)))
    else:
        stride = 1

    data = {
        "nt": int(np.ceil(orig_nt / stride)) if orig_nt > 0 else 0,
        "dt": orig_dt * stride,
        "nx": getattr(cell, "nx", 0),
        "ny": getattr(cell, "ny", 0),
        "nstack": getattr(cell, "nstack", 0),
        "T_record": _downsample_time_axis(getattr(cell, "T_record", None), stride),
        "SoC_Cell_record": _downsample_time_axis(getattr(cell, "SoC_Cell_record", None), stride),
        "SoC_ele_record": _downsample_time_axis(getattr(cell, "SoC_ele_record", None), stride),
        "I_record": _downsample_time_axis(getattr(cell, "I_record", None), stride),
        "I_ele_record": _downsample_time_axis(getattr(cell, "I_ele_record", None), stride),
        "U_pndiff_plot": _downsample_time_axis(getattr(cell, "U_pndiff_plot", None), stride),
        "I0_record": _downsample_time_axis(getattr(cell, "I0_record", None), stride),
        "SoC": getattr(cell, "SoC", None),
        "V_record": _downsample_time_axis(getattr(cell, "V_record", None), stride),
        "t_record": _downsample_time_axis(getattr(cell, "t_record", None), stride),
        "xi": getattr(cell, "xi", None),
        "yi": getattr(cell, "yi", None),
        "zi": getattr(cell, "zi", None),
        "xi_4T": getattr(cell, "xi_4T", None),
        "yi_4T": getattr(cell, "yi_4T", None),
        "zi_4T": getattr(cell, "zi_4T", None),
        "Al_4T": getattr(cell, "Al_4T", None),
        "Cu_4T": getattr(cell, "Cu_4T", None),
        "Elb_4T": getattr(cell, "Elb_4T", None),
        "Elr_4T": getattr(cell, "Elr_4T", None),
        "LG_Jellyroll": getattr(cell, "LG_Jellyroll", None),
        "Axy_ele": getattr(cell, "Axy_ele", None),
        "List_node2ele_4T": getattr(cell, "List_node2ele_4T", None),
        "scalefactor_z": getattr(cell, "scalefactor_z", None),
        "Spiral_Sep_s_real": getattr(cell, "Spiral_Sep_s_real", None),
        "Spiral_Sep_s": getattr(cell, "Spiral_Sep_s", None),
        "SpiralandStripe_Sep_s_real": getattr(cell, "SpiralandStripe_Sep_s_real", None),
        "Lx_electrodes_real": getattr(cell, "Lx_electrodes_real", None),
        "Ly_electrodes_real": getattr(cell, "Ly_electrodes_real", None),
        "status_FormFactor": getattr(cell, "status_FormFactor", None),
        "q_4T_record": _downsample_time_axis(getattr(cell, "q_4T_record", None), stride),
        "V_stencil_4T_ALL": getattr(cell, "V_stencil_4T_ALL", None),
        "ntotal_4T": getattr(cell, "ntotal_4T", None),
    }

    # Save only non-None arrays/values
    save_dict = {k: v for k, v in data.items() if v is not None}
    if not save_dict:
        raise RuntimeError("No simulation data available to save.")

    np.savez_compressed(output_path, **save_dict)
    print(f"Saved results to {output_path}")
    print("Saved keys:", sorted(save_dict.keys()))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
