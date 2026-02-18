import ast
import struct
import zipfile
from functools import reduce
import operator as op

PATH = r"d:\PyECN Project\PyECN_ThinkClock\pyecn\browser_visualizations\uploads\results_1771400737.npz"


def read_npy_header(zf, name):
    with zf.open(name, "r") as f:
        magic = f.read(6)
        if magic != b"\x93NUMPY":
            raise ValueError("not npy")
        major, minor = f.read(2)
        if major == 1:
            header_len = struct.unpack("<H", f.read(2))[0]
        else:
            header_len = struct.unpack("<I", f.read(4))[0]
        header = f.read(header_len).decode("latin1")
        info = ast.literal_eval(header)
        return info, f


def dtype_struct_fmt(descr):
    endian = "<" if descr[0] in "<|=" else descr[0]
    kind = descr[1] if descr[0] in "<|=>" else descr[0]
    size = int(descr[2:]) if descr[0] in "<|=>" else int(descr[1:])
    if kind == "i":
        return endian + {1: "b", 2: "h", 4: "i", 8: "q"}[size]
    if kind == "u":
        return endian + {1: "B", 2: "H", 4: "I", 8: "Q"}[size]
    if kind == "f":
        return endian + {4: "f", 8: "d"}[size]
    raise ValueError("unsupported dtype " + descr)


def read_scalar(zf, name):
    info, f = read_npy_header(zf, name)
    descr = info["descr"]
    value_bytes = f.read(struct.calcsize(dtype_struct_fmt(descr)))
    return struct.unpack(dtype_struct_fmt(descr), value_bytes)[0]


def read_size(zf, name):
    info, _ = read_npy_header(zf, name)
    shape = info["shape"]
    return 1 if shape == () else reduce(op.mul, shape, 1)


with zipfile.ZipFile(PATH, "r") as zf:
    names = set(zf.namelist())
    ny = int(read_scalar(zf, "ny.npy")) if "ny.npy" in names else None
    for key in ["Al_4T", "Cu_4T", "Elb_4T", "Elr_4T"]:
        name = f"{key}.npy"
        if name not in names:
            print(key, "missing")
            continue
        size = read_size(zf, name)
        n_h = size // ny if ny else None
        print(key, "len", size, "nx", n_h, "ny", ny)
