"""
mesh_descriptors/mesh_io.py
============================
Mesh loaders for OBJ, PLY (ASCII & binary), OFF, STL.
No trimesh or open3d required — pure Python + numpy.

Returns
-------
vertices : (V, 3) float64
faces    : (F, 3) int64
semantics: (F,)   int64   — per-face label (0 = unknown if not in file)

Supported formats
-----------------
  .obj   — Wavefront OBJ  (v / f lines; ignores mtl / texcoords)
  .ply   — Stanford PLY   (ASCII and little-endian binary)
  .off   — Object File Format
  .stl   — STL ASCII and binary
"""

import os
import re
import struct
import numpy as np
from typing import Tuple, Optional

MeshData = Tuple[np.ndarray, np.ndarray, np.ndarray]   # verts, faces, sems


# ─────────────────────────────────────────────────────────────────────────────
#  OBJ
# ─────────────────────────────────────────────────────────────────────────────

def load_obj(path: str) -> MeshData:
    verts, faces = [], []
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.split()[1:]
                # Each part can be "v", "v/vt", "v/vt/vn", "v//vn"
                idxs = [int(p.split("/")[0]) for p in parts]
                # Convert 1-based to 0-based; negative indices count from end
                n = len(verts)
                idxs = [i - 1 if i > 0 else n + i for i in idxs]
                # Fan triangulation for polygons
                for k in range(1, len(idxs) - 1):
                    faces.append([idxs[0], idxs[k], idxs[k + 1]])

    if not verts:
        raise ValueError(f"No vertices found in {path}")
    if not faces:
        raise ValueError(f"No faces found in {path}")

    V = np.array(verts, dtype=np.float64)
    F = np.array(faces, dtype=np.int64)
    S = np.zeros(len(F), dtype=np.int64)
    return V, F, S


# ─────────────────────────────────────────────────────────────────────────────
#  PLY
# ─────────────────────────────────────────────────────────────────────────────

def load_ply(path: str) -> MeshData:
    with open(path, "rb") as fh:
        # ── Parse header ──────────────────────────────────────────────────
        header_lines = []
        while True:
            raw = fh.readline()
            line = raw.decode("ascii", errors="replace").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        fmt = "ascii"
        for l in header_lines:
            if l.startswith("format"):
                if "binary_little_endian" in l:
                    fmt = "binary_le"
                elif "binary_big_endian" in l:
                    fmt = "binary_be"

        # Parse element counts and property types
        elements = {}   # name -> {count, properties: [(name, dtype)]}
        cur_elem = None
        for l in header_lines:
            if l.startswith("element "):
                parts = l.split()
                cur_elem = parts[1]
                elements[cur_elem] = {"count": int(parts[2]), "properties": []}
            elif l.startswith("property list") and cur_elem:
                parts = l.split()
                # property list <count_type> <val_type> <name>
                elements[cur_elem]["properties"].append(
                    (parts[4], "list", parts[2], parts[3]))
            elif l.startswith("property ") and cur_elem:
                parts = l.split()
                elements[cur_elem]["properties"].append((parts[2], parts[1]))

        _PLY_TYPES = {
            "char": "i1", "uchar": "u1", "short": "i2", "ushort": "u2",
            "int": "i4", "uint": "u4", "float": "f4", "double": "f8",
            "int8": "i1", "uint8": "u1", "int16": "i2", "uint16": "u2",
            "int32": "i4", "uint32": "u4", "float32": "f4", "float64": "f8",
        }

        def ply_dtype(t):
            return np.dtype(_PLY_TYPES.get(t, "f4"))

        data_start = fh.tell()

        if fmt == "ascii":
            rest = fh.read().decode("ascii", errors="replace")
            tokens = rest.split()
            ti = 0

            def read_tok(dtype):
                nonlocal ti
                v = tokens[ti]; ti += 1
                return dtype(v)

            verts, faces, sems = _ply_parse_ascii(elements, tokens, _PLY_TYPES)
        else:
            endian = "<" if fmt == "binary_le" else ">"
            verts, faces, sems = _ply_parse_binary(elements, fh, endian, _PLY_TYPES)

    V = np.array(verts, dtype=np.float64)
    F = np.array(faces, dtype=np.int64)
    S = np.array(sems,  dtype=np.int64) if sems else np.zeros(len(F), dtype=np.int64)
    return V, F, S


def _ply_parse_ascii(elements, tokens, type_map):
    verts, faces, sems = [], [], []
    ti = 0

    for elem_name, elem in elements.items():
        count = elem["count"]
        props = elem["properties"]

        for _ in range(count):
            row = {}
            for prop in props:
                if prop[1] == "list":
                    _, _, cnt_t, val_t = prop[0], prop[1], prop[2], prop[3]
                    n = int(tokens[ti]); ti += 1
                    vals = [float(tokens[ti + k]) for k in range(n)]; ti += n
                    row[prop[0]] = vals
                else:
                    name, dtype_str = prop[0], prop[1]
                    cast = float if "float" in dtype_str or "double" in dtype_str else int
                    row[name] = cast(tokens[ti]); ti += 1

            if elem_name == "vertex":
                verts.append([row.get("x", 0), row.get("y", 0), row.get("z", 0)])
            elif elem_name == "face":
                idxs = None
                for p in props:
                    if p[1] == "list":
                        idxs = [int(v) for v in row[p[0]]]
                        break
                if idxs and len(idxs) >= 3:
                    for k in range(1, len(idxs) - 1):
                        faces.append([idxs[0], idxs[k], idxs[k + 1]])
                        label = row.get("label", row.get("semantic", row.get("class", 0)))
                        sems.append(int(label))

    return verts, faces, sems


def _ply_parse_binary(elements, fh, endian, type_map):
    _STRUCT = {
        "char": "b", "uchar": "B", "short": "h", "ushort": "H",
        "int": "i", "uint": "I", "float": "f", "double": "d",
        "int8": "b", "uint8": "B", "int16": "h", "uint16": "H",
        "int32": "i", "uint32": "I", "float32": "f", "float64": "d",
    }
    _SIZE = {"b":1,"B":1,"h":2,"H":2,"i":4,"I":4,"f":4,"d":8}

    verts, faces, sems = [], [], []

    for elem_name, elem in elements.items():
        count = elem["count"]
        props = elem["properties"]

        for _ in range(count):
            row = {}
            for prop in props:
                if prop[1] == "list":
                    cnt_fmt = endian + _STRUCT[prop[2]]
                    cnt_size = _SIZE[_STRUCT[prop[2]]]
                    n = struct.unpack(cnt_fmt, fh.read(cnt_size))[0]
                    val_fmt = endian + str(n) + _STRUCT[prop[3]]
                    val_size = n * _SIZE[_STRUCT[prop[3]]]
                    vals = list(struct.unpack(val_fmt, fh.read(val_size)))
                    row[prop[0]] = vals
                else:
                    name, dtype_str = prop[0], prop[1]
                    fmt_c = _STRUCT.get(dtype_str, "f")
                    sz = _SIZE[fmt_c]
                    val = struct.unpack(endian + fmt_c, fh.read(sz))[0]
                    row[name] = val

            if elem_name == "vertex":
                verts.append([row.get("x", 0), row.get("y", 0), row.get("z", 0)])
            elif elem_name == "face":
                idxs = None
                for p in props:
                    if p[1] == "list":
                        idxs = [int(v) for v in row[p[0]]]
                        break
                if idxs and len(idxs) >= 3:
                    for k in range(1, len(idxs) - 1):
                        faces.append([idxs[0], idxs[k], idxs[k + 1]])
                        label = row.get("label", row.get("semantic", row.get("class", 0)))
                        sems.append(int(label))

    return verts, faces, sems


# ─────────────────────────────────────────────────────────────────────────────
#  OFF
# ─────────────────────────────────────────────────────────────────────────────

def load_off(path: str) -> MeshData:
    with open(path, "r", errors="replace") as fh:
        lines = [l.strip() for l in fh if l.strip() and not l.startswith("#")]

    i = 0
    if lines[i].upper().startswith("OFF"):
        i += 1

    counts = lines[i].split(); i += 1
    n_verts, n_faces = int(counts[0]), int(counts[1])

    verts = []
    for _ in range(n_verts):
        parts = lines[i].split(); i += 1
        verts.append([float(parts[0]), float(parts[1]), float(parts[2])])

    faces = []
    for _ in range(n_faces):
        parts = lines[i].split(); i += 1
        n = int(parts[0])
        idxs = [int(parts[k + 1]) for k in range(n)]
        for k in range(1, n - 1):
            faces.append([idxs[0], idxs[k], idxs[k + 1]])

    V = np.array(verts, dtype=np.float64)
    F = np.array(faces, dtype=np.int64)
    S = np.zeros(len(F), dtype=np.int64)
    return V, F, S


# ─────────────────────────────────────────────────────────────────────────────
#  STL
# ─────────────────────────────────────────────────────────────────────────────

def load_stl(path: str) -> MeshData:
    with open(path, "rb") as fh:
        header = fh.read(80)
        # Detect binary vs ASCII
        try:
            if header[:5].decode("ascii").lower() == "solid":
                # Could still be binary — check file size
                n_tri = struct.unpack("<I", fh.read(4))[0]
                expected = 80 + 4 + n_tri * 50
                actual   = os.path.getsize(path)
                if abs(expected - actual) < 10:
                    return _load_stl_binary(path)
                else:
                    return _load_stl_ascii(path)
            else:
                return _load_stl_binary(path)
        except Exception:
            return _load_stl_ascii(path)


def _load_stl_binary(path: str) -> MeshData:
    with open(path, "rb") as fh:
        fh.read(80)   # header
        n_tri = struct.unpack("<I", fh.read(4))[0]
        verts, faces = [], []
        for i in range(n_tri):
            data = struct.unpack("<12fH", fh.read(50))
            # normal: data[0:3], v1: data[3:6], v2: data[6:9], v3: data[9:12]
            base = len(verts)
            verts.append(list(data[3:6]))
            verts.append(list(data[6:9]))
            verts.append(list(data[9:12]))
            faces.append([base, base + 1, base + 2])

    V = np.array(verts, dtype=np.float64)
    F = np.array(faces, dtype=np.int64)
    S = np.zeros(len(F), dtype=np.int64)
    return V, F, S


def _load_stl_ascii(path: str) -> MeshData:
    verts, faces = [], []
    with open(path, "r", errors="replace") as fh:
        buf = []
        for line in fh:
            line = line.strip()
            if line.startswith("vertex "):
                parts = line.split()
                buf.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("endloop"):
                if len(buf) == 3:
                    base = len(verts)
                    verts.extend(buf)
                    faces.append([base, base + 1, base + 2])
                buf = []

    V = np.array(verts, dtype=np.float64)
    F = np.array(faces, dtype=np.int64)
    S = np.zeros(len(F), dtype=np.int64)
    return V, F, S


# ─────────────────────────────────────────────────────────────────────────────
#  Unified loader
# ─────────────────────────────────────────────────────────────────────────────

_LOADERS = {
    ".obj": load_obj,
    ".ply": load_ply,
    ".off": load_off,
    ".stl": load_stl,
}


def load_mesh(path: str) -> MeshData:
    """
    Load a mesh from file. Supports .obj, .ply (ASCII/binary), .off, .stl.

    Returns
    -------
    vertices : (V, 3) float64
    faces    : (F, 3) int64   — triangulated
    semantics: (F,)   int64   — per-face label (0 = unknown)
    """
    ext = os.path.splitext(path)[1].lower()
    if ext not in _LOADERS:
        raise ValueError(f"Unsupported format '{ext}'. "
                         f"Supported: {list(_LOADERS.keys())}")
    verts, faces, sems = _LOADERS[ext](path)

    # Sanity checks
    if len(verts) == 0:
        raise ValueError(f"Mesh has no vertices: {path}")
    if len(faces) == 0:
        raise ValueError(f"Mesh has no faces: {path}")
    if faces.max() >= len(verts):
        raise ValueError(f"Face index out of range in {path}")

    return verts, faces, sems


def mesh_info(path: str) -> str:
    """Quick summary string without full load (just counts)."""
    try:
        v, f, s = load_mesh(path)
        sem_counts = {int(k): int((s == k).sum()) for k in np.unique(s)}
        return (f"{os.path.basename(path)}  |  "
                f"{len(v)} vertices  {len(f)} faces  "
                f"sem_classes={sem_counts}")
    except Exception as e:
        return f"{path}  ERROR: {e}"
