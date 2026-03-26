# Mesh Descriptor Framework — Setup & Usage

## 1. Create the conda environment

```bash
conda env create -f environment.yml
conda activate mesh_descriptors
```

## 2. Verify the install

```bash
python -c "import numpy, scipy, sklearn, matplotlib; print('Core OK')"
python -c "import trimesh; print('trimesh OK')"
python -c "import open3d; print('open3d OK')"
```

## 3. Run the full benchmark (synthetic scenes)

```bash
python run_benchmark.py
```

Optional flags:
```bash
python run_benchmark.py --scene UrbanBlock MultiHeight
python run_benchmark.py --desc NDS ScanContextMesh
python run_benchmark.py --no-plots
```

## 4. Match your own mesh files

```bash
# OBJ, PLY (ASCII or binary), OFF, STL all supported
python match_meshes.py stereo.ply lidar.ply

# Specific descriptors + save plots
python match_meshes.py stereo.ply lidar.ply --desc NDS ScanContextMesh --save-plots

# Just inspect file contents
python match_meshes.py stereo.ply lidar.ply --info-only

# List all available descriptors
python match_meshes.py --list-descriptors
```

## 5. Use in Python directly

```python
from mesh_io import load_mesh
from descriptors import get_descriptor, DESCRIPTOR_REGISTRY
from matching import cosine_similarity

# Load your meshes
verts_a, faces_a, sems_a = load_mesh("stereo.ply")
verts_b, faces_b, sems_b = load_mesh("lidar.ply")

# Compute a descriptor
desc = get_descriptor("NDS")
d_a = desc.compute(verts_a, faces_a, sems_a)
d_b = desc.compute(verts_b, faces_b, sems_b)

# Match
score = cosine_similarity(d_a, d_b)
print(f"Cosine similarity: {score:.4f}")

# Or run all descriptors at once
for name, desc in DESCRIPTOR_REGISTRY.items():
    da = desc.compute(verts_a, faces_a, sems_a)
    db = desc.compute(verts_b, faces_b, sems_b)
    print(f"{name:<22} cosine={cosine_similarity(da, db):.4f}")
```

## 6. Loading trimesh / open3d meshes (optional)

If you have `trimesh` or `open3d` installed, you can load meshes via those
libraries and pass the arrays directly:

```python
import trimesh
import numpy as np

mesh = trimesh.load("my_mesh.ply")
verts = np.array(mesh.vertices)
faces = np.array(mesh.faces)
sems  = np.zeros(len(faces), dtype=np.int64)  # or your labels

from descriptors import get_descriptor
d = get_descriptor("NDS").compute(verts, faces, sems)
```

## File overview

| File                  | Purpose                                           |
|-----------------------|---------------------------------------------------|
| `descriptors.py`      | All 7 descriptors (NDS, FPFH, SpinImage, etc.)   |
| `mesh_io.py`          | Pure-numpy OBJ/PLY/OFF/STL loader                |
| `synthetic_scenes.py` | 5 synthetic stereo+LiDAR scene generators        |
| `matching.py`         | Similarity metrics + evaluation engine           |
| `visualisation.py`    | Heatmaps, radar, descriptor plots                |
| `run_benchmark.py`    | Full benchmark CLI                               |
| `match_meshes.py`     | Two-file matching CLI                            |
| `environment.yml`     | Conda environment definition                     |

## Semantic label conventions (for NDS descriptor)

| ID | Class       |
|----|-------------|
| 0  | Unknown     |
| 1  | Ground      |
| 2  | Wall        |
| 3  | Roof        |
| 4  | Vegetation  |
| 5  | Vehicle     |

Per-face labels are read from the `label`, `semantic`, or `class` property
in PLY files automatically.
