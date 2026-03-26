"""
mesh_descriptors/synthetic_scenes.py
=====================================
Synthetic mesh generators for testing cross-modal matching.

Each scene function returns a dict:
  {
    'stereo_mesh':  (vertices, faces, semantics),   # drone / stereo
    'lidar_mesh':   (vertices, faces, semantics),   # ground / LiDAR
    'name':         str,
    'description':  str,
  }

Stereo meshes are denser + noisier; LiDAR meshes are coarser + blockier,
simulating real-world sensor characteristics.
"""

import numpy as np
from typing import Tuple, Dict, Optional

# Semantic class IDs
SEM_UNKNOWN   = 0
SEM_GROUND    = 1
SEM_WALL      = 2
SEM_ROOF      = 3
SEM_VEGETATION= 4
SEM_VEHICLE   = 5


def _add_noise(vertices: np.ndarray, std: float, rng) -> np.ndarray:
    return vertices + rng.normal(0, std, vertices.shape)


def _make_quad(v0, v1, v2, v3):
    """Two triangles from a quad (v0,v1,v2,v3)."""
    return [(v0, v1, v2), (v0, v2, v3)]


def _grid_plane(x0, x1, y0, y1, z, nx, ny, rng, noise=0.0, sem=SEM_GROUND):
    """Flat grid of triangles."""
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    XX, YY = np.meshgrid(xs, ys)
    verts = np.column_stack([XX.ravel(), YY.ravel(),
                              np.full(XX.size, z)])
    if noise > 0:
        verts += rng.normal(0, noise, verts.shape)
    faces = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a = j * nx + i
            b = a + 1
            c = a + nx
            d = c + 1
            faces += [(a, c, b), (b, c, d)]
    sems = np.full(len(faces), sem)
    return verts, np.array(faces), sems


def _box(cx, cy, cz, w, d, h, res, rng, noise=0.0,
         sem_walls=SEM_WALL, sem_roof=SEM_ROOF):
    """A simple box building."""
    all_v, all_f, all_s = [], [], []
    offset = 0

    def add_quad_grid(p0, p1, p2, p3, nx, ny, sem):
        nonlocal offset
        us = np.linspace(0, 1, nx)
        vs = np.linspace(0, 1, ny)
        UU, VV = np.meshgrid(us, vs)
        pts = (p0[None, None, :]
               + UU[:, :, None] * (p1 - p0)[None, None, :]
               + VV[:, :, None] * (p3 - p0)[None, None, :])
        pts = pts.reshape(-1, 3)
        if noise > 0:
            pts += rng.normal(0, noise, pts.shape)
        all_v.append(pts)
        fs = []
        for j in range(ny - 1):
            for i in range(nx - 1):
                a = offset + j * nx + i
                b = a + 1; c = a + nx; dd = c + 1
                fs += [(a, b, c), (b, dd, c)]
        all_f.append(np.array(fs))
        all_s.append(np.full(len(fs), sem))
        offset += len(pts)

    x0, x1 = cx - w/2, cx + w/2
    y0, y1 = cy - d/2, cy + d/2
    z0, z1 = cz, cz + h
    n = max(2, res)

    p = np.array
    # 4 walls
    add_quad_grid(p([x0,y0,z0]), p([x1,y0,z0]), p([x1,y0,z1]), p([x0,y0,z1]), n, n, sem_walls)
    add_quad_grid(p([x1,y1,z0]), p([x0,y1,z0]), p([x0,y1,z1]), p([x1,y1,z1]), n, n, sem_walls)
    add_quad_grid(p([x1,y0,z0]), p([x1,y1,z0]), p([x1,y1,z1]), p([x1,y0,z1]), n, n, sem_walls)
    add_quad_grid(p([x0,y1,z0]), p([x0,y0,z0]), p([x0,y0,z1]), p([x0,y1,z1]), n, n, sem_walls)
    # roof
    add_quad_grid(p([x0,y0,z1]), p([x1,y0,z1]), p([x1,y1,z1]), p([x0,y1,z1]), n, n, sem_roof)

    verts = np.vstack(all_v)
    faces = np.vstack(all_f)
    sems  = np.concatenate(all_s)
    return verts, faces, sems


def _combine(*meshes):
    """Combine multiple (verts, faces, sems) tuples into one mesh."""
    all_v, all_f, all_s = [], [], []
    offset = 0
    for v, f, s in meshes:
        all_v.append(v)
        all_f.append(f + offset)
        all_s.append(s)
        offset += len(v)
    return np.vstack(all_v), np.vstack(all_f), np.concatenate(all_s)


# ─────────────────────────────────────────────────────────────────────────────
#  Scene definitions
# ─────────────────────────────────────────────────────────────────────────────

def scene_urban_block(seed: int = 42) -> Dict:
    """
    Simple urban block: flat ground + 3 buildings.
    Stereo (drone): top-down view, dense, noisy.
    LiDAR (ground): ground-level sweep, coarser, blocky.
    Same geometry, different noise/resolution/viewpoint.
    """
    rng = np.random.default_rng(seed)

    # --- Stereo mesh: drone sees mostly roofs + tops of walls ---
    ground_s = _grid_plane(-15, 15, -15, 15, 0, 30, 30, rng, noise=0.05, sem=SEM_GROUND)
    b1_s = _box( 5,  5, 0, 6, 6, 4, 8, rng, noise=0.08)
    b2_s = _box(-5,  5, 0, 4, 6, 5, 6, rng, noise=0.08)
    b3_s = _box( 5, -5, 0, 5, 5, 3, 7, rng, noise=0.08)
    stereo = _combine(ground_s, b1_s, b2_s, b3_s)

    # --- LiDAR mesh: ground robot, coarser grid, less roof coverage ---
    ground_l = _grid_plane(-15, 15, -15, 15, 0, 15, 15, rng, noise=0.02, sem=SEM_GROUND)
    b1_l = _box( 5,  5, 0, 6, 6, 4, 4, rng, noise=0.03)
    b2_l = _box(-5,  5, 0, 4, 6, 5, 3, rng, noise=0.03)
    b3_l = _box( 5, -5, 0, 5, 5, 3, 4, rng, noise=0.03)
    lidar = _combine(ground_l, b1_l, b2_l, b3_l)

    return dict(name="UrbanBlock",
                description="Flat ground + 3 buildings. Stereo=drone top-down, LiDAR=ground sweep.",
                stereo_mesh=stereo, lidar_mesh=lidar)


def scene_outdoor_corridor(seed: int = 42) -> Dict:
    """
    Outdoor corridor between two walls.
    Tests viewpoint difference: drone sees tops of walls, ground robot sees wall faces.
    """
    rng = np.random.default_rng(seed)

    def wall(x, y0, y1, h, res, noise, rng):
        p = np.array
        return _box.__wrapped__ if False else _box(x, (y0+y1)/2, 0, 0.5, y1-y0, h, res, rng, noise)

    ground_s = _grid_plane(-2, 12, -6, 6, 0, 20, 16, rng, noise=0.05, sem=SEM_GROUND)
    wall1_s  = _box(-0.25, 0, 0, 0.5, 10, 3, 8, rng, noise=0.1)
    wall2_s  = _box(-0.25, 4, 0, 0.5, 10, 3, 8, rng, noise=0.1)
    stereo   = _combine(ground_s, wall1_s, wall2_s)

    ground_l = _grid_plane(-2, 12, -6, 6, 0, 12, 10, rng, noise=0.02, sem=SEM_GROUND)
    wall1_l  = _box(-0.25, 0, 0, 0.5, 10, 3, 4, rng, noise=0.03)
    wall2_l  = _box(-0.25, 4, 0, 0.5, 10, 3, 4, rng, noise=0.03)
    lidar    = _combine(ground_l, wall1_l, wall2_l)

    return dict(name="OutdoorCorridor",
                description="Two parallel walls forming a corridor. Tests severe viewpoint gap.",
                stereo_mesh=stereo, lidar_mesh=lidar)


def scene_open_plaza(seed: int = 42) -> Dict:
    """
    Open plaza: mostly flat ground + scattered low objects.
    Mostly ground-plane → tests normal histogram (dominated by upward normals).
    """
    rng = np.random.default_rng(seed)

    ground_s = _grid_plane(-20, 20, -20, 20, 0, 40, 40, rng, noise=0.04, sem=SEM_GROUND)
    # Low objects
    obj1_s   = _box( 8, 8, 0, 2, 2, 1, 8, rng, noise=0.08)
    obj2_s   = _box(-8, 5, 0, 3, 1, 0.8, 8, rng, noise=0.08)
    stereo   = _combine(ground_s, obj1_s, obj2_s)

    ground_l = _grid_plane(-20, 20, -20, 20, 0, 20, 20, rng, noise=0.02, sem=SEM_GROUND)
    obj1_l   = _box( 8, 8, 0, 2, 2, 1, 4, rng, noise=0.02)
    obj2_l   = _box(-8, 5, 0, 3, 1, 0.8, 4, rng, noise=0.02)
    lidar    = _combine(ground_l, obj1_l, obj2_l)

    return dict(name="OpenPlaza",
                description="Flat plaza with small objects. Ground-plane dominated.",
                stereo_mesh=stereo, lidar_mesh=lidar)


def scene_multi_height(seed: int = 42) -> Dict:
    """
    Scene with terrain height variation + buildings at different heights.
    Tests curvature & dihedral descriptors on non-planar ground.
    """
    rng = np.random.default_rng(seed)

    def _wavy_ground(x0, x1, y0, y1, nx, ny, amp, noise, rng, sem=SEM_GROUND):
        xs = np.linspace(x0, x1, nx)
        ys = np.linspace(y0, y1, ny)
        XX, YY = np.meshgrid(xs, ys)
        ZZ = amp * np.sin(XX / 5) * np.cos(YY / 5)
        verts = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])
        if noise > 0:
            verts += rng.normal(0, noise, verts.shape)
        faces = []
        for j in range(ny - 1):
            for i in range(nx - 1):
                a = j*nx+i; b=a+1; c=a+nx; d=c+1
                faces += [(a, c, b), (b, c, d)]
        sems = np.full(len(faces), sem)
        return verts, np.array(faces), sems

    ground_s = _wavy_ground(-12,12,-12,12, 30, 30, 1.5, 0.06, rng)
    b1_s     = _box( 6, 6, 1.2, 5, 5, 6, 8, rng, noise=0.1)
    stereo   = _combine(ground_s, b1_s)

    ground_l = _wavy_ground(-12,12,-12,12, 16, 16, 1.5, 0.02, rng)
    b1_l     = _box( 6, 6, 1.2, 5, 5, 6, 5, rng, noise=0.03)
    lidar    = _combine(ground_l, b1_l)

    return dict(name="MultiHeight",
                description="Wavy terrain + tall building. Tests curvature-based descriptors.",
                stereo_mesh=stereo, lidar_mesh=lidar)


def scene_negative_control(seed: int = 42) -> Dict:
    """
    NEGATIVE CONTROL: Two completely different scenes.
    Descriptors should give low similarity — useful for calibrating thresholds.
    """
    rng = np.random.default_rng(seed)

    # Scene A: buildings
    ground_a = _grid_plane(-10, 10, -10, 10, 0, 20, 20, rng, noise=0.04, sem=SEM_GROUND)
    b1 = _box(4, 4, 0, 5, 5, 5, 6, rng, noise=0.07)
    stereo = _combine(ground_a, b1)

    # Scene B: rolling hills (no buildings)
    def _hill(x0, x1, y0, y1, nx, ny, rng):
        xs = np.linspace(x0, x1, nx)
        ys = np.linspace(y0, y1, ny)
        XX, YY = np.meshgrid(xs, ys)
        ZZ = 3 * np.exp(-(XX**2 + YY**2) / 20)
        verts = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])
        faces = []
        for j in range(ny - 1):
            for i in range(nx - 1):
                a=j*nx+i; b=a+1; c=a+nx; d=c+1
                faces += [(a, c, b), (b, c, d)]
        sems = np.full(len(faces), SEM_VEGETATION)
        return verts, np.array(faces), sems

    hill = _hill(-10, 10, -10, 10, 20, 20, rng)
    lidar = hill

    return dict(name="NegativeControl",
                description="Completely different scenes: buildings vs hills. Should give low similarity.",
                stereo_mesh=stereo, lidar_mesh=lidar)


SCENE_REGISTRY = {
    "UrbanBlock":      scene_urban_block,
    "OutdoorCorridor": scene_outdoor_corridor,
    "OpenPlaza":       scene_open_plaza,
    "MultiHeight":     scene_multi_height,
    "NegativeControl": scene_negative_control,
}
