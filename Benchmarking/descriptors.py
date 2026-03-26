"""
mesh_descriptors/descriptors.py
================================
Handcrafted mesh descriptors for cross-modal place recognition.
All descriptors operate on triangle meshes (vertices + faces).

Implemented descriptors
-----------------------
1.  NormalHistogram      – orientation of face normals on unit sphere
2.  DihedralHistogram    – distribution of dihedral angles along edges
3.  CurvatureHistogram   – Gaussian (angle-deficit) and mean curvature
4.  SpinImage            – Johnson & Hebert 1997 spin-image descriptor
5.  MeshFPFH             – FPFH-inspired angular histogram on mesh faces
6.  ScanContextMesh      – Scan-Context style polar grid with mesh features
7.  NDS (our proposal)   – Normal + Dihedral + Semantic place descriptor
"""

import numpy as np
from scipy.spatial import cKDTree
from sklearn.preprocessing import normalize as sk_normalize
from typing import Optional, Dict, Tuple, List
import warnings


# ─────────────────────────────────────────────────────────────────────────────
#  Low-level mesh geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Unit normals for every triangle face."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(n, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return n / norms


def compute_face_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Scalar area of every triangle face."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    return 0.5 * np.linalg.norm(cross, axis=1)


def compute_face_centroids(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    return (vertices[faces[:, 0]] + vertices[faces[:, 1]] + vertices[faces[:, 2]]) / 3.0


def build_edge_face_map(faces: np.ndarray) -> Dict[Tuple, List[int]]:
    """Build mapping from (v_i, v_j) sorted edge → list of face indices."""
    edge_map: Dict[Tuple, List[int]] = {}
    for fi, (a, b, c) in enumerate(faces):
        for e in [(min(a, b), max(a, b)),
                  (min(b, c), max(b, c)),
                  (min(a, c), max(a, c))]:
            edge_map.setdefault(e, []).append(fi)
    return edge_map


def compute_dihedral_angles(face_normals: np.ndarray,
                             faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (angles, face_pairs) where angles[i] is the dihedral angle
    (in radians, 0..π) at the shared edge between face_pairs[i,0] and
    face_pairs[i,1].
    """
    edge_map = build_edge_face_map(faces)
    angles, pairs = [], []
    for shared_faces in edge_map.values():
        if len(shared_faces) == 2:
            fi, fj = shared_faces
            cos_a = np.clip(np.dot(face_normals[fi], face_normals[fj]), -1.0, 1.0)
            angles.append(np.arccos(cos_a))
            pairs.append((fi, fj))
    return np.array(angles), np.array(pairs)


def compute_gaussian_curvature(vertices: np.ndarray,
                                faces: np.ndarray) -> np.ndarray:
    """
    Discrete Gaussian curvature via angle deficit: K(v) = 2π − Σ θ_i
    Normalised by mixed Voronoi area.
    Returns per-vertex Gaussian curvature.
    """
    n_verts = len(vertices)
    angle_sum = np.zeros(n_verts)
    voronoi_area = np.zeros(n_verts)

    for fi, (a, b, c) in enumerate(faces):
        verts = [vertices[a], vertices[b], vertices[c]]
        idx = [a, b, c]
        # angles at each corner
        for i in range(3):
            vi = verts[i]
            vj = verts[(i + 1) % 3]
            vk = verts[(i + 2) % 3]
            e1 = vj - vi
            e2 = vk - vi
            l1, l2 = np.linalg.norm(e1), np.linalg.norm(e2)
            if l1 < 1e-12 or l2 < 1e-12:
                continue
            cos_a = np.clip(np.dot(e1, e2) / (l1 * l2), -1.0, 1.0)
            angle = np.arccos(cos_a)
            angle_sum[idx[i]] += angle
            voronoi_area[idx[i]] += 0.5 * l1 * l2 * np.sin(angle) / 3.0

    with np.errstate(divide='ignore', invalid='ignore'):
        K = np.where(voronoi_area > 1e-12, (2 * np.pi - angle_sum) / voronoi_area, 0.0)
    return K


def compute_mean_curvature(vertices: np.ndarray,
                            faces: np.ndarray,
                            face_normals: np.ndarray) -> np.ndarray:
    """
    Approximate mean curvature from dihedral angles:
    H_edge ≈ |e| * dihedral_angle / (2 * area)
    Accumulated per vertex.
    """
    n_verts = len(vertices)
    H = np.zeros(n_verts)
    area = np.zeros(n_verts)

    edge_map = build_edge_face_map(faces)
    face_areas = compute_face_areas(vertices, faces)

    for (vi, vj), shared in edge_map.items():
        if len(shared) != 2:
            continue
        fi, fj = shared
        cos_a = np.clip(np.dot(face_normals[fi], face_normals[fj]), -1.0, 1.0)
        dihedral = np.arccos(cos_a)
        edge_len = np.linalg.norm(vertices[vi] - vertices[vj])
        contribution = 0.5 * edge_len * dihedral
        for v in (vi, vj):
            H[v] += contribution
            area[v] += (face_areas[fi] + face_areas[fj]) / 6.0

    with np.errstate(divide='ignore', invalid='ignore'):
        H = np.where(area > 1e-12, H / area, 0.0)
    return H


def icosahedron_normals() -> np.ndarray:
    """20 face-normal directions of a unit icosahedron — used for sphere binning."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1,  phi, 0], [1,  phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1,  phi], [0,  1,  phi], [0, -1, -phi], [0,  1, -phi],
        [ phi, 0, -1], [phi,  0,  1], [-phi, 0, -1], [-phi,  0,  1],
    ], dtype=float)
    verts /= np.linalg.norm(verts[0])
    faces = np.array([
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
    ])
    return compute_face_normals(verts, faces)   # 20 × 3


ICO_NORMALS = icosahedron_normals()             # cached


def bin_normals_icosphere(normals: np.ndarray, n_bins: int = 20) -> np.ndarray:
    """Map unit normals to nearest icosahedron face → histogram of length n_bins."""
    assert normals.ndim == 2 and normals.shape[1] == 3
    bins = ICO_NORMALS[:n_bins]
    # dot product similarity; use abs() to treat opposite normals as same bin
    dots = np.abs(normals @ bins.T)          # (N, n_bins)
    assignments = np.argmax(dots, axis=1)
    hist, _ = np.histogram(assignments, bins=np.arange(n_bins + 1))
    return hist.astype(float)


# ─────────────────────────────────────────────────────────────────────────────
#  Base class
# ─────────────────────────────────────────────────────────────────────────────

class MeshDescriptor:
    """Abstract base class for all mesh descriptors."""
    name: str = "base"

    def compute(self, vertices: np.ndarray, faces: np.ndarray,
                semantics: Optional[np.ndarray] = None) -> np.ndarray:
        raise NotImplementedError

    def describe(self) -> str:
        return self.__doc__ or self.name


# ─────────────────────────────────────────────────────────────────────────────
#  1. Normal Histogram Descriptor
# ─────────────────────────────────────────────────────────────────────────────

class NormalHistogram(MeshDescriptor):
    """
    Global histogram of face-normal orientations on the unit sphere.
    Uses icosahedral discretisation (20 bins by default) weighted by face area.
    Sensor-agnostic: captures dominant surface orientations regardless of resolution.
    """
    name = "NormalHistogram"

    def __init__(self, n_bins: int = 20, area_weighted: bool = True):
        self.n_bins = n_bins
        self.area_weighted = area_weighted

    def compute(self, vertices, faces, semantics=None):
        normals = compute_face_normals(vertices, faces)
        bins = ICO_NORMALS[:self.n_bins]
        dots = np.abs(normals @ bins.T)
        assignments = np.argmax(dots, axis=1)

        if self.area_weighted:
            areas = compute_face_areas(vertices, faces)
        else:
            areas = np.ones(len(faces))

        hist = np.zeros(self.n_bins)
        np.add.at(hist, assignments, areas)
        total = hist.sum()
        if total > 0:
            hist /= total
        return hist


# ─────────────────────────────────────────────────────────────────────────────
#  2. Dihedral Angle Histogram
# ─────────────────────────────────────────────────────────────────────────────

class DihedralHistogram(MeshDescriptor):
    """
    Histogram of dihedral angles along all shared edges.
    Smooth surfaces → angles cluster near π.
    Sharp edges / complex geometry → angles spread toward 0.
    Sensor-agnostic: encodes surface bending, not sampling density.
    """
    name = "DihedralHistogram"

    def __init__(self, n_bins: int = 18):
        self.n_bins = n_bins

    def compute(self, vertices, faces, semantics=None):
        normals = compute_face_normals(vertices, faces)
        angles, _ = compute_dihedral_angles(normals, faces)
        if len(angles) == 0:
            return np.zeros(self.n_bins)
        hist, _ = np.histogram(angles, bins=self.n_bins, range=(0, np.pi))
        hist = hist.astype(float)
        if hist.sum() > 0:
            hist /= hist.sum()
        return hist


# ─────────────────────────────────────────────────────────────────────────────
#  3. Curvature Histogram
# ─────────────────────────────────────────────────────────────────────────────

class CurvatureHistogram(MeshDescriptor):
    """
    Concatenated histogram of Gaussian curvature (angle-deficit) and
    mean curvature (dihedral-based). Captures local surface shape.
    """
    name = "CurvatureHistogram"

    def __init__(self, n_bins: int = 10, clip_percentile: float = 95.0):
        self.n_bins = n_bins
        self.clip = clip_percentile

    def compute(self, vertices, faces, semantics=None):
        normals = compute_face_normals(vertices, faces)
        K = compute_gaussian_curvature(vertices, faces)
        H = compute_mean_curvature(vertices, faces, normals)

        def _hist(vals):
            lo, hi = np.percentile(vals, 100 - self.clip), np.percentile(vals, self.clip)
            if hi <= lo:
                return np.zeros(self.n_bins)
            h, _ = np.histogram(np.clip(vals, lo, hi), bins=self.n_bins, range=(lo, hi))
            h = h.astype(float)
            if h.sum() > 0:
                h /= h.sum()
            return h

        return np.concatenate([_hist(K), _hist(H)])


# ─────────────────────────────────────────────────────────────────────────────
#  4. Spin Image (Johnson & Hebert, 1999)
# ─────────────────────────────────────────────────────────────────────────────

class SpinImage(MeshDescriptor):
    """
    Spin Image descriptor (Johnson & Hebert 1997/1999).
    For each oriented point (position + normal), accumulates nearby
    vertices in a 2D (α, β) cylindrical coordinate system:
      α = radial distance from the spin axis (the normal)
      β = signed height along the normal

    Sensitive to mesh resolution — use with coarse images (low res) for
    cross-modal robustness.
    """
    name = "SpinImage"

    def __init__(self, image_width: int = 8, image_height: int = 8,
                 support_angle_deg: float = 60.0, bin_size: float = 0.5):
        self.W = image_width
        self.H = image_height
        self.support_cos = np.cos(np.deg2rad(support_angle_deg))
        self.bin_size = bin_size

    def _spin_for_point(self, center: np.ndarray, normal: np.ndarray,
                         all_pts: np.ndarray) -> np.ndarray:
        img = np.zeros((self.H, self.W))
        delta = all_pts - center
        beta = delta @ normal                              # height along axis
        alpha = np.sqrt(np.maximum(
            np.sum(delta**2, axis=1) - beta**2, 0.0))    # radial distance

        # Support angle filter
        norms_d = np.linalg.norm(delta, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            cos_a = np.where(norms_d > 1e-12, beta / norms_d, 0.0)
        mask = cos_a >= self.support_cos

        ai = np.floor(alpha[mask] / self.bin_size).astype(int)
        bi = np.floor((beta[mask] + self.H * self.bin_size / 2) / self.bin_size).astype(int)
        valid = (ai >= 0) & (ai < self.W) & (bi >= 0) & (bi < self.H)
        np.add.at(img, (bi[valid], ai[valid]), 1)
        return img.ravel()

    def compute(self, vertices, faces, semantics=None):
        normals = compute_face_normals(vertices, faces)
        centroids = compute_face_centroids(vertices, faces)

        # Use only a subset for speed (up to 200 keypoints)
        n = len(centroids)
        idx = np.random.choice(n, min(n, 200), replace=False)
        spins = [self._spin_for_point(centroids[i], normals[i], centroids)
                 for i in idx]
        if not spins:
            return np.zeros(self.W * self.H)
        # Aggregate: mean spin image
        mean_spin = np.mean(spins, axis=0)
        if mean_spin.sum() > 0:
            mean_spin /= mean_spin.sum()
        return mean_spin


# ─────────────────────────────────────────────────────────────────────────────
#  5. Mesh-FPFH  (FPFH-inspired, adapted for triangle meshes)
# ─────────────────────────────────────────────────────────────────────────────

class MeshFPFH(MeshDescriptor):
    """
    FPFH-inspired descriptor adapted for triangle meshes (Rusu et al., 2009).
    For each face, computes Darboux-frame angles (α, φ, θ) with neighbouring
    faces, histogramming the angular relationships between normals.
    11 bins each × 3 angles = 33-dim descriptor.
    """
    name = "MeshFPFH"

    def __init__(self, n_bins: int = 11, n_rings: int = 2):
        self.n_bins = n_bins
        self.n_rings = n_rings

    def _spfh(self, fi: int, normals: np.ndarray, centroids: np.ndarray,
              tree: cKDTree, k: int = 8) -> np.ndarray:
        """Simplified Point Feature Histogram for face fi."""
        dists, neighbors = tree.query(centroids[fi], k=k + 1)
        neighbors = neighbors[1:]   # exclude self

        alpha_hist = np.zeros(self.n_bins)
        phi_hist   = np.zeros(self.n_bins)
        theta_hist = np.zeros(self.n_bins)

        n_s = normals[fi]
        p_s = centroids[fi]
        count = 0

        for fj in neighbors:
            n_t = normals[fj]
            p_t = centroids[fj]
            d = p_t - p_s
            d_norm = np.linalg.norm(d)
            if d_norm < 1e-12:
                continue
            d_unit = d / d_norm

            # Darboux frame
            u = n_s
            v = np.cross(d_unit, u)
            w = np.cross(u, v)

            alpha = np.dot(v, n_t)                         # ∈ [-1, 1]
            phi   = np.dot(u, d_unit)                      # ∈ [-1, 1]
            theta = np.arctan2(np.dot(w, n_t),
                               np.dot(u, n_t)) / np.pi     # ∈ [-1, 1]

            for val, hist in [(alpha, alpha_hist),
                              (phi,   phi_hist),
                              (theta, theta_hist)]:
                bin_idx = int(np.clip((val + 1) / 2 * self.n_bins, 0, self.n_bins - 1))
                hist[bin_idx] += 1
            count += 1

        if count > 0:
            alpha_hist /= count
            phi_hist   /= count
            theta_hist /= count

        return np.concatenate([alpha_hist, phi_hist, theta_hist])

    def compute(self, vertices, faces, semantics=None):
        normals   = compute_face_normals(vertices, faces)
        centroids = compute_face_centroids(vertices, faces)
        tree      = cKDTree(centroids)

        n = len(centroids)
        idx = np.random.choice(n, min(n, 300), replace=False)
        spfhs = np.array([self._spfh(i, normals, centroids, tree) for i in idx])

        # FPFH: weighted sum of neighbouring SPFH (here: just mean for speed)
        desc = spfhs.mean(axis=0) if len(spfhs) > 0 else np.zeros(3 * self.n_bins)
        if desc.sum() > 0:
            desc /= desc.sum()
        return desc


# ─────────────────────────────────────────────────────────────────────────────
#  6. Scan Context (Mesh variant)
# ─────────────────────────────────────────────────────────────────────────────

class ScanContextMesh(MeshDescriptor):
    """
    Scan-Context style descriptor adapted for meshes (Kim & Kim, IROS 2018).
    Divides space into a polar grid (n_rings × n_sectors).
    Instead of max-height, encodes: mean normal elevation, mean dihedral angle,
    and face density per cell — stacked into a 2D image then flattened.

    Rotation-invariant via column-shifted matching (handled externally).
    """
    name = "ScanContextMesh"

    def __init__(self, n_rings: int = 8, n_sectors: int = 16,
                 max_radius: Optional[float] = None):
        self.R = n_rings
        self.S = n_sectors
        self.max_radius = max_radius

    def compute(self, vertices, faces, semantics=None):
        normals   = compute_face_normals(vertices, faces)
        centroids = compute_face_centroids(vertices, faces)

        # Compute dihedral angles per-face (mean over adjacent edges)
        dihedral_per_face = np.full(len(faces), np.pi / 2)
        _, pairs = compute_dihedral_angles(normals, faces)
        dihedral_vals, _ = compute_dihedral_angles(normals, faces)
        if len(pairs) > 0:
            face_dihedral_acc   = np.zeros(len(faces))
            face_dihedral_count = np.zeros(len(faces))
            for (fi, fj), ang in zip(pairs, dihedral_vals):
                face_dihedral_acc[fi]   += ang
                face_dihedral_acc[fj]   += ang
                face_dihedral_count[fi] += 1
                face_dihedral_count[fj] += 1
            mask = face_dihedral_count > 0
            dihedral_per_face[mask] = (face_dihedral_acc[mask] /
                                       face_dihedral_count[mask])

        # Project to XY plane; use centroid XY for binning
        cx, cy = centroids[:, 0], centroids[:, 1]
        # Centre
        cx = cx - cx.mean()
        cy = cy - cy.mean()

        r = np.sqrt(cx**2 + cy**2)
        max_r = self.max_radius if self.max_radius else (np.percentile(r, 95) + 1e-6)
        theta = np.arctan2(cy, cx) + np.pi     # [0, 2π]

        ring_idx    = np.floor(r / max_r * self.R).astype(int).clip(0, self.R - 1)
        sector_idx  = np.floor(theta / (2 * np.pi) * self.S).astype(int).clip(0, self.S - 1)

        # 3-channel cell map: normal elevation (z), dihedral, density
        sc_elev  = np.zeros((self.R, self.S))
        sc_dihed = np.zeros((self.R, self.S))
        sc_count = np.zeros((self.R, self.S))

        np.add.at(sc_elev,  (ring_idx, sector_idx), normals[:, 2])
        np.add.at(sc_dihed, (ring_idx, sector_idx), dihedral_per_face)
        np.add.at(sc_count, (ring_idx, sector_idx), 1)

        mask = sc_count > 0
        sc_elev[mask]  /= sc_count[mask]
        sc_dihed[mask] /= sc_count[mask]
        sc_count       /= (sc_count.max() + 1e-6)

        sc = np.stack([sc_elev, sc_dihed, sc_count], axis=-1)    # R × S × 3
        desc = sc.ravel()
        # Normalise
        if np.linalg.norm(desc) > 0:
            desc = desc / np.linalg.norm(desc)
        return desc


# ─────────────────────────────────────────────────────────────────────────────
#  7. NDS – Normal + Dihedral + Semantic place descriptor (our proposal)
# ─────────────────────────────────────────────────────────────────────────────

class NDSDescriptor(MeshDescriptor):
    """
    Normal-Dihedral-Semantic (NDS) descriptor — the proposed cross-modal baseline.

    Pipeline
    --------
    1. Divide mesh into a polar grid (Scan-Context style)
    2. Per cell, compute:
       a) Normal orientation histogram on icosphere (n_normal_bins)
       b) Dihedral angle histogram (n_dihedral_bins)
       c) Semantic composition histogram (n_sem_classes) — if labels provided
    3. Concatenate per-cell vectors → flatten to global descriptor
    4. L2-normalise

    Cross-modal robustness
    ----------------------
    - Histograms absorb resolution differences (stereo vs LiDAR mesh density)
    - Surface properties (normals, dihedral) are independent of sensor
    - Semantics is the most sensor-agnostic signal when available
    """
    name = "NDS"

    def __init__(self, n_rings: int = 4, n_sectors: int = 8,
                 n_normal_bins: int = 20, n_dihedral_bins: int = 10,
                 n_sem_classes: int = 10, max_radius: Optional[float] = None):
        self.R = n_rings
        self.S = n_sectors
        self.n_nb = n_normal_bins
        self.n_db = n_dihedral_bins
        self.n_sc = n_sem_classes
        self.max_radius = max_radius

        self.cell_dim = n_normal_bins + n_dihedral_bins + n_sem_classes
        self.descriptor_dim = n_rings * n_sectors * self.cell_dim

    def compute(self, vertices, faces, semantics=None):
        """
        Parameters
        ----------
        vertices : (V, 3) float
        faces    : (F, 3) int
        semantics: (F,) int  — per-face class labels [0..n_sem_classes-1], or None
        """
        normals   = compute_face_normals(vertices, faces)
        centroids = compute_face_centroids(vertices, faces)
        areas     = compute_face_areas(vertices, faces)

        # Dihedral per face (mean over adjacent edges)
        dihedral_per_face = np.full(len(faces), np.pi / 2)
        dihedral_vals, pairs = compute_dihedral_angles(normals, faces)
        if len(pairs) > 0:
            acc   = np.zeros(len(faces))
            count = np.zeros(len(faces))
            for (fi, fj), ang in zip(pairs, dihedral_vals):
                acc[fi] += ang; acc[fj] += ang
                count[fi] += 1; count[fj] += 1
            m = count > 0
            dihedral_per_face[m] = acc[m] / count[m]

        # Polar binning in XY
        cx, cy = centroids[:, 0] - centroids[:, 0].mean(), \
                 centroids[:, 1] - centroids[:, 1].mean()
        r = np.sqrt(cx**2 + cy**2)
        max_r = self.max_radius if self.max_radius else (np.percentile(r, 95) + 1e-6)
        theta = (np.arctan2(cy, cx) + np.pi)   # [0, 2π]

        ring_idx   = np.floor(r / max_r * self.R).astype(int).clip(0, self.R - 1)
        sector_idx = np.floor(theta / (2 * np.pi) * self.S).astype(int).clip(0, self.S - 1)

        descriptor = np.zeros((self.R, self.S, self.cell_dim))

        for ri in range(self.R):
            for si in range(self.S):
                mask = (ring_idx == ri) & (sector_idx == si)
                if not np.any(mask):
                    continue

                cell_normals   = normals[mask]
                cell_dihedrals = dihedral_per_face[mask]
                cell_areas     = areas[mask]

                # a) Normal histogram (area-weighted)
                ico_bins = ICO_NORMALS[:self.n_nb]
                dots = np.abs(cell_normals @ ico_bins.T)
                asgn = np.argmax(dots, axis=1)
                nh   = np.zeros(self.n_nb)
                np.add.at(nh, asgn, cell_areas)
                if nh.sum() > 0: nh /= nh.sum()

                # b) Dihedral histogram
                dh, _ = np.histogram(cell_dihedrals, bins=self.n_db, range=(0, np.pi))
                dh = dh.astype(float)
                if dh.sum() > 0: dh /= dh.sum()

                # c) Semantic histogram
                sh = np.zeros(self.n_sc)
                if semantics is not None:
                    cell_sem = semantics[mask]
                    for label in range(self.n_sc):
                        sh[label] = np.sum(cell_areas[cell_sem == label])
                    if sh.sum() > 0: sh /= sh.sum()
                else:
                    sh[0] = 1.0   # unknown class

                descriptor[ri, si, :] = np.concatenate([nh, dh, sh])

        desc = descriptor.ravel()
        norm = np.linalg.norm(desc)
        if norm > 0:
            desc /= norm
        return desc


# ─────────────────────────────────────────────────────────────────────────────
#  Registry
# ─────────────────────────────────────────────────────────────────────────────

DESCRIPTOR_REGISTRY: Dict[str, MeshDescriptor] = {
    "NormalHistogram":   NormalHistogram(),
    "DihedralHistogram": DihedralHistogram(),
    "CurvatureHistogram": CurvatureHistogram(),
    "SpinImage":         SpinImage(),
    "MeshFPFH":          MeshFPFH(),
    "ScanContextMesh":   ScanContextMesh(),
    "NDS":               NDSDescriptor(),
}


def get_descriptor(name: str) -> MeshDescriptor:
    if name not in DESCRIPTOR_REGISTRY:
        raise ValueError(f"Unknown descriptor '{name}'. "
                         f"Available: {list(DESCRIPTOR_REGISTRY.keys())}")
    return DESCRIPTOR_REGISTRY[name]
