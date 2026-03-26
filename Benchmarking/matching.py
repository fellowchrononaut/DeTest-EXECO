"""
mesh_descriptors/matching.py
=============================
Matching, similarity metrics, and evaluation utilities.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
import time


# ─────────────────────────────────────────────────────────────────────────────
#  Similarity / distance metrics
# ─────────────────────────────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def l2_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Convert L2 distance to a [0,1] similarity score."""
    d = l2_distance(a, b)
    return float(1.0 / (1.0 + d))


def chi2_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Chi-squared distance for histograms."""
    denom = a + b + 1e-10
    return float(0.5 * np.sum((a - b)**2 / denom))


def chi2_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.exp(-chi2_distance(a, b)))


def bhattacharyya_coefficient(a: np.ndarray, b: np.ndarray) -> float:
    """Bhattacharyya coefficient for normalised histograms. Range [0,1]."""
    a_n = a / (a.sum() + 1e-10)
    b_n = b / (b.sum() + 1e-10)
    return float(np.sum(np.sqrt(np.maximum(0.0, a_n * b_n))))


METRIC_REGISTRY = {
    "cosine":          cosine_similarity,
    "l2":              l2_similarity,
    "chi2":            chi2_similarity,
    "bhattacharyya":   bhattacharyya_coefficient,
}


def scan_context_distance(a: np.ndarray, b: np.ndarray,
                          n_sectors: int = 16) -> float:
    """
    Rotation-invariant Scan Context distance.
    Tries all column shifts and returns minimum cosine distance.
    a, b must be flat Scan Context descriptors.
    """
    # Reshape into (n_rings * n_channels, n_sectors) for column shifting
    len_a = len(a)
    if len_a % n_sectors != 0:
        return 1 - cosine_similarity(a, b)

    cols = n_sectors
    rows = len_a // cols
    ma = a.reshape(rows, cols)
    mb = b.reshape(rows, cols)

    best = 1.0
    for shift in range(cols):
        mb_shifted = np.roll(mb, shift, axis=1)
        sim = cosine_similarity(ma.ravel(), mb_shifted.ravel())
        best = min(best, 1 - sim)
    return best


# ─────────────────────────────────────────────────────────────────────────────
#  Descriptor comparison result
# ─────────────────────────────────────────────────────────────────────────────

class MatchResult:
    def __init__(self, descriptor_name: str, scene_name: str,
                 desc_stereo: np.ndarray, desc_lidar: np.ndarray,
                 compute_time_stereo_ms: float, compute_time_lidar_ms: float):
        self.descriptor_name = descriptor_name
        self.scene_name = scene_name
        self.desc_stereo = desc_stereo
        self.desc_lidar  = desc_lidar
        self.dim = len(desc_stereo)
        self.compute_time_stereo_ms = compute_time_stereo_ms
        self.compute_time_lidar_ms  = compute_time_lidar_ms

        # Compute all metrics
        self.scores: Dict[str, float] = {}
        for name, fn in METRIC_REGISTRY.items():
            try:
                self.scores[name] = fn(desc_stereo, desc_lidar)
            except Exception:
                self.scores[name] = float('nan')

    @property
    def primary_score(self) -> float:
        return self.scores.get("cosine", 0.0)

    def __repr__(self):
        s = (f"MatchResult({self.descriptor_name} | {self.scene_name} | "
             f"dim={self.dim} | cosine={self.primary_score:.4f})")
        return s


# ─────────────────────────────────────────────────────────────────────────────
#  Batch evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_descriptor_on_scene(descriptor, scene: Dict) -> MatchResult:
    """Run a single descriptor on a single scene and return a MatchResult."""
    v_s, f_s, sem_s = scene['stereo_mesh']
    v_l, f_l, sem_l = scene['lidar_mesh']

    t0 = time.perf_counter()
    desc_s = descriptor.compute(v_s, f_s, sem_s)
    t1 = time.perf_counter()
    desc_l = descriptor.compute(v_l, f_l, sem_l)
    t2 = time.perf_counter()

    return MatchResult(
        descriptor_name=descriptor.name,
        scene_name=scene['name'],
        desc_stereo=desc_s,
        desc_lidar=desc_l,
        compute_time_stereo_ms=(t1 - t0) * 1000,
        compute_time_lidar_ms=(t2 - t1) * 1000,
    )


def run_full_benchmark(descriptors: List, scenes: List[Dict]) -> List[MatchResult]:
    """Run all descriptors × all scenes."""
    results = []
    for scene in scenes:
        for desc in descriptors:
            try:
                r = evaluate_descriptor_on_scene(desc, scene)
                results.append(r)
            except Exception as e:
                print(f"  ERROR: {desc.name} on {scene['name']}: {e}")
    return results


def results_to_matrix(results: List[MatchResult],
                       metric: str = "cosine") -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Convert results list into a matrix [descriptors × scenes].
    Returns (matrix, descriptor_names, scene_names).
    """
    desc_names  = sorted(set(r.descriptor_name for r in results))
    scene_names = sorted(set(r.scene_name for r in results))
    mat = np.full((len(desc_names), len(scene_names)), np.nan)

    desc_idx  = {n: i for i, n in enumerate(desc_names)}
    scene_idx = {n: i for i, n in enumerate(scene_names)}

    for r in results:
        di = desc_idx[r.descriptor_name]
        si = scene_idx[r.scene_name]
        mat[di, si] = r.scores.get(metric, np.nan)

    return mat, desc_names, scene_names


def print_summary_table(results: List[MatchResult], metric: str = "cosine"):
    mat, desc_names, scene_names = results_to_matrix(results, metric)
    col_w = max(max(len(s) for s in scene_names), 14)
    desc_w = max(max(len(d) for d in desc_names), 20)

    header = f"{'Descriptor':<{desc_w}}" + "".join(f"  {s:>{col_w}}" for s in scene_names)
    print("\n" + "=" * len(header))
    print(f"Metric: {metric.upper()}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for i, desc in enumerate(desc_names):
        row = f"{desc:<{desc_w}}"
        for j in range(len(scene_names)):
            v = mat[i, j]
            row += f"  {v:>{col_w}.4f}" if not np.isnan(v) else f"  {'nan':>{col_w}}"
        print(row)
    print("=" * len(header) + "\n")
