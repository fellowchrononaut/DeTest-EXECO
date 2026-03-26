"""
mesh_descriptors/visualisation.py
===================================
Plotting utilities: descriptor vectors, similarity matrices, mesh previews.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import List, Dict, Optional, Tuple
from matching import MatchResult, results_to_matrix


PALETTE = {
    "bg":      "#0d1117",
    "surface": "#161b22",
    "border":  "#30363d",
    "accent1": "#58a6ff",
    "accent2": "#3fb950",
    "accent3": "#f0883e",
    "accent4": "#ff7b72",
    "text":    "#e6edf3",
    "subtext": "#8b949e",
}

SEM_COLORS = {
    0: "#444444",   # unknown
    1: "#8b6914",   # ground
    2: "#1f6feb",   # wall
    3: "#da3633",   # roof
    4: "#238636",   # vegetation
    5: "#a371f7",   # vehicle
}

DESC_COLORS = [
    "#58a6ff", "#3fb950", "#f0883e", "#ff7b72",
    "#d2a8ff", "#ffa657", "#79c0ff", "#56d364",
]

plt.rcParams.update({
    "figure.facecolor":   PALETTE["bg"],
    "axes.facecolor":     PALETTE["surface"],
    "axes.edgecolor":     PALETTE["border"],
    "axes.labelcolor":    PALETTE["text"],
    "xtick.color":        PALETTE["subtext"],
    "ytick.color":        PALETTE["subtext"],
    "text.color":         PALETTE["text"],
    "grid.color":         PALETTE["border"],
    "grid.linestyle":     "--",
    "grid.linewidth":     0.5,
    "font.family":        "monospace",
})


def _ax_style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PALETTE["surface"])
    for sp in ax.spines.values():
        sp.set_color(PALETTE["border"])
    if title:  ax.set_title(title, color=PALETTE["text"], fontsize=10, pad=8)
    if xlabel: ax.set_xlabel(xlabel, color=PALETTE["subtext"], fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color=PALETTE["subtext"], fontsize=9)
    ax.grid(True, alpha=0.4)


# ─────────────────────────────────────────────────────────────────────────────
#  Mesh preview
# ─────────────────────────────────────────────────────────────────────────────

def plot_mesh_pair(scene: Dict, max_faces: int = 3000,
                   save_path: Optional[str] = None):
    """Side-by-side 3D preview of stereo and LiDAR mesh (coloured by semantics)."""
    fig = plt.figure(figsize=(14, 6), facecolor=PALETTE["bg"])
    fig.suptitle(f"Scene: {scene['name']}  |  {scene['description']}",
                 color=PALETTE["text"], fontsize=11, y=1.01)

    for col, (key, label) in enumerate([("stereo_mesh", "Stereo (Drone)"),
                                        ("lidar_mesh",  "LiDAR (Ground Robot)")]):
        verts, faces, sems = scene[key]
        ax = fig.add_subplot(1, 2, col + 1, projection='3d')
        ax.set_facecolor(PALETTE["bg"])

        # Sub-sample for display speed
        n = len(faces)
        idx = np.random.choice(n, min(n, max_faces), replace=False)
        f_sub = faces[idx]
        s_sub = sems[idx]

        tris  = verts[f_sub]   # (F, 3, 3)
        colors = [SEM_COLORS.get(s, "#888888") for s in s_sub]
        poly = Poly3DCollection(tris, alpha=0.6, linewidth=0)
        poly.set_facecolors(colors)
        ax.add_collection3d(poly)

        lims = [(verts[:, i].min(), verts[:, i].max()) for i in range(3)]
        for i, (lo, hi) in enumerate(lims):
            pad = (hi - lo) * 0.05 + 0.5
            getattr(ax, f"set_{'xyz'[i]}lim")(lo - pad, hi + pad)

        ax.set_title(f"{label}\n{len(verts)} verts · {len(faces)} faces",
                     color=PALETTE["text"], fontsize=10)
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor(PALETTE["border"])
        ax.tick_params(colors=PALETTE["subtext"], labelsize=7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight",
                    facecolor=PALETTE["bg"])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Descriptor vector comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_descriptor_comparison(results: List[MatchResult],
                                scene_name: str,
                                save_path: Optional[str] = None):
    """
    For each descriptor, plot stereo vs LiDAR descriptor vectors side by side,
    along with all similarity scores.
    """
    scene_results = [r for r in results if r.scene_name == scene_name]
    if not scene_results:
        print(f"No results for scene '{scene_name}'")
        return

    n = len(scene_results)
    fig = plt.figure(figsize=(16, 3.5 * n), facecolor=PALETTE["bg"])
    fig.suptitle(f"Descriptor Vectors — Scene: {scene_name}",
                 color=PALETTE["text"], fontsize=13, y=1.002)

    gs = gridspec.GridSpec(n, 3, figure=fig, wspace=0.4, hspace=0.6,
                           width_ratios=[3, 3, 1.2])

    for row, r in enumerate(scene_results):
        # Stereo descriptor
        ax_s = fig.add_subplot(gs[row, 0])
        xs = np.arange(len(r.desc_stereo))
        ax_s.bar(xs, r.desc_stereo, color=PALETTE["accent1"], alpha=0.8, width=1.0)
        _ax_style(ax_s, title=f"{r.descriptor_name}  |  Stereo (Drone)",
                  xlabel=f"dim={r.dim}", ylabel="value")

        # LiDAR descriptor
        ax_l = fig.add_subplot(gs[row, 1])
        ax_l.bar(xs[:len(r.desc_lidar)], r.desc_lidar,
                 color=PALETTE["accent2"], alpha=0.8, width=1.0)
        _ax_style(ax_l, title=f"{r.descriptor_name}  |  LiDAR (Ground Robot)",
                  xlabel=f"dim={len(r.desc_lidar)}", ylabel="value")

        # Score table
        ax_t = fig.add_subplot(gs[row, 2])
        ax_t.axis('off')
        metrics = list(r.scores.keys())
        values  = [f"{r.scores[m]:.4f}" for m in metrics]
        times   = [f"{r.compute_time_stereo_ms:.1f}ms",
                   f"{r.compute_time_lidar_ms:.1f}ms"]
        table_data = [[m, v] for m, v in zip(metrics, values)]
        table_data += [["---", "---"], ["t_stereo", times[0]], ["t_lidar", times[1]]]
        tbl = ax_t.table(cellText=table_data, colLabels=["Metric", "Score"],
                         loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        for (ri, ci), cell in tbl.get_celld().items():
            cell.set_facecolor(PALETTE["surface"] if ri > 0 else PALETTE["border"])
            cell.set_edgecolor(PALETTE["border"])
            cell.set_text_props(color=PALETTE["text"])
        ax_t.set_title("Scores", color=PALETTE["text"], fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight",
                    facecolor=PALETTE["bg"])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Heatmap: all descriptors × all scenes
# ─────────────────────────────────────────────────────────────────────────────

def plot_similarity_heatmap(results: List[MatchResult],
                             metric: str = "cosine",
                             save_path: Optional[str] = None):
    """Heatmap of similarity scores: rows=descriptors, columns=scenes."""
    mat, desc_names, scene_names = results_to_matrix(results, metric)

    fig, ax = plt.subplots(figsize=(max(8, len(scene_names) * 2),
                                    max(5, len(desc_names) * 1.2)),
                            facecolor=PALETTE["bg"])

    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                   interpolation="nearest")

    # Annotate cells
    for i in range(len(desc_names)):
        for j in range(len(scene_names)):
            v = mat[i, j]
            if not np.isnan(v):
                txt_color = "black" if 0.3 < v < 0.85 else "white"
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        color=txt_color, fontsize=9, fontweight="bold")

    ax.set_xticks(range(len(scene_names)))
    ax.set_xticklabels(scene_names, rotation=30, ha="right",
                       color=PALETTE["text"], fontsize=9)
    ax.set_yticks(range(len(desc_names)))
    ax.set_yticklabels(desc_names, color=PALETTE["text"], fontsize=9)

    for sp in ax.spines.values():
        sp.set_color(PALETTE["border"])

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color=PALETTE["subtext"])
    cbar.ax.tick_params(colors=PALETTE["subtext"], labelsize=8)
    cbar.set_label(f"{metric.upper()} similarity", color=PALETTE["text"], fontsize=10)

    ax.set_title(f"Cross-Modal Match Quality  ({metric.upper()})\n"
                 "Higher = more similar Stereo↔LiDAR descriptor",
                 color=PALETTE["text"], fontsize=12, pad=12)
    ax.set_facecolor(PALETTE["surface"])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight",
                    facecolor=PALETTE["bg"])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Timing bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_timing(results: List[MatchResult], save_path: Optional[str] = None):
    """Bar chart of computation times per descriptor."""
    desc_names = sorted(set(r.descriptor_name for r in results))
    stereo_times = []
    lidar_times  = []
    for d in desc_names:
        dr = [r for r in results if r.descriptor_name == d]
        stereo_times.append(np.mean([r.compute_time_stereo_ms for r in dr]))
        lidar_times.append(np.mean([r.compute_time_lidar_ms for r in dr]))

    x = np.arange(len(desc_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=PALETTE["bg"])
    ax.bar(x - width/2, stereo_times, width, label="Stereo (Drone)",
           color=PALETTE["accent1"], alpha=0.85)
    ax.bar(x + width/2, lidar_times,  width, label="LiDAR (Ground)",
           color=PALETTE["accent2"], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(desc_names, rotation=25, ha="right",
                       color=PALETTE["text"], fontsize=9)
    _ax_style(ax, title="Mean Descriptor Computation Time",
              xlabel="Descriptor", ylabel="Time (ms)")
    ax.legend(facecolor=PALETTE["surface"], edgecolor=PALETTE["border"],
              labelcolor=PALETTE["text"], fontsize=9)
    ax.set_yscale("log")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight",
                    facecolor=PALETTE["bg"])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Radar / spider chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_radar(results: List[MatchResult],
               metric: str = "cosine",
               save_path: Optional[str] = None):
    """
    Radar chart: for each descriptor, how well it does across all scenes.
    Reveals which descriptors are consistent vs scene-dependent.
    """
    mat, desc_names, scene_names = results_to_matrix(results, metric)
    N = len(scene_names)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True),
                            facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["surface"])
    ax.spines["polar"].set_color(PALETTE["border"])

    for i, (desc, color) in enumerate(zip(desc_names,
                                           DESC_COLORS[:len(desc_names)])):
        values = mat[i, :].tolist()
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=2, label=desc)
        ax.fill(angles, values, color=color, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(scene_names, color=PALETTE["text"], fontsize=9)
    ax.yaxis.set_tick_params(labelcolor=PALETTE["subtext"])
    ax.set_ylim(0, 1)
    ax.set_title(f"Cross-Modal Descriptor Consistency  ({metric.upper()})",
                 color=PALETTE["text"], fontsize=12, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1),
              facecolor=PALETTE["surface"], edgecolor=PALETTE["border"],
              labelcolor=PALETTE["text"], fontsize=8)
    ax.grid(color=PALETTE["border"], linestyle="--", linewidth=0.6)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight",
                    facecolor=PALETTE["bg"])
    return fig
