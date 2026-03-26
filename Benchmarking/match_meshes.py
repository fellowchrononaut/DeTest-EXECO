#!/usr/bin/env python3
"""
mesh_descriptors/match_meshes.py
==================================
Match two real mesh files using any descriptor from the framework.

Usage examples
--------------
  # Compare two files with all descriptors, cosine metric
  python match_meshes.py stereo.ply lidar.ply

  # Use only NDS and ScanContextMesh
  python match_meshes.py stereo.obj lidar.obj --desc NDS ScanContextMesh

  # Specify metric and save plots
  python match_meshes.py a.ply b.ply --metric cosine chi2 --save-plots

  # Info only (no descriptor computation)
  python match_meshes.py stereo.ply lidar.ply --info-only

  # List available descriptors
  python match_meshes.py --list-descriptors
"""

import sys
import os
import argparse
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from mesh_io import load_mesh, mesh_info
from descriptors import DESCRIPTOR_REGISTRY
from matching import METRIC_REGISTRY, cosine_similarity


def parse_args():
    p = argparse.ArgumentParser(
        description="Match two mesh files using handcrafted descriptors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("mesh_a", nargs="?", help="First mesh (e.g. stereo .ply/.obj)")
    p.add_argument("mesh_b", nargs="?", help="Second mesh (e.g. lidar .ply/.obj)")
    p.add_argument("--desc", nargs="+", default=None,
                   help="Descriptors to use (default: all). "
                        "Use --list-descriptors to see options.")
    p.add_argument("--metric", nargs="+",
                   default=["cosine", "l2", "chi2"],
                   choices=list(METRIC_REGISTRY.keys()),
                   help="Similarity metrics to report")
    p.add_argument("--save-plots", action="store_true",
                   help="Save comparison plots to ./match_output/")
    p.add_argument("--info-only", action="store_true",
                   help="Print mesh stats only, skip descriptor computation")
    p.add_argument("--list-descriptors", action="store_true",
                   help="Print available descriptors and exit")
    p.add_argument("--out", default="./match_output",
                   help="Output directory for plots (default: ./match_output)")
    return p.parse_args()


def print_mesh_info(path: str, label: str):
    print(f"\n  [{label}]  {mesh_info(path)}")


def run_match(mesh_a_path: str, mesh_b_path: str,
              desc_keys: list, metrics: list,
              save_plots: bool, out_dir: str):

    print(f"\n{'═'*62}")
    print(f"  Mesh A:  {os.path.basename(mesh_a_path)}")
    print(f"  Mesh B:  {os.path.basename(mesh_b_path)}")
    print(f"{'═'*62}")

    # Load meshes
    print("\nLoading meshes ...")
    t0 = time.perf_counter()
    va, fa, sa = load_mesh(mesh_a_path)
    t1 = time.perf_counter()
    vb, fb, sb = load_mesh(mesh_b_path)
    t2 = time.perf_counter()

    print(f"  Mesh A: {len(va):>7} vertices  {len(fa):>7} faces  "
          f"({(t1-t0)*1000:.0f} ms)")
    print(f"  Mesh B: {len(vb):>7} vertices  {len(fb):>7} faces  "
          f"({(t2-t1)*1000:.0f} ms)")

    # Header
    col = 14
    desc_col = 22
    header = (f"\n{'Descriptor':<{desc_col}}  {'dim':>5}  "
              + "  ".join(f"{m:>{col}}" for m in metrics)
              + f"  {'t_A (ms)':>9}  {'t_B (ms)':>9}")
    print(header)
    print("─" * len(header))

    results = []

    for key in desc_keys:
        desc = DESCRIPTOR_REGISTRY[key]
        try:
            t0 = time.perf_counter()
            d_a = desc.compute(va, fa, sa)
            t1 = time.perf_counter()
            d_b = desc.compute(vb, fb, sb)
            t2 = time.perf_counter()

            scores = {}
            for m in metrics:
                try:
                    scores[m] = METRIC_REGISTRY[m](d_a, d_b)
                except Exception:
                    scores[m] = float("nan")

            ta_ms = (t1 - t0) * 1000
            tb_ms = (t2 - t1) * 1000

            score_str = "  ".join(f"{scores[m]:>{col}.4f}" for m in metrics)
            print(f"  {key:<{desc_col}}  {len(d_a):>5}  {score_str}  "
                  f"  {ta_ms:>9.1f}  {tb_ms:>9.1f}")

            results.append({
                "name": key, "d_a": d_a, "d_b": d_b,
                "scores": scores, "ta": ta_ms, "tb": tb_ms,
            })

        except Exception as e:
            print(f"  {key:<{desc_col}}  ERROR: {e}")

    print("─" * len(header))

    # Best descriptor summary
    if results:
        primary = "cosine" if "cosine" in metrics else metrics[0]
        best = max(results, key=lambda r: r["scores"].get(primary, 0))
        print(f"\n  Best match ({primary}): {best['name']}  "
              f"score={best['scores'][primary]:.4f}")

    # Save plots
    if save_plots and results:
        _save_plots(results, mesh_a_path, mesh_b_path, metrics, out_dir)

    return results


def _save_plots(results, path_a, path_b, metrics, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    os.makedirs(out_dir, exist_ok=True)

    BG = "#0d1117"
    SRF = "#161b22"
    BRD = "#30363d"
    TXT = "#e6edf3"
    SUB = "#8b949e"
    C_A = "#58a6ff"
    C_B = "#3fb950"

    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": SRF,
        "axes.edgecolor": BRD, "text.color": TXT,
        "xtick.color": SUB, "ytick.color": SUB,
        "font.family": "monospace",
    })

    n = len(results)
    fig = plt.figure(figsize=(16, 3.2 * n + 1.5), facecolor=BG)
    fig.suptitle(
        f"Descriptor Match  |  A: {os.path.basename(path_a)}  ↔  "
        f"B: {os.path.basename(path_b)}",
        color=TXT, fontsize=12, y=1.002,
    )

    gs = gridspec.GridSpec(n, 3, figure=fig, wspace=0.42, hspace=0.65,
                           width_ratios=[3, 3, 1.4])

    for row, r in enumerate(results):
        # Descriptor A
        ax_a = fig.add_subplot(gs[row, 0])
        xs = np.arange(len(r["d_a"]))
        ax_a.bar(xs, r["d_a"], color=C_A, alpha=0.85, width=1.0)
        ax_a.set_title(f"{r['name']}  —  Mesh A",
                       color=TXT, fontsize=9, pad=6)
        ax_a.set_xlabel(f"dim={len(r['d_a'])}", color=SUB, fontsize=8)
        for sp in ax_a.spines.values(): sp.set_color(BRD)
        ax_a.grid(True, alpha=0.3, color=BRD)

        # Descriptor B
        ax_b = fig.add_subplot(gs[row, 1])
        ax_b.bar(np.arange(len(r["d_b"])), r["d_b"],
                 color=C_B, alpha=0.85, width=1.0)
        ax_b.set_title(f"{r['name']}  —  Mesh B",
                       color=TXT, fontsize=9, pad=6)
        ax_b.set_xlabel(f"dim={len(r['d_b'])}", color=SUB, fontsize=8)
        for sp in ax_b.spines.values(): sp.set_color(BRD)
        ax_b.grid(True, alpha=0.3, color=BRD)

        # Scores table
        ax_t = fig.add_subplot(gs[row, 2])
        ax_t.axis("off")
        table_data = [[m, f"{r['scores'][m]:.4f}"] for m in metrics]
        table_data += [["─────", "─────"],
                       ["t_A (ms)", f"{r['ta']:.1f}"],
                       ["t_B (ms)", f"{r['tb']:.1f}"]]
        tbl = ax_t.table(cellText=table_data, colLabels=["Metric", "Score"],
                         loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        for (ri, ci), cell in tbl.get_celld().items():
            bg = BRD if ri == 0 else SRF
            cell.set_facecolor(bg)
            cell.set_edgecolor(BRD)
            cell.set_text_props(color=TXT)
        ax_t.set_title("Scores", color=TXT, fontsize=9, pad=6)

    plt.tight_layout()
    out_path = os.path.join(out_dir,
        f"match_{os.path.splitext(os.path.basename(path_a))[0]}"
        f"_vs_{os.path.splitext(os.path.basename(path_b))[0]}.png")
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"\n  Plot saved → {out_path}")

    # Bar chart: descriptor scores side-by-side
    primary = "cosine" if "cosine" in metrics else metrics[0]
    fig2, ax = plt.subplots(figsize=(max(7, n * 1.4), 4), facecolor=BG)
    ax.set_facecolor(SRF)
    names  = [r["name"] for r in results]
    scores = [r["scores"].get(primary, 0) for r in results]
    colors = [C_A if s >= 0.8 else ("#f0883e" if s >= 0.5 else "#ff7b72")
              for s in scores]
    ax.bar(names, scores, color=colors, alpha=0.85, edgecolor=BRD)
    ax.axhline(0.8, color="#f0883e", linestyle="--", linewidth=1.2,
               label="0.8 threshold")
    ax.set_ylim(0, 1.05)
    ax.set_xticklabels(names, rotation=25, ha="right", color=TXT, fontsize=9)
    for sp in ax.spines.values(): sp.set_color(BRD)
    ax.set_title(f"{primary.upper()} similarity  A ↔ B", color=TXT, fontsize=11)
    ax.set_ylabel(primary, color=SUB, fontsize=9)
    ax.grid(True, axis="y", alpha=0.3, color=BRD)
    ax.legend(facecolor=SRF, edgecolor=BRD, labelcolor=TXT, fontsize=8)
    plt.tight_layout()
    bar_path = os.path.join(out_dir, "match_scores_bar.png")
    fig2.savefig(bar_path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig2)
    print(f"  Plot saved → {bar_path}")


def main():
    args = parse_args()

    if args.list_descriptors:
        print("\nAvailable descriptors:")
        for k, d in DESCRIPTOR_REGISTRY.items():
            doc = (d.__doc__ or "").strip().split("\n")[0]
            print(f"  {k:<22}  {doc[:60]}")
        return

    if not args.mesh_a or not args.mesh_b:
        print("Usage: python match_meshes.py <mesh_a> <mesh_b> [options]")
        print("       python match_meshes.py --list-descriptors")
        return

    for path in [args.mesh_a, args.mesh_b]:
        if not os.path.exists(path):
            print(f"ERROR: file not found: {path}")
            sys.exit(1)

    if args.info_only:
        print_mesh_info(args.mesh_a, "Mesh A")
        print_mesh_info(args.mesh_b, "Mesh B")
        return

    desc_keys = args.desc or list(DESCRIPTOR_REGISTRY.keys())
    unknown = [k for k in desc_keys if k not in DESCRIPTOR_REGISTRY]
    if unknown:
        print(f"ERROR: unknown descriptors: {unknown}")
        print(f"Available: {list(DESCRIPTOR_REGISTRY.keys())}")
        sys.exit(1)

    run_match(args.mesh_a, args.mesh_b,
              desc_keys=desc_keys,
              metrics=args.metric,
              save_plots=args.save_plots,
              out_dir=args.out)


if __name__ == "__main__":
    main()
