#!/usr/bin/env python3
"""
mesh_descriptors/run_benchmark.py
===================================
Main entry-point. Runs all descriptors × all scenes, prints results,
and saves a set of publication-quality plots.

Usage
-----
  python run_benchmark.py                   # full benchmark, all plots
  python run_benchmark.py --scene UrbanBlock
  python run_benchmark.py --desc NDS ScanContextMesh
  python run_benchmark.py --metric cosine chi2
  python run_benchmark.py --no-plots
"""

import sys
import os
import argparse
import numpy as np

# Make local imports work regardless of CWD
sys.path.insert(0, os.path.dirname(__file__))

from descriptors import DESCRIPTOR_REGISTRY, get_descriptor
from synthetic_scenes import SCENE_REGISTRY
from matching import run_full_benchmark, print_summary_table


def parse_args():
    p = argparse.ArgumentParser(description="Mesh descriptor benchmark")
    p.add_argument("--scene", nargs="+", default=None,
                   choices=list(SCENE_REGISTRY.keys()),
                   help="Scenes to include (default: all)")
    p.add_argument("--desc", nargs="+", default=None,
                   choices=list(DESCRIPTOR_REGISTRY.keys()),
                   help="Descriptors to include (default: all)")
    p.add_argument("--metric", nargs="+",
                   default=["cosine", "l2", "chi2", "bhattacharyya"],
                   help="Similarity metrics to report")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip plot generation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="/mnt/user-data/outputs",
                   help="Output directory for plots")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    # ── Select scenes ──────────────────────────────────────────────────────
    scene_keys = args.scene or list(SCENE_REGISTRY.keys())
    scenes = []
    for k in scene_keys:
        print(f"  Building scene: {k} ...", end=" ", flush=True)
        try:
            sc = SCENE_REGISTRY[k](seed=args.seed)
            v, f, s = sc['stereo_mesh']
            vl, fl, sl = sc['lidar_mesh']
            print(f"stereo={len(f)} faces  lidar={len(fl)} faces")
            scenes.append(sc)
        except Exception as e:
            print(f"FAILED: {e}")

    # ── Select descriptors ─────────────────────────────────────────────────
    desc_keys = args.desc or list(DESCRIPTOR_REGISTRY.keys())
    descriptors = [DESCRIPTOR_REGISTRY[k] for k in desc_keys]

    print(f"\nRunning {len(descriptors)} descriptors × {len(scenes)} scenes ...\n")

    # ── Run benchmark ──────────────────────────────────────────────────────
    results = run_full_benchmark(descriptors, scenes)

    # ── Print tables ───────────────────────────────────────────────────────
    for metric in args.metric:
        print_summary_table(results, metric=metric)

    # ── Per-result detail ──────────────────────────────────────────────────
    print("\n── Per-descriptor detail ──────────────────────────────────────")
    for r in sorted(results, key=lambda x: (x.scene_name, x.descriptor_name)):
        scores_str = "  ".join(f"{m}={r.scores[m]:.4f}" for m in args.metric)
        print(f"  {r.scene_name:<18} {r.descriptor_name:<22} "
              f"dim={r.dim:<6} {scores_str}  "
              f"t={r.compute_time_stereo_ms:.1f}/{r.compute_time_lidar_ms:.1f}ms")

    # ── Plots ──────────────────────────────────────────────────────────────
    if not args.no_plots:
        import matplotlib
        matplotlib.use("Agg")
        from visualisation import (plot_mesh_pair, plot_descriptor_comparison,
                                   plot_similarity_heatmap, plot_timing, plot_radar)

        print("\nGenerating plots ...")

        # 1. Mesh previews
        for sc in scenes:
            fig = plot_mesh_pair(sc)
            path = os.path.join(args.out, f"mesh_{sc['name']}.png")
            fig.savefig(path, dpi=130, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            import matplotlib.pyplot as plt
            plt.close(fig)
            print(f"  Saved {path}")

        # 2. Descriptor comparison per scene
        for sc in scenes:
            fig = plot_descriptor_comparison(results, sc['name'])
            if fig:
                path = os.path.join(args.out, f"descriptors_{sc['name']}.png")
                fig.savefig(path, dpi=130, bbox_inches="tight",
                            facecolor=fig.get_facecolor())
                import matplotlib.pyplot as plt
                plt.close(fig)
                print(f"  Saved {path}")

        # 3. Heatmap
        for metric in args.metric:
            import matplotlib.pyplot as plt
            fig = plot_similarity_heatmap(results, metric=metric)
            path = os.path.join(args.out, f"heatmap_{metric}.png")
            fig.savefig(path, dpi=130, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"  Saved {path}")

        # 4. Timing
        import matplotlib.pyplot as plt
        fig = plot_timing(results)
        path = os.path.join(args.out, "timing.png")
        fig.savefig(path, dpi=130, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  Saved {path}")

        # 5. Radar chart
        fig = plot_radar(results, metric="cosine")
        path = os.path.join(args.out, "radar_cosine.png")
        fig.savefig(path, dpi=130, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  Saved {path}")

        print(f"\nAll plots saved to {args.out}/")


if __name__ == "__main__":
    main()
