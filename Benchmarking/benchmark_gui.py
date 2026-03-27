#!/usr/bin/env python3
"""
Benchmarking/benchmark_gui.py
==============================
Interactive GUI: load real mesh pairs (.ply/.obj/.off/.stl), run descriptors,
explore match quality with selectable visualisation modes per descriptor.

Usage
-----
  python benchmark_gui.py                        # empty, add pairs in GUI
  python benchmark_gui.py mesh_a.ply mesh_b.ply  # preload one pair
"""
import sys
import os
import threading
import time as _time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import queue as _queue

from mesh_io import load_mesh
from descriptors import DESCRIPTOR_REGISTRY
from matching import METRIC_REGISTRY

try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    HAS_O3D = False


# ─── colour theme ────────────────────────────────────────────────────────────

DARK = {
    "bg":      "#0d1117",
    "surface": "#161b22",
    "border":  "#30363d",
    "accent1": "#58a6ff",
    "accent2": "#3fb950",
    "accent3": "#f0883e",
    "accent4": "#ff7b72",
    "text":    "#e6edf3",
    "subtext": "#8b949e",
    "select":  "#1f6feb",
}

ALL_METRICS = list(METRIC_REGISTRY.keys())   # cosine, l2, chi2, bhattacharyya


# ─────────────────────────────────────────────────────────────────────────────

class BenchmarkApp(tk.Tk):
    def __init__(self, initial_pair=None):
        super().__init__()
        self.title("Mesh Descriptor Benchmark")
        self.geometry("1440x860")
        self.minsize(900, 600)
        self.configure(bg=DARK["bg"])

        self.pairs_var: list = []          # list of (path_a, path_b, label)
        self.results:   list = []          # accumulated result dicts
        self.mesh_cache: dict = {}         # path -> (V, F, S)
        self._cell_cache: dict = {}        # (desc, path_a, path_b) -> cell list
        self._iid_to_result: dict = {}     # treeview iid -> result dict
        self._selected_result = None

        self._build_ui()
        self._apply_dark_theme()

        if initial_pair:
            self._add_pair(*initial_pair)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # Outer horizontal pane: left controls | right detail
        outer = tk.PanedWindow(self, orient=tk.HORIZONTAL,
                               bg=DARK["bg"], sashwidth=5,
                               relief=tk.FLAT)
        outer.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # ── Left panel ────────────────────────────────────────────────────
        left = tk.Frame(outer, bg=DARK["bg"])
        outer.add(left, minsize=300, width=380)

        self._build_pairs_section(left)
        self._build_desc_section(left)
        self._build_run_section(left)
        self._build_results_table(left)

        # ── Right panel ───────────────────────────────────────────────────
        right = tk.Frame(outer, bg=DARK["bg"])
        outer.add(right, minsize=560)

        self._build_detail_view(right)

    # ─── helpers ─────────────────────────────────────────────────────────────

    def _section_header(self, parent, text):
        tk.Frame(parent, bg=DARK["border"], height=1).pack(fill=tk.X, padx=4,
                                                            pady=(10, 0))
        tk.Label(parent, text=text, bg=DARK["bg"], fg=DARK["subtext"],
                 font=("monospace", 8, "bold"), anchor="w", padx=6
                 ).pack(fill=tk.X)

    def _btn(self, parent, text, cmd, fg=None, **kw):
        return tk.Button(parent, text=text, command=cmd,
                         bg=DARK["surface"], fg=fg or DARK["text"],
                         relief=tk.FLAT, padx=8, pady=3,
                         font=("monospace", 9), cursor="hand2",
                         activebackground=DARK["border"],
                         activeforeground=DARK["text"], **kw)

    # ─── pairs section ────────────────────────────────────────────────────────

    def _build_pairs_section(self, parent):
        self._section_header(parent, "MESH PAIRS")

        ctrl = tk.Frame(parent, bg=DARK["bg"])
        ctrl.pack(fill=tk.X, padx=4, pady=2)
        self._btn(ctrl, "+ Add Pair", self._add_pair_dialog,
                  fg=DARK["accent1"]).pack(side=tk.LEFT, padx=2)
        self._btn(ctrl, "Remove", self._remove_pair,
                  fg=DARK["accent4"]).pack(side=tk.LEFT, padx=2)

        frame = tk.Frame(parent, bg=DARK["border"], bd=1)
        frame.pack(fill=tk.X, padx=4, pady=2)
        self.pairs_lb = tk.Listbox(
            frame, bg=DARK["surface"], fg=DARK["text"],
            selectbackground=DARK["select"], relief=tk.FLAT, height=5,
            font=("monospace", 8), activestyle="none")
        self.pairs_lb.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)

    def _add_pair_dialog(self):
        path_a = filedialog.askopenfilename(
            title="Select Mesh A",
            filetypes=[("Mesh files", "*.ply *.obj *.off *.stl"), ("All", "*.*")])
        if not path_a:
            return
        path_b = filedialog.askopenfilename(
            title="Select Mesh B (to match against)",
            filetypes=[("Mesh files", "*.ply *.obj *.off *.stl"), ("All", "*.*")])
        if not path_b:
            return
        self._add_pair(path_a, path_b)

    def _add_pair(self, path_a, path_b):
        label = f"{os.path.basename(path_a)}  ↔  {os.path.basename(path_b)}"
        self.pairs_var.append((path_a, path_b, label))
        self.pairs_lb.insert(tk.END, f"  {label}")

    def _remove_pair(self):
        sel = self.pairs_lb.curselection()
        if sel:
            idx = sel[0]
            self.pairs_lb.delete(idx)
            self.pairs_var.pop(idx)

    # ─── descriptors + metric section ────────────────────────────────────────

    def _build_desc_section(self, parent):
        self._section_header(parent, "DESCRIPTORS")

        frame = tk.Frame(parent, bg=DARK["bg"])
        frame.pack(fill=tk.X, padx=6, pady=2)

        self.desc_vars: dict = {}
        cols = 2
        for i, key in enumerate(DESCRIPTOR_REGISTRY.keys()):
            var = tk.BooleanVar(value=True)
            self.desc_vars[key] = var
            tk.Checkbutton(
                frame, text=key, variable=var,
                bg=DARK["bg"], fg=DARK["text"],
                selectcolor=DARK["surface"],
                activebackground=DARK["bg"], activeforeground=DARK["accent1"],
                font=("monospace", 9), relief=tk.FLAT
            ).grid(row=i // cols, column=i % cols, sticky="w", padx=2)

        mf = tk.Frame(parent, bg=DARK["bg"])
        mf.pack(fill=tk.X, padx=6, pady=(6, 2))
        tk.Label(mf, text="Primary metric:", bg=DARK["bg"], fg=DARK["subtext"],
                 font=("monospace", 9)).pack(side=tk.LEFT)
        self.metric_var = tk.StringVar(value="cosine")
        ttk.Combobox(mf, textvariable=self.metric_var, values=ALL_METRICS,
                     width=14, state="readonly",
                     font=("monospace", 9)).pack(side=tk.LEFT, padx=6)

    # ─── run section ──────────────────────────────────────────────────────────

    def _build_run_section(self, parent):
        self._section_header(parent, "")

        rf = tk.Frame(parent, bg=DARK["bg"])
        rf.pack(fill=tk.X, padx=4, pady=4)

        self.run_btn = tk.Button(
            rf, text="▶  Run Benchmark", command=self._run_benchmark,
            bg=DARK["accent1"], fg=DARK["bg"],
            relief=tk.FLAT, padx=12, pady=5,
            font=("monospace", 10, "bold"), cursor="hand2",
            activebackground="#79c0ff", activeforeground=DARK["bg"])
        self.run_btn.pack(side=tk.LEFT, padx=2)

        self._btn(rf, "💾  Save", self._save_session,
                  fg=DARK["accent2"]).pack(side=tk.LEFT, padx=2)
        self._btn(rf, "📂  Load", self._load_session,
                  fg=DARK["accent3"]).pack(side=tk.LEFT, padx=2)

        self.status_lbl = tk.Label(rf, text="", bg=DARK["bg"],
                                   fg=DARK["subtext"], font=("monospace", 9))
        self.status_lbl.pack(side=tk.LEFT, padx=8)

        self.progress = ttk.Progressbar(parent, mode="determinate")
        self.progress.pack(fill=tk.X, padx=4, pady=2)

    # ─── results table ────────────────────────────────────────────────────────

    def _build_results_table(self, parent):
        self._section_header(parent, "RESULTS  (click row to inspect)")

        frame = tk.Frame(parent, bg=DARK["border"])
        frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)

        cols = ("pair", "descriptor",
                "cosine", "l2", "chi2", "bhattacharyya",
                "t_a_ms", "t_b_ms")
        widths = {"pair": 110, "descriptor": 110,
                  "cosine": 64, "l2": 64, "chi2": 64, "bhattacharyya": 76,
                  "t_a_ms": 58, "t_b_ms": 58}
        headers = {"pair": "Pair", "descriptor": "Descriptor",
                   "cosine": "cosine", "l2": "l2",
                   "chi2": "chi2", "bhattacharyya": "bhatt.",
                   "t_a_ms": "t_A(ms)", "t_b_ms": "t_B(ms)"}

        self.tree = ttk.Treeview(frame, columns=cols, show="headings",
                                 selectmode="browse")
        for c in cols:
            self.tree.heading(c, text=headers[c],
                              command=lambda _c=c: self._sort_by(_c))
            self.tree.column(c, width=widths[c], anchor="center", stretch=False)

        vsb = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree.bind("<<TreeviewSelect>>", self._on_result_select)
        self._sort_col = "cosine"
        self._sort_asc = False

    # ─── detail / visualisation panel ────────────────────────────────────────

    def _build_detail_view(self, parent):
        # Mode radio buttons
        mode_frame = tk.Frame(parent, bg=DARK["bg"])
        mode_frame.pack(fill=tk.X, padx=6, pady=(4, 0))

        tk.Label(mode_frame, text="View mode:", bg=DARK["bg"],
                 fg=DARK["subtext"], font=("monospace", 9)
                 ).pack(side=tk.LEFT, padx=(0, 6))

        self.view_mode = tk.StringVar(value="side-by-side")
        for val, label in [("side-by-side", "Side by Side"),
                            ("overlay",      "Overlay"),
                            ("difference",   "Difference")]:
            tk.Radiobutton(
                mode_frame, text=label, variable=self.view_mode, value=val,
                command=self._refresh_detail,
                bg=DARK["bg"], fg=DARK["text"], selectcolor=DARK["surface"],
                activebackground=DARK["bg"], activeforeground=DARK["accent1"],
                font=("monospace", 9), relief=tk.FLAT
            ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            mode_frame, text="⬡  3D View", command=self._open_3d_viewer,
            bg=DARK["surface"], fg=DARK["accent3"],
            relief=tk.FLAT, padx=10, pady=2,
            font=("monospace", 9), cursor="hand2",
            activebackground=DARK["border"], activeforeground=DARK["accent3"],
        ).pack(side=tk.RIGHT, padx=4)

        # Matplotlib canvas
        self.fig = Figure(facecolor=DARK["bg"])
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self._draw_placeholder()

    # ─── dark-theme styling ───────────────────────────────────────────────────

    def _apply_dark_theme(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Treeview",
                        background=DARK["surface"], foreground=DARK["text"],
                        rowheight=22, fieldbackground=DARK["surface"],
                        borderwidth=0, font=("monospace", 8))
        style.configure("Treeview.Heading",
                        background=DARK["border"], foreground=DARK["text"],
                        relief="flat", font=("monospace", 8, "bold"))
        style.map("Treeview",
                  background=[("selected", DARK["select"])],
                  foreground=[("selected", DARK["text"])])
        style.configure("TCombobox",
                        fieldbackground=DARK["surface"],
                        background=DARK["surface"],
                        foreground=DARK["text"])
        style.configure("TProgressbar",
                        troughcolor=DARK["surface"], background=DARK["accent1"],
                        borderwidth=0)
        style.configure("TScrollbar",
                        background=DARK["surface"], troughcolor=DARK["bg"],
                        borderwidth=0, arrowsize=12)

    # ─── benchmark execution ─────────────────────────────────────────────────

    def _run_benchmark(self):
        if not self.pairs_var:
            messagebox.showwarning("No pairs", "Add at least one mesh pair first.")
            return
        desc_keys = [k for k, v in self.desc_vars.items() if v.get()]
        if not desc_keys:
            messagebox.showwarning("No descriptors", "Select at least one descriptor.")
            return

        self.run_btn.config(state=tk.DISABLED)
        self.results.clear()
        self._clear_table()

        total = len(self.pairs_var) * len(desc_keys)
        self.progress.config(maximum=total, value=0)

        threading.Thread(
            target=self._benchmark_worker,
            args=(list(self.pairs_var), desc_keys),
            daemon=True).start()

    def _benchmark_worker(self, pairs, desc_keys):
        done = 0
        done_lock = threading.Lock()

        for path_a, path_b, label in pairs:
            self._set_status(f"Loading  {os.path.basename(path_a)} ...")
            try:
                if path_a not in self.mesh_cache:
                    self.mesh_cache[path_a] = load_mesh(path_a)
                if path_b not in self.mesh_cache:
                    self.mesh_cache[path_b] = load_mesh(path_b)
                va, fa, sa = self.mesh_cache[path_a]
                vb, fb, sb = self.mesh_cache[path_b]
            except Exception as e:
                self._set_status(f"ERROR loading: {e}")
                with done_lock:
                    done += len(desc_keys)
                self.after(0, self._set_progress, done)
                continue

            def _run_one_desc(key, _va=va, _fa=fa, _sa=sa,
                              _vb=vb, _fb=fb, _sb=sb,
                              _label=label, _pa=path_a, _pb=path_b):
                desc = DESCRIPTOR_REGISTRY[key]
                try:
                    t0 = _time.perf_counter()
                    d_a = desc.compute(_va, _fa, _sa)
                    t1 = _time.perf_counter()
                    d_b = desc.compute(_vb, _fb, _sb)
                    t2 = _time.perf_counter()
                    scores = {}
                    for m, fn in METRIC_REGISTRY.items():
                        try:
                            scores[m] = fn(d_a, d_b)
                        except Exception:
                            scores[m] = float("nan")
                    return {
                        "pair_label": _label, "path_a": _pa, "path_b": _pb,
                        "descriptor": key, "d_a": d_a, "d_b": d_b,
                        "scores": scores,
                        "ta_ms": (t1 - t0) * 1000,
                        "tb_ms": (t2 - t1) * 1000,
                    }
                except Exception as e:
                    return {
                        "pair_label": _label, "path_a": _pa, "path_b": _pb,
                        "descriptor": key,
                        "d_a": np.array([]), "d_b": np.array([]),
                        "scores": {m: float("nan") for m in METRIC_REGISTRY},
                        "ta_ms": 0.0, "tb_ms": 0.0,
                        "error": str(e),
                    }

            n_workers = min(len(desc_keys), os.cpu_count() or 4)
            self._set_status(
                f"Running {len(desc_keys)} descriptors "
                f"({n_workers} threads)  ·  {os.path.basename(path_a)[:22]}")

            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_run_one_desc, k): k for k in desc_keys}
                for future in as_completed(futures):
                    result = future.result()
                    self.results.append(result)
                    self.after(0, self._add_table_row, result)
                    with done_lock:
                        done += 1
                    self.after(0, self._set_progress, done)

        # Pre-compute cell correspondences so Save always includes them.
        # Cells are stored without display offset — offset is applied at render time.
        successful = [r for r in self.results if not r.get("error")]
        if successful:
            n_cell = len(successful)
            self.after(0, lambda: self.progress.config(maximum=n_cell, value=0))
            cell_done = [0]
            cell_lock = threading.Lock()

            def _pre_cell(r):
                key = (r["descriptor"], r["path_a"], r["path_b"])
                if key not in self._cell_cache:
                    try:
                        va_, fa_, sa_ = self.mesh_cache[r["path_a"]]
                        vb_, fb_, sb_ = self.mesh_cache[r["path_b"]]
                        self._cell_cache[key] = _compute_cell_correspondences(
                            r["descriptor"], va_, fa_, sa_, vb_, fb_, sb_)
                    except Exception:
                        pass
                with cell_lock:
                    cell_done[0] += 1
                    n = cell_done[0]
                self._set_status(
                    f"Correspondence lines  {n}/{n_cell}  "
                    f"({r['descriptor']} · {os.path.basename(r['path_a'])[:18]})")
                self.after(0, self._set_progress, n)

            n_workers = min(n_cell, os.cpu_count() or 4)
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                list(pool.map(_pre_cell, successful))

        self.after(0, self._benchmark_done)

    def _set_status(self, msg):
        self.after(0, lambda: self.status_lbl.config(text=msg))

    def _set_progress(self, val):
        self.progress["value"] = val

    def _benchmark_done(self):
        self.run_btn.config(state=tk.NORMAL)
        self.status_lbl.config(text=f"Done — {len(self.results)} results")

    # ─── session save / load ──────────────────────────────────────────────────

    def _save_session(self):
        import pickle
        if not self.results:
            messagebox.showinfo("Nothing to save", "Run the benchmark first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save Session",
            defaultextension=".pkl",
            filetypes=[("Benchmark session", "*.pkl"), ("All", "*.*")])
        if not path:
            return
        session = {
            "version": 1,
            "results":    self.results,
            "cell_cache": self._cell_cache,
            "pairs_var":  self.pairs_var,
        }
        try:
            with open(path, "wb") as fh:
                pickle.dump(session, fh, protocol=4)
            self.status_lbl.config(
                text=f"Saved {len(self.results)} results → {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def _load_session(self):
        import pickle
        path = filedialog.askopenfilename(
            title="Load Session",
            filetypes=[("Benchmark session", "*.pkl"), ("All", "*.*")])
        if not path:
            return
        try:
            with open(path, "rb") as fh:
                session = pickle.load(fh)
        except Exception as e:
            messagebox.showerror("Load failed", str(e))
            return

        if session.get("version", 0) != 1:
            messagebox.showwarning("Unknown format",
                                   "Session file version not recognised — trying anyway.")

        self.results = session.get("results", [])
        self._cell_cache.update(session.get("cell_cache", {}))

        # Restore pairs listbox (skip duplicates)
        existing = {(pa, pb) for pa, pb, _ in self.pairs_var}
        for pa, pb, lbl in session.get("pairs_var", []):
            if (pa, pb) not in existing:
                self.pairs_var.append((pa, pb, lbl))
                self.pairs_lb.insert(tk.END, f"  {lbl}")
                existing.add((pa, pb))

        # Repopulate results table
        self._clear_table()
        for r in self.results:
            self._add_table_row(r)

        n_cells = len(self._cell_cache)
        self.status_lbl.config(
            text=f"Loaded {len(self.results)} results, "
                 f"{n_cells} cached correspondence sets — {os.path.basename(path)}")

    # ─── table management ────────────────────────────────────────────────────

    def _clear_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        self._iid_to_result.clear()

    def _add_table_row(self, r):
        scores = r["scores"]
        pair_short = os.path.basename(r["path_a"])[:18]

        if r.get("error"):
            # Show error inline so it's never silently dropped
            err_short = r["error"][:28]
            vals = (pair_short, r["descriptor"],
                    f"ERR: {err_short}", "", "", "", "", "")
            iid = self.tree.insert("", tk.END, values=vals)
            self._iid_to_result[iid] = r
            self.tree.item(iid, tags=("err",))
            self.tree.tag_configure("err", foreground=DARK["accent4"])
            return

        vals = (
            pair_short,
            r["descriptor"],
            _fmt(scores.get("cosine")),
            _fmt(scores.get("l2")),
            _fmt(scores.get("chi2")),
            _fmt(scores.get("bhattacharyya")),
            f"{r['ta_ms']:.1f}",
            f"{r['tb_ms']:.1f}",
        )
        iid = self.tree.insert("", tk.END, values=vals)
        self._iid_to_result[iid] = r

        # Colour code by primary metric
        score = scores.get(self.metric_var.get(), scores.get("cosine", 0)) or 0
        tag = "good" if score >= 0.8 else ("mid" if score >= 0.5 else "low")
        self.tree.item(iid, tags=(tag,))
        self.tree.tag_configure("good", foreground=DARK["accent2"])
        self.tree.tag_configure("mid",  foreground=DARK["accent3"])
        self.tree.tag_configure("low",  foreground=DARK["accent4"])

    def _sort_by(self, col):
        self._sort_asc = (not self._sort_asc) if col == self._sort_col else False
        self._sort_col = col

        def key(r):
            if col in r["scores"]:
                v = r["scores"][col]
                return v if not np.isnan(v) else -999
            return {"t_a_ms": r["ta_ms"], "t_b_ms": r["tb_ms"],
                    "descriptor": r["descriptor"],
                    "pair": r["pair_label"]}.get(col, 0)

        try:
            sorted_results = sorted(self.results, key=key,
                                    reverse=not self._sort_asc)
        except Exception:
            sorted_results = self.results

        self._clear_table()
        for r in sorted_results:
            self._add_table_row(r)

    # ─── detail view ─────────────────────────────────────────────────────────

    def _on_result_select(self, _event):
        sel = self.tree.selection()
        if not sel:
            return
        r = self._iid_to_result.get(sel[0])
        if r is not None:
            self._selected_result = r
            self._refresh_detail()

    def _refresh_detail(self):
        if self._selected_result is None:
            return
        self._draw_detail(self._selected_result, self.view_mode.get())

    def _draw_detail(self, r, mode):
        self.fig.clear()
        self.fig.patch.set_facecolor(DARK["bg"])

        d_a = np.asarray(r["d_a"], dtype=float)
        d_b = np.asarray(r["d_b"], dtype=float)

        # Pad to equal length for comparison
        n = max(len(d_a), len(d_b))
        a_pad = np.zeros(n); a_pad[:len(d_a)] = d_a
        b_pad = np.zeros(n); b_pad[:len(d_b)] = d_b
        xs = np.arange(n)

        scores = r["scores"]
        title = (f"{r['descriptor']}   |   "
                 + "   ".join(f"{m}={v:.4f}" for m, v in scores.items()
                              if not np.isnan(v)))

        if mode == "side-by-side":
            gs = self.fig.add_gridspec(1, 3, wspace=0.4,
                                        left=0.05, right=0.98,
                                        top=0.86, bottom=0.10)
            ax_a = self.fig.add_subplot(gs[0, 0])
            ax_b = self.fig.add_subplot(gs[0, 1])
            ax_t = self.fig.add_subplot(gs[0, 2])

            ax_a.bar(xs, a_pad, color=DARK["accent1"], alpha=0.85, width=1.0)
            _style_ax(ax_a,
                      f"Mesh A — {os.path.basename(r['path_a'])}",
                      f"dim = {len(d_a)}", "value")

            ax_b.bar(xs, b_pad, color=DARK["accent2"], alpha=0.85, width=1.0)
            _style_ax(ax_b,
                      f"Mesh B — {os.path.basename(r['path_b'])}",
                      f"dim = {len(d_b)}", "")

            # Scores table
            ax_t.set_facecolor(DARK["surface"])
            ax_t.axis("off")
            rows = [[m, f"{scores[m]:.4f}"] for m in ALL_METRICS
                    if m in scores]
            rows += [["───", "───"],
                     ["t_A", f"{r['ta_ms']:.1f} ms"],
                     ["t_B", f"{r['tb_ms']:.1f} ms"]]
            tbl = ax_t.table(cellText=rows, colLabels=["Metric", "Score"],
                              loc="center", cellLoc="center")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(9)
            for (ri, ci), cell in tbl.get_celld().items():
                cell.set_facecolor(DARK["border"] if ri == 0 else DARK["surface"])
                cell.set_edgecolor(DARK["border"])
                cell.set_text_props(color=DARK["text"])
            ax_t.set_title("Scores", color=DARK["text"], fontsize=10, pad=8)

        elif mode == "overlay":
            ax = self.fig.add_subplot(111)
            w = 0.4
            ax.bar(xs - w / 2, a_pad, width=w, color=DARK["accent1"],
                   alpha=0.85, label=f"A: {os.path.basename(r['path_a'])}")
            ax.bar(xs + w / 2, b_pad, width=w, color=DARK["accent2"],
                   alpha=0.85, label=f"B: {os.path.basename(r['path_b'])}")
            ax.legend(facecolor=DARK["surface"], edgecolor=DARK["border"],
                      labelcolor=DARK["text"], fontsize=9)
            _style_ax(ax, f"Overlay — {r['descriptor']}", "dimension index", "value")

        elif mode == "difference":
            diff = a_pad - b_pad
            colors = [DARK["accent2"] if v >= 0 else DARK["accent4"]
                      for v in diff]
            ax = self.fig.add_subplot(111)
            ax.bar(xs, diff, color=colors, alpha=0.85, width=1.0)
            ax.axhline(0, color=DARK["border"], linewidth=1.0)
            _style_ax(ax,
                      f"Difference (A − B) — {r['descriptor']}",
                      "dimension index", "A − B")
            ax.text(0.98, 0.02,
                    f"MAE = {np.abs(diff).mean():.4f}",
                    ha="right", va="bottom", color=DARK["subtext"],
                    fontsize=8, fontfamily="monospace",
                    transform=ax.transAxes)

        self.fig.suptitle(title, color=DARK["text"], fontsize=9,
                          fontfamily="monospace", y=0.99)
        self.canvas.draw()

    def _open_3d_viewer(self):
        valid = [r for r in self.results
                 if not r.get("error") and len(r.get("d_a", [])) > 0]
        if not valid:
            messagebox.showinfo("No results", "Run the benchmark first.")
            return
        MeshViewerWindow(self, valid, self.mesh_cache, self._cell_cache)

    def _draw_placeholder(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor(DARK["surface"])
        for sp in ax.spines.values():
            sp.set_color(DARK["border"])
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.5, 0.5,
                "Add mesh pairs, run benchmark,\nthen click a result row to inspect.",
                ha="center", va="center", color=DARK["subtext"],
                fontsize=12, fontfamily="monospace",
                transform=ax.transAxes, linespacing=1.8)
        self.canvas.draw()


# ─── module-level helpers ─────────────────────────────────────────────────────

def _fmt(v):
    return f"{v:.4f}" if v is not None and not np.isnan(v) else "—"


def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(DARK["surface"])
    for sp in ax.spines.values():
        sp.set_color(DARK["border"])
    ax.tick_params(colors=DARK["subtext"], labelsize=7)
    ax.grid(True, alpha=0.3, color=DARK["border"])
    if title:
        ax.set_title(title, color=DARK["text"], fontsize=9, pad=6,
                     fontfamily="monospace")
    if xlabel:
        ax.set_xlabel(xlabel, color=DARK["subtext"], fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=DARK["subtext"], fontsize=8)


# ─── 3D viewer helpers ───────────────────────────────────────────────────────

def _desc_face_colors(desc_name, vertices, faces, semantics):
    """Return one (R,G,B,1) tuple per face styled for the descriptor."""
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    from descriptors import (compute_face_normals, compute_face_centroids,
                              compute_dihedral_angles, compute_gaussian_curvature,
                              ICO_NORMALS)
    n = len(faces)

    if desc_name in ("ScanContextMesh", "NDS"):
        desc_obj = DESCRIPTOR_REGISTRY[desc_name]
        n_rings, n_sectors = desc_obj.R, desc_obj.S
        cents = compute_face_centroids(vertices, faces)
        cx = cents[:, 0] - cents[:, 0].mean()
        cy = cents[:, 1] - cents[:, 1].mean()
        r  = np.sqrt(cx**2 + cy**2)
        max_r = np.percentile(r, 95) + 1e-6
        theta = np.arctan2(cy, cx) + np.pi
        ri = np.floor(r / max_r * n_rings).astype(int).clip(0, n_rings - 1)
        si = np.floor(theta / (2 * np.pi) * n_sectors).astype(int).clip(0, n_sectors - 1)
        # Use tab20b for better visibility on both dark and light backgrounds
        cell_idx = ri * n_sectors + si
        n_cells  = n_rings * n_sectors
        cmap = cm.get_cmap("gist_rainbow", n_cells)
        return [cmap(int(c) % n_cells) for c in cell_idx]

    elif desc_name == "NormalHistogram":
        normals = compute_face_normals(vertices, faces)
        bins = np.argmax(np.abs(normals @ ICO_NORMALS.T), axis=1)
        cmap = cm.get_cmap("tab20", 20)
        return [cmap(int(b)) for b in bins]

    elif desc_name == "DihedralHistogram":
        normals = compute_face_normals(vertices, faces)
        dvals, pairs = compute_dihedral_angles(normals, faces)
        face_d = np.full(n, np.pi / 2)
        if len(pairs):
            acc, cnt = np.zeros(n), np.zeros(n)
            for (fi, fj), ang in zip(pairs, dvals):
                acc[fi] += ang; acc[fj] += ang
                cnt[fi] += 1;  cnt[fj] += 1
            m = cnt > 0
            face_d[m] = acc[m] / cnt[m]
        cmap = cm.get_cmap("coolwarm")
        return [cmap(Normalize(0, np.pi)(v)) for v in face_d]

    elif desc_name == "CurvatureHistogram":
        K = compute_gaussian_curvature(vertices, faces)
        face_K = K[faces].mean(axis=1)
        p5, p95 = np.percentile(face_K, 5), np.percentile(face_K, 95)
        norm = Normalize(p5, p95)
        cmap = cm.get_cmap("RdYlGn")
        return [cmap(float(np.clip(norm(v), 0, 1))) for v in face_K]

    else:  # SpinImage, MeshFPFH – radial distance from centroid
        cents = compute_face_centroids(vertices, faces)
        cx = cents[:, 0] - cents[:, 0].mean()
        cy = cents[:, 1] - cents[:, 1].mean()
        r  = np.sqrt(cx**2 + cy**2)
        norm = Normalize(r.min(), r.max() + 1e-10)
        cmap = cm.get_cmap("plasma")
        return [cmap(norm(v)) for v in r]


def _compute_cell_correspondences(desc_name, va, fa, sa, vb, fb, sb,
                                   n_rings=4, n_sectors=6, min_faces=8,
                                   on_progress=None):
    """
    Universal spatial correspondences for ANY descriptor.

    Divides both meshes into a polar grid, computes the descriptor on each
    cell's local faces, and matches cells by cosine similarity.

    on_progress(done, total) is called (from worker threads) after each cell
    completes — use it to drive a progress bar.
    """
    from descriptors import compute_face_centroids

    def polar_assign(verts, faces_arr):
        cents = compute_face_centroids(verts, faces_arr)
        cx = cents[:, 0] - cents[:, 0].mean()
        cy = cents[:, 1] - cents[:, 1].mean()
        r  = np.sqrt(cx**2 + cy**2)
        max_r = np.percentile(r, 95) + 1e-6
        theta = np.arctan2(cy, cx) + np.pi
        ri = np.floor(r / max_r * n_rings).astype(int).clip(0, n_rings - 1)
        si = np.floor(theta / (2 * np.pi) * n_sectors).astype(int).clip(0, n_sectors - 1)
        return ri, si, cents

    ri_a, si_a, cents_a = polar_assign(va, fa)
    ri_b, si_b, cents_b = polar_assign(vb, fb)

    desc = DESCRIPTOR_REGISTRY[desc_name]
    cell_args = [(ri, si) for ri in range(n_rings) for si in range(n_sectors)]
    total = len(cell_args)
    done_count = [0]
    done_lock  = threading.Lock()

    def _cell_sim(ri, si):
        ma = (ri_a == ri) & (si_a == si)
        mb = (ri_b == ri) & (si_b == si)
        result = None
        if ma.sum() >= min_faces and mb.sum() >= min_faces:
            ca = cents_a[ma].mean(axis=0)
            cb = cents_b[mb].mean(axis=0)
            try:
                d_a = desc.compute(va, fa[ma], sa[ma])
                d_b = desc.compute(vb, fb[mb], sb[mb])
                na, nb = np.linalg.norm(d_a), np.linalg.norm(d_b)
                sim = float(np.dot(d_a, d_b) / (na * nb)) if na > 1e-10 and nb > 1e-10 else 0.0
            except Exception:
                sim = 0.0
            result = (ca, cb, max(0.0, sim))
        if on_progress is not None:
            with done_lock:
                done_count[0] += 1
                n = done_count[0]
            on_progress(n, total)
        return result

    n_workers = min(total, os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_cell_sim, ri, si) for ri, si in cell_args]
        raw = [f.result() for f in futures]

    return [c for c in raw if c is not None]


_COLOR_DESC = {
    "NormalHistogram":    "icosphere normal bin  (tab20)",
    "DihedralHistogram":  "avg dihedral angle  (blue=flat → red=sharp)",
    "CurvatureHistogram": "Gaussian curvature  (green=flat → red=curved)",
    "SpinImage":          "radial distance from centroid  (plasma)",
    "MeshFPFH":           "radial distance from centroid  (plasma)",
    "ScanContextMesh":    "polar grid cell  (gist_rainbow)",
    "NDS":                "polar grid cell  (gist_rainbow)",
}


# ─── Open3D helpers (used when HAS_O3D) ──────────────────────────────────────

def _o3d_colored_mesh(vertices, faces, colors, max_faces=60_000):
    """Build an Open3D TriangleMesh with per-face colours via vertex duplication."""
    n = len(faces)
    if n > max_faces:
        idx = np.random.choice(n, max_faces, replace=False)
        faces  = faces[idx]
        colors = [colors[i] for i in idx]
    # Duplicate vertices so each face owns its 3 vertices → clean per-face colour
    vdup = vertices[faces.reshape(-1)].astype(np.float64)           # (F*3, 3)
    fdup = np.arange(len(faces) * 3).reshape(-1, 3)                 # (F,   3)
    rgb  = np.clip([[c[0], c[1], c[2]] for c in colors], 0, 1)      # (F,   3)
    cdup = np.repeat(rgb, 3, axis=0).astype(np.float64)             # (F*3, 3)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices      = o3d.utility.Vector3dVector(vdup)
    mesh.triangles     = o3d.utility.Vector3iVector(fdup)
    mesh.vertex_colors = o3d.utility.Vector3dVector(cdup)
    mesh.compute_vertex_normals()
    return mesh


def _o3d_lineset(cells, max_lines=50):
    """Build an Open3D LineSet from (centroid_a, centroid_b, sim) triples."""
    import matplotlib.cm as cm
    cmap = cm.get_cmap("RdYlGn")
    top  = sorted(cells, key=lambda c: c[2], reverse=True)[:max_lines]
    if not top:
        return None
    pts, lines, cols = [], [], []
    for i, (ca, cb, sim) in enumerate(top):
        pts  += [ca.tolist(), cb.tolist()]
        lines.append([2 * i, 2 * i + 1])
        c = cmap(sim)
        cols.append([c[0], c[1], c[2]])
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.array(pts, dtype=np.float64))
    ls.lines  = o3d.utility.Vector2iVector(np.array(lines))
    ls.colors = o3d.utility.Vector3dVector(np.array(cols))
    return ls


# ─── matplotlib fallback helpers ─────────────────────────────────────────────

def _mpl_render_mesh(ax, vertices, faces, colors, alpha=0.82, max_faces=1500):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    n = len(faces)
    if n > max_faces:
        idx  = np.random.choice(n, max_faces, replace=False)
        fsub = faces[idx]
        csub = [colors[i] for i in idx]
    else:
        fsub, csub = faces, list(colors)
    rgba = [(c[0], c[1], c[2], alpha) for c in csub]
    poly = Poly3DCollection(vertices[fsub], linewidth=0)
    poly.set_facecolors(rgba)
    ax.add_collection3d(poly)


def _mpl_draw_lines(ax, cells, max_lines=40):
    """Draw lines; return artist tuples (line, dot_a, dot_b, col, base_lw, ca, cb, sim)."""
    import matplotlib.cm as cm
    cmap    = cm.get_cmap("RdYlGn")
    artists = []
    for ca, cb, sim in sorted(cells, key=lambda c: c[2], reverse=True)[:max_lines]:
        col     = cmap(sim)
        base_lw = 2.5 + 2.5 * sim
        line, = ax.plot([ca[0], cb[0]], [ca[1], cb[1]], [ca[2], cb[2]],
                        color=col, linewidth=base_lw, alpha=0.9, zorder=10, picker=5)
        dot_a = ax.scatter(*ca, color=col, s=40, alpha=1.0, depthshade=False,
                            zorder=11, edgecolors="white", linewidths=0.5)
        dot_b = ax.scatter(*cb, color=col, s=40, alpha=1.0, depthshade=False,
                            zorder=11, edgecolors="white", linewidths=0.5)
        artists.append((line, dot_a, dot_b, col, base_lw, ca, cb, sim))
    return artists


# ─── Open3D interactivity helpers ────────────────────────────────────────────

def _o3d_cylinder_between(p1, p2, radius, color):
    """Thick tube between two 3-D points (used for highlighted correspondence)."""
    p1, p2 = np.array(p1, float), np.array(p2, float)
    d      = p2 - p1
    length = np.linalg.norm(d)
    if length < 1e-8:
        return None
    cyl = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius, height=length, resolution=16, split=1)
    cyl.compute_vertex_normals()
    cyl.paint_uniform_color(list(color[:3]))
    # Rotate from Z-axis to d
    z   = np.array([0.0, 0.0, 1.0])
    dn  = d / length
    cp  = np.cross(z, dn)
    cp_n = np.linalg.norm(cp)
    dot  = float(np.dot(z, dn))
    if cp_n < 1e-8:
        R = np.eye(3) if dot > 0 else np.diag([1., -1., -1.])
    else:
        k  = cp / cp_n
        a  = np.arctan2(cp_n, dot)
        K  = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R  = np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)
    cyl.rotate(R, center=[0, 0, 0])
    cyl.translate((p1 + p2) / 2)
    return cyl


def _o3d_highlight_geoms(cells_sorted, sel_idx, mesh_scale, max_lines=50):
    """
    Build a list of O3D geometries for a given selection state.
    sel_idx == -1  →  all lines equally bright (reset state).
    sel_idx >= 0   →  line sel_idx shown as thick tube; others dimmed.
    """
    import matplotlib.cm as cm
    cmap = cm.get_cmap("RdYlGn")
    top  = cells_sorted[:max_lines]
    radius = max(mesh_scale * 0.012, 0.03)

    if sel_idx == -1:
        ls = _o3d_lineset(top, max_lines=max_lines)
        return [ls] if ls else []

    geoms = []
    # ── dim background for unselected lines ──────────────────────────────
    pts, segs, cols = [], [], []
    for i, (ca, cb, sim) in enumerate(top):
        if i == sel_idx:
            continue
        base = len(pts)
        pts  += [ca.tolist(), cb.tolist()]
        segs.append([base, base + 1])
        cols.append([0.75, 0.75, 0.75])
    if pts:
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(np.array(pts, float))
        ls.lines  = o3d.utility.Vector2iVector(np.array(segs))
        ls.colors = o3d.utility.Vector3dVector(np.array(cols))
        geoms.append(ls)

    # ── highlighted line as thick cylinder + endpoint spheres ────────────
    ca, cb, sim = top[sel_idx]
    col = list(cmap(sim))[:3]
    cyl = _o3d_cylinder_between(ca, cb, radius=radius, color=col)
    if cyl:
        geoms.append(cyl)
    for pt in (ca, cb):
        s = o3d.geometry.TriangleMesh.create_sphere(radius=radius * 2.5)
        s.compute_vertex_normals()
        s.paint_uniform_color(col)
        s.translate(pt.tolist())
        geoms.append(s)

    return geoms


# ─── 3D viewer window ─────────────────────────────────────────────────────────

class MeshViewerWindow(tk.Toplevel):
    """3D viewer: GPU-accelerated via Open3D (if installed), else matplotlib fallback."""

    def __init__(self, parent, results, mesh_cache, cell_cache):
        super().__init__(parent)
        self.title("3D Match Viewer" + ("  [Open3D]" if HAS_O3D else "  [matplotlib]"))
        self.geometry("1400x820")
        self.minsize(800, 500)
        self.configure(bg=DARK["bg"])

        self.valid = results
        self.mesh_cache = mesh_cache
        self._color_cache: dict = {}
        self._cell_cache  = cell_cache   # shared with BenchmarkApp — survives viewer re-opens

        # O3D interactivity state
        self._o3d_queue:           _queue.Queue = _queue.Queue(maxsize=1)
        self._o3d_mesh_geoms:      list         = []   # [mesh_a, mesh_b]
        self._o3d_mesh_geom_cache: dict         = {}   # (desc,path_a,path_b,side) -> TriangleMesh
        self._o3d_running:         bool         = False
        self._cells_sorted:        list         = []
        self._mesh_scale:          float        = 1.0
        self._sel_idx:             int          = -1   # -1 = all lines shown

        # Matplotlib pick state
        self._line_artists: list = []
        self._pick_cid  = None
        self._press_cid = None
        self._pick_happened = False
        self._mpl_sel_idx: int = -1

        self._build_ui()

    def _build_ui(self):
        # ── Selector row ──────────────────────────────────────────────────
        top = tk.Frame(self, bg=DARK["bg"])
        top.pack(fill=tk.X, padx=8, pady=(6, 2))

        tk.Label(top, text="Result:", bg=DARK["bg"],
                 fg=DARK["subtext"], font=("monospace", 9)).pack(side=tk.LEFT)

        labels = [
            f"{r['descriptor']}   "
            f"{os.path.basename(r['path_a'])} ↔ {os.path.basename(r['path_b'])}"
            for r in self.valid
        ]
        self.sel_var = tk.StringVar()
        self.combo = ttk.Combobox(self, textvariable=self.sel_var,
                                   values=labels, width=74,
                                   state="readonly", font=("monospace", 9))
        self.combo.pack(fill=tk.X, padx=8, pady=2)
        self.combo.bind("<<ComboboxSelected>>", lambda _: self._refresh())

        # Info / legend
        self.info_lbl = tk.Label(self, text="", bg=DARK["bg"],
                                  fg=DARK["subtext"], font=("monospace", 8),
                                  anchor="w")
        self.info_lbl.pack(fill=tk.X, padx=10)

        if HAS_O3D:
            self._build_o3d_ui()
        else:
            self._build_mpl_ui()

        # Apply style
        style = ttk.Style(self)
        style.configure("TCombobox",
                        fieldbackground=DARK["surface"],
                        background=DARK["surface"],
                        foreground=DARK["text"])

        if self.valid:
            self.combo.current(0)
            self._refresh()

    # ── Open3D path ───────────────────────────────────────────────────────────

    def _build_o3d_ui(self):
        """Side panel with correspondence list + reset button alongside the O3D note."""
        pane = tk.PanedWindow(self, orient=tk.HORIZONTAL,
                              bg=DARK["bg"], sashwidth=4)
        pane.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Left: line list
        left = tk.Frame(pane, bg=DARK["bg"])
        pane.add(left, minsize=200, width=230)

        tk.Label(left, text="Correspondence lines\n(click to highlight)",
                 bg=DARK["bg"], fg=DARK["subtext"],
                 font=("monospace", 8), justify="center").pack(pady=(4, 2))

        lbframe = tk.Frame(left, bg=DARK["border"])
        lbframe.pack(fill=tk.BOTH, expand=True, padx=4)

        self.line_lb = tk.Listbox(
            lbframe, bg=DARK["surface"], fg=DARK["text"],
            selectbackground=DARK["select"], relief=tk.FLAT,
            font=("monospace", 9), activestyle="none")
        vsb = ttk.Scrollbar(lbframe, orient="vertical",
                             command=self.line_lb.yview)
        self.line_lb.configure(yscrollcommand=vsb.set)
        self.line_lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.line_lb.bind("<<ListboxSelect>>", self._on_line_select)

        tk.Button(
            left, text="Reset  (show all)",
            command=self._reset_selection,
            bg=DARK["surface"], fg=DARK["subtext"],
            relief=tk.FLAT, padx=6, pady=3,
            font=("monospace", 8), cursor="hand2",
        ).pack(fill=tk.X, padx=4, pady=4)

        # Right: instruction label
        right = tk.Frame(pane, bg=DARK["bg"])
        pane.add(right, minsize=400)
        tk.Label(
            right,
            text="Open3D window opens separately  (GPU OpenGL).\n\n"
                 "• Drag  →  rotate\n"
                 "• Scroll  →  zoom\n"
                 "• Shift+drag  →  pan\n\n"
                 "Click a line in the list on the left\n"
                 "to isolate it in the 3D window.",
            bg=DARK["bg"], fg=DARK["subtext"],
            font=("monospace", 10), justify="left", anchor="nw",
        ).pack(padx=20, pady=20, anchor="nw")

    def _show_o3d(self, r):
        va, fa, sa = self.mesh_cache[r["path_a"]]
        vb, fb, sb = self.mesh_cache[r["path_b"]]
        desc_name  = r["descriptor"]

        # ── face colours (cached) ─────────────────────────────────────────
        col_key_a = (desc_name, r["path_a"])
        col_key_b = (desc_name, r["path_b"])
        if col_key_a not in self._color_cache:
            self._color_cache[col_key_a] = _desc_face_colors(desc_name, va, fa, sa)
        if col_key_b not in self._color_cache:
            self._color_cache[col_key_b] = _desc_face_colors(desc_name, vb, fb, sb)

        xa_range = va[:, 0].max() - va[:, 0].min()
        xb_range = vb[:, 0].max() - vb[:, 0].min()
        offset   = np.array([xa_range + max(xa_range, xb_range) * 0.25, 0.0, 0.0])
        vb_off   = vb + offset

        self._mesh_scale = float(np.linalg.norm(
            np.array([va[:, i].max() - va[:, i].min() for i in range(3)])))

        # ── TriangleMesh objects (cached — avoids vertex duplication + normal
        #    computation on every descriptor switch) ─────────────────────────
        geom_key_a = (desc_name, r["path_a"])
        geom_key_b = (desc_name, r["path_a"], r["path_b"])   # includes path_a because offset depends on va
        if geom_key_a not in self._o3d_mesh_geom_cache:
            self._o3d_mesh_geom_cache[geom_key_a] = _o3d_colored_mesh(
                va, fa, self._color_cache[col_key_a])
        if geom_key_b not in self._o3d_mesh_geom_cache:
            self._o3d_mesh_geom_cache[geom_key_b] = _o3d_colored_mesh(
                vb_off, fb, self._color_cache[col_key_b])
        mesh_a = self._o3d_mesh_geom_cache[geom_key_a]
        mesh_b = self._o3d_mesh_geom_cache[geom_key_b]

        # ── correspondence lines ──────────────────────────────────────────
        cells_raw = self._get_cells(r, va, fa, sa, vb, fb, sb, desc_name)
        cells = [(ca, cb + offset, sim) for ca, cb, sim in cells_raw]
        self._cells_sorted = sorted(cells, key=lambda c: c[2], reverse=True)
        self._sel_idx = -1

        # ── populate sidebar listbox ──────────────────────────────────────
        self.line_lb.delete(0, tk.END)
        import matplotlib.cm as cm
        cmap = cm.get_cmap("RdYlGn")
        for i, (_, _, sim) in enumerate(self._cells_sorted[:50]):
            bar = "█" * int(sim * 10)
            self.line_lb.insert(tk.END, f"  #{i+1:>2}  sim={sim:.3f}  {bar}")
            col = cmap(sim)
            hex_col = "#{:02x}{:02x}{:02x}".format(
                int(col[0] * 255), int(col[1] * 255), int(col[2] * 255))
            self.line_lb.itemconfig(i, fg=hex_col)

        title = (f"{desc_name}  |  "
                 + "  ".join(f"{m}={v:.3f}" for m, v in r["scores"].items()
                              if not np.isnan(v)))

        init_ls    = _o3d_lineset(self._cells_sorted)
        line_geoms = [init_ls] if init_ls else []

        if self._o3d_running:
            # ── window already open: push scene update, no new thread ─────
            self._push_o3d({"meshes": [mesh_a, mesh_b], "lines": line_geoms})
        else:
            # ── first open: launch the Visualizer in a background thread ──
            q = self._o3d_queue
            # Drain any stale item
            try:
                q.get_nowait()
            except _queue.Empty:
                pass

            def run(_ma=mesh_a, _mb=mesh_b, _ll=line_geoms):
                vis = o3d.visualization.Visualizer()
                vis.create_window(window_name=title, width=1280, height=800)
                state = {"meshes": [_ma, _mb], "lines": list(_ll)}
                for g in state["meshes"] + state["lines"]:
                    vis.add_geometry(g)

                def on_frame(v):
                    try:
                        update = q.get_nowait()
                    except _queue.Empty:
                        return False
                    if "meshes" in update:
                        for g in state["meshes"]:
                            v.remove_geometry(g, reset_bounding_box=False)
                        for g in update["meshes"]:
                            v.add_geometry(g, reset_bounding_box=False)
                        state["meshes"] = update["meshes"]
                    if "lines" in update:
                        for g in state["lines"]:
                            v.remove_geometry(g, reset_bounding_box=False)
                        for g in update["lines"]:
                            v.add_geometry(g, reset_bounding_box=False)
                        state["lines"] = update["lines"]
                    v.update_renderer()
                    return False

                vis.register_animation_callback(on_frame)
                self._o3d_running = True
                vis.run()
                self._o3d_running = False
                vis.destroy_window()

            threading.Thread(target=run, daemon=True).start()

        n = len(self._cells_sorted)
        self.info_lbl.config(
            text=f"Color: {_COLOR_DESC.get(desc_name, '')}   "
                 f"|   {n} lines — click list to isolate one   "
                 f"|   drag=rotate  scroll=zoom  shift+drag=pan")

    def _on_line_select(self, _event):
        sel = self.line_lb.curselection()
        if not sel:
            return
        idx = sel[0]
        if idx == self._sel_idx:
            self._reset_selection()
            return
        self._sel_idx = idx
        geoms = _o3d_highlight_geoms(
            self._cells_sorted, idx, self._mesh_scale)
        self._push_o3d({"lines": geoms})
        ca, cb, sim = self._cells_sorted[idx]
        self.info_lbl.config(
            text=f"Line #{idx+1}  sim={sim:.4f}  —  "
                 f"click same row or 'Reset' to show all")

    def _reset_selection(self):
        self._sel_idx = -1
        self.line_lb.selection_clear(0, tk.END)
        geoms = _o3d_highlight_geoms(
            self._cells_sorted, -1, self._mesh_scale)
        self._push_o3d({"lines": geoms})
        self.info_lbl.config(text="All correspondence lines shown")

    def _push_o3d(self, geoms):
        """Thread-safe push to O3D animation queue; drops stale item if full."""
        try:
            self._o3d_queue.put_nowait(geoms)
        except _queue.Full:
            try:
                self._o3d_queue.get_nowait()
            except _queue.Empty:
                pass
            self._o3d_queue.put_nowait(geoms)

    # ── Matplotlib fallback path ──────────────────────────────────────────────

    def _build_mpl_ui(self):
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

        # Light background for matplotlib — much better contrast
        self.fig3d = Figure(facecolor="#e8e8e8")
        self.canvas3d = FigureCanvasTkAgg(self.fig3d, master=self)

        tb_frame = tk.Frame(self, bg="#e0e0e0")
        tb_frame.pack(fill=tk.X, padx=4)
        toolbar = NavigationToolbar2Tk(self.canvas3d, tb_frame)
        toolbar.config(bg="#e0e0e0")
        toolbar.update()

        self.canvas3d.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    def _show_mpl(self, r):
        self.fig3d.clear()
        self.fig3d.patch.set_facecolor("#e8e8e8")

        va, fa, sa = self.mesh_cache[r["path_a"]]
        vb, fb, sb = self.mesh_cache[r["path_b"]]
        desc_name  = r["descriptor"]

        key_a = (desc_name, r["path_a"])
        key_b = (desc_name, r["path_b"])
        if key_a not in self._color_cache:
            self._color_cache[key_a] = _desc_face_colors(desc_name, va, fa, sa)
        if key_b not in self._color_cache:
            self._color_cache[key_b] = _desc_face_colors(desc_name, vb, fb, sb)

        xa_range = va[:, 0].max() - va[:, 0].min()
        xb_range = vb[:, 0].max() - vb[:, 0].min()
        offset   = np.array([xa_range + max(xa_range, xb_range) * 0.25, 0.0, 0.0])
        vb_off   = vb + offset

        ax = self.fig3d.add_subplot(111, projection="3d")
        ax.set_facecolor("#f5f5f5")

        _mpl_render_mesh(ax, va,     fa, self._color_cache[key_a])
        _mpl_render_mesh(ax, vb_off, fb, self._color_cache[key_b])

        # Labels
        for v, lbl, col in [
            (va,     f"A: {os.path.basename(r['path_a'])}", "#1a6ef5"),
            (vb_off, f"B: {os.path.basename(r['path_b'])}", "#1aa050"),
        ]:
            ax.text(v[:, 0].mean(), v[:, 1].mean(), v[:, 2].max() * 1.06,
                    lbl, color=col, fontsize=9, ha="center",
                    fontfamily="monospace", fontweight="bold")

        cells_raw = self._get_cells(r, va, fa, sa, vb, fb, sb, desc_name)
        cells = [(ca, cb + offset, sim) for ca, cb, sim in cells_raw]
        self._cells_sorted = sorted(cells, key=lambda c: c[2], reverse=True)
        self._mpl_sel_idx  = -1

        # Disconnect stale handlers before redrawing
        if self._pick_cid is not None:
            self.canvas3d.mpl_disconnect(self._pick_cid)
        if self._press_cid is not None:
            self.canvas3d.mpl_disconnect(self._press_cid)

        self._line_artists = _mpl_draw_lines(ax, self._cells_sorted)

        self._pick_cid  = self.canvas3d.mpl_connect("pick_event",         self._on_pick)
        self._press_cid = self.canvas3d.mpl_connect("button_press_event", self._on_press)

        all_v = np.vstack([va, vb_off])
        for i, axis in enumerate("xyz"):
            lo, hi = all_v[:, i].min(), all_v[:, i].max()
            pad = (hi - lo) * 0.05 + 0.5
            getattr(ax, f"set_{axis}lim")(lo - pad, hi + pad)

        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = True
            pane.set_facecolor("#dcdcdc")
            pane.set_edgecolor("#aaaaaa")
        ax.tick_params(colors="#555555", labelsize=6)

        scores = r["scores"]
        sc_str = "   ".join(f"{m}={v:.3f}" for m, v in scores.items()
                             if not np.isnan(v))
        self.fig3d.suptitle(f"{desc_name}  |  {sc_str}",
                             color="#111111", fontsize=10,
                             fontfamily="monospace")

        n_lines = len(self._cells_sorted)
        self.info_lbl.config(
            text=f"Color: {_COLOR_DESC.get(desc_name, '')}   "
                 f"|   {n_lines} lines — click a line to isolate it   "
                 f"|   drag=rotate  scroll=zoom")
        self.canvas3d.draw()

    # ── Matplotlib pick interaction ───────────────────────────────────────────

    def _on_pick(self, event):
        """Handle pick_event: highlight the clicked line, or toggle off."""
        self._pick_happened = True
        for idx, (line, dot_a, dot_b, col, base_lw, ca, cb, sim) in enumerate(self._line_artists):
            if event.artist is line:
                if self._mpl_sel_idx == idx:
                    self._reset_highlight()
                else:
                    self._highlight_line(idx)
                break

    def _on_press(self, event):
        """Handle button_press_event: if no pick fired, reset highlight."""
        if not self._pick_happened:
            self._reset_highlight()
        self._pick_happened = False

    def _highlight_line(self, idx):
        """Dim all lines except idx; make idx thick and opaque."""
        self._mpl_sel_idx = idx
        for i, (line, dot_a, dot_b, col, base_lw, ca, cb, sim) in enumerate(self._line_artists):
            if i == idx:
                line.set_linewidth(base_lw + 2.5)
                line.set_alpha(1.0)
                dot_a.set_alpha(1.0)
                dot_b.set_alpha(1.0)
            else:
                line.set_linewidth(0.8)
                line.set_alpha(0.12)
                dot_a.set_alpha(0.08)
                dot_b.set_alpha(0.08)
        ca, cb, sim = self._cells_sorted[idx]
        self.info_lbl.config(
            text=f"Line #{idx+1}  sim={sim:.4f}  —  click same line or click empty area to show all")
        self.canvas3d.draw_idle()

    def _reset_highlight(self):
        """Restore all lines to their original weight/alpha."""
        self._mpl_sel_idx = -1
        for line, dot_a, dot_b, col, base_lw, ca, cb, sim in self._line_artists:
            line.set_linewidth(base_lw)
            line.set_alpha(0.9)
            dot_a.set_alpha(1.0)
            dot_b.set_alpha(1.0)
        n = len(self._cells_sorted)
        self.info_lbl.config(
            text=f"{n} correspondence lines   |   click a line to isolate it")
        self.canvas3d.draw_idle()

    # ── shared helpers ────────────────────────────────────────────────────────

    def _get_cells(self, r, va, fa, sa, vb, fb, sb, desc_name):
        """
        Cell correspondences, cached per (desc, pair).
        Cells are stored WITHOUT display offset — callers add offset to cb at render time.
        """
        key = (desc_name, r["path_a"], r["path_b"])
        if key not in self._cell_cache:
            n_total = 4 * 6   # n_rings × n_sectors

            def _progress(done, total):
                self.after(0, self.info_lbl.config,
                           {"text": f"Computing correspondence lines …  {done}/{total} cells"})

            self.info_lbl.config(
                text=f"Computing correspondence lines …  0/{n_total} cells")
            self.update()
            self._cell_cache[key] = _compute_cell_correspondences(
                desc_name, va, fa, sa, vb, fb, sb, on_progress=_progress)
        return self._cell_cache[key]

    def _refresh(self):
        idx = self.combo.current()
        if not (0 <= idx < len(self.valid)):
            return
        r = self.valid[idx]
        self.config(cursor="watch")
        self.update()
        try:
            if HAS_O3D:
                self._show_o3d(r)
            else:
                self._show_mpl(r)
        finally:
            self.config(cursor="")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    initial_pair = None
    if len(sys.argv) == 3:
        for p in sys.argv[1:]:
            if not os.path.exists(p):
                print(f"ERROR: file not found: {p}")
                sys.exit(1)
        initial_pair = (sys.argv[1], sys.argv[2])
    elif len(sys.argv) > 1:
        print("Usage: python benchmark_gui.py [mesh_a mesh_b]")
        sys.exit(1)

    app = BenchmarkApp(initial_pair=initial_pair)
    app.mainloop()


if __name__ == "__main__":
    main()
