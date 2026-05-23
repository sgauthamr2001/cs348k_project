#!/usr/bin/env python3
"""plot_results.py — generate Checkpoint 2 figures from the CSV results.
Usage:  python3 plot_results.py [data_dir]   (default: ./data)
Writes PNGs into ./plots/
"""
import csv, os, sys
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"
OUT  = "plots"
os.makedirs(OUT, exist_ok=True)


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


# ---------- E2 — synthetic pointer chase: ns/access local vs remote ----------
def plot_e2():
    rows = read_csv(f"{DATA}/e2_pointer_chase.csv")
    sizes = sorted({int(r["bytes"]) for r in rows})
    by_mb = defaultdict(dict)
    for r in rows:
        by_mb[int(r["membind"])][int(r["bytes"])] = float(r["ns_per_chase"])
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(sizes))
    w = 0.35
    local  = [by_mb[0][s] for s in sizes]
    remote = [by_mb[1][s] for s in sizes]
    ratios = [r/l for l, r in zip(local, remote)]
    ax.bar(x - w/2, local,  w, label="local DRAM (membind=0)",  color="#3b7dd8")
    ax.bar(x + w/2, remote, w, label="remote DRAM (membind=1)", color="#d8743b")
    for i, ratio in enumerate(ratios):
        ax.text(i, max(local[i], remote[i]) + 5, f"{ratio:.2f}×", ha="center", fontsize=10, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s//10**6} MB" for s in sizes])
    ax.set_ylabel("ns per random pointer-chase")
    ax.set_title("E2 — Synthetic NUMA latency upper bound\n(random pointer chase, no compute)")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/e2_pointer_chase.png", dpi=150)
    plt.close()
    print(f"wrote {OUT}/e2_pointer_chase.png")


# ---------- E3 — RTIOW single-thread local vs remote (null result) ----------
def plot_e3():
    rows = read_csv(f"{DATA}/e3_AB.csv")
    grids = sorted({int(r["grid"]) for r in rows})
    by_cfg = defaultdict(dict)
    for r in rows:
        by_cfg[r["config"]][int(r["grid"])] = (int(r["setup_ms"]), int(r["render_ms"]))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    x = np.arange(len(grids))
    w = 0.35
    for ax, idx, title in [(ax1, 0, "Setup (single-threaded BVH build)"),
                           (ax2, 1, "Render (single-threaded ray tracing)")]:
        A = [by_cfg["A_local"][g][idx]  for g in grids]
        B = [by_cfg["B_remote"][g][idx] for g in grids]
        ax.bar(x - w/2, A, w, label="A: data local",  color="#3b7dd8")
        ax.bar(x + w/2, B, w, label="B: data remote", color="#d8743b")
        for i, (a, b) in enumerate(zip(A, B)):
            r = b/a if a > 0 else 0
            ax.text(i, max(a, b) * 1.02, f"{r:.2f}×", ha="center", fontsize=9, weight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"grid={g}" for g in grids])
        ax.set_ylabel("ms")
        ax.set_title(title)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle("E3 — Single-thread NUMA penalty for RTIOW\nBuild is mildly memory-bound (10-17%); render is fully compute-hidden (<5%)")
    plt.tight_layout()
    plt.savefig(f"{OUT}/e3_AB_numa.png", dpi=150)
    plt.close()
    print(f"wrote {OUT}/e3_AB_numa.png")


# ---------- E4 — Multi-thread C (interleave) vs D (sharded) heatmap ----------
def plot_e4():
    rows = read_csv(f"{DATA}/e4_CD.csv")
    grids  = sorted({int(r["grid"])  for r in rows})
    depths = sorted({int(r["depth"]) for r in rows})
    by_cfg = defaultdict(dict)
    for r in rows:
        by_cfg[r["config"]][(int(r["grid"]), int(r["depth"]))] = {
            "setup": int(r["setup_ms"]), "render": int(r["render_ms"]), "wall": int(r["wall_ms"])
        }
    # D/C ratio for render and wall
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, key, title in [(axes[0], "render", "D/C render ratio (lower = D faster)"),
                           (axes[1], "wall",   "D/C wall ratio (includes BVH build)")]:
        mat = np.zeros((len(depths), len(grids)))
        for i, d in enumerate(depths):
            for j, g in enumerate(grids):
                c = by_cfg["C"][(g, d)][key]
                dv = by_cfg["D"][(g, d)][key]
                mat[i, j] = dv / c if c > 0 else 0
        im = ax.imshow(mat, cmap="RdBu_r", aspect="auto", vmin=0.4, vmax=1.6)
        ax.set_xticks(range(len(grids))); ax.set_xticklabels([f"g={g}" for g in grids])
        ax.set_yticks(range(len(depths))); ax.set_yticklabels([f"d={d}" for d in depths])
        ax.set_xlabel("scene size (GRID_HALF)")
        ax.set_ylabel("max bounce depth")
        ax.set_title(title)
        for i in range(len(depths)):
            for j in range(len(grids)):
                ax.text(j, i, f"{mat[i,j]:.2f}×", ha="center", va="center",
                        color="white" if abs(mat[i,j]-1.0) > 0.25 else "black",
                        fontsize=10, weight="bold")
        plt.colorbar(im, ax=ax)
    plt.suptitle("E4 — Multi-thread sharded (D) vs interleaved (C), normal RTIOW\n"
                 "D wins decisively on render across all cells (~1.6-1.8× speedup)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{OUT}/e4_CD_heatmap.png", dpi=150)
    plt.close()
    print(f"wrote {OUT}/e4_CD_heatmap.png")


# ---------- E5 — Shadow-only mode (memory-bound regime) ----------
def plot_e5():
    rows = read_csv(f"{DATA}/e5_shadow.csv")
    grids   = sorted({int(r["grid"])    for r in rows})
    samples = sorted({int(r["samples"]) for r in rows})
    by_cfg = defaultdict(dict)
    for r in rows:
        by_cfg[r["config"]][(int(r["grid"]), int(r["samples"]))] = int(r["render_ms"])
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(grids) * len(samples))
    labels, c_vals, d_vals = [], [], []
    for s in samples:
        for g in grids:
            labels.append(f"g={g}\ns={s}")
            c_vals.append(by_cfg["C"][(g, s)])
            d_vals.append(by_cfg["D"][(g, s)])
    w = 0.35
    ax.bar(x - w/2, c_vals, w, label="C: interleave (32 threads, full image)",  color="#3b7dd8")
    ax.bar(x + w/2, d_vals, w, label="D: sharded (2×16 threads, NUMA-local)",   color="#43c25e")
    for i, (c, d) in enumerate(zip(c_vals, d_vals)):
        ax.text(i, max(c, d) * 1.02, f"{d/c:.2f}×", ha="center", fontsize=9, weight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("render ms (lower is better)")
    ax.set_title("E5 — Shadow-only RTIOW (memory-bound regime: hit/miss, no shading)\n"
                 "Less compute per ray exposes the NUMA cost more; sharding wins by ~1.9-2.0×")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/e5_shadow.png", dpi=150)
    plt.close()
    print(f"wrote {OUT}/e5_shadow.png")


# ---------- E4 + E5 combined summary: render-only D/C across all configs ----------
def plot_summary():
    e4 = read_csv(f"{DATA}/e4_CD.csv")
    e5 = read_csv(f"{DATA}/e5_shadow.csv")
    e4_by_cfg = defaultdict(dict)
    for r in e4:
        e4_by_cfg[r["config"]][(int(r["grid"]), int(r["depth"]))] = int(r["render_ms"])
    e5_by_cfg = defaultdict(dict)
    for r in e5:
        e5_by_cfg[r["config"]][(int(r["grid"]), int(r["samples"]))] = int(r["render_ms"])

    e4_ratios, e5_ratios = [], []
    for (g, d), r in e4_by_cfg["D"].items():
        e4_ratios.append(r / e4_by_cfg["C"][(g, d)])
    for (g, s), r in e5_by_cfg["D"].items():
        e5_ratios.append(r / e5_by_cfg["C"][(g, s)])

    fig, ax = plt.subplots(figsize=(8, 4))
    bp = ax.boxplot([e4_ratios, e5_ratios],
                     labels=["E4 — normal RTIOW\n(scatter + recursion)",
                             "E5 — shadow-only\n(hit/miss, no shading)"],
                     widths=0.5, patch_artist=True)
    bp["boxes"][0].set_facecolor("#3b7dd8aa")
    bp["boxes"][1].set_facecolor("#43c25eaa")
    ax.axhline(1.0, color="gray", linestyle="--", label="parity (D = C)")
    ax.axhline(1/1.7, color="red", linestyle=":", label="hardware NUMA upper bound (1/1.7)")
    ax.set_ylabel("D/C render-time ratio (lower = D faster)")
    ax.set_title("Sharded (D) vs interleaved (C) — render-only ratio across all sweep cells\n"
                 "D consistently approaches the hardware NUMA latency ratio; shadow-only goes a bit further")
    ax.set_ylim(0, 1.2)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/summary_boxplot.png", dpi=150)
    plt.close()
    print(f"wrote {OUT}/summary_boxplot.png")


if __name__ == "__main__":
    plot_e2()
    plot_e3()
    plot_e4()
    plot_e5()
    plot_summary()
    print(f"\nAll plots written to {OUT}/")
