#!/bin/bash
# debug_D.sh — verify that the sharded D config actually renders the FULL image
# (not just half), by stitching s0+s1 and comparing non-black pixel coverage to
# the unified C reference render. Use when you suspect a shard silently failed.
#
# Usage:  ./debug_D.sh
# Needs:  rtiow_native_mt built; a 2-socket NUMA node; python3.
set -e

[ -x rtiow_native_mt ] || {
    echo "Building rtiow_native_mt..."
    module load gcc/14.2.0 2>/dev/null || true
    g++ -std=c++20 -O3 -march=native -fopenmp rtiow_native.cpp -o rtiow_native_mt
}

# Auto-detect per-socket CPU count from cpuset (handles --exclusive vs shared)
PER_SOCKET=$(($(nproc) / 2))
S0=$PER_SOCKET
S1=$PER_SOCKET
echo "available: socket0=$S0  socket1=$S1"

mkdir -p debug/images
rm -f debug/images/*.ppm debug/*.log
GRID=500; DEPTH=10; SAMPLES=4

echo ""
echo "=== Launching D shards (s0 + s1 in parallel) ==="
( OMP_NUM_THREADS=$S0 numactl --cpunodebind=0 --membind=0 \
    env RTIOW_SHARD_BAND=1 RTIOW_SHARD_OF=2 RTIOW_SHARD_ID=0 RTIOW_SHARD_AXIS=1 \
        RTIOW_GRID_HALF=$GRID RTIOW_SAMPLES=$SAMPLES RTIOW_MAX_DEPTH=$DEPTH \
    ./rtiow_native_mt "debug/images/D_s0.ppm" > debug/s0.log 2>&1
    echo "S0_EXIT=$?" >> debug/s0.log ) &
( OMP_NUM_THREADS=$S1 numactl --cpunodebind=1 --membind=1 \
    env RTIOW_SHARD_BAND=1 RTIOW_SHARD_OF=2 RTIOW_SHARD_ID=1 RTIOW_SHARD_AXIS=1 \
        RTIOW_GRID_HALF=$GRID RTIOW_SAMPLES=$SAMPLES RTIOW_MAX_DEPTH=$DEPTH \
    ./rtiow_native_mt "debug/images/D_s1.ppm" > debug/s1.log 2>&1
    echo "S1_EXIT=$?" >> debug/s1.log ) &
wait

echo ""
echo "=== Reference C run (single process, --interleave=all) ==="
OMP_NUM_THREADS=$((S0+S1)) numactl --interleave=all \
    env RTIOW_GRID_HALF=$GRID RTIOW_SAMPLES=$SAMPLES RTIOW_MAX_DEPTH=$DEPTH \
    ./rtiow_native_mt "debug/images/C.ppm" > debug/c.log 2>&1
echo "C_EXIT=$?" >> debug/c.log

echo ""
echo "=== Files ==="; ls -la debug/images/
echo ""
echo "=== Exit codes ==="; grep "_EXIT=" debug/*.log
echo ""
echo "=== BENCH_STATS (both shards must have one) ==="
echo "--- s0 ---"; grep "BENCH_STATS" debug/s0.log || echo "  MISSING — s0 did not finish cleanly"
echo "--- s1 ---"; grep "BENCH_STATS" debug/s1.log || echo "  MISSING — s1 did not finish cleanly"
echo "--- c  ---"; grep "BENCH_STATS" debug/c.log

echo ""
echo "=== Pixel-coverage equivalence (the actual correctness test) ==="
python3 << 'PYEOF'
import os, statistics
def rd(p):
    if not os.path.exists(p) or os.path.getsize(p) < 100:
        return None
    with open(p) as f: lines = f.readlines()
    return [list(map(int, l.split())) for l in lines[3:]]

s0 = rd('debug/images/D_s0.ppm')
s1 = rd('debug/images/D_s1.ppm')
c  = rd('debug/images/C.ppm')
print(f"  D_s0.ppm:  {'OK' if s0 else 'MISSING'}")
print(f"  D_s1.ppm:  {'OK' if s1 else 'MISSING'}")
print(f"  C.ppm:     {'OK' if c else 'MISSING'}")

if s0 and s1 and c:
    combined = [[a+b for a,b in zip(p0,p1)] for p0,p1 in zip(s0,s1)]
    with open('debug/images/D_combined.ppm','w') as f:
        f.write(f"P3\n1200 675\n255\n")
        for r in combined: f.write(' '.join(map(str,r))+'\n')

    BLACK = [0,0,0]; T = len(c)
    n_c  = sum(1 for p in c  if p != BLACK)
    n_s0 = sum(1 for p in s0 if p != BLACK)
    n_s1 = sum(1 for p in s1 if p != BLACK)
    n_co = sum(1 for p in combined if p != BLACK)

    print()
    print(f"  Total pixels: {T}")
    print(f"  C non-black:        {n_c:>7}  ({100*n_c/T:.1f}%)")
    print(f"  s0 non-black:       {n_s0:>7}  ({100*n_s0/T:.1f}%)  [expect ~{n_c//2}]")
    print(f"  s1 non-black:       {n_s1:>7}  ({100*n_s1/T:.1f}%)  [expect ~{n_c//2}]")
    print(f"  combined non-black: {n_co:>7}  ({100*n_co/T:.1f}%)  [expect ~{n_c}]")

    # Additional brightness check
    def bright(p): return (p[0] + p[1] + p[2]) / 3.0
    cb = [bright(p) for p in c        if p != BLACK]
    db = [bright(p) for p in combined if p != BLACK]
    print()
    print(f"  Mean brightness over rendered pixels:")
    print(f"    C        = {statistics.mean(cb):6.2f}  (stdev {statistics.stdev(cb):6.2f})")
    print(f"    combined = {statistics.mean(db):6.2f}  (stdev {statistics.stdev(db):6.2f})")

    diff = abs(n_co - n_c) / max(n_c, 1)
    bright_diff = abs(statistics.mean(cb) - statistics.mean(db)) / statistics.mean(cb)
    print()
    if diff < 0.05 and bright_diff < 0.10:
        print("  *** PASS: D rendered the full image with correct per-pixel work. ***")
        print("  ***       Sharded measurements are valid.                       ***")
    elif n_s1 < 100:
        print("  *** FAIL: s1's PPM is missing — investigate s1.log above.       ***")
    else:
        print(f"  *** WARNING: coverage off by {100*diff:.1f}% or brightness off by {100*bright_diff:.1f}%. ***")
PYEOF
