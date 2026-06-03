#!/bin/bash
# cpF_run.sh -- FINAL unified experiment. Three binaries, two scenes, all
# placement strategies, plus a direct NUMA-penalty profiler. Everything runs
# on one 4-socket node so distances {near=12, far=32} are controlled (same
# cores/clock/DRAM). See PLAN.md for the full description of each setup.
set -u
STAMP=$(date +%Y%m%d_%H%M%S); OUT=cpF_${STAMP}; mkdir -p "$OUT"
echo "===================================================================="
echo " FINAL experiment  host=$(hostname)  date=$(date)"
echo "===================================================================="
numactl --hardware > "$OUT/hardware.txt"
numactl --hardware | grep -A8 "node distances"
NPROC=$(nproc); MAXNODES=$(numactl --hardware | awk '/available:/ {print $2}')
echo "nproc(cpuset)=$NPROC numa_nodes=$MAXNODES"

echo ""; echo "[build]"
for b in numa_probe rtiow_hairball rtiow_spheres; do
  g++ -O3 -std=c++17 -pthread ${b}.cpp -lnuma -o $b 2>"$OUT/build_$b.log" || { echo "$b build FAILED"; cat "$OUT/build_$b.log"; exit 1; }
done
echo "  binaries ready"

# ---- near/far node + threads ----
DISTROW=$(numactl --hardware | awk '/^ *0:/{print; exit}')
read -ra D <<< "$(echo "$DISTROW" | sed 's/^ *0: *//')"
NEAR=-1; FAR=-1; ND=9999; FD=-1
for j in "${!D[@]}"; do d=${D[$j]}; [[ "$d" == "10" ]] && continue
  if [[ $d -lt $ND ]]; then ND=$d; NEAR=$j; fi
  if [[ $d -gt $FD ]]; then FD=$d; FAR=$j; fi; done
echo "  near 0,$NEAR (dist $ND)   far 0,$FAR (dist $FD)"
X=$(( NPROC / MAXNODES )); [[ $X -lt 2 ]] && X=2; X2=$(( X * 2 ))
echo "  X=$X per node, full-machine pair uses 2X=$X2 threads (all modes equal)"

# ================= EXPERIMENT 1: raw NUMA penalty =================
echo ""; echo "############ E1: NUMA penalty profiler (latency + bandwidth) ############"
NP_LAT_MB=${NP_LAT_MB:-512} NP_BW_MB=${NP_BW_MB:-1024} NP_CHASE=${NP_CHASE:-100} NP_THREADS=$X \
  ./numa_probe 2>&1 | tee "$OUT/e1_numa_probe.txt"

# ================= correctness gate for both renderers =================
echo ""; echo "############ correctness gate (mutex vs local must bit-match) ############"
gate(){ # $1 binary $2 scene-args
  local bin=$1; shift
  for Q in mutex local; do RTIOW_QUEUE=$Q RTIOW_MODE=partition $@ RTIOW_IMAGE_WIDTH=300 \
    RTIOW_SAMPLES=1 RTIOW_MAX_DEPTH=1 RTIOW_COMPUTE_LIGHT=1 RTIOW_NSOCKETS=2 RTIOW_SEED=42 \
    ./$bin "$OUT/gate_${bin}_$Q.ppm" >/dev/null 2>&1; done
  local a=$(md5sum "$OUT/gate_${bin}_mutex.ppm"|awk '{print $1}'); local b=$(md5sum "$OUT/gate_${bin}_local.ppm"|awk '{print $1}')
  [[ "$a" == "$b" ]] && echo "  $bin GATE OK ($a)" || { echo "  *** $bin GATE FAIL $a != $b -- ABORT ***"; exit 1; }
}
gate rtiow_spheres  "RTIOW_GRID_HALF=11"
gate rtiow_hairball "RTIOW_NSTRANDS=200 RTIOW_SEGMENTS=60 RTIOW_SIDES=8"

# ================= EXPERIMENT 2: placement matrix, both scenes =================
# Full machine, EQUAL threads (2X), local queue. N/R/P/H x dist x SCENE-SIZE,
# 3 reps, depth fixed. SIZE is the key axis: it controls whether the BVH spills
# L3, i.e. whether the workload is actually memory-bound. We sweep spheres from
# cache-resident to L3-spilling, and hairball likewise -- so the compute-bound
# (spheres) vs memory-bound (hairball) claim is shown ACROSS scale, not just at
# one size.
#   SPHERE_SIZES = grid_half values. 100 ~40K sph (cache-resident),
#       500 ~1M sph / ~120MB BVH (spills L3), 1000 ~4M (deep spill).
#   HAIR_SIZES   = strands (segs/sides fixed). 400 ~0.8M tris, 800 ~2.3M, 1500 ~6.75M.
# Scaling study: size points spanning cache-resident -> L3-spilling -> deep
# DRAM-bound. Hairball goes up to ~0.8 GB BVH (6000 strands ~17M tris). Cap is
# 2^28 prims/BVH (hit id is 28-bit); 6000 strands is well under.
SPHERE_SIZES=${CPF_SPHERE_SIZES:-"100 300 500 800 1200"}
HAIR_SIZES=${CPF_HAIR_SIZES:-"200 800 1600 3000 6000"}
DEPTH=${CPF_DEPTH:-5}; REPS=${CPF_REPS:-3}; TMO=${CPF_TIMEOUT:-400}
# Per-socket L3 (shared) -- the working-set threshold. getconf is reliable;
# fall back to lscpu, then a Broadwell default.
L3B=$(getconf LEVEL3_CACHE_SIZE 2>/dev/null)
if [[ -z "$L3B" || "$L3B" -eq 0 ]]; then
  L3B=$(lscpu 2>/dev/null | awk '/L3 cache/{v=$0; gsub(/[^0-9.]/,"",$NF); m=$NF;
        if(v ~ /MiB|MB/) print m*1048576; else if(v ~ /KiB|K/) print m*1024; else print m; exit}')
fi
[[ -z "$L3B" || "$L3B" -eq 0 ]] && L3B=$((40*1048576))
L3MB=$(awk -v b="$L3B" 'BEGIN{printf "%.0f", b/1048576}')
echo "  per-socket L3 = ${L3MB} MB (working-set threshold for cache vs NUMA)"
CSV="$OUT/e2_matrix.csv"
echo "rep,scene,size,mode,dist,depth,spawned,prims,bvh_MB,render_ms,total_ms,fwd_rate,note" > "$CSV"

cell(){ # rep scene size mode dist node
  local REP=$1 SCENE=$2 SIZE=$3 MODE=$4 DIST=$5 NODE=$6
  local bin sceneargs primkey
  if [[ "$SCENE" == spheres ]]; then bin=rtiow_spheres; sceneargs="RTIOW_GRID_HALF=$SIZE"; primkey=n_spheres
  else bin=rtiow_hairball; sceneargs="RTIOW_NSTRANDS=$SIZE RTIOW_SEGMENTS=120 RTIOW_SIDES=12"; primkey=n_triangles; fi
  local TAG="${SCENE}_sz${SIZE}_${MODE}_${DIST}_r${REP}"; local LOG="$OUT/${TAG}.log"
  timeout ${TMO}s numactl --cpunodebind=0,$NODE --membind=0,$NODE \
    env RTIOW_QUEUE=local RTIOW_MODE=$MODE $sceneargs RTIOW_IMAGE_WIDTH=1200 \
        RTIOW_SAMPLES=4 RTIOW_MAX_DEPTH=$DEPTH RTIOW_LEAF_MAX=4 RTIOW_NSOCKETS=2 \
        RTIOW_NTHREADS=$X2 RTIOW_REPL_DEPTH=8 \
        ./$bin "$OUT/${TAG}.ppm" > "$LOG" 2>&1
  local bs=$(grep '^BENCH_STATS' "$LOG"|head -1)
  local r=$(echo "$bs"|grep -oE 'render_ms=-?[0-9]+'|cut -d= -f2)
  local t=$(echo "$bs"|grep -oE 'total_ms=-?[0-9]+'|cut -d= -f2)
  local f=$(echo "$bs"|grep -oE 'forwarding_rate=[0-9.]+'|cut -d= -f2)
  local pr=$(echo "$bs"|grep -oE "${primkey}=[0-9]+"|cut -d= -f2)
  local bb=$(echo "$bs"|grep -oE 'bvh_bytes_per_socket=[0-9]+'|cut -d= -f2)
  local bmb=$(awk -v n="${bb:-0}" 'BEGIN{printf "%.0f", n/1048576}')
  local sp=$(grep -oE 'spawned [0-9]+ workers' "$LOG"|grep -oE '[0-9]+'|head -1)
  echo "$REP,$SCENE,$SIZE,$MODE,$DIST,$DEPTH,${sp:-NA},${pr:-NA},${bmb:-NA},${r:-NA},${t:-NA},${f:-NA},$bin" >> "$CSV"
  printf "    r%s %-8s sz=%-5s %-2s %-4s sp=%-4s bvh=%-5sMB render=%-7s\n" "$REP" "$SCENE" "$SIZE" "$MODE" "$DIST" "${sp:-NA}" "${bmb:-NA}" "${r:-NA}"
}

echo ""; echo "############ E2: placement matrix (N/R/P/H) x SCENE-SIZE, both scenes (depth=$DEPTH) ############"
for REP in $(seq 1 $REPS); do
  echo "  ===== rep $REP ====="
  for SCENE in spheres hairball; do
    if [[ "$SCENE" == spheres ]]; then SIZES="$SPHERE_SIZES"; else SIZES="$HAIR_SIZES"; fi
    for SIZE in $SIZES; do
      for D in near far; do
        if [[ "$D" == far ]]; then NODE=$FAR; else NODE=$NEAR; fi
        for MODE in naive replicate partition hybrid; do
          cell $REP $SCENE $SIZE $MODE $D $NODE
        done
      done
    done
  done
done

echo ""; echo "############ MEDIAN render_ms + ratios vs naive (per scene,size,dist) + REGIME ############"
awk -F, -v L3="$L3MB" '
NR>1 && $10!="NA" && $10!="-1" {
  k=$2"|"$3"|"$5"|"$4; n[k]++; v[k"#"n[k]]=$10;
  # full working set = naive per-socket bvh (whole interleaved tree);
  # half working set = partition per-socket bvh (its half of the scene)
  if($4=="naive")     Gfull[$2"|"$3]=$9+0
  if($4=="partition") Ghalf[$2"|"$3]=$9+0
  SC[$2]=1; DI[$5]=1; SZSET[$2"|"($3+0)]=$3+0
}
function med(k, a,c,i,j,t,ky){c=0;for(i=1;i<=20;i++){ky=k"#"i;if(ky in v)a[++c]=v[ky]}
  if(c==0)return"";for(i=1;i<=c;i++)for(j=i+1;j<=c;j++)if(a[j]<a[i]){t=a[i];a[i]=a[j];a[j]=t}
  return (c%2)?a[int(c/2)+1]:int((a[c/2]+a[c/2+1])/2)}
function regime(full,half){
  if(full=="" ) return "?"
  if(full < L3) return "cacheRes"      # whole tree fits L3 -> NUMA irrelevant
  if(half!="" && half < L3) return "CONFOUND"  # half fits L3 but full does not -> partition win is cache, not NUMA
  return "DRAMbnd"                     # even half exceeds L3 -> clean NUMA regime
}
END{
  printf "  (per-socket L3 = %s MB; regime: cacheRes=<L3, CONFOUND=half<L3<full, DRAMbnd=half>L3)\n", L3
  ns=asorti(SC,ss); nd=asorti(DI,dd)
  for(si=1;si<=ns;si++){sc=ss[si]
   delete SZ; m=0; for(key in SZSET){ split(key,pp,"|"); if(pp[1]==sc) SZ[++m]=SZSET[key] }
   for(a=1;a<=m;a++)for(b=a+1;b<=m;b++)if(SZ[b]<SZ[a]){t=SZ[a];SZ[a]=SZ[b];SZ[b]=t}
   for(di=1;di<=nd;di++){dist=dd[di]
    printf "\n  scene=%s  distance=%s   (depth fixed)\n", sc, dist
    printf "    %-7s %-8s %-9s %-7s %-7s %-7s %-7s | R/N   P/N   H/N\n","size","fullMB","regime","N","R","P","H"
    for(zi=1;zi<=m;zi++){ sz=SZ[zi]
      N=med(sc"|"sz"|"dist"|naive"); R=med(sc"|"sz"|"dist"|replicate")
      P=med(sc"|"sz"|"dist"|partition"); H=med(sc"|"sz"|"dist"|hybrid")
      rn=(N>0&&R>0)?sprintf("%.2f",R/N):"-"; pn=(N>0&&P>0)?sprintf("%.2f",P/N):"-"; hn=(N>0&&H>0)?sprintf("%.2f",H/N):"-"
      reg=regime(Gfull[sc"|"sz], Ghalf[sc"|"sz])
      printf "    %-7s %-8s %-9s %-7s %-7s %-7s %-7s | %-5s %-5s %-5s\n",sz,Gfull[sc"|"sz],reg,N,R,P,H,rn,pn,hn
    }
   }
  }
  print ""
  print "  Trust the NUMA claim only on DRAMbnd rows (even partitions half-tree > L3)."
  print "  CONFOUND rows: partition fits L3 but naive does not -> any P/N<1 is a cache"
  print "    effect, NOT NUMA locality. cacheRes rows: NUMA irrelevant (all fits L3)."
  print "  spheres (compute-bound): R/N,P/N,H/N ~1.0 even on DRAMbnd rows."
  print "  hairball (memory-bound): R/H<1 on DRAMbnd rows, more at dist32 = real NUMA."
}' "$CSV"

# ================= EXPERIMENT 3: DRAM-capacity sweep (the partition raison d'etre) =================
# DRAM is huge here, so without a constraint partition never reaches its
# defining regime (scene > one socket's memory, where R is IMPOSSIBLE). We
# model a per-socket DRAM budget (RTIOW_SOCKET_BUDGET_MB) and sweep it down at
# the LARGEST scene of each type, for all 4 modes. The binary predicts each
# mode's per-socket footprint and declares INFEASIBLE if it exceeds the budget:
#   N/P/H footprint ~ G/n_sockets ;  R footprint = G (full copy per socket).
# So as the budget shrinks, R drops out FIRST (at G), while N/P/H survive to
# ~G/ns. This is the capacity frontier: the regime where partition is the only
# performant strategy that fits.
# NOTE: the budget is a software feasibility MODEL (predict+gate), not a real
# OS cap -- it characterizes the frontier; a true memory-pressure perf study
# would need cgroups (impractical on a shared bigmem node).
echo ""; echo "############ E3: DRAM-capacity feasibility sweep (largest scene, far dist) ############"
E3CSV="$OUT/e3_capacity.csv"
echo "scene,size,mode,budget_MB,feasible,per_socket_pred_MB,render_ms,note" > "$E3CSV"
SPHERE_BIG=$(echo $SPHERE_SIZES | awk '{print $NF}')
HAIR_BIG=$(echo $HAIR_SIZES | awk '{print $NF}')
BUDGETS=${CPF_BUDGETS:-"0 2048 1024 768 512 384 256"}
e3cell(){ # scene size mode budget node
  local SCENE=$1 SIZE=$2 MODE=$3 BUD=$4 NODE=$5
  local bin sceneargs
  if [[ "$SCENE" == spheres ]]; then bin=rtiow_spheres; sceneargs="RTIOW_GRID_HALF=$SIZE"
  else bin=rtiow_hairball; sceneargs="RTIOW_NSTRANDS=$SIZE RTIOW_SEGMENTS=120 RTIOW_SIDES=12"; fi
  local TAG="cap_${SCENE}_${MODE}_B${BUD}"; local LOG="$OUT/${TAG}.log"
  timeout ${TMO}s numactl --cpunodebind=0,$NODE --membind=0,$NODE \
    env RTIOW_QUEUE=local RTIOW_MODE=$MODE $sceneargs RTIOW_IMAGE_WIDTH=1200 \
        RTIOW_SAMPLES=4 RTIOW_MAX_DEPTH=$DEPTH RTIOW_LEAF_MAX=4 RTIOW_NSOCKETS=2 \
        RTIOW_NTHREADS=$X2 RTIOW_REPL_DEPTH=8 RTIOW_SOCKET_BUDGET_MB=$BUD \
        ./$bin "$OUT/${TAG}.ppm" > "$LOG" 2>&1
  local feas=$(grep '^FEASIBILITY' "$LOG"|head -1)
  local fz=$(echo "$feas"|grep -oE 'feasible=[01]'|cut -d= -f2)
  local ps=$(echo "$feas"|grep -oE 'per_socket_pred_MB=[0-9.]+'|cut -d= -f2)
  local r="NA"; [[ "$fz" == "1" ]] && r=$(grep '^BENCH_STATS' "$LOG"|head -1|grep -oE 'render_ms=-?[0-9]+'|cut -d= -f2)
  echo "$SCENE,$SIZE,$MODE,$BUD,${fz:-NA},${ps:-NA},${r:-NA},$bin" >> "$E3CSV"
  printf "    %-8s %-2s budget=%-5s feasible=%-2s per_socket=%-7s render=%-6s\n" "$SCENE" "$MODE" "$BUD" "${fz:-NA}" "${ps:-NA}" "${r:-NA}"
}
for cfg in "spheres $SPHERE_BIG" "hairball $HAIR_BIG"; do
  set -- $cfg; SCENE=$1; SIZE=$2
  echo "  --- $SCENE size=$SIZE (far node $FAR) ---"
  for BUD in $BUDGETS; do for MODE in naive replicate partition hybrid; do e3cell $SCENE $SIZE $MODE $BUD $FAR; done; done
done
echo ""; echo "############ E3 FEASIBILITY FRONTIER (1=fits, 0=infeasible) ############"
awk -F, 'NR>1 { f[$1"|"$4"|"$3]=$5; ps[$1"|"$3]=$6; SC[$1]=1; BU[$4]=1 }
END{
  ns=asorti(SC,ss); nb=asorti(BU,bb,"@ind_num_desc")
  for(si=1;si<=ns;si++){sc=ss[si]
    printf "\n  scene=%s   per-socket footprint: N/P/H=%s/%s/%s MB  R=%s MB\n", sc,
      ps[sc"|naive"], ps[sc"|partition"], ps[sc"|hybrid"], ps[sc"|replicate"]
    printf "    %-9s %-3s %-3s %-3s %-3s\n","budgetMB","N","R","P","H"
    for(bi=1;bi<=nb;bi++){ b=bb[bi]; lbl=(b=="0"?"unlim":b)
      printf "    %-9s %-3s %-3s %-3s %-3s\n", lbl,
        f[sc"|"b"|naive"], f[sc"|"b"|replicate"], f[sc"|"b"|partition"], f[sc"|"b"|hybrid"] }
  }
  print ""
  print "  As budget shrinks R (full copy/socket) goes infeasible FIRST; N/P/H (~G/ns)"
  print "  survive longer. The band where R=0 but P=1 is partitions raison detre:"
  print "  scene exceeds per-socket DRAM -> replication impossible -> partition is the"
  print "  only local-data strategy that fits. (Budget is a feasibility model; see PLAN.)"
}' "$E3CSV"

tar -czf "${OUT}.tar.gz" "$OUT" 2>/dev/null
echo ""; echo "DONE. Archive: $(du -sh "${OUT}.tar.gz" 2>/dev/null|cut -f1) ${OUT}.tar.gz"
exit 0
