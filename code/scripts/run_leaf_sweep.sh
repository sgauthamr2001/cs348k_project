#!/bin/bash
# cpLv2_run.sh -- LEAF-geometry replication sweep.
# cpF showed: for the memory-bound hairball, full replication (R) wins (R/N~0.75
# at dist32) but top-K NODE hybrid (H) does NOT (H~N), because the bandwidth is
# in the leaf GEOMETRY, not the upper tree. This experiment tests the fix:
# replicate the hottest FRACTION of leaf VERTICES per socket (ranked by primary
# visibility), and sweep that fraction. Question: does a small hot fraction
# recover most of R's benefit at a fraction of R's memory? (the Pareto knee)
#
# leafhybrid replicates verts only; tris/nodes stay interleaved -> isolates the
# leaf-geometry contribution. Reference points: naive(N), replicate(R, full
# geometry per socket), hybrid(H, top-K NODES). All at equal 2X threads, local
# queue, controlled distance {near=12, far=32}, 3-rep median.
set -u
STAMP=$(date +%Y%m%d_%H%M%S); OUT=cpLv2_${STAMP}; mkdir -p "$OUT"
echo "===================================================================="
echo " COHERENCE-PRESERVING leaf-replication sweep (v2)  host=$(hostname)  date=$(date)"
echo "===================================================================="
numactl --hardware > "$OUT/hardware.txt"
numactl --hardware | grep -A8 "node distances"
# SMT/core layout (answers: are the 32/28 workers cores or hyperthreads?)
lscpu 2>/dev/null | grep -iE '^CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|^Socket\(s\)' | sed 's/^/  SMT: /' | tee "$OUT/lscpu.txt"
NPROC=$(nproc); MAXNODES=$(numactl --hardware | awk '/available:/ {print $2}')
echo "nproc(cpuset)=$NPROC numa_nodes=$MAXNODES"

echo ""; echo "[build]"
g++ -O3 -std=c++17 -pthread rtiow_hairball.cpp -lnuma -o rtiow_hairball 2>"$OUT/build.log" || { echo "build FAILED"; cat "$OUT/build.log"; exit 1; }
echo "  binary ready"

# ---- near/far node + threads (same scheme as cpF) ----
DISTROW=$(numactl --hardware | awk '/^ *0:/{print; exit}')
read -ra D <<< "$(echo "$DISTROW" | sed 's/^ *0: *//')"
NEAR=-1; FAR=-1; ND=9999; FD=-1
for j in "${!D[@]}"; do d=${D[$j]}; [[ "$d" == "10" ]] && continue
  if [[ $d -lt $ND ]]; then ND=$d; NEAR=$j; fi
  if [[ $d -gt $FD ]]; then FD=$d; FAR=$j; fi; done
echo "  near 0,$NEAR (dist $ND)   far 0,$FAR (dist $FD)"
X=$(( NPROC / MAXNODES )); [[ $X -lt 2 ]] && X=2; X2=$(( X * 2 ))
echo "  X=$X per node, full-machine pair uses 2X=$X2 threads (all modes equal)"

# ---- correctness gate: deterministic depth-1 compute-light, naive vs leaf must bit-match ----
echo ""; echo "[correctness gate] naive vs leafhybrid(0.5) must bit-match (deterministic config)"
GC="RTIOW_QUEUE=local RTIOW_NSTRANDS=200 RTIOW_SEGMENTS=60 RTIOW_SIDES=8 RTIOW_IMAGE_WIDTH=300 RTIOW_SAMPLES=1 RTIOW_MAX_DEPTH=1 RTIOW_COMPUTE_LIGHT=1 RTIOW_NSOCKETS=2 RTIOW_NTHREADS=4 RTIOW_SEED=42"
env $GC RTIOW_MODE=naive ./rtiow_hairball "$OUT/g_n.ppm" >/dev/null 2>&1
env $GC RTIOW_MODE=leafhybrid RTIOW_LEAF_REPL_FRAC=0.5 ./rtiow_hairball "$OUT/g_l.ppm" >/dev/null 2>&1
GN=$(md5sum "$OUT/g_n.ppm"|awk '{print $1}'); GL=$(md5sum "$OUT/g_l.ppm"|awk '{print $1}')
if [[ "$GN" == "$GL" ]]; then echo "  GATE OK ($GN)"; else echo "  *** GATE FAIL $GN != $GL -- ABORT ***"; exit 1; fi

SIZES=${CPL_SIZES:-"3000 6000"}
FRACS=${CPL_FRACS:-"0.02 0.05 0.10 0.25 0.50 1.00"}
DEPTH=${CPL_DEPTH:-5}; REPS=${CPL_REPS:-3}; TMO=${CPL_TIMEOUT:-400}
SEG=120; SID=12
echo ""; echo "  sizes=[$SIZES] fracs=[$FRACS] depth=$DEPTH reps=$REPS"

CSV="$OUT/leaf_sweep.csv"
echo "rep,size,mode,frac,dist,depth,spawned,bvh_MB,leaf_MB,coverage,render_ms,total_ms,fwd_rate" > "$CSV"

cell(){ # rep size mode frac dist node
  local REP=$1 SIZE=$2 MODE=$3 FRAC=$4 DIST=$5 NODE=$6
  local fenv=""; [[ "$MODE" == leafhybrid ]] && fenv="RTIOW_LEAF_REPL_FRAC=$FRAC"
  local fl="${FRAC//./p}"
  local TAG="sz${SIZE}_${MODE}_${fl}_${DIST}_r${REP}"; local LOG="$OUT/${TAG}.log"
  timeout ${TMO}s numactl --cpunodebind=0,$NODE --membind=0,$NODE \
    env RTIOW_QUEUE=local RTIOW_MODE=$MODE $fenv RTIOW_NSTRANDS=$SIZE RTIOW_SEGMENTS=$SEG RTIOW_SIDES=$SID \
        RTIOW_IMAGE_WIDTH=1200 RTIOW_SAMPLES=4 RTIOW_MAX_DEPTH=$DEPTH RTIOW_LEAF_MAX=4 \
        RTIOW_NSOCKETS=2 RTIOW_NTHREADS=$X2 RTIOW_REPL_DEPTH=8 \
        ./rtiow_hairball "$OUT/${TAG}.ppm" > "$LOG" 2>&1
  local bs=$(grep '^BENCH_STATS' "$LOG"|head -1)
  local r=$(echo "$bs"|grep -oE 'render_ms=-?[0-9]+'|cut -d= -f2)
  local t=$(echo "$bs"|grep -oE 'total_ms=-?[0-9]+'|cut -d= -f2)
  local f=$(echo "$bs"|grep -oE 'forwarding_rate=[0-9.]+'|cut -d= -f2)
  local bb=$(echo "$bs"|grep -oE 'bvh_bytes_per_socket=[0-9]+'|cut -d= -f2)
  local bmb=$(awk -v n="${bb:-0}" 'BEGIN{printf "%.0f", n/1048576}')
  local lmb=$(echo "$bs"|grep -oE 'leaf_repl_MB=[0-9.]+'|cut -d= -f2)
  local cov=$(grep -oE 'access_coverage=[0-9.]+' "$LOG"|head -1|cut -d= -f2)
  local sp=$(grep -oE 'spawned [0-9]+ workers' "$LOG"|grep -oE '[0-9]+'|head -1)
  echo "$REP,$SIZE,$MODE,$FRAC,$DIST,$DEPTH,${sp:-NA},${bmb:-NA},${lmb:-0},${cov:-NA},${r:-NA},${t:-NA},${f:-NA}" >> "$CSV"
  printf "    r%s sz=%-5s %-10s f=%-4s %-4s leafMB=%-6s cov=%-5s render=%-7s\n" \
    "$REP" "$SIZE" "$MODE" "$FRAC" "$DIST" "${lmb:-0}" "${cov:-NA}" "${r:-NA}"
}

echo ""; echo "############ leaf-replication sweep (N / R / H(nodes) / leaf-frac) ############"
for REP in $(seq 1 $REPS); do
  echo "  ===== rep $REP ====="
  for SIZE in $SIZES; do
    for DI in near far; do
      if [[ "$DI" == far ]]; then NODE=$FAR; else NODE=$NEAR; fi
      cell $REP $SIZE naive     NA  $DI $NODE
      cell $REP $SIZE replicate NA  $DI $NODE
      cell $REP $SIZE hybrid    NA  $DI $NODE
      for F in $FRACS; do cell $REP $SIZE leafhybrid $F $DI $NODE; done
    done
  done
done

echo ""; echo "############ SUMMARY: median render_ms, ratio vs naive, vs replicated-MB ############"
gawk -F, '
NR>1 && $11!="NA" && $11!="-1" {
  key=$2"|"$5"|"$3"|"$4; n[key]++; v[key"#"n[key]]=$11;
  lmb[$2"|"$5"|"$3"|"$4]=$9; cov[$2"|"$5"|"$3"|"$4]=$10;
  SZ[$2]=1; DI[$5]=1;
  if($3=="leafhybrid") FR[$4]=1;
}
function med(k, a,c,i,j,t,ky){c=0;for(i=1;i<=20;i++){ky=k"#"i;if(ky in v)a[++c]=v[ky]}
  if(c==0)return"";for(i=1;i<=c;i++)for(j=i+1;j<=c;j++)if(a[j]<a[i]){t=a[i];a[i]=a[j];a[j]=t}
  return (c%2)?a[int(c/2)+1]:int((a[c/2]+a[c/2+1])/2)}
END{
  nz=asorti(SZ,sz); nd=asorti(DI,dd); nf=asorti(FR,ff,"@val_num_asc")
  for(zi=1;zi<=nz;zi++){s=sz[zi]
   for(di=1;di<=nd;di++){d=dd[di]
    N=med(s"|"d"|naive|NA"); R=med(s"|"d"|replicate|NA"); H=med(s"|"d"|hybrid|NA")
    printf "\n  hairball %s strands, distance=%s   (N=%s  R=%s  H_nodes=%s ms)\n",s,d,N,R,H
    rrn=(N>0&&R>0)?sprintf("%.2f",R/N):"-"; hhn=(N>0&&H>0)?sprintf("%.2f",H/N):"-"
    printf "    full replicate R/N=%s   top-K node hybrid H/N=%s\n",rrn,hhn
    printf "    %-7s %-9s %-9s %-7s %-7s\n","frac","leaf_MB","coverage","render","L/N"
    printf "    %-7s %-9s %-9s %-7s %-7s\n","0(N)","0",   "0.000",  N,"1.00"
    for(fi=1;fi<=nf;fi++){fr=ff[fi]; k=s"|"d"|leafhybrid|"fr
      L=med(k); ln=(N>0&&L>0)?sprintf("%.2f",L/N):"-"
      printf "    %-7s %-9s %-9s %-7s %-7s\n",fr,lmb[k],cov[k],L,ln
    }
    printf "    %-7s %-9s %-9s %-7s %-7s   <- full geometry replicate (ceiling)\n","R(all)","-","1.000",R,rrn
   }
  }
  print ""
  print "  Read: if L/N drops to ~R/N at a SMALL frac (small leaf_MB), then you"
  print "  need NOT replicate all geometry -- the hot leaf fraction recovers R."
  print "  Compare the frac where L/N plateaus against the node-hybrid H/N (~1.0):"
  print "  leaf replication should beat node replication for this mesh workload."
}' "$CSV"

tar -czf "${OUT}.tar.gz" "$OUT" 2>/dev/null
echo ""; echo "DONE. Archive: $(du -sh "${OUT}.tar.gz" 2>/dev/null|cut -f1) ${OUT}.tar.gz"
exit 0
