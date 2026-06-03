#!/bin/bash
#SBATCH --job-name=cpF
#SBATCH --output=cpF_%j.out
#SBATCH --error=cpF_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --partition=bigmem
# bigmem: NO --exclusive; --cpus-per-task=64 expands the cpuset (QOS cap 128).
# Needs the 4-socket distance-10/12/32 node. Confirm nproc>1 + both GATE OK.
set -u
echo "######################################################################"
echo "# FINAL experiment  job=${SLURM_JOB_ID:-NA}  node=$(hostname)  date=$(date)"
echo "######################################################################"
cd "$SLURM_SUBMIT_DIR"
echo "nproc=$(nproc)"; numactl --hardware | grep -A8 "node distances"
module load gcc/14.2.0 2>/dev/null || echo "(no gcc module load)"
which g++; g++ --version | head -1
chmod +x cpF_run.sh
export CPF_REPS=3
./cpF_run.sh
echo "# DONE  job=${SLURM_JOB_ID:-NA}  $(date)"
