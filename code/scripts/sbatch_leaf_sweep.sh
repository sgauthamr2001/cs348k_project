#!/bin/bash
#SBATCH --job-name=cpLv2
#SBATCH --output=cpLv2_%j.out
#SBATCH --error=cpLv2_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --partition=bigmem
set -u
echo "######################################################################"
echo "# cpLv2 coherence-preserving leaf-repl  job=${SLURM_JOB_ID:-NA}  node=$(hostname)  date=$(date)"
echo "######################################################################"
cd "$SLURM_SUBMIT_DIR"
echo "nproc=$(nproc)"; numactl --hardware | grep -A8 "node distances"
module load gcc/14.2.0 2>/dev/null || echo "(no gcc module load)"
which g++; g++ --version | head -1
chmod +x cpLv2_run.sh
export CPL_REPS=3
./cpLv2_run.sh
echo "# DONE  job=${SLURM_JOB_ID:-NA}  $(date)"
