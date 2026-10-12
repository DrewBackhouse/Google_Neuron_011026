#!/bin/bash
#SBATCH --job-name=TN_Sim_Scan
#SBATCH --partition=cpu
#SBATCH --array=0-17                 # 3 bond dims x 3 timesteps x 2 TEBD orders
#SBATCH --cpus-per-task=9            # One core per J value (J_List has 9 entries)
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out

# Submit from the project directory with:  mkdir -p logs && sbatch run_scan.sh

BONDS=(64 128 256)
STEPS=(0.1 0.05 0.025)
ORDERS=(2 4)

i=$SLURM_ARRAY_TASK_ID
ORDER=${ORDERS[$(( i % 2 ))]}
STEP=${STEPS[$(( (i / 2) % 3 ))]}
BOND=${BONDS[$(( i / 6 ))]}

# Parallelism comes from running J values in separate processes, so stop each one spawning its own BLAS threads
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1

module load python/3.11.6-gcc-13.2.0                # Check the exact module name on CREATE with: module avail anaconda3
source ~/GoogleQVM/bin/activate

cd "$SLURM_SUBMIT_DIR"
echo "Task $i: max_bond=$BOND timestep=$STEP tebd_order=$ORDER on $(hostname)"
python TN_Sim_Scan.py --max_bond "$BOND" --timestep "$STEP" --tebd_order "$ORDER" --workers "$SLURM_CPUS_PER_TASK"
