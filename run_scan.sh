#!/bin/bash
#SBATCH --job-name=TN_Sim_Scan
#SBATCH --partition=cpu
#SBATCH --array=0-11                 # 4 bond dims x 3 timesteps x 1 TEBD order
#SBATCH --cpus-per-task=10           # One core per J value (J_List has 10 entries)
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out

# Submit from the project directory with:  mkdir -p logs && sbatch run_scan.sh

BONDS=(32 64 128 256)
STEPS=(0.1 0.05 0.025)
ORDERS=(2)

# Index maths derived from the array lengths, so --array must be 0 to (#BONDS x #STEPS x #ORDERS - 1)
i=$SLURM_ARRAY_TASK_ID
nO=${#ORDERS[@]}; nS=${#STEPS[@]}
ORDER=${ORDERS[$(( i % nO ))]}
STEP=${STEPS[$(( (i / nO) % nS ))]}
BOND=${BONDS[$(( i / (nO * nS) ))]}

# Parallelism comes from running J values in separate processes, so stop each one spawning its own BLAS threads
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1

module load python/3.11.6-gcc-13.2.0                # Check the exact module name on CREATE with: module avail python
source ~/GoogleQVM/bin/activate

cd "$SLURM_SUBMIT_DIR"
echo "Task $i: max_bond=$BOND timestep=$STEP tebd_order=$ORDER on $(hostname)"
python TN_Sim_Scan.py --max_bond "$BOND" --timestep "$STEP" --tebd_order "$ORDER" --workers "$SLURM_CPUS_PER_TASK"
