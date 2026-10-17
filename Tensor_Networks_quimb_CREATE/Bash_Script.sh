#!/bin/bash
#SBATCH --job-name=TN_Sim_Scan
#SBATCH --partition=cpu
#SBATCH --array=0-0                 # #LS x #JS x #BONDS x #STEPS x #ORDERS - 1 (set automatically when submitted with submit.sh)
#SBATCH --cpus-per-task=1           # One core per (L, J, bond, step, order) task
#SBATCH --mem=2G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%A_%a.out

# Submit from the Tensor_Networks_quimb_CREATE directory with submit.sh, e.g.:  ./submit.sh --ls "4" --bonds "128" --steps "0.05 0.025" --mem 2G --time 1-12:00:00
# Each run also writes its own log next to its .npz in Output/ (see Log_Path in Simulation.py)

# Lists are space separated (no commas). LS, BONDS and STEPS can be overridden at submission, the values here are the defaults
LS=(${LS:-4})
JS=(0.1 0.2 0.3 0.4 0.5)
BONDS=(${BONDS:-32 64 128})
STEPS=(${STEPS:-0.1 0.05})
ORDERS=(2)

# Index maths derived from the array lengths, so --array must be 0 to (#LS x #JS x #BONDS x #STEPS x #ORDERS - 1)
i=$SLURM_ARRAY_TASK_ID
nJ=${#JS[@]}; nO=${#ORDERS[@]}; nS=${#STEPS[@]}; nB=${#BONDS[@]}
J=${JS[$(( i % nJ ))]}
ORDER=${ORDERS[$(( (i / nJ) % nO ))]}
STEP=${STEPS[$(( (i / (nJ * nO)) % nS ))]}
BOND=${BONDS[$(( (i / (nJ * nO * nS)) % nB ))]}
L=${LS[$(( i / (nJ * nO * nS * nB) ))]}

# One process per task, so stop it spawning extra BLAS threads beyond its allocated core
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1

module load python/3.11.6-gcc-13.2.0                # Check the exact module name on CREATE with: module avail python
source ~/GoogleQVM/bin/activate

cd "$SLURM_SUBMIT_DIR"
echo "Task $i: L=$L J=$J max_bond=$BOND timestep=$STEP tebd_order=$ORDER on $(hostname)"
python Simulation.py --L "$L" --J "$J" --max_bond "$BOND" --timestep "$STEP" --tebd_order "$ORDER"
