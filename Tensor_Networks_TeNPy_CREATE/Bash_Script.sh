#!/bin/bash
#SBATCH --job-name=TeNPy_Sim_Scan
#SBATCH --partition=cpu
#SBATCH --array=0-0                 # #LS x #NS x #INITS x #JS x #BONDS x #STEPS x #ORDERS x #CBES x #CBE0S - 1 (set automatically when submitted with submit.sh)
#SBATCH --cpus-per-task=1           # One core per (L, N, init, J, bond, step, order, cbe, cbe0) task
#SBATCH --mem=2G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%A_%a.out

# Submit from the Tensor_Networks_TeNPy_CREATE directory with submit.sh, e.g.:  ./submit.sh --ls "4" --bonds "128" --steps "0.05 0.025" --mem 2G --time 1-12:00:00
# Each run also writes its own log next to its .npz in Output/ (see Log_Path in Simulation.py)

# Lists are space separated (no commas). LS, NS, INITS, JS, BONDS, STEPS, CBES and CBE0S can be overridden at submission, the values here are the defaults
# NS are numbers of Fock states per bosonic mode. Like L, each one gets its own Output folder
# INITS are initial spin states, 0 = up, 1 = down, padded with 0 to L (e.g. "0 001 0101"). Each one gets its own Output folder
# CBES / CBE0S are the QR-TEBD cbe_expand / cbe_expand_0 (none = leave cbe_expand_0 out). Ignored by the PBC (L > 2) TDVP runs, so submit those with a single CBES value
# BC (pbc or obc) and TOTAL_TIME (total evolution time, sets the T in the Output folder name) are single values for the whole array, not lists
BC=${BC:-pbc}
TOTAL_TIME=${TOTAL_TIME:-100}
LS=(${LS:-4})
NS=(${NS:-8})
INITS=(${INITS:-0})
JS=(${JS:-0.1 0.2 0.3 0.4 0.5})
BONDS=(${BONDS:-32 64 128})
STEPS=(${STEPS:-0.1 0.05})
ORDERS=(2)
CBES=(${CBES:-0.1})
CBE0S=(${CBE0S:-none})

# Index maths derived from the array lengths, so --array must be 0 to (#LS x #NS x #INITS x #JS x #BONDS x #STEPS x #ORDERS x #CBES x #CBE0S - 1)
i=$SLURM_ARRAY_TASK_ID
nJ=${#JS[@]}; nO=${#ORDERS[@]}; nS=${#STEPS[@]}; nB=${#BONDS[@]}; nC=${#CBES[@]}; nC0=${#CBE0S[@]}; nI=${#INITS[@]}; nN=${#NS[@]}
J=${JS[$(( i % nJ ))]};          i=$(( i / nJ ))
ORDER=${ORDERS[$(( i % nO ))]};  i=$(( i / nO ))
STEP=${STEPS[$(( i % nS ))]};    i=$(( i / nS ))
BOND=${BONDS[$(( i % nB ))]};    i=$(( i / nB ))
CBE=${CBES[$(( i % nC ))]};      i=$(( i / nC ))
CBE0=${CBE0S[$(( i % nC0 ))]};   i=$(( i / nC0 ))
INIT=${INITS[$(( i % nI ))]};    i=$(( i / nI ))
NFOCK=${NS[$(( i % nN ))]};      i=$(( i / nN ))
L=${LS[$i]}
CBE0_ARG=(); [ "$CBE0" != "none" ] && CBE0_ARG=(--cbe_expand_0 "$CBE0")

# One process per task, so limit BLAS threads to the allocated cores (1 unless submitted with --cpus-per-task)
NT=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT MKL_NUM_THREADS=$NT NUMBA_NUM_THREADS=$NT

module load python/3.11.6-gcc-13.2.0                # Must match the module in setup_venv.sh. Check the exact name on CREATE with: module avail python
cd "$SLURM_SUBMIT_DIR"
source TeNPyvenv/bin/activate                       # Built on the cluster by setup_venv.sh (submit.sh runs it if missing)
echo "Task $SLURM_ARRAY_TASK_ID: bc=$BC T=$TOTAL_TIME L=$L N=$NFOCK init=$INIT J=$J max_bond=$BOND timestep=$STEP tebd_order=$ORDER cbe_expand=$CBE cbe_expand_0=$CBE0 threads=$NT on $(hostname)"
python Simulation.py --bc "$BC" --T "$TOTAL_TIME" --L "$L" --N "$NFOCK" --init "$INIT" --J "$J" --max_bond "$BOND" --timestep "$STEP" --tebd_order "$ORDER" --cbe_expand "$CBE" "${CBE0_ARG[@]}"
