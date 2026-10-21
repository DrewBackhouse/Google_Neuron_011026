#!/bin/bash
# Submits Bash_Script.sh as a job array, working out --array from the number of (L, N, init, J, bond, step, order, cbe, cbe0) combinations
# Usage:  ./submit.sh --bc obc --total_time 200 --ls "4 6" --ns "8" --inits "0 001 0101" --js "0 0.1 0.2" --bonds "128" --steps "0.05 0.025" --cbes "0.1 0.2" --cbe0s "none 1.0" --mem 2G --time 1-12:00:00
# --ns are numbers of Fock states, --inits are initial spin states (0 = up, 1 = down, padded with 0 to L), each saved in its own Output folder
# --ls / --ns / --inits / --js / --bonds / --steps / --cbes / --cbe0s are space separated lists (--cbe0s "none" leaves cbe_expand_0 out), left out they use the defaults in Bash_Script.sh (ORDERS is always taken from there)
# --bc is pbc (default) or obc for the whole array
# --total_time is the total evolution time for the whole array (default 100), not to be confused with sbatch's --time wall clock limit
# Every other option (--mem, --time, ...) is passed straight to sbatch, overriding the #SBATCH lines in Bash_Script.sh
# Memory and time apply to every task in the array, so submit runs with very different requirements separately
# The job is named from L, N, BC, init and total time, e.g. L6_N8_PBC_S1_T100(lists joined with -, e.g. L4-6), unless --job-name / -J is given
# squeue cuts names to 8 characters, see them in full with: squeue -u $USER -o "%.12i %.24j %.8T %.10M %R"

cd "$(dirname "$0")"

SBATCH_ARGS=()
NAME_GIVEN=0
while [ $# -gt 0 ]; do
    case $1 in
        --bc)    export BC=$2; shift 2 ;;
        --total_time) export TOTAL_TIME=$2; shift 2 ;;
        --ls)    export LS=$2; shift 2 ;;
        --ns)    export NS=$2; shift 2 ;;
        --job-name*|-J*) NAME_GIVEN=1; SBATCH_ARGS+=("$1"); shift ;; # The value (if separate) goes to sbatch on the next pass
        --inits) export INITS=$2; shift 2 ;;
        --js)    export JS=$2; shift 2 ;;
        --bonds) export BONDS=$2; shift 2 ;;
        --steps) export STEPS=$2; shift 2 ;;
        --cbes)  export CBES=$2; shift 2 ;;
        --cbe0s) export CBE0S=$2; shift 2 ;;
        *)       SBATCH_ARGS+=("$1"); shift ;;
    esac
done

# Evaluate the list definitions from Bash_Script.sh in a subshell, so both scripts always agree on the number of tasks and the defaults in the job name
read N JOB_NAME < <(eval "$(grep -E '^(BC|TOTAL_TIME|LS|NS|INITS|JS|BONDS|STEPS|ORDERS|CBES|CBE0S)=' Bash_Script.sh)"
    join() { local IFS=-; echo "$*"; }
    echo $(( ${#LS[@]} * ${#NS[@]} * ${#INITS[@]} * ${#JS[@]} * ${#BONDS[@]} * ${#STEPS[@]} * ${#ORDERS[@]} * ${#CBES[@]} * ${#CBE0S[@]} )) \
         "L$(join "${LS[@]}")_N$(join "${NS[@]}")_$(echo "$BC" | tr a-z A-Z)_S$(join "${INITS[@]}")_T$TOTAL_TIME")
[ $NAME_GIVEN = 0 ] && SBATCH_ARGS=(--job-name="$JOB_NAME" "${SBATCH_ARGS[@]}")

if [ ! -x TeNPyvenv/bin/python ]; then # A venv made on a Mac won't run on the cluster, so it is built here on first use
    ./setup_venv.sh || { echo "Building TeNPyvenv failed, not submitting"; exit 1; }
fi

mkdir -p logs
echo "Submitting $N tasks: BC=${BC:-default} TOTAL_TIME=${TOTAL_TIME:-default} LS=(${LS:-default}) NS=(${NS:-default}) INITS=(${INITS:-default}) JS=(${JS:-default}) BONDS=(${BONDS:-default}) STEPS=(${STEPS:-default}) CBES=(${CBES:-default}) CBE0S=(${CBE0S:-default}) ${SBATCH_ARGS[*]}"
sbatch --array=0-$(( N - 1 )) --export=ALL "${SBATCH_ARGS[@]}" Bash_Script.sh
