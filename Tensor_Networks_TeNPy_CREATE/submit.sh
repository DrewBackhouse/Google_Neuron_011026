#!/bin/bash
# Submits Bash_Script.sh as a job array, working out --array from the number of (L, J, bond, step, order, cbe, cbe0) combinations
# Usage:  ./submit.sh --bc obc --ls "4 6" --js "0 0.1 0.2" --bonds "128" --steps "0.05 0.025" --cbes "0.1 0.2" --cbe0s "none 1.0" --mem 2G --time 1-12:00:00
# --ls / --js / --bonds / --steps / --cbes / --cbe0s are space separated lists (--cbe0s "none" leaves cbe_expand_0 out), left out they use the defaults in Bash_Script.sh (ORDERS is always taken from there)
# --bc is pbc (default) or obc for the whole array
# Every other option (--mem, --time, ...) is passed straight to sbatch, overriding the #SBATCH lines in Bash_Script.sh
# Memory and time apply to every task in the array, so submit runs with very different requirements separately

cd "$(dirname "$0")"

SBATCH_ARGS=()
while [ $# -gt 0 ]; do
    case $1 in
        --bc)    export BC=$2; shift 2 ;;
        --ls)    export LS=$2; shift 2 ;;
        --js)    export JS=$2; shift 2 ;;
        --bonds) export BONDS=$2; shift 2 ;;
        --steps) export STEPS=$2; shift 2 ;;
        --cbes)  export CBES=$2; shift 2 ;;
        --cbe0s) export CBE0S=$2; shift 2 ;;
        *)       SBATCH_ARGS+=("$1"); shift ;;
    esac
done

# Evaluate the list definitions from Bash_Script.sh in a subshell, so both scripts always agree on the number of tasks
N=$(eval "$(grep -E '^(LS|JS|BONDS|STEPS|ORDERS|CBES|CBE0S)=' Bash_Script.sh)"; echo $(( ${#LS[@]} * ${#JS[@]} * ${#BONDS[@]} * ${#STEPS[@]} * ${#ORDERS[@]} * ${#CBES[@]} * ${#CBE0S[@]} )))

if [ ! -x TeNPyvenv/bin/python ]; then # A venv made on a Mac won't run on the cluster, so it is built here on first use
    ./setup_venv.sh || { echo "Building TeNPyvenv failed, not submitting"; exit 1; }
fi

mkdir -p logs
echo "Submitting $N tasks: BC=${BC:-default} LS=(${LS:-default}) JS=(${JS:-default}) BONDS=(${BONDS:-default}) STEPS=(${STEPS:-default}) CBES=(${CBES:-default}) CBE0S=(${CBE0S:-default}) ${SBATCH_ARGS[*]}"
sbatch --array=0-$(( N - 1 )) --export=ALL "${SBATCH_ARGS[@]}" Bash_Script.sh
