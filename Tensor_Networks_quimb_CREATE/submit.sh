#!/bin/bash
# Submits Bash_Script.sh as a job array, working out --array from the number of (L, J, bond, step, order) combinations
# Usage:  ./submit.sh --ls "4 6" --bonds "128" --steps "0.05 0.025" --mem 2G --time 1-12:00:00
# --ls / --bonds / --steps are space separated lists, left out they use the defaults in Bash_Script.sh (JS and ORDERS are always taken from there)
# Every other option (--mem, --time, ...) is passed straight to sbatch, overriding the #SBATCH lines in Bash_Script.sh
# Memory and time apply to every task in the array, so submit runs with very different requirements separately

cd "$(dirname "$0")"

SBATCH_ARGS=()
while [ $# -gt 0 ]; do
    case $1 in
        --ls)    export LS=$2; shift 2 ;;
        --bonds) export BONDS=$2; shift 2 ;;
        --steps) export STEPS=$2; shift 2 ;;
        *)       SBATCH_ARGS+=("$1"); shift ;;
    esac
done

# Evaluate the list definitions from Bash_Script.sh in a subshell, so both scripts always agree on the number of tasks
N=$(eval "$(grep -E '^(LS|JS|BONDS|STEPS|ORDERS)=' Bash_Script.sh)"; echo $(( ${#LS[@]} * ${#JS[@]} * ${#BONDS[@]} * ${#STEPS[@]} * ${#ORDERS[@]} )))

mkdir -p logs
echo "Submitting $N tasks: LS=(${LS:-default}) BONDS=(${BONDS:-default}) STEPS=(${STEPS:-default}) ${SBATCH_ARGS[*]}"
sbatch --array=0-$(( N - 1 )) --export=ALL "${SBATCH_ARGS[@]}" Bash_Script.sh
