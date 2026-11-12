#!/usr/bin/env bash
# SSH to a Slurm node where you already have a running job.
#
# Usage: slurm_ssh [NODE]
#   NODE   node to ssh into; omit to list your nodes (or go straight there if only one)
#
# Compute nodes use pam_slurm_adopt: ssh only works on nodes running one of your jobs.

mapfile -t my_nodes < <(squeue -u "$USER" -h -t RUNNING -o "%N" | xargs -r -n1 scontrol show hostnames | sort -u)

list_nodes() {
    echo "Nodes with your running jobs:"
    squeue -u "$USER" -h -t RUNNING -o "%N|%i|%j|%M" | sort |
        awk -F'|' '{ printf "  %-10s job %-8s %-25s %s elapsed\n", $1, $2, $3, $4 }'
}

node=$1
if [ -z "$node" ]; then
    case ${#my_nodes[@]} in
        0) echo "You have no running jobs, so there is no node to ssh into." >&2; exit 1 ;;
        1) node=${my_nodes[0]} ;;
        *) list_nodes; exit 0 ;;
    esac
fi

if [[ " ${my_nodes[*]} " != *" $node "* ]]; then
    echo "You have no running job on '$node', so ssh would be refused." >&2
    [ ${#my_nodes[@]} -gt 0 ] && list_nodes >&2
    exit 1
fi

exec ssh "$node"
