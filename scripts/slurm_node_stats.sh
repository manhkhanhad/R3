#!/usr/bin/env bash
# Show GPU / CPU / RAM / VRAM availability (free / total) for every Slurm node.
#
# Usage: slurm_info [-a] [-p PARTITION] [-g GPU_TYPE] [-n]
#   -a            only nodes with at least one free GPU
#   -p PARTITION  only nodes in this partition
#   -g GPU_TYPE   only nodes with this GPU type (e.g. a100, h100, a40)
#   -n            sort by node name (default: most free GPUs first)
#
# VRAM per GPU is taken from the node's feature tag (e.g. "48gb"). Slurm hands out
# whole GPUs, so "free VRAM" = free GPUs x VRAM per GPU.

only_avail=0; part=""; gtype=""; sort_free=1
while getopts "ap:g:nh" opt; do
    case $opt in
        a) only_avail=1 ;;
        p) part=$OPTARG ;;
        g) gtype=$OPTARG ;;
        n) sort_free=0 ;;
        *) sed -n '2,11p' "$0"; exit 0 ;;
    esac
done

scontrol show node -o | awk -v only_avail="$only_avail" -v part="$part" \
    -v gtype="$gtype" -v sort_free="$sort_free" '
function field(key,   i) {
    for (i = 1; i <= NF; i++) if (index($i, key "=") == 1) return substr($i, length(key) + 2)
    return ""
}
function tres(str, key,   n, a, i) {
    n = split(str, a, ",")
    for (i = 1; i <= n; i++) if (index(a[i], key "=") == 1) return substr(a[i], length(key) + 2)
    return 0
}
function to_gb(v) {                     # "500000M" / "1.5T" / "2G" -> GB
    if (v ~ /T$/) return v * 1024
    if (v ~ /G$/) return v + 0
    return v / 1024                     # M or plain number (MB)
}
{
    node  = field("NodeName")
    state = field("State")
    parts = field("Partitions")
    feats = field("AvailableFeatures")
    gres  = field("Gres")
    cfg   = field("CfgTRES")
    alloc = field("AllocTRES")

    if (part != "" && ("," parts ",") !~ ("," part ",")) next

    gpu_type = "-"
    if (match(gres, /gpu:[^:(,]+:/)) gpu_type = substr(gres, RSTART + 4, RLENGTH - 5)
    if (gtype != "" && gpu_type != gtype) next

    gpu_tot   = tres(cfg,   "gres/gpu") + 0
    gpu_alloc = tres(alloc, "gres/gpu") + 0
    cpu_tot   = field("CPUTot") + 0
    cpu_alloc = field("CPUAlloc") + 0
    mem_tot   = field("RealMemory") / 1024
    mem_alloc = field("AllocMem")  / 1024

    vram = 0
    n = split(feats, f, ",")
    for (i = 1; i <= n; i++) if (f[i] ~ /^[0-9]+gb$/) vram = f[i] + 0

    # Nodes that cannot take jobs count as having nothing free
    down = (state ~ /DOWN|DRAIN|FAIL|MAINT|RESERVED|NOT_RESPONDING|INVALID/)
    gpu_free = down ? 0 : gpu_tot - gpu_alloc
    cpu_free = down ? 0 : cpu_tot - cpu_alloc
    mem_free = down ? 0 : mem_tot - mem_alloc

    if (only_avail && gpu_free <= 0) next

    line = sprintf("%-10s %-14s %-16s  %6d %6d  %6d %6d  %6.0f %6.0f  %6d %6d  %s",
        node, substr(tolower(state), 1, 14), substr(gpu_type, 1, 16),
        gpu_free, gpu_tot, cpu_free, cpu_tot, mem_free, mem_tot,
        gpu_free * vram, gpu_tot * vram, parts)
    rows[++nrows] = line; key[nrows] = gpu_free

    sum_gpu_f += gpu_free; sum_gpu_t += gpu_tot
    sum_cpu_f += cpu_free; sum_cpu_t += cpu_tot
    sum_mem_f += mem_free; sum_mem_t += mem_tot
    sum_vr_f  += gpu_free * vram; sum_vr_t += gpu_tot * vram
}
END {
    # insertion sort by free GPUs (descending) unless -n is given
    if (sort_free) for (i = 2; i <= nrows; i++) {
        r = rows[i]; k = key[i]
        for (j = i - 1; j >= 1 && key[j] < k; j--) { rows[j+1] = rows[j]; key[j+1] = key[j] }
        rows[j+1] = r; key[j+1] = k
    }
    printf "%-10s %-14s %-16s  %-13s  %-13s  %-13s  %-13s\n",
        "", "", "", "---- GPU ----", "---- CPU ----", "-- RAM(GB) --", "-- VRAM(GB) -"
    printf "%-10s %-14s %-16s  %6s %6s  %6s %6s  %6s %6s  %6s %6s  %s\n",
        "NODE", "STATE", "GPU_TYPE", "AVAIL", "TOTAL", "AVAIL", "TOTAL",
        "AVAIL", "TOTAL", "AVAIL", "TOTAL", "PARTITIONS"
    for (i = 1; i <= nrows; i++) print rows[i]
    print "-------------------------------------------------------------------------------------------------"
    printf "%-42s  %6d %6d  %6d %6d  %6.0f %6.0f  %6d %6d\n", "TOTAL (" nrows+0 " nodes)",
        sum_gpu_f, sum_gpu_t, sum_cpu_f, sum_cpu_t, sum_mem_f, sum_mem_t, sum_vr_f, sum_vr_t
}'
