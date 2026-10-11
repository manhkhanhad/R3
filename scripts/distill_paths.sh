#!/usr/bin/env bash
# Paths for the latent-distillation jobs (document/latent_distillation.md); sourced after
# collab_env.sh with the mode (full|smoke) as $1.  Smoke runs use the first 512 rows of
# each split, copied so that the rationale files stay aligned with the CSV rows.
distill_root=${scratch}/latent_distill
if [[ "${1:-full}" == smoke ]]; then
  data_root=${distill_root}/smoke_data
  rationales_root=${distill_root}/smoke_rationales
  for split in train valid test; do
    mkdir -p "${data_root}/${split}"
    src=$(ls ${data}/${split}/*.csv)
    [[ -f "${data_root}/${split}/$(basename "$src")" ]] || head -n 513 "$src" > "${data_root}/${split}/$(basename "$src")"
  done
else
  data_root=${data}
  rationales_root=${distill_root}/rationales
fi
info_file=$(ls ${data}/info/${category}*.txt)
