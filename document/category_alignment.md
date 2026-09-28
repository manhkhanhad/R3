# Musical Instruments category alignment

The category target is the normalized CLS embedding of a product's full Amazon
category path, encoded with `Alibaba-NLP/gte-large-en-v1.5` (1024 dimensions).
The extraction script joins metadata to examples by ASIN, decodes HTML entities,
and stores one row per `item_id`. Items without a category path have `valid=False`
and do not contribute to the alignment loss.

Generate the target file from the repository root:

```bash
python src/latent/category_alignment_data.py \
  --metadata data/Musical_Instruments/meta_Musical_Instruments.json \
  --split data/Musical_Instruments/{train,valid,test}/Musical_Instruments_5_2015-1-2018-11.csv \
  --output data/Musical_Instruments/info/Musical_Instruments_gte_large_en_v1_5_categories.pt \
  --device cuda
```

Run the original SFT entry point with category alignment enabled:

```bash
python src/latent/latent_attention_train.py \
  --base_model /path/to/recommender/base-model \
  --train_file data/Musical_Instruments/train/Musical_Instruments_5_2015-1-2018-11.csv \
  --eval_file data/Musical_Instruments/valid/Musical_Instruments_5_2015-1-2018-11.csv \
  --output_dir /path/to/output \
  --category Musical_Instruments --sample -1 \
  --category_embeddings_file data/Musical_Instruments/info/Musical_Instruments_gte_large_en_v1_5_categories.pt \
  --category_alignment_weight 0.1
```

The new projection maps the generated R3 latent to the 1024-dimensional GTE
space. Both projected latents and stored targets are unit normalized, so the
per-item squared L2 loss is bounded between zero and four. The GTE encoder is
used only during preprocessing. Setting `--category_alignment_weight 0` keeps
the original training loss and dataset path.
