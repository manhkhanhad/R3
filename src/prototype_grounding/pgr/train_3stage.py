"""3-stage training curriculum driver (design doc: Training curriculum, Phase 1-3).

Stage 1 -- backbone as-is, no prototype machinery. A thin call-through to the existing,
unmodified `latent_attention_train.train()` (same function every other R3_*.sbatch run
in this repo uses) -- Phase 3's design doc note is explicit that this reduces to
"start from that" rather than reimplementing dataset/model loading a second time.
Skip this stage entirely (pass --stage1_checkpoint to stage3/run_all) if a suitable
E1 checkpoint already exists.

Stage 2 -- mapping pretrained alone, backbone frozen. Two independent things:
  (a) proj_up/proj_down' reconstruction pretraining against the full prototype bank
      (no forward pass through R3 needed -- reuses phase2_pretrain.pretrain_projector).
  (b) commitment-loss warm-start for the query heads (W_1..W_n), using real r's
      extracted from the frozen Stage-1 checkpoint on a sample of training rows.
      This is the piece phase2_pretrain.py's smoke test explicitly deferred, since no
      Stage-1 checkpoint existed yet when that was written.

Stage 3 -- joint fine-tune, backbone unfrozen, prototypes concatenated for the first
time. Loads the Stage-1 checkpoint into PrototypeGuidedLatentModel (new prototype
parameters come back freshly initialized; overwritten with Stage-2's warm-started
weights immediately after), attaches the bank, and trains with
L = L_E1 + mu*L_recons + nu*L_commit via PrototypeGuidedTrainer.

Run:
    python3 train_3stage.py stage1 --train_file ... --output_dir ckpt/stage1
    python3 train_3stage.py stage2 --stage1_checkpoint ckpt/stage1 --train_file ... --output_dir ckpt/stage2
    python3 train_3stage.py stage3 --stage1_checkpoint ckpt/stage1 --stage2_mapping ckpt/stage2/mapping.pt \\
        --train_file ... --eval_file ... --output_dir ckpt/stage3
    python3 train_3stage.py run_all --train_file ... --eval_file ... --output_dir ckpt/
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "latent"))

import fire
import torch
import transformers
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, EarlyStoppingCallback

from bank import load_prototype_bank
from layers import LatentModel  # noqa: E402  (src/latent/layers.py)
from model import PrototypeGuidedLatentModel, extract_r
from phase2_pretrain import pretrain_projector
from reasoning_dataset import LatentRDataset  # noqa: E402
from retrieval import PrototypeRetriever
from losses import commitment_loss, query_diversity_loss, usage_balance_loss
from trainer import PrototypeGuidedTrainer

DEFAULT_BANK_PATH = Path(__file__).resolve().parent / "data" / (
    "CDs_and_Vinyl_5_2015-10-2018-11_train_evidence_bank.augmented.jsonl"
)

# Same mapping latent_attention_train.train() applies before constructing LatentRDataset
# (it converts category via this dict, then passes the *display string* through) --
# LatentRDataset uses `category` verbatim in prompt text, so every call site here must
# apply the same mapping stage1 does, or the prompt text silently diverges between stages.
CATEGORY_DISPLAY = {
    "Office_Products": "office products", "Books": "books", "steam": "games",
    "CDs_and_Vinyl": "musics", "Toys_and_Games": "toys and games", "Video_Games": "video games",
    "Musical_Instruments": "music instruments", "Sports_and_Outdoors": "sports and outdoors",
    "Pet_Supplies": "pet supplies", "Arts_Crafts_and_Sewing": "arts products", "Movies": "movie",
    "yelp": "resturant",
}


def stage1(
    base_model: str,
    train_file: str,
    eval_file: str,
    output_dir: str,
    **train_kwargs,
) -> None:
    """Thin call-through to the existing latent_attention_train.train(). See module docstring."""
    from latent_attention_train import train as base_train  # local import: heavy, only needed here

    base_train(
        base_model=base_model, train_file=train_file, eval_file=eval_file,
        output_dir=output_dir, **train_kwargs,
    )


def stage2(
    stage1_checkpoint: str,
    train_file: str,
    output_dir: str,
    bank_path: str = str(DEFAULT_BANK_PATH),
    category: str = "CDs_and_Vinyl",
    n_heads: int = 3,
    k_per_head: int = 3,
    lam: float = 0.5,
    recon_steps: int = 2000,
    recon_lr: float = 1e-3,
    commit_steps: int = 500,
    commit_lr: float = 1e-3,
    diversity_weight: float = 1.0,
    balance_weight: float = 1.0,
    warmstart_batch_size: int = 128,
    r_sample_size: int = 512,
    r_batch_size: int = 8,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 0,
) -> None:
    """Mapping pretrained alone. Saves retriever + proto_projector weights to
    <output_dir>/mapping.pt, plus the bank_dim/n_heads/k_per_head/lam config needed
    to reconstruct matching modules in Stage 3."""
    print(f"Loading bank from {bank_path}")
    bank = load_prototype_bank(bank_path)
    n_verified = sum(1 for t in bank.tiers if t == "verified")
    print(f"n prototypes = {len(bank)} ({n_verified} verified), dim = {bank.dim}")

    print(f"Loading frozen Stage-1 checkpoint from {stage1_checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(stage1_checkpoint, trust_remote_code=True)
    frozen_model = LatentModel.from_pretrained(stage1_checkpoint, torch_dtype=torch.bfloat16).to(device)
    frozen_model.eval()
    hidden_size = frozen_model.config.hidden_size

    print("\n=== Stage 2a: proj_up/proj_down' reconstruction pretraining ===")
    projector = pretrain_projector(bank, steps=recon_steps, lr=recon_lr, hidden_size=hidden_size)

    print(f"\n=== Extracting r from {r_sample_size} training rows (frozen Stage-1 checkpoint) ===")
    dataset = LatentRDataset(
        train_file=train_file, tokenizer=tokenizer, sample=r_sample_size, seed=seed,
        category=CATEGORY_DISPLAY.get(category, category), test=True,  # test=True: no labels needed, just r
    )
    r_list = []
    with torch.no_grad():
        for start in range(0, len(dataset), r_batch_size):
            batch = [dataset[i] for i in range(start, min(start + r_batch_size, len(dataset)))]
            max_len = max(len(b["input_ids"]) for b in batch)
            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
            input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
            attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
            for i, b in enumerate(batch):
                L = len(b["input_ids"])
                input_ids[i, :L] = torch.tensor(b["input_ids"])
                attention_mask[i, :L] = torch.tensor(b["attention_mask"])
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            r_list.append(extract_r(frozen_model, input_ids, attention_mask).float().cpu())
    r_all = torch.cat(r_list, dim=0)
    print(f"extracted r: {r_all.shape}")

    print("\n=== Stage 2b: commitment-loss warm-start for query heads ===")
    # commitment_loss alone has no term rewarding a query for actually varying with its
    # input r, so it's free to collapse toward a constant output that always retrieves
    # the same fixed high-quality prototype -- confirmed happening in production (see
    # losses.py's module docstring). query_diversity_loss (penalizing query-vector
    # similarity within a batch) was the first fix attempted; it converges cleanly at
    # the loss level but was confirmed, across three separate real trials, NOT to fix
    # practical retrieval diversity -- it optimizes a proxy (query-vector spread) that a
    # population collapsed onto a handful of discrete modes can satisfy cheaply within
    # any small random batch. balance_weight > 0 adds usage_balance_loss instead/as well
    # -- a Switch-Transformer-style load-balancing loss ported from MoE routing collapse,
    # which penalizes the actual *retrieval usage distribution* (which bank items get
    # selected) rather than query-vector similarity.
    #
    # usage_balance_loss needs a large reference batch (warmstart_batch_size) to make
    # "how often does each item actually win" a meaningful statistic -- but it only
    # needs raw queries, not MMR's actual selection, so it's computed from a *separate*,
    # cheap, fully-vectorized retriever.query_proj() call, decoupled from
    # retriever.retrieve()'s much smaller batch. Learned this the hard way: the first
    # version of this ran retrieve() itself at warmstart_batch_size=128 -- retrieve()
    # calls mmr_select in a pure per-example Python loop with O(k*N) comparisons each
    # involving a GPU-sync .item() call, so 128 * 800 steps * ~3*N comparisons blew up
    # to hundreds of millions of .item() calls and the process was still running after
    # 51 minutes wall-clock with no sign of finishing -- killed rather than a real fix.
    # commit_loss/diversity_loss still need retrieve()'s actual MMR selection (the top-1
    # target, and the selected queries), so they stay on a small, cheap batch.
    retriever = PrototypeRetriever(
        hidden_size=hidden_size, bank_dim=bank.dim, n_heads=n_heads, k_per_head=k_per_head, lam=lam,
    )
    opt = torch.optim.Adam(retriever.parameters(), lr=commit_lr)
    mmr_batch_size = min(32, r_all.shape[0])
    balance_batch_size = min(warmstart_batch_size, r_all.shape[0])
    for step in range(commit_steps):
        opt.zero_grad()

        idx = torch.randint(0, r_all.shape[0], (mmr_batch_size,))
        r_batch = r_all[idx]
        results = retriever.retrieve(r_batch, bank)
        queries = torch.stack([res.queries for res in results])
        top1_idx = torch.stack([res.indices[:, 0] for res in results])
        targets = bank.embeddings[top1_idx]
        commit = commitment_loss(queries, targets)
        diversity = query_diversity_loss(queries)

        balance_idx = torch.randint(0, r_all.shape[0], (balance_batch_size,))
        balance_queries = retriever.query_proj(r_all[balance_idx])  # cheap, vectorized, no MMR
        balance = usage_balance_loss(balance_queries, bank.embeddings)

        loss = commit + diversity_weight * diversity + balance_weight * balance
        loss.backward()
        opt.step()
        if step % max(1, commit_steps // 10) == 0 or step == commit_steps - 1:
            print(f"  step {step:5d}  L_commit = {commit.item():.6f}  L_diversity = {diversity.item():.6f}  "
                  f"L_balance = {balance.item():.6f}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "retriever_state_dict": retriever.state_dict(),
        "proto_projector_state_dict": projector.state_dict(),
        "config": {"bank_dim": bank.dim, "n_heads": n_heads, "k_per_head": k_per_head, "lam": lam},
    }, out_dir / "mapping.pt")
    print(f"\nsaved mapping weights to {out_dir / 'mapping.pt'}")


def stage3(
    stage1_checkpoint: str,
    stage2_mapping: str,
    train_file: str,
    eval_file: str,
    output_dir: str,
    bank_path: str = str(DEFAULT_BANK_PATH),
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512,
    sample: int = 1024,
    recon_weight: float = 1.0,
    commit_weight: float = 0.1,
    diversity_weight: float = 1.0,
    balance_weight: float = 1.0,
    category: str = "CDs_and_Vinyl",
    seed: int = 0,
) -> None:
    """Joint fine-tune: backbone unfrozen, prototypes concatenated for the first time."""
    print(f"Loading bank from {bank_path}")
    bank = load_prototype_bank(bank_path)

    print(f"Loading mapping weights from {stage2_mapping}")
    mapping = torch.load(stage2_mapping, map_location="cpu")
    mcfg = mapping["config"]
    assert mcfg["bank_dim"] == bank.dim, (
        f"stage2 mapping was trained against bank_dim={mcfg['bank_dim']}, "
        f"but the bank at {bank_path} is dim={bank.dim} -- these must match, "
        f"re-run stage2 against this bank first"
    )

    print(f"Loading Stage-1 checkpoint from {stage1_checkpoint}, attaching prototype machinery")
    tokenizer = AutoTokenizer.from_pretrained(stage1_checkpoint, trust_remote_code=True)
    config = transformers.AutoConfig.from_pretrained(stage1_checkpoint, trust_remote_code=True)
    config.prototype_guided_reasoning = mcfg
    model = PrototypeGuidedLatentModel.from_pretrained(
        stage1_checkpoint, config=config, torch_dtype=torch.bfloat16,
    )
    # from_pretrained randomly initializes retriever/proto_projector (new params not in
    # the Stage-1 checkpoint) -- overwrite with Stage-2's warm-started weights now.
    model.retriever.load_state_dict(mapping["retriever_state_dict"])
    model.proto_projector.load_state_dict(mapping["proto_projector_state_dict"])
    model.set_prototype_bank(bank)
    model.config.loss_type = "ce"

    category_display = CATEGORY_DISPLAY.get(category, category)
    train_data = LatentRDataset(
        train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len,
        sample=sample, seed=seed, category=category_display,
    )
    val_data = LatentRDataset(
        train_file=eval_file, tokenizer=tokenizer, max_len=cutoff_len,
        sample=sample, category=category_display,
    )
    from datasets import Dataset as HFDataset
    hf_train_dataset = HFDataset.from_dict({k: [v[k] for v in train_data] for k in train_data[0].keys()})
    hf_val_dataset = HFDataset.from_dict({k: [v[k] for v in val_data] for k in val_data[0].keys()})

    gradient_accumulation_steps = batch_size // micro_batch_size
    trainer = PrototypeGuidedTrainer(
        model=model,
        train_dataset=hf_train_dataset,
        eval_dataset=hf_val_dataset,
        processing_class=tokenizer,
        recon_weight=recon_weight,
        commit_weight=commit_weight,
        diversity_weight=diversity_weight,
        balance_weight=balance_weight,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=20,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True,
            logging_steps=1,
            optim="adamw_torch",
            eval_strategy="epoch",
            save_strategy="epoch",
            output_dir=output_dir,
            save_total_limit=5,
            load_best_model_at_end=True,
            report_to="none",
            remove_unused_columns=False,
        ),
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )
    model.config.use_cache = False
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def run_all(
    base_model: str,
    train_file: str,
    eval_file: str,
    output_dir: str,
    bank_path: str = str(DEFAULT_BANK_PATH),
    **kwargs,
) -> None:
    """Stage 1 -> 2 -> 3 in one call, each into <output_dir>/stage{1,2,3}."""
    out = Path(output_dir)
    stage1(base_model=base_model, train_file=train_file, eval_file=eval_file,
           output_dir=str(out / "stage1"), **{k: v for k, v in kwargs.items() if k in (
               "sample", "seed", "end_k", "batch_size", "micro_batch_size", "num_epochs",
               "learning_rate", "cutoff_len", "category",
           )})
    stage2(stage1_checkpoint=str(out / "stage1"), train_file=train_file,
           output_dir=str(out / "stage2"), bank_path=bank_path,
           **{k: v for k, v in kwargs.items() if k in (
               "category", "n_heads", "k_per_head", "lam", "recon_steps", "recon_lr",
               "commit_steps", "commit_lr", "diversity_weight", "balance_weight",
               "warmstart_batch_size", "r_sample_size", "r_batch_size", "seed",
           )})
    stage3(stage1_checkpoint=str(out / "stage1"), stage2_mapping=str(out / "stage2" / "mapping.pt"),
           train_file=train_file, eval_file=eval_file, output_dir=str(out / "stage3"),
           bank_path=bank_path, **{k: v for k, v in kwargs.items() if k in (
               "batch_size", "micro_batch_size", "num_epochs", "learning_rate", "cutoff_len",
               "sample", "recon_weight", "commit_weight", "diversity_weight", "balance_weight",
               "category", "seed",
           )})


if __name__ == "__main__":
    fire.Fire({"stage1": stage1, "stage2": stage2, "stage3": stage3, "run_all": run_all})
