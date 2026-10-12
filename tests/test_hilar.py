import sys
from pathlib import Path

import torch
from transformers import Qwen2Config

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "latent"))
sys.path.insert(0, str(ROOT / "src" / "latent_distill"))

from hilar_latent import HiLaRLatentModel  # noqa: E402
from hilar_quantizer import ResidualQuantizer, temporal_loss, time_prompt, window_targets  # noqa: E402

VOCAB, HIDDEN, K, D = 40, 32, 4, 16
THOUGHT = VOCAB - 1


def test_window_targets_are_chronological_suffix_unions_and_next_item():
    assert window_targets([1, 2, 3, 4, 5, 6, 7, 8], 9, 4) == [[1, 2, 3, 4, 5, 6, 7, 8], [3, 4, 5, 6, 7, 8], [5, 6, 7, 8], [9]]
    assert window_targets([1], 9, 4) == [[1], [], [], [9]]  # short history: later windows are empty


def test_time_prompt_has_dates_in_order():
    p = time_prompt(["Capo", "Strings"], [1420070400, 1450000000])
    assert p.index("[2015-01-01] Capo") < p.index("Strings")


def test_residual_quantizer_cumulates_and_trains_codebooks():
    torch.manual_seed(0)
    h = torch.randn(256, D)
    rq = ResidualQuantizer(D, levels=3, codes=8, items=10)
    rq.kmeans_init(h, iters=5)
    e, codes, vq = rq(h)
    assert e.shape == (256, 3, D) and codes.shape == (256, 3)
    errs = [(h - e[:, k]).square().mean().item() for k in range(3)]
    assert errs[0] > errs[1] > errs[2]  # each level reduces the residual
    g = torch.Generator().manual_seed(0)
    targets = [[[1, 2], [2], [3]] for _ in range(256)]
    loss = temporal_loss(rq, e, targets, 10, 4, g) + vq + (h - e[:, -1]).square().mean()
    loss.backward()
    assert rq.codebooks.grad.abs().sum() > 0 and rq.decoders[0].weight.grad.abs().sum() > 0


def tiny():
    config = Qwen2Config(vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=64, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64, eos_token_id=1)
    config.latent_larec = {"k": K, "alpha": 0.0, "gamma": 0.0}
    config.latent_hilar = {"lambda_align": 0.1, "target_dim": D}
    torch.manual_seed(0)
    model = HiLaRLatentModel(config)
    model.head.reset(0)
    model.reset_projection(0)
    model.set_targets(torch.randn(10, K, D))
    return model


def batch():
    ids = torch.tensor([[2, 3, 4] + [THOUGHT] * K + [5, 6, 1], [7, 8, 9] + [THOUGHT] * K + [10, 11, 1]])
    labels = ids.clone()
    labels[:, :3 + K] = -100
    return ids, torch.ones_like(ids), labels


def test_alignment_only_in_training_and_reaches_latents():
    model = tiny()
    ids, mask, labels = batch()
    rows = torch.tensor([3, 7])
    model.eval()
    plain = model(input_ids=ids, attention_mask=mask, labels=labels, larec_row=rows).loss
    model.train()
    out = model(input_ids=ids, attention_mask=mask, labels=labels, larec_row=rows)
    tau = model._trace["heads"]
    expected = torch.nn.functional.mse_loss(model.align_proj_q(tau), model.targets[rows])
    torch.testing.assert_close(out.loss, plain + 0.1 * expected, rtol=1e-4, atol=1e-4)
    out.loss.backward()
    assert model.align_proj_q.weight.grad.abs().sum() > 0
    assert model.head.down.weight.grad.abs().sum() > 0  # the alignment shapes the latent states
    assert model.model.layers[0].mlp.up_proj.weight.grad.abs().sum() > 0


def test_latent_steps_match_levels():
    model = tiny().eval()
    ids, mask, _ = batch()
    with torch.no_grad():
        model.generate_embs(ids, mask)
    assert model._trace["heads"].shape == (2, K, HIDDEN)


def test_rollout_replay_and_truncation():
    from hilar_grpo import title_rewards
    model = tiny().eval()
    ids, mask, _ = batch()
    ids = ids[:, :3 + K]
    mask = mask[:, :3 + K]
    with torch.no_grad():
        model.generate_embs(ids, mask)
        mu = model._trace["heads"].clone()
        sigma = torch.tensor([0.12, 0.08, 0.05, 0.025])
        eps = torch.randn(2, K, HIDDEN)
        model.rollout_sigma, model.rollout_eps = sigma, eps
        sampled = model.generate_embs(ids, mask)
        taus = model._trace["heads"].clone()
        model.rollout_sigma = model.rollout_eps = None
    torch.testing.assert_close(taus[:, 0], mu[:, 0] + 0.12 * eps[:, 0])  # first step: same state, added noise
    model.replay_tau = taus  # replaying the sampled chain reproduces its inputs and its means
    replayed = model.generate_embs(ids, mask)
    torch.testing.assert_close(replayed, sampled)
    assert model._trace["mus"].requires_grad
    model.replay_tau, model.keep_levels = taus, 1
    with torch.no_grad():
        partial = model.generate_embs(ids, mask)
    model.replay_tau = model.keep_levels = None
    torch.testing.assert_close(partial[:, 3], sampled[:, 3])
    torch.testing.assert_close(partial[:, 4:3 + K], model.model.embed_tokens(ids)[:, 4:3 + K])  # later slots empty
    assert title_rewards('"Capo X"', "Capo X") == (1.0, 1.0, 1.0)
    hit, prefix, f1 = title_rewards("Capo Y", "Capo X")
    assert hit == 0 and 0 < prefix < 1 and f1 == 0.5


def test_sampled_chain_survives_generate():
    model = tiny().eval()
    ids, mask, _ = batch()
    ids, mask = ids[:, :3 + K], mask[:, :3 + K]
    model.rollout_sigma = torch.tensor([0.12, 0.08, 0.05, 0.025])
    model.rollout_eps = torch.randn(2, K, HIDDEN)
    with torch.no_grad():
        model.generate(ids, attention_mask=mask, max_new_tokens=3, do_sample=False, pad_token_id=0)
    model.rollout_sigma = model.rollout_eps = None
    assert model.last_chain is not None and model.last_chain["heads"].shape == (2, K, HIDDEN)


def test_quantizer_main_runs_from_cached_user_states(tmp_path):
    import json

    import pandas as pd
    from hilar_quantizer import main
    n = 40
    pd.DataFrame({"history_item_id": [str([i % 7, (i + 1) % 7, (i + 2) % 7]) for i in range(n)],
                  "history_item_title": [str(["a", "b", "c"])] * n,
                  "history_timestamp": [str([1, 2, 3])] * n,
                  "item_id": [(i + 3) % 11 for i in range(n)]}).to_csv(tmp_path / "train.csv", index=False)
    torch.save({"h": torch.randn(n, D).to(torch.bfloat16)}, tmp_path / "hu.pt")
    main(csv=str(tmp_path / "train.csv"), out=str(tmp_path), levels=K, codes=4, epochs=1, batch_size=16,
         negatives=3, holdout=0.25, device="cpu")
    stats = json.load(open(tmp_path / "stats.json"))
    assert 0 <= stats["holdout_levelK_hr10"] <= 1 and stats["n_items"] == 11
    t = torch.load(tmp_path / "targets.pt", weights_only=False)
    assert t["e"].shape == (n, K, D) and t["codes"].shape == (n, K)
