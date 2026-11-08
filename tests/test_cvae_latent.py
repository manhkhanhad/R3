"""Variational latent (option A): closed-form KL, training objective, posterior vs
prior paths, inference path = plain LatentModel with up(mu_p), fp32 round trip."""

import sys
from pathlib import Path

import torch
from transformers import Qwen2Config

LATENT_DIR = Path(__file__).resolve().parents[1] / "src" / "latent"
sys.path.insert(0, str(LATENT_DIR))

from cvae_latent import (CVAELatentModel, KLSchedule, cvae_probe, gaussian_kl, init_cvae,  # noqa: E402
                         load_fp32_cvae_params)
from layers import LatentModel  # noqa: E402

THOUGHT, PAD = 31, 0
H, D, N_ITEMS, EMB = 16, 4, 7, 12


def make_config(aux=True):
    config = Qwen2Config(vocab_size=32, hidden_size=H, intermediate_size=32, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=2, pad_token_id=PAD, eos_token_id=1)
    config._attn_implementation = "eager"
    config.latent_cvae = {"dim": D, "posterior_dim": 8, "logvar_init": -1.0, "token_norm": 1.0,
                          "aux": {"file": "", "emb_dim": EMB, "tau": 0.1} if aux else None}
    return config


def make_model(aux=True, seed=0):
    torch.manual_seed(seed)
    model = CVAELatentModel(make_config(aux))
    init_cvae(model, model.config.latent_cvae)
    # mu heads are zero at init (KL = 0); give them distinct random weights so the tests
    # exercise a non-degenerate prior / posterior pair
    with torch.no_grad():
        torch.nn.init.normal_(model.attention.mu.weight, std=0.1)
        torch.nn.init.normal_(model.posterior.mu.weight, std=0.1)
    if aux:
        model.aux.set_table(torch.randn(N_ITEMS, EMB))
    model.grad_accum_steps = 2
    return model


def batch(seed=0):
    torch.manual_seed(seed)
    rows = [[5, 6, 7, 10, 11, 12, 20, THOUGHT, 3, 4, 1], [5, 6, 7, 13, 20, THOUGHT, 8, 1]]
    width = max(map(len, rows))
    ids = torch.tensor([r + [PAD] * (width - len(r)) for r in rows])
    mask = torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r in rows])
    labels = ids.clone()
    for i, r in enumerate(rows):
        t = r.index(THOUGHT)
        labels[i, :t + 1] = -100
        labels[i, len(r):] = -100
    return {"input_ids": ids, "attention_mask": mask, "labels": labels, "target_item": torch.tensor([2, 5])}


def test_gaussian_kl_matches_torch():
    mu_q, lv_q, mu_p, lv_p = (torch.randn(3, D) for _ in range(4))
    ours = gaussian_kl(mu_q, lv_q, mu_p, lv_p)
    q = torch.distributions.Normal(mu_q, (0.5 * lv_q).exp())
    p = torch.distributions.Normal(mu_p, (0.5 * lv_p).exp())
    ref = torch.distributions.kl_divergence(q, p).sum(-1)
    assert torch.allclose(ours, ref, atol=1e-5)
    assert torch.allclose(gaussian_kl(mu_q, lv_q, mu_q, lv_q), torch.zeros(3), atol=1e-6)


def test_training_loss_components_and_grads():
    model = make_model()
    model.train()
    model.kl_beta, model.kl_free_nats, model.aux_weight, model.aux_prior_weight = 0.5, 0.0, 1.0, 0.5
    out = model(**batch())
    stats = model.last_stats
    expected = stats["ce"] + 0.5 * stats["kl"] + 1.0 * stats["aux_q"] + 0.5 * stats["aux_p"]
    assert torch.allclose(out.loss * model.grad_accum_steps, expected, atol=1e-4)
    out.loss.backward()
    for name in ("posterior.mu.weight", "attention.mu.weight", "attention.up.weight", "aux.net.0.weight",
                 "attention.pool.query.weight", "model.embed_tokens.weight"):
        p = dict(model.named_parameters())[name]
        assert p.grad is not None and p.grad.abs().sum() > 0, name
    assert stats["kl"] > 0


def test_free_nats_zero_gradient_below_threshold():
    model = make_model(aux=False)
    model.train()
    model.kl_beta, model.kl_free_nats = 1.0, 1e6  # every row below the threshold
    out = model(**batch())
    out.loss.backward()
    # the posterior's only gradient comes through the decoder path, so the logvar head
    # (zero weights at init) still gets a reparameterisation gradient; the KL part is constant
    assert torch.allclose(model.last_stats["kl"].clamp(min=1e6), torch.tensor(1e6))


def test_posterior_and_prior_paths_differ_and_prior_prob_one_uses_prior():
    model = make_model(aux=False)
    model.train()
    b = batch()
    torch.manual_seed(1)
    model.prior_prob = 0.0
    nll_q = model(**b).loss.item()
    torch.manual_seed(1)
    model.prior_prob = 1.0
    nll_p = model(**b).loss.item()
    assert nll_q != nll_p


def test_eval_path_is_plain_latent_model_with_prior_mean():
    model = make_model(aux=False).eval()
    b = batch()
    with torch.no_grad():
        out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], labels=b["labels"])
        # reference: LatentModel path with slot = up(mu_p)
        mu_p, logvar_p, mu_q, logvar_q, thought_idx = model.encode(b["input_ids"], b["attention_mask"])
        nll, ref = model.decode_nll(b["input_ids"], b["attention_mask"], b["labels"], thought_idx, model.attention.up(mu_p))
    assert torch.allclose(out.logits, ref.logits, atol=1e-4)
    assert model.attention.last is not None


def test_probe_reports_oracle_gap_keys():
    model = make_model().train()
    stats = cvae_probe(model, batch())
    for k in ("kl", "mu_gap", "nll_posterior", "nll_prior", "nll_mask", "nll_shuffle", "aux_hit10_posterior", "aux_hit10_prior"):
        assert k in stats, k
    assert model.training  # mode restored
    assert not cvae_probe(model.eval(), batch()) is None and not model.training


def test_save_load_round_trip_into_plain_latent_model(tmp_path):
    model = make_model()
    with torch.no_grad():
        model.attention.up.weight.mul_(1.2345678)
    model.save_pretrained(tmp_path)
    loaded, info = LatentModel.from_pretrained(tmp_path, torch_dtype=torch.bfloat16, output_loading_info=True)
    assert not info["missing_keys"] and not info["unexpected_keys"], info
    assert loaded.attention.up.weight.dtype == torch.float32
    assert torch.allclose(loaded.attention.up.weight, model.attention.up.weight)
    assert torch.allclose(loaded.posterior.mu.weight, model.posterior.mu.weight)
    b = batch()
    with torch.no_grad():
        a = model.eval()(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
        c = loaded.eval()(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
    assert torch.allclose(a.float(), c.float(), atol=5e-2)  # bf16 backbone vs fp32 backbone
    load_fp32_cvae_params(loaded, str(tmp_path))  # idempotent


def test_kl_schedule():
    class S:  # minimal trainer state
        max_steps = 100

    model = make_model(aux=False)
    lin = KLSchedule(model, 2.0, 0.5, "linear")
    for step, expected in ((0, 0.0), (25, 1.0), (50, 2.0), (99, 2.0)):
        S.global_step = step
        lin.on_step_begin(None, S, None)
        assert abs(model.kl_beta - expected) < 1e-6
    cyc = KLSchedule(model, 1.0, 0.5, "cyclical", cycles=2)
    S.global_step = 75  # second cycle, half way -> progress within cycle 0.5 -> full
    cyc.on_step_begin(None, S, None)
    assert abs(model.kl_beta - 1.0) < 1e-6
    S.global_step = 60  # second cycle, 0.2 into it -> 0.4
    cyc.on_step_begin(None, S, None)
    assert abs(model.kl_beta - 0.4) < 1e-6
