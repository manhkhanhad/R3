"""Mock-model test of ContrastGuidanceProcessor bookkeeping (CPU, no weights)."""
import sys, types
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "prototype_grounding"))
from latent_contrast import ContrastGuidanceProcessor

V, D, NB, B, L = 11, 4, 3, 2, 5


class MockOut:
    def __init__(self, logits, past):
        self.logits, self.past_key_values = logits, past


class MockModel:
    """Logits depend deterministically on (sum of embedded/dedicated inputs, cache length)
    so that cache reordering errors change outputs."""
    def __init__(self):
        self.calls = []

    def __call__(self, inputs_embeds=None, input_ids=None, attention_mask=None,
                 past_key_values=None, use_cache=True, latent_bottleneck_ids=None,
                 num_logits_to_keep=0):
        if inputs_embeds is not None:  # prefill: past = per-row sum of embeddings
            state = inputs_embeds.sum(dim=(1, 2)) + attention_mask.sum(dim=1).float()
            n = inputs_embeds.size(0)
        else:  # decode: fold last token into per-row state
            state = past_key_values + input_ids[:, -1].float()
            n = input_ids.size(0)
        logits = state.unsqueeze(1).unsqueeze(2) * torch.linspace(-1, 1, V).view(1, 1, V)
        logits = logits.expand(n, 1, V).clone()
        self.calls.append(("prefill" if inputs_embeds is not None else "decode", n))
        return MockOut(logits, state)  # past is a plain tensor -> index_select fallback


def test_reorder_and_mask_growth():
    torch.manual_seed(0)
    model = MockModel()
    prompt_ids = torch.randint(1, V, (B, L))
    prompt_mask = torch.ones(B, L, dtype=torch.long)
    null_embs = torch.randn(B, L, D)
    proc = ContrastGuidanceProcessor(model, null_embs, prompt_mask, prompt_ids, num_beams=NB, w=3.0)

    # step 1: prefill — sequences are the expanded prompts
    seqs = prompt_ids.repeat_interleave(NB, 0)
    scores = torch.randn(B * NB, V)
    out1 = proc(seqs, scores)
    assert out1.shape == scores.shape and proc.mask.shape == (B * NB, L)
    past_after_prefill = proc.past.clone()

    # step 2: append tokens, reorder beams within each group
    perm = torch.cat([torch.tensor([2, 0, 1]), torch.tensor([1, 2, 0]) + NB])
    new_tok = torch.randint(1, V, (B * NB, 1))
    seqs2 = torch.cat([seqs[perm], new_tok], dim=1)
    out2 = proc(seqs2, torch.randn(B * NB, V))
    assert proc.mask.shape == (B * NB, L + 1)
    # the null state must follow the SAME beams: past == reordered prefill state + new token
    expect = past_after_prefill[perm] + new_tok[:, 0].float()
    assert torch.allclose(proc.past, expect), (proc.past, expect)

    # step 3: another reorder, including a duplicated beam (beam search can copy a beam)
    perm3 = torch.cat([torch.tensor([1, 1, 0]), torch.tensor([0, 2, 2]) + NB])
    # duplicated prefixes are legitimate only if their sequences are identical — force that:
    seqs3_base = seqs2.clone()
    seqs3_base[0] = seqs2[1]; seqs3_base[2] = seqs2[0]
    seqs3_base[NB + 1] = seqs2[NB + 2]
    tok3 = torch.randint(1, V, (B * NB, 1))
    seqs3 = torch.cat([seqs3_base, tok3], dim=1)
    proc.prev_ids = seqs2
    out3 = proc(seqs3, torch.randn(B * NB, V))
    assert out3.shape == (B * NB, V)
    s = proc.summary()
    assert s["steps"] == 3 and s["mean_abs_delta"] >= 0
    print("OK: prefill/reorder/mask-growth/stats all consistent")


def test_guidance_identity_at_w1():
    """w=1 must return the conditional distribution untouched (up to normalization)."""
    model = MockModel()
    prompt_ids = torch.randint(1, V, (1, L))
    proc = ContrastGuidanceProcessor(model, torch.randn(1, L, D), torch.ones(1, L, dtype=torch.long),
                                     prompt_ids, num_beams=1, w=1.0)
    scores = torch.randn(1, V)
    out = proc(prompt_ids, scores)
    assert torch.allclose(out, torch.log_softmax(scores, -1), atol=1e-5)
    print("OK: w=1 is the identity on normalized conditional scores")


test_reorder_and_mask_growth()
test_guidance_identity_at_w1()


def test_cross_group_copy():
    """A finished batch item's rows point at absolute row 0 (BeamSearchScorer):
    the reorder must find the prefix across groups."""
    model = MockModel()
    prompt_ids = torch.randint(1, V, (B, L))
    proc = ContrastGuidanceProcessor(model, torch.randn(B, L, D), torch.ones(B, L, dtype=torch.long),
                                     prompt_ids, num_beams=NB, w=2.0)
    seqs = prompt_ids.repeat_interleave(NB, 0)
    proc(seqs, torch.randn(B * NB, V))
    past0 = proc.past.clone()
    # group 1 is "done": all its rows copy absolute row 0 (group 0's first beam)
    perm = torch.tensor([0, 1, 2, 0, 0, 0])
    tok = torch.randint(1, V, (B * NB, 1))
    seqs2 = torch.cat([seqs[perm], tok], dim=1)
    proc(seqs2, torch.randn(B * NB, V))
    assert torch.allclose(proc.past, past0[perm] + tok[:, 0].float())
    print("OK: cross-group copy (finished batch item) handled")


test_cross_group_copy()
