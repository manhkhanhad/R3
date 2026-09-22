"""Vocabulary-native continuous latent projection for R3.

The projection maps a pooled final-layer reasoning state back through the
language-model head, keeps the top-k vocabulary probabilities, and returns
their weighted mixture in the input-token embedding space.  It has no new
trainable parameters; the existing LM head and token embeddings define the
interface.
"""

import torch
import torch.nn.functional as F


def project_to_soft_vocabulary(model, latent: torch.Tensor, spec: dict):
    """Return ``(soft_latent, diagnostics)`` for a Qwen causal-LM instance."""
    top_k = int(spec.get("top_k", 64))
    temperature = float(spec.get("temperature", 1.0))
    exclude_last_n = int(spec.get("exclude_last_n", 1))

    if top_k <= 0:
        raise ValueError(f"latent soft-vocabulary top_k must be positive, got {top_k}")
    if temperature <= 0:
        raise ValueError(
            f"latent soft-vocabulary temperature must be positive, got {temperature}"
        )
    vocab_size = model.model.embed_tokens.num_embeddings
    eligible_size = vocab_size - exclude_last_n
    if exclude_last_n < 0 or eligible_size <= 0:
        raise ValueError(
            "latent soft-vocabulary exclude_last_n must leave at least one token; "
            f"vocab_size={vocab_size}, exclude_last_n={exclude_last_n}"
        )

    # The pooled reasoning state is not itself a normal final transformer
    # state.  Apply the model's final normalization before the pretrained LM
    # head, matching the hidden-to-vocabulary interface used by Qwen.
    normalized = model.model.norm(latent)
    logits = model.lm_head(normalized)[..., :eligible_size]
    selected_logits, selected_ids = torch.topk(
        logits, k=min(top_k, eligible_size), dim=-1
    )
    probabilities = F.softmax(selected_logits.float() / temperature, dim=-1)
    selected_embeddings = F.embedding(
        selected_ids, model.model.embed_tokens.weight
    ).float()
    soft_latent = torch.sum(
        probabilities.unsqueeze(-1) * selected_embeddings, dim=-2
    ).to(latent.dtype)

    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
    diagnostics = {
        "entropy": entropy.mean().detach(),
        "max_probability": probabilities.max(dim=-1).values.mean().detach(),
        "raw_norm": latent.float().norm(dim=-1).mean().detach(),
        "soft_norm": soft_latent.float().norm(dim=-1).mean().detach(),
    }
    return soft_latent, diagnostics
