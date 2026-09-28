"""Attempt 11: SASRec-similarity-weighted collaborative InfoNCE on plain R3.

Attempt 10 took positives from exact metadata matches (same user, same target
item).  Here both relations are replaced by soft collaborative neighborhoods
from a frozen SASRec trained on the same split:

* sequence relation: candidates whose SASRec history embedding is close to the
  anchor's (similar behavior sequences, possibly from different users);
* item relation: candidates whose target item's SASRec item embedding is close
  to the anchor's target (the same item has similarity 1).

The SASRec cosine similarity of each positive weights its InfoNCE term, so a
closer neighbor matters more than a marginal one.  The R3 encoder, latent,
decoder, and inference path are unchanged; SASRec is used only to select and
weight training pairs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast

from collaborative_contrastive_attention import CollaborativeContrastiveLatentModel
from layers import LatentModel


class SASRecCollaborativeLatentModel(LatentModel):
    # Shared plain-R3 helpers: thought slot lookup, per-example quality, and
    # the history-plus-latent recommendation pass.
    _thought_positions = CollaborativeContrastiveLatentModel._thought_positions
    _loss_and_quality = staticmethod(CollaborativeContrastiveLatentModel._loss_and_quality)
    _run_second_pass = CollaborativeContrastiveLatentModel._run_second_pass
    _sample = staticmethod(CollaborativeContrastiveLatentModel._sample)

    def __init__(self, config):
        super().__init__(config)
        settings = getattr(config, "sasrec_collaborative", {}) or {}
        hidden = config.hidden_size
        projection_dim = int(settings.get("projection_dim", 256))
        sasrec_dim = int(settings.get("sasrec_dim", 128))
        num_sasrec_items = int(settings.get("num_sasrec_items", 0))
        self.bank_size = int(settings.get("bank_size", 10_000))

        self.sequence_projection = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, projection_dim, bias=False)
        )
        self.item_projection = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, projection_dim, bias=False)
        )

        # Frozen, normalized SASRec item table; filled by the trainer.
        self.register_buffer(
            "sasrec_item_table", torch.zeros(num_sasrec_items, sasrec_dim), persistent=False
        )
        # Run-local FIFO of recent high-quality latents and their SASRec keys.
        for name, shape, dtype in (
            ("bank_r", (self.bank_size, hidden), torch.float16),
            ("bank_sequence", (self.bank_size, sasrec_dim), torch.float16),
            ("bank_item", (self.bank_size,), torch.int64),
            ("bank_user", (self.bank_size,), torch.int64),
            ("bank_sample", (self.bank_size,), torch.int64),
        ):
            self.register_buffer(name, torch.zeros(shape, dtype=dtype), persistent=False)
        self.register_buffer("bank_pointer", torch.zeros((), dtype=torch.int64), persistent=False)
        self.register_buffer("bank_count", torch.zeros((), dtype=torch.int64), persistent=False)

        self.temperature = 0.1
        self.sequence_contrastive_weight = 0.1
        self.item_contrastive_weight = 0.1
        self.max_sequence_positives = 16
        self.max_item_positives = 16
        self.num_negatives = 256
        self.min_positive_similarity = 0.3
        self.max_negative_similarity = 0.2
        self.similarity_weight_power = 1.0
        self.exclude_same_user_positives = False
        self.positive_min_token_accuracy = 1.0
        self.positive_max_nll = float("inf")
        self.grad_accum_steps = 1

        self.last_ce_loss = None
        self.last_sequence_contrastive_loss = None
        self.last_item_contrastive_loss = None
        self.last_sequence_positive_rate = None
        self.last_item_positive_rate = None
        self.last_sequence_positive_similarity = None
        self.last_item_positive_similarity = None
        self.last_queue_accept_rate = None
        self.last_queue_count = None

    _BANK_FIELDS = ("bank_r", "bank_sequence", "bank_item", "bank_user", "bank_sample")

    def _bank_view(self):
        """Return valid queue entries (oldest to newest) as a field dict."""
        count = int(self.bank_count.item())
        if count < self.bank_size:
            order = torch.arange(count, device=self.bank_r.device)
        else:
            pointer = int(self.bank_pointer.item())
            order = torch.cat([
                torch.arange(pointer, self.bank_size, device=self.bank_r.device),
                torch.arange(0, pointer, device=self.bank_r.device),
            ])
        return {name: getattr(self, name)[order] for name in self._BANK_FIELDS}

    @torch.no_grad()
    def _push_bank(self, r, sequence, item, user_hash, sample_hash, quality):
        keep = torch.nonzero(quality, as_tuple=False).flatten()
        if keep.numel() == 0:
            return
        values = dict(zip(self._BANK_FIELDS, (r, sequence, item, user_hash, sample_hash)))
        values = {
            name: value.detach()[keep][-self.bank_size:].to(
                device=getattr(self, name).device, dtype=getattr(self, name).dtype
            )
            for name, value in values.items()
        }
        size = values["bank_r"].size(0)
        pointer = int(self.bank_pointer.item())
        first = min(size, self.bank_size - pointer)
        for name, value in values.items():
            buffer = getattr(self, name)
            buffer[pointer:pointer + first].copy_(value[:first])
            if size > first:
                buffer[:size - first].copy_(value[first:])
        self.bank_pointer.fill_((pointer + size) % self.bank_size)
        self.bank_count.fill_(min(self.bank_size, int(self.bank_count.item()) + size))

    def _item_similarity(self, anchor_item, candidate_item):
        table = self.sasrec_item_table
        anchor = table[anchor_item.clamp(min=0)]
        candidate = table[candidate_item.clamp(min=0)]
        similarity = anchor.float() @ candidate.float().T
        unknown = (anchor_item[:, None] < 0) | (candidate_item[None, :] < 0)
        return similarity.masked_fill(unknown, float("nan"))

    def _relation_loss(self, anchors, candidates, similarity, eligible, negatives, max_positives):
        """Similarity-weighted InfoNCE for one relation.

        Each selected positive ``p`` is contrasted against the anchor's shared
        negatives, and its term is weighted by ``sim_p ** power``.  Weights are
        normalized over the whole micro-batch, so both a closer neighbor of
        one anchor and an anchor with closer neighbors count for more.
        """
        zero = anchors.sum() * 0.0
        terms, weights, positive_sims = [], [], []
        has_positive = torch.zeros(anchors.size(0), device=anchors.device)
        for row in range(anchors.size(0)):
            pos_idx = torch.nonzero(eligible[row], as_tuple=False).flatten()
            neg_idx = self._sample(torch.nonzero(negatives[row], as_tuple=False).flatten(), self.num_negatives)
            if pos_idx.numel() == 0 or neg_idx.numel() == 0:
                continue
            if pos_idx.numel() > max_positives:
                pos_idx = pos_idx[similarity[row, pos_idx].topk(max_positives).indices]
            has_positive[row] = 1.0
            pos_scores = anchors[row] @ candidates[pos_idx].T / self.temperature
            neg_scores = anchors[row] @ candidates[neg_idx].T / self.temperature
            # -log exp(s_p) / (exp(s_p) + sum_n exp(s_n)) per positive; the
            # positives do not compete with one another.
            denominator = torch.logaddexp(pos_scores, torch.logsumexp(neg_scores, dim=0))
            terms.append(denominator - pos_scores)
            sims = similarity[row, pos_idx].float()
            weights.append(sims.clamp(min=0.0) ** self.similarity_weight_power)
            positive_sims.append(sims)
        if not terms:
            return zero, has_positive.mean(), zero.detach()
        terms, weights = torch.cat(terms), torch.cat(weights)
        loss = (weights * terms).sum() / weights.sum().clamp(min=1e-8)
        return loss, has_positive.mean(), torch.cat(positive_sims).mean().detach()

    def _collaborative_losses(self, r, sequence, item, user_hash, sample_hash, quality):
        bank = self._bank_view()
        current = torch.nonzero(quality, as_tuple=False).flatten()
        candidate_r = torch.cat([bank["bank_r"].to(r.device), r.detach()[current]])
        candidate_sequence = torch.cat([bank["bank_sequence"].to(r.device), sequence[current]])
        candidate_item = torch.cat([bank["bank_item"].to(r.device), item[current]])
        candidate_user = torch.cat([bank["bank_user"].to(r.device), user_hash[current]])
        candidate_sample = torch.cat([bank["bank_sample"].to(r.device), sample_hash[current]])
        if candidate_r.size(0) == 0:
            zero = r.sum() * 0.0
            return zero, zero, zero.detach(), zero.detach(), zero.detach(), zero.detach()

        sequence_sim = sequence.float() @ candidate_sequence.float().T
        item_sim = self._item_similarity(item, candidate_item)
        not_self = sample_hash[:, None] != candidate_sample[None, :]
        allowed = not_self
        if self.exclude_same_user_positives:
            allowed = allowed & (user_hash[:, None] != candidate_user[None, :])
        # NaN (unknown item) compares False, so it is neither positive nor negative.
        sequence_eligible = allowed & (sequence_sim >= self.min_positive_similarity)
        item_eligible = allowed & (item_sim >= self.min_positive_similarity)
        # A negative must be dissimilar under both relations, so neither loss
        # pushes away a pair that the other loss treats as related.
        negatives = (
            not_self
            & (sequence_sim < self.max_negative_similarity)
            & (item_sim < self.max_negative_similarity)
        )

        results = []
        for projection, similarity, eligible, max_positives in (
            (self.sequence_projection, sequence_sim, sequence_eligible, self.max_sequence_positives),
            (self.item_projection, item_sim, item_eligible, self.max_item_positives),
        ):
            dtype = next(projection.parameters()).dtype
            anchors = F.normalize(projection(r.to(dtype)).float(), dim=-1)
            candidates = F.normalize(projection(candidate_r.to(dtype)).float(), dim=-1)
            results.append(self._relation_loss(
                anchors, candidates, similarity, eligible, negatives, max_positives
            ))
        (sequence_loss, sequence_rate, sequence_pos_sim), (item_loss, item_rate, item_pos_sim) = results
        return sequence_loss, item_loss, sequence_rate, item_rate, sequence_pos_sim, item_pos_sim

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        num_logits_to_keep=0,
        user_hash=None,
        target_hash=None,
        sample_hash=None,
        sasrec_sequence=None,
        sasrec_item=None,
        **kwargs,
    ):
        is_training_prefill = (
            self.training and labels is not None and input_ids is not None
            and inputs_embeds is None and attention_mask is not None
            and input_ids.size() == attention_mask.size()
            and sample_hash is not None and sasrec_sequence is not None
            and sasrec_item is not None
        )
        if not is_training_prefill:
            return super().forward(
                input_ids=input_ids, attention_mask=attention_mask,
                position_ids=position_ids, past_key_values=past_key_values,
                inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict, cache_position=cache_position,
                num_logits_to_keep=num_logits_to_keep, **kwargs,
            )

        embeds = self.generate_embs(input_ids, attention_mask)
        thought_idx = self._thought_positions(input_ids)
        rows = torch.arange(input_ids.size(0), device=input_ids.device)
        r = embeds[rows, thought_idx].float()
        out = self._run_second_pass(
            embeds, attention_mask, position_ids, cache_position, num_logits_to_keep
        )
        ce_loss, example_nll, token_accuracy = self._loss_and_quality(out.logits, labels)
        quality = (
            (token_accuracy >= self.positive_min_token_accuracy)
            & (example_nll <= self.positive_max_nll)
        )

        sequence = sasrec_sequence.float()
        (
            sequence_loss, item_loss, sequence_rate, item_rate, sequence_pos_sim, item_pos_sim,
        ) = self._collaborative_losses(r, sequence, sasrec_item, user_hash, sample_hash, quality)
        self._push_bank(r, sequence, sasrec_item, user_hash, sample_hash, quality)

        self.last_ce_loss = ce_loss.detach()
        self.last_sequence_contrastive_loss = sequence_loss.detach()
        self.last_item_contrastive_loss = item_loss.detach()
        self.last_sequence_positive_rate = sequence_rate
        self.last_item_positive_rate = item_rate
        self.last_sequence_positive_similarity = sequence_pos_sim
        self.last_item_positive_similarity = item_pos_sim
        self.last_queue_accept_rate = quality.float().mean().detach()
        self.last_queue_count = self.bank_count.detach().clone()

        total_loss = (
            ce_loss
            + self.sequence_contrastive_weight * sequence_loss
            + self.item_contrastive_weight * item_loss
        ) / self.grad_accum_steps
        return CausalLMOutputWithPast(
            loss=total_loss,
            logits=out.logits,
            past_key_values=out.past_key_values,
            hidden_states=out.hidden_states,
            attentions=out.attentions,
        )
