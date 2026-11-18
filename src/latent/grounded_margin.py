"""Margin + WIDER + behaviour-grounded donors (collaborative reasoning) for R3.

Combines three pieces that already exist separately:

* ``grounded_reasoning.py`` (L5/L10): r0 = R3 pooling over the user's own history;
  r1 = r0 + s * ||r0|| * dir(sum_j a_j V(r0_j, e(y_j))) over time-causal donors j
  (other users' r0 and their observed next item).  r1 goes into the thought slot.
  Auxiliary item-grounding CE and behaviour-supervised retrieval KL.
* ``margin_loss_attention.py`` (Attempt 2): -logP(y|x,r1) + m <= -logP(y|x,r1_other),
  r1_other = another in-batch row's grounded thought, detached.
* ``wider_reg.py``: cross-user WIDER penalty on the slot vector (r1; stored by
  LatentModel.generate_embs).

Motivation: margin + WIDER (Toys 356322) made the decoder read r's content
(shuffle -0.073 HR@10) at R3-level ranking, but r = f(x) carries nothing the
history lacks.  Donor outcomes do (H1), and the decoder now reads the slot.

All three terms and CE are summed, then divided by grad_accum_steps once (HF 4.48
does not divide; see feedback-hf-grad-accum-aux-loss).  Inference is the grounded
LatentModel path (eval sets query metadata; GROUNDING_MODE selects donor controls).
"""

import torch
import torch.nn.functional as F
from transformers import Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from grounded_reasoning import GroundedReasoningLatentModel
from margin_loss_attention import MarginLatentModel


class MarginGroundedLatentModel(GroundedReasoningLatentModel):
    def __init__(self, config):
        super().__init__(config)
        self.margin = 0.2
        self.margin_weight = 1.0

    _per_example_nll = staticmethod(MarginLatentModel._per_example_nll)
    _thought_positions = MarginLatentModel._thought_positions

    def keep_new_modules_fp32(self):
        """bf16 froze the evidence scale / LayerNorms in L4-L6; keep the new modules in fp32."""
        layer = self.attention
        for module in (layer.query, layer.key, layer.value, layer.item_head):
            module.float()
        layer.evidence_scale.data = layer.evidence_scale.data.float()

    def _decode(self, embeds, attention_mask):
        return Qwen2ForCausalLM.forward(
            self, inputs_embeds=embeds, attention_mask=attention_mask,
            use_cache=False, return_dict=True,
        )

    def forward(self, input_ids=None, attention_mask=None, labels=None, query_time=None,
                query_user=None, target_item=None, outcome_time=None, **kwargs):
        training_prefill = (
            self.training and labels is not None and input_ids is not None
            and attention_mask is not None and kwargs.get("inputs_embeds") is None
        )
        if not training_prefill:
            return super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                                   query_time=query_time, query_user=query_user,
                                   target_item=target_item, outcome_time=outcome_time, **kwargs)
        layer = self.attention
        layer.query_time, layer.query_user, layer.target_item = query_time, query_user, target_item
        try:
            embeds = self.generate_embs(input_ids, attention_mask)   # r1 in the slot; WIDER stores r1
            thought_idx = self._thought_positions(input_ids)
            rows = torch.arange(input_ids.size(0), device=input_ids.device)

            pos_out = self._decode(embeds, attention_mask)
            pos_nll = self._per_example_nll(pos_out.logits, labels)
            ce = pos_nll.mean()

            if input_ids.size(0) > 1 and self.margin_weight > 0:
                perm = torch.roll(rows, shifts=1)
                neg_embeds = embeds.clone()
                neg_embeds[rows, thought_idx] = embeds[perm, thought_idx[perm]].detach()
                neg_nll = self._per_example_nll(self._decode(neg_embeds, attention_mask).logits, labels)
                margin_loss = F.relu(self.margin + pos_nll - neg_nll).mean()
            else:
                margin_loss = ce.new_zeros(())

            loss = ce + self.margin_weight * margin_loss
            aux = layer.aux_loss or {}
            if aux:
                loss = loss + self.item_weight * aux["item"] + self.retrieval_weight * aux["retrieval"]
            wider = None
            if getattr(self.config, "latent_wider", None):
                from wider_reg import wider_state
                state = wider_state(self)
                wider = state.penalty() if state is not None else None
                if wider is not None:
                    loss = loss + state.weight * wider
            layer.push(layer.last_raw, target_item, query_user, outcome_time)
            self.last_losses = {
                "ce_loss": ce.detach(), "margin_loss": margin_loss.detach(),
                **({"wider_loss": wider.detach()} if wider is not None else {}),
                **{f"{k}_loss": v.detach() for k, v in aux.items()}, **layer.stats,
            }
            return CausalLMOutputWithPast(loss=loss / self.grad_accum_steps, logits=pos_out.logits)
        finally:
            layer.query_time = layer.query_user = layer.target_item = None

    @torch.no_grad()
    def build_slot_bank(self, loader, device, limit):
        """Grounded slot vectors r1 (knn donors) of the first `limit` train rows, for mean / shuffle_user."""
        layer = self.attention
        self.eval()
        vectors, users = [], []
        for batch in loader:
            layer.query_time = batch["query_time"].to(device)
            layer.query_user = batch["query_user"].to(device)
            ids = batch["input_ids"].to(device)
            embeds = self.generate_embs(ids, batch["attention_mask"].to(device))
            vectors.append(embeds[torch.arange(ids.size(0), device=device), self._thought_positions(ids)].float().cpu())
            users.append(batch["query_user"].cpu())
            if sum(len(v) for v in vectors) >= limit:
                break
        layer.query_time = layer.query_user = None
        return torch.cat(vectors)[:limit], torch.cat(users)[:limit]
