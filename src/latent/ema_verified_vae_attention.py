"""Attempt 7c: 7b decomposition with an EMA latent-consumer verifier.

The deployed student always trains on the ordinary ``history + latent`` path.
The 7b aggregate KL and mu-consistency anchor remain unchanged.  Instead of
the attention curriculum, a parameter-frozen EMA copy of the decoder receives
``mu_ema + z_i`` under a reasoning-only mask and must predict the target.  Its
gradient flows through the teacher computation into ``z_i`` but never updates
the EMA teacher directly.

The EMA teacher is a training-only module.  It is deliberately excluded from
``state_dict`` so normal checkpoints retain the same size and load through the
repository's standard ``LatentModel`` ranking evaluator.  On resume, it is
reinitialized from the resumed student checkpoint before training continues.
"""

import copy

import torch
import torch.nn.functional as F
from transformers import Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from vae_decomposed_attention import VAEDecomposedLatentModel


class EMAVerifiedVAEDecomposedLatentModel(VAEDecomposedLatentModel):
    def __init__(self, config):
        super().__init__(config)
        self.verify_weight = 0.05
        self.teacher_ema_momentum = 0.999
        self.ema_teacher = None
        self.ema_teacher_updates = 0
        self.last_verify_loss = None

    def initialize_ema_teacher(self) -> None:
        """Create the training-only teacher after tokenizer resize/loading."""
        if self.ema_teacher is not None:
            return
        teacher = copy.deepcopy(self)
        teacher.ema_teacher = None
        teacher.verify_weight = 0.0
        teacher.requires_grad_(False)
        teacher.eval()
        teacher.config.use_cache = False
        self.ema_teacher = teacher

    @torch.no_grad()
    def update_ema_teacher(self) -> None:
        """EMA-update only the decoder parameters used by verification."""
        if self.ema_teacher is None:
            raise RuntimeError("EMA teacher has not been initialized")
        student_parameters = {
            name: parameter
            for name, parameter in self.named_parameters()
            if not name.startswith("ema_teacher.")
        }
        momentum = self.teacher_ema_momentum
        for name, teacher_parameter in self.ema_teacher.named_parameters():
            if not (name.startswith("model.") or name.startswith("lm_head.")):
                continue
            student_parameter = student_parameters[name]
            teacher_parameter.mul_(momentum).add_(
                student_parameter.detach(), alpha=1.0 - momentum
            )
        self.ema_teacher_updates += 1

    def train(self, mode: bool = True):
        result = super().train(mode)
        if self.ema_teacher is not None:
            # Trainer repeatedly calls model.train(); the verifier must remain
            # deterministic and parameter-frozen throughout those transitions.
            self.ema_teacher.eval()
        return result

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        # Module.state_dict recursively invokes this override on the copied
        # teacher too, sometimes with a non-empty prefix, so handle both forms.
        for key in list(state):
            if key.startswith("ema_teacher.") or ".ema_teacher." in key:
                state.pop(key)
        return state

    @staticmethod
    def _target_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = labels[:, 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    def verification_loss(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        labels: torch.LongTensor,
        thought_idx: torch.LongTensor,
        z: torch.Tensor,
        position_ids=None,
        cache_position=None,
        num_logits_to_keep: int = 0,
    ) -> torch.Tensor:
        """History-blind target CE under the frozen EMA decoder.

        ``mu_ema`` supplies a shared, valid-scale scaffold.  Because it is
        detached and identical for every sample, all sample-specific target
        information reaching this branch must pass through ``z``.
        """
        if self.ema_teacher is None:
            raise RuntimeError("call initialize_ema_teacher() before training")

        teacher_embeds = self.ema_teacher.model.embed_tokens(input_ids).clone()
        verify_latent = self.attention.mu_ema.detach().unsqueeze(0) + z
        batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
        teacher_embeds[batch_idx, thought_idx] = verify_latent.to(teacher_embeds.dtype)
        teacher_attention = self._scheduled_attention_bias(
            attention_mask, thought_idx, teacher_embeds.dtype, "reasoning_only"
        )
        teacher_out = Qwen2ForCausalLM.forward(
            self.ema_teacher,
            inputs_embeds=teacher_embeds,
            attention_mask=teacher_attention,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
        )
        return self._target_ce(teacher_out.logits, labels)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
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
        num_logits_to_keep: int = 0,
        **kwargs,
    ):
        is_prefill = (
            input_ids is not None
            and inputs_embeds is None
            and attention_mask is not None
            and input_ids.size() == attention_mask.size()
        )
        if not (is_prefill and self.training and labels is not None):
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                num_logits_to_keep=num_logits_to_keep,
                **kwargs,
            )

        thought_idx = self._thought_positions(input_ids)
        student_embeds = self.generate_embs(input_ids, attention_mask)
        mu, z = self.attention.last_mu, self.attention.last_z

        # Unlike 7b, the deployed student path is never trained under an
        # artificial history-only/reasoning-only mode.
        self.last_schedule_mode = "both"
        student_out = self._run_second_pass(
            student_embeds,
            attention_mask,
            position_ids,
            cache_position,
            num_logits_to_keep,
        )
        ce_loss = self._target_ce(student_out.logits, labels)

        kl_loss = self._queued_kl(z)
        if self.mu_consistency_weight > 0 and bool(self.attention.mu_ema_initialized):
            mu_consistency = (mu - self.attention.mu_ema.unsqueeze(0)).pow(2).mean()
        else:
            mu_consistency = ce_loss.new_zeros(())

        if self.verify_weight > 0:
            verify_loss = self.verification_loss(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                thought_idx=thought_idx,
                z=z,
                position_ids=position_ids,
                cache_position=cache_position,
                num_logits_to_keep=num_logits_to_keep,
            )
        else:
            verify_loss = ce_loss.new_zeros(())

        self.last_ce_loss = ce_loss.detach()
        self.last_deployment_ce_loss = ce_loss.detach()
        self.last_kl_loss = kl_loss.detach()
        self.last_mu_consistency_loss = mu_consistency.detach()
        self.last_verify_loss = verify_loss.detach()

        total_loss = (
            ce_loss
            + self.kl_weight * kl_loss
            + self.mu_consistency_weight * mu_consistency
            + self.verify_weight * verify_loss
        ) / self.grad_accum_steps

        return CausalLMOutputWithPast(
            loss=total_loss,
            logits=student_out.logits,
            past_key_values=student_out.past_key_values,
            hidden_states=student_out.hidden_states,
            attentions=student_out.attentions,
        )
