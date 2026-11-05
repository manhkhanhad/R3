import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Optional, Tuple
from transformers.modeling_outputs import CausalLMOutputWithPast

from transformers import Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb


class VanillaRoPE(nn.Module):
    def __init__(self, hidden_size, device=None):
        super(VanillaRoPE, self).__init__()
        base = 1000
        dim = hidden_size
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class SelfAttentionLayer(torch.nn.Module):
    def __init__(self, hidden_size, end_k = -1):
        super().__init__()
        self.hidden_size = hidden_size
        # need the end_k states to calculate
        # if set to -1, all of hidden_states needed
        self.end_k = end_k

        # 定义Q, K, V的线性变换层
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.scale_score = 1.0/(self.hidden_size ** 0.5)
        self.rope = VanillaRoPE(hidden_size)

    @staticmethod
    def mask_to_weights(attention_mask, thought_id_idx, end_k=-1):
        if thought_id_idx is not None and attention_mask.size(0) != thought_id_idx.size(0):
            raise ValueError("attention_mask must have the same size as thought_id_idx")

        if end_k != -1:
            # 把thought_id前end_k个位置的内容考虑进去
            # 这里的代码有一点问题：也就是我们认为end_k不会长过前面的句子长度
            # 否则会把padding的部分置为1
            for i in range(attention_mask.size(0)):
                idx = thought_id_idx[i].item()
                start_idx = max(0, idx - end_k)
                attention_mask[i].zero_()
                attention_mask[i, start_idx:idx] = 1
        else:
            # 把thought_id前所有的内容都考虑进去 (要屏蔽padding和thought_id后面的部分)
            # 这里想一下
            # 在训练的时候，padding在右侧
            # 在推理的时候，padding在左侧
            # 而在推理的时候，我们传进来的attention_mask实际上是正确的，所以不需要全部置0
            # 训练时候，代码可以改为: attention_mask[i, idx:] = 0
            # 推理时候, idx == seq_len, 也是可以和训练时候一样去改，所以修改
            for i in range(attention_mask.size(0)):
                idx = thought_id_idx[i].item()
                attention_mask[i, idx:] = 0

        # Convert mask to weights: 1 -> 0, 0 -> -inf
        attention_weight = torch.zeros_like(attention_mask, dtype=torch.float32)
        attention_weight = attention_weight.masked_fill(~attention_mask.bool(), float('-inf'))
        return attention_weight

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor,
                thought_id_idx) -> torch.Tensor:
        # hidden_states = hidden_states.to(torch.float32)
        attention_mask = SelfAttentionLayer.mask_to_weights(attention_mask.clone(), thought_id_idx, self.end_k)

        cos, sin = self.rope(
            hidden_states,
            position_ids=torch.arange(0, hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
        )

        batch_size, seq_len, _ = hidden_states.shape

        # 生成Q, K, V
        Q = self.query(hidden_states)  # [batch_size, seq_len, embed_dim]
        K = self.key(hidden_states)
        V = self.value(hidden_states)

        # 加入位置编码
        Q, K = apply_rotary_pos_emb(Q, K, cos.squeeze(0), sin.squeeze(0), unsqueeze_dim=0)

        # 确定目标位置索引
        if thought_id_idx is None:
            # 取每个样本的最后一个位置（seq_len - 1）
            indices = torch.full((batch_size,), seq_len - 1, device=hidden_states.device)
        else:
            indices = thought_id_idx - 1
            if (indices < 0).any() or (indices >= seq_len).any():
                raise ValueError("thought_id_idx-1 exceeds valid sequence indices")

        # 提取目标位置的Q向量 [batch_size, 1, embed_dim]
        Q_selected = Q[torch.arange(batch_size), indices].unsqueeze(1)

        # 计算注意力分数 [batch_size, 1, seq_len]
        attn_scores = torch.matmul(Q_selected, K.transpose(-2, -1)) * self.scale_score

        # 应用attention_mask（形状需匹配为[batch_size, 1, seq_len]）
        attn_scores += attention_mask.unsqueeze(1)  # 广播至[batch_size, 1, seq_len]

        # 计算注意力权重与输出
        attn_weights = F.softmax(attn_scores, dim=-1)
        output_selected = torch.matmul(attn_weights, V)  # [batch_size, 1, embed_dim]
        output = output_selected.squeeze(1) # [batch_size, embed_dim]
        return output


class Noise(nn.Module):
    def __init__(self, hidden_size):
        super(Noise, self).__init__()
        # 这里的初始化就放在from_pretrained之后去做
        # 对于 log_var  model.mlp.log_var = torch.nn.Parameter(torch.tensor([float('-inf')] * model.config.hidden_size))
        # 对于 mu       nn.init.eyes
        self.mu = nn.Linear(hidden_size, hidden_size, bias=False)
        self.log_var = torch.nn.Parameter(torch.tensor([float('-10')] * hidden_size))

    def forward(self, x):
        x = self.mu(x)
        std = torch.exp(self.log_var / 2)
        return x + torch.randn_like(x).mul(std)


class LatentModel(Qwen2ForCausalLM):

    def __init__(self, config):
        super().__init__(config)
        # self.mlp = Noise(config.hidden_size)
        thoughts = getattr(config, "latent_continuous_thoughts", None)
        if thoughts:
            from continuous_thoughts import ContinuousThoughts
            self.thoughts = ContinuousThoughts(config.hidden_size, thoughts)
        steer = getattr(config, "latent_steered_reading", None)
        if steer:
            if not thoughts or getattr(config, "latent_bottleneck", None):
                raise ValueError("steered reading needs continuous thoughts and no latent bottleneck")
            from steered_reading import SteeredReading
            self.steer = SteeredReading(config.hidden_size, steer)
        decomposition = getattr(config, "latent_decomposition", None)
        if decomposition:
            from decomposed_attention import build_decomposed_attention
            self.attention = build_decomposed_attention(config.hidden_size, decomposition)
        else:
            self.attention = SelfAttentionLayer(config.hidden_size)

        residual = getattr(config, "latent_residual_correction", None)
        if residual:
            from residual_correction_attention import wrap_with_residual_correction
            self.attention = wrap_with_residual_correction(
                self.attention, config.hidden_size, residual
            )

        neighbor = getattr(config, "latent_neighbor_refinement", None)
        if neighbor:
            if neighbor.get("projected", False):
                from projected_neighbor_refinement import wrap_with_projected_neighbor_refinement
                self.attention = wrap_with_projected_neighbor_refinement(
                    self.attention, config.hidden_size, neighbor
                )
            else:
                from neighbor_refinement_attention import wrap_with_neighbor_refinement
                self.attention = wrap_with_neighbor_refinement(
                    self.attention, config.hidden_size, neighbor
                )

        grounded = getattr(config, "latent_grounded_reasoning", None)
        if grounded:
            from grounded_reasoning import GroundedReasoningLayer
            self.attention = GroundedReasoningLayer(self.attention, config.hidden_size, grounded)

        intent = getattr(config, "latent_intent_reasoning", None)
        if intent:
            from intent_reasoning import IntentReasoningLayer
            self.attention = IntentReasoningLayer(self.attention, config.hidden_size, intent)

        self.last_soft_vocab_entropy = None
        self.last_soft_vocab_max_probability = None
        self.last_soft_vocab_raw_norm = None
        self.last_soft_vocab_norm = None

    def project_reasoning_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Map ``latent`` to input-embedding space when enabled in config.

        Keeping this selection in the base checkpoint class ensures training,
        ordinary ranking evaluation, and generation all use the same latent.
        With no config flag, the baseline path is exactly the identity.
        """
        spec = getattr(self.config, "latent_soft_vocabulary", None)
        if not spec:
            return latent

        from soft_vocabulary import project_to_soft_vocabulary

        soft_latent, diagnostics = project_to_soft_vocabulary(self, latent, spec)
        self.last_soft_vocab_entropy = diagnostics["entropy"]
        self.last_soft_vocab_max_probability = diagnostics["max_probability"]
        self.last_soft_vocab_raw_norm = diagnostics["raw_norm"]
        self.last_soft_vocab_norm = diagnostics["soft_norm"]
        return soft_latent

    def observe_reasoning_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Optional training hook; the baseline leaves the latent unchanged."""
        return latent

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        target = model[0] if isinstance(model, tuple) else model
        if getattr(target.config, "latent_continuous_thoughts", None):
            from continuous_thoughts import load_fp32_thought_params
            load_fp32_thought_params(target, str(pretrained_model_name_or_path))
        if getattr(target.config, "latent_steered_reading", None):
            from steered_reading import load_fp32_steer_params
            load_fp32_steer_params(target, str(pretrained_model_name_or_path))
        if getattr(target.config, "latent_attention_share", None):
            from attention_share import enable_attention_share
            enable_attention_share(target)
        return model

    def pooling_mask(self, input_ids, attention_mask):
        """Mask for the r pooling: unchanged, or history title tokens only
        (config.latent_pool_history, history_pooling.py)."""
        spec = getattr(self.config, "latent_pool_history", None)
        if not spec:
            return attention_mask
        from history_pooling import history_title_mask
        thought_id = self.model.embed_tokens.num_embeddings - 1
        return history_title_mask(input_ids, attention_mask, spec, thought_id).to(attention_mask.dtype)

    def generate_embs(self, input_ids, attention_mask):
        if getattr(self.config, "latent_continuous_thoughts", None):
            from continuous_thoughts import first_thought_positions
            thought_token_id = self.model.embed_tokens.num_embeddings - 1
            first = first_thought_positions(input_ids, thought_token_id, self.thoughts.k)
            thoughts = self.thoughts(self.model, input_ids, attention_mask, first)
            input_embs = self.model.embed_tokens(input_ids)
            rows = torch.arange(input_ids.size(0), device=input_ids.device)[:, None]
            cols = first[:, None] + torch.arange(self.thoughts.k, device=input_ids.device)[None]
            input_embs = input_embs.index_put((rows, cols), thoughts.to(input_embs.dtype))
            if getattr(self.config, "latent_steered_reading", None):
                index, sim = self.steer.batch_tensors(input_ids.size(0), input_ids.device)
                self.steer.block = None
                if index is not None and self.thoughts.source == "constant":
                    # Fixed-rule selection (recent / all) needs no latent or first pass.
                    self.steer.select_by_rule(index)
                elif index is not None:
                    self.steer(thoughts.mean(1), self.thoughts.last_prefix_hidden,
                               self.thoughts.last_prefix_positions, index, sim)
            return input_embs
        output_mid = super().forward(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # generate embeddings
        thought_ids = self.model.embed_tokens.num_embeddings - 1  # thought id flag
        where_thought_ids = torch.nonzero(input_ids == thought_ids)
        hidden_states = output_mid['hidden_states']
        input_embs = self.model.embed_tokens(input_ids)

        reasoning_latent = self.attention(
            hidden_states=hidden_states[-1], attention_mask=self.pooling_mask(input_ids, attention_mask),
            thought_id_idx=where_thought_ids[:, 1],
        )
        reasoning_latent = self.observe_reasoning_latent(reasoning_latent)
        reasoning_latent = self.project_reasoning_latent(reasoning_latent)
        input_embs[where_thought_ids[:, 0], where_thought_ids[:, 1]] = reasoning_latent.to(
            input_embs.dtype
        )

        return input_embs


    def _latent_bottleneck_inputs(self, spec, prompt_ids, attention_mask, input_ids, inputs_embeds):
        """Swap the 2D padding mask for the latent-only 4D mask (latent_bottleneck.py).

        ``prompt_ids`` is the full prompt (prefill, or an explicit
        ``latent_bottleneck_ids`` for inputs_embeds-only scoring); ``None`` means a
        KV-cache decode step, which reuses the block cached at prefill.
        """
        from latent_bottleneck import prompt_block, bottleneck_mask_and_positions

        thought_token_id = self.model.embed_tokens.num_embeddings - 1
        if prompt_ids is not None:
            thoughts = getattr(self.config, "latent_continuous_thoughts", None)
            block, thought_pos = prompt_block(
                prompt_ids, attention_mask, thought_token_id, spec["visible_prefix_ids"],
                n_thoughts=int(thoughts["k"]) if thoughts else 1,
            )
            self._bottleneck_block, self._bottleneck_thought_pos = block, thought_pos
        else:
            block = getattr(self, "_bottleneck_block", None)
            thought_pos = getattr(self, "_bottleneck_thought_pos", None)
            if block is None or block.size(0) != attention_mask.size(0):
                raise RuntimeError(
                    "latent bottleneck: decode step without a matching prefill "
                    "(pass latent_bottleneck_ids for inputs_embeds-only calls)"
                )
        query_len = (input_ids if input_ids is not None else inputs_embeds).size(1)
        dtype = inputs_embeds.dtype if inputs_embeds is not None else self.model.embed_tokens.weight.dtype
        gist_k = 0
        if spec.get("gist"):
            thoughts = getattr(self.config, "latent_continuous_thoughts", None)
            gist_k = int(thoughts["k"]) if thoughts else 1
        return bottleneck_mask_and_positions(attention_mask, block, thought_pos, query_len, dtype,
                                             gist_k=gist_k, delete_latents=bool(spec.get("delete_latents")),
                                             history_bias=spec.get("history_bias"))

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        bottleneck_ids = kwargs.pop("latent_bottleneck_ids", None)
        item_index = kwargs.pop("item_index", None)
        item_sim = kwargs.pop("item_sim", None)
        steer = getattr(self.config, "latent_steered_reading", None)
        is_prefill = (
            input_ids is not None and inputs_embeds is None
            and attention_mask is not None and input_ids.size() == attention_mask.size()
        )
        plain_attention_mask = attention_mask
        bottleneck = getattr(self.config, "latent_bottleneck", None)
        if bottleneck and attention_mask is not None and attention_mask.dim() == 2:
            attention_mask, position_ids = self._latent_bottleneck_inputs(
                bottleneck, input_ids if is_prefill else bottleneck_ids,
                attention_mask, input_ids, inputs_embeds,
            )
        if is_prefill and getattr(self, "_latent_share", None) is not None:
            from attention_share import set_share_positions
            set_share_positions(self, input_ids, self.model.embed_tokens.num_embeddings - 1)
        if is_prefill:
            if steer and item_index is not None:
                self.steer.item_index, self.steer.item_sim = item_index, item_sim
            inputs_embeds = self.generate_embs(input_ids, plain_attention_mask)
            input_ids = None
        if steer and plain_attention_mask is not None and plain_attention_mask.dim() == 2:
            # Prefill selects the items (generate_embs); decoding steps reuse the block.
            if self.steer.block is None:
                raise RuntimeError("steered reading: no item selection for this batch (item_index not set)")
            from steered_reading import steered_mask_and_positions
            query_len = (inputs_embeds if inputs_embeds is not None else input_ids).size(1)
            attention_mask, position_ids = steered_mask_and_positions(
                plain_attention_mask, self.steer.block, query_len,
                inputs_embeds.dtype if inputs_embeds is not None else self.model.embed_tokens.weight.dtype)
        # elif input_ids is not None and inputs_embeds is None and input_ids.size()!=attention_mask.size():
            # 这个时候是在做生成任务，已经有了cache所以用input_ids即可
        output = super().forward(
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
        if steer and is_prefill and item_index is not None:
            self.steer.item_index = self.steer.item_sim = None  # training batch: do not leak into the next call
            aux = self.steer.last_aux
            if labels is not None and aux is not None and self.steer.aux_weight > 0 and self.training:
                output.loss = output.loss + self.steer.aux_weight * aux
        return output
