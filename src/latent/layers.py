import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Optional, Tuple
from transformers.modeling_outputs import CausalLMOutputWithPast

from transformers import Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, rotate_half

from diffusion import (
    AttentionPooling,
    DenoiseMLP,
    TimestepEmbedding,
    cosine_alpha_bar,
    reverse_process,
)


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

    def compute_kv(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor,
                   thought_id_idx) -> dict:
        """Precompute K/V, rotary tables and the mask weights over hidden_states.

        hidden_states does not depend on the latent, so this can be computed once
        and reused across the R thinking steps.
        """
        attention_weight = SelfAttentionLayer.mask_to_weights(
            attention_mask.clone(), thought_id_idx, self.end_k)

        cos, sin = self.rope(
            hidden_states,
            position_ids=torch.arange(0, hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
        )

        K = self.key(hidden_states)
        V = self.value(hidden_states)
        # rotary on K only (queries are rotated in forward); same math as
        # apply_rotary_pos_emb with unsqueeze_dim=0
        K = K * cos + rotate_half(K) * sin
        return {"K": K, "V": V, "attention_weight": attention_weight, "cos": cos, "sin": sin}

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor,
                thought_id_idx, query_states: Optional[torch.Tensor] = None,
                cached_kv: Optional[dict] = None) -> torch.Tensor:
        """query_states: optional external query [batch_size, hidden] (e.g. tau_{r-1});
        defaults to the hidden state right before the thought slot, as before.
        cached_kv: result of compute_kv, to reuse K/V across thinking steps."""
        if cached_kv is None:
            cached_kv = self.compute_kv(hidden_states, attention_mask, thought_id_idx)
        K, V = cached_kv["K"], cached_kv["V"]
        attention_weight = cached_kv["attention_weight"]
        cos, sin = cached_kv["cos"], cached_kv["sin"]

        batch_size, seq_len, _ = hidden_states.shape

        # 确定目标位置索引
        if thought_id_idx is None:
            # 取每个样本的最后一个位置（seq_len - 1）
            indices = torch.full((batch_size,), seq_len - 1, device=hidden_states.device)
        else:
            indices = thought_id_idx - 1
            if (indices < 0).any() or (indices >= seq_len).any():
                raise ValueError("thought_id_idx-1 exceeds valid sequence indices")

        if query_states is None:
            Q = self.query(hidden_states)  # [batch_size, seq_len, embed_dim]
            Q = Q * cos + rotate_half(Q) * sin
            Q_selected = Q[torch.arange(batch_size), indices].unsqueeze(1)
        else:
            Q = self.query(query_states)  # [batch_size, embed_dim]
            cos_sel = cos.squeeze(0)[indices]
            sin_sel = sin.squeeze(0)[indices]
            Q = Q * cos_sel + rotate_half(Q) * sin_sel
            Q_selected = Q.unsqueeze(1)

        # 计算注意力分数 [batch_size, 1, seq_len]
        attn_scores = torch.matmul(Q_selected, K.transpose(-2, -1)) * self.scale_score

        # 应用attention_mask（形状需匹配为[batch_size, 1, seq_len]）
        attn_scores = attn_scores + attention_weight.unsqueeze(1)  # 广播至[batch_size, 1, seq_len]

        # 计算注意力权重与输出
        attn_weights = F.softmax(attn_scores, dim=-1).to(V.dtype)
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
    """Think-then-Diffuse latent model (DiffuReason, arXiv 2602.09744).

    R autoregressive thinking tokens are computed internally from the prompt's
    last hidden states; the last one is refined through a learned reverse
    diffusion process conditioned on the pooled thoughts; only the refined
    anchor mu is written into the single <|Thought|> slot.
    """

    def __init__(self, config):
        super().__init__(config)
        # DiffuReason hyperparameters, persisted in config.json.
        # Defaults reproduce the original LatentR3 behavior exactly:
        # R=1 (slot = pooler output, no thinking loop) and T=0 (no diffusion).
        config.latent_R = getattr(config, "latent_R", 1)
        config.diff_T = getattr(config, "diff_T", 0)
        config.diff_s = getattr(config, "diff_s", 0.008)
        config.diff_alpha = getattr(config, "diff_alpha", 0.0)
        config.diff_loss = getattr(config, "diff_loss", "proj_mse")
        config.use_schedule_scaling = getattr(config, "use_schedule_scaling", True)

        d = config.hidden_size
        self.attention = SelfAttentionLayer(d)  # thinking attention (Eq. 1)
        self.think_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.cond_pool = AttentionPooling(d)
        self.denoiser = DenoiseMLP(d)
        self.time_embed = TimestepEmbedding(d)
        self.diff_proj = nn.Linear(d, d)  # loss-only head for the MSE anchor loss
        self.register_buffer(
            "alpha_bar", cosine_alpha_bar(max(config.diff_T, 1), config.diff_s), persistent=False)

        # eval-time behavior: sample eps_init (paper default) unless forced to zero
        self.eps_init_zero = False
        self._last_diff = None

    def load_item_embeddings(self, path):
        data = torch.load(path, map_location="cpu")
        emb = data["embeddings"] if isinstance(data, dict) else data
        assert emb.size(-1) == self.config.hidden_size, \
            f"item embedding dim {emb.size(-1)} != hidden {self.config.hidden_size}"
        self.register_buffer("item_emb", emb.float().to(self.device), persistent=False)

    def generate_embs(self, input_ids, attention_mask, return_diffusion=False,
                      replay_noises=None, generator=None):
        output_mid = super().forward(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # generate embeddings
        thought_ids = self.model.embed_tokens.num_embeddings - 1  # thought id flag
        where_thought_ids = torch.nonzero(input_ids == thought_ids)
        hidden_states = output_mid['hidden_states']
        input_embs = self.model.embed_tokens(input_ids)

        H = hidden_states[-1]
        thought_idx = where_thought_ids[:, 1]
        batch_size = input_ids.size(0)

        # ---- Think: tau_1 = pooler(H) (the original LatentR3 latent);
        # tau_r = MLP(Attn(tau_{r-1}, H)) for r >= 2 (Eq. 1) ----
        cached_kv = self.attention.compute_kv(H, attention_mask, thought_idx)
        tau = self.attention(hidden_states=H, attention_mask=attention_mask,
                             thought_id_idx=thought_idx, cached_kv=cached_kv)
        taus = [tau]
        for _ in range(2, self.config.latent_R + 1):
            tau = self.think_mlp(self.attention(
                hidden_states=H, attention_mask=attention_mask,
                thought_id_idx=thought_idx, query_states=tau, cached_kv=cached_kv))
            taus.append(tau)
        tau_R = taus[-1]

        # ---- Diffuse: x_T = tau_R + eps_init -> T reverse steps -> (mu, sigma) ----
        if self.config.diff_T > 0:
            # config.diff_T/diff_s may have been set after __init__ (train script);
            # refresh the schedule buffer if it no longer matches
            if self.alpha_bar.numel() != self.config.diff_T + 1:
                self.alpha_bar = cosine_alpha_bar(
                    self.config.diff_T, self.config.diff_s).to(tau_R.device)
            c = self.cond_pool(torch.stack(taus, dim=1))  # (Eq. 2)
            eps_init = None
            if replay_noises is None and self.eps_init_zero:
                eps_init = torch.zeros_like(tau_R)
            mu, sigma, used_noises = reverse_process(
                self.denoiser, self.time_embed, tau_R, c,
                T=self.config.diff_T, alpha_bar=self.alpha_bar,
                training=self.training,
                use_schedule_scaling=self.config.use_schedule_scaling,
                eps_init=eps_init, noises=replay_noises, generator=generator,
            )
        else:  # T=0: no refinement (original LatentR3 behavior)
            c = None
            mu = tau_R
            sigma = torch.full_like(tau_R, 1e-3, dtype=torch.float32)
            used_noises = []

        input_embs[where_thought_ids[:, 0], thought_idx] = mu.to(input_embs.dtype)

        self._last_diff = {
            "mu": mu, "sigma": sigma, "c": c, "tau_R": tau_R, "noises": used_noises,
        }
        if return_diffusion:
            return input_embs, self._last_diff
        return input_embs


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
        num_logits_to_keep: Optional[int] = None,
        logits_to_keep: Optional[int] = None,
        item_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if input_ids is not None and inputs_embeds is None and input_ids.size() == attention_mask.size():
            inputs_embeds = self.generate_embs(input_ids, attention_mask)
            input_ids = None
        # elif input_ids is not None and inputs_embeds is None and input_ids.size()!=attention_mask.size():
            # 这个时候是在做生成任务，已经有了cache所以用input_ids即可
        ltk = logits_to_keep if logits_to_keep is not None else (
            num_logits_to_keep if num_logits_to_keep is not None else 0)
        # transformers renamed num_logits_to_keep -> logits_to_keep (~4.49);
        # accept both here and forward whichever the installed parent expects
        parent_params = inspect.signature(Qwen2ForCausalLM.forward).parameters
        ltk_kw = "num_logits_to_keep" if "num_logits_to_keep" in parent_params else "logits_to_keep"
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
            **{ltk_kw: ltk},
            **kwargs,
        )

        # Optional MSE anchor loss (Eq. 7), off by default (diff_alpha=0)
        if (labels is not None and item_ids is not None
                and getattr(self, "item_emb", None) is not None
                and self.config.diff_alpha > 0 and self._last_diff is not None):
            mu = self._last_diff["mu"]
            v_target = self.item_emb[item_ids]  # [B, d] fp32, unit-norm
            if self.config.diff_loss == "proj_mse":
                pred = self.diff_proj(mu).float()
                l_diff = ((pred - v_target) ** 2).sum(dim=-1).mean()
            elif self.config.diff_loss == "mse":
                l_diff = ((mu.float() - v_target) ** 2).sum(dim=-1).mean()
            elif self.config.diff_loss == "cosine":
                l_diff = (1 - F.cosine_similarity(mu.float(), v_target, dim=-1)).mean()
            else:
                raise ValueError(f"unknown diff_loss: {self.config.diff_loss}")
            self._log_ce = output.loss.detach().item()
            self._log_ldiff = l_diff.detach().item()
            output.loss = output.loss + self.config.diff_alpha * l_diff

        return output



