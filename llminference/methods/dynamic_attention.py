# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""Approximate nearest neighbour methods that approximate `Q @ K.T`."""

from dataclasses import dataclass
from functools import partial
from typing import Any, List, Optional, Tuple, Union, cast

import torch
from torch import Tensor, nn
from transformers.models.gemma.configuration_gemma import GemmaConfig
from transformers.models.gemma.modeling_gemma import GemmaAttention, GemmaForCausalLM
from transformers.models.gpt_neox.configuration_gpt_neox import GPTNeoXConfig
from transformers.models.gpt_neox.modeling_gpt_neox import (
    GPTNeoXAttention,
    GPTNeoXForCausalLM,
)
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaForCausalLM
from transformers.models.mistral.configuration_mistral import MistralConfig
from transformers.models.mistral.modeling_mistral import (
    MistralAttention,
    MistralForCausalLM,
)

from .. import utility
from ..models import gemma_attention, llama_attention, mistral_attention


def gather(t: Tensor, dim: int, i: Tensor) -> Tensor:
    """A broadcasting version of torch.gather."""
    dim += (dim < 0) * t.ndim
    return t.gather(dim, i.expand(*t.shape[:dim], i.shape[dim], *t.shape[dim + 1 :]))


class LowRank(nn.Module):
    """Use a random orthonormal projection to down-project Q & K."""

    @dataclass
    class Settings:
        rank: int
        name: str = "low_rank"

    def __init__(self, settings: Settings, n_kv_heads: int, head_size: int):
        super().__init__()
        self.settings = settings
        self.weight = nn.Parameter(torch.empty(n_kv_heads, 1, head_size, settings.rank))
        for i in range(n_kv_heads):  # can't batch this!
            nn.init.orthogonal_(self.weight[i, 0])

    def forward(self, query: Tensor, key: Tensor) -> Tensor:
        """Compute approximate score for each (query, key).

        query -- (batch, n_kv_heads, n_heads_per_kv, query, head_size)

        key -- (batch, n_kv_heads, 1, key, head_size)

        returns -- (batch, n_kv_heads, n_heads_per_kv, query, key)
        """
        query_proj = query.to(self.weight.dtype) @ self.weight
        key_proj = key.to(self.weight.dtype) @ self.weight
        return cast(
            Tensor,
            (query_proj.div(query.shape[-1] ** 0.5) @ key_proj.transpose(-1, -2)),
        )


class SparseQ(nn.Module):
    """Gather the top (absolute) components of Q from Q & K."""

    @dataclass
    class Settings:
        rank: int
        name: str = "sparse_q"

    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings

    def forward(self, query: Tensor, key: Tensor) -> Tensor:
        """Compute approximate score for each (query, key).

        query -- (batch, n_kv_heads, n_heads_per_kv, 1, head_size)

        key -- (batch, n_kv_heads, 1, key, head_size)

        returns -- (batch, n_kv_heads, n_heads_per_kv, 1, key)
        """
        assert query.shape[-2] == 1, "no support for multiple queries"
        head_size = query.shape[-1]

        # Sum the magnitudes within KV groups before top-k
        # shape -- (batch, n_kv_heads, 1, 1, rank)
        topk = query.abs().sum(dim=2, keepdim=True).topk(dim=-1, k=self.settings.rank)

        query_proj = gather(query, -1, topk.indices)
        key_proj = gather(key, -1, topk.indices)

        # Scale could be:
        #  - sqrt(head_size) -- if we think our approximation is exact
        #  - sqrt(rank)      -- if our approximation is no better than random
        #  - sqrt(q_coverage * head_size) -- used below
        #       q_coverage estimates the variance of Q K^T from the approximated
        #       product, and the L1 coverage of Q by the topk components
        scale = (
            query_proj.abs()
            .sum(-1)
            .div_(query.abs().sum(-1))
            .mul_(head_size)
            .pow_(0.5)
            .unsqueeze(-1)
        )
        return query_proj.div(scale) @ key_proj.transpose(-1, -2)


ScoreSettings = Union[LowRank.Settings, SparseQ.Settings]


@dataclass
class Settings:
    k: int
    local_k: int
    reallocate_to_mean_value: bool
    sparsity: float
    score: ScoreSettings
    global_stats: Optional[dict] = None

    def __init__(
        self,
        k: int,
        local_k: int,
        reallocate_to_mean_value: bool,
        sparsity: float,
        score: Union[ScoreSettings, str],
        global_stats: Optional[dict],
        **args: Any,
    ):
        if isinstance(score, str):
            ctor: Any = dict(low_rank=LowRank.Settings, sparse_q=SparseQ.Settings)[
                score
            ]
            score_settings: ScoreSettings = ctor(**args)
        else:
            assert not args, (
                "ann_attention.Setting only accepts **args when `score` is a string"
            )
            score_settings = score
        self.k = k
        self.local_k = local_k
        self.sparsity = sparsity
        self.reallocate_to_mean_value = reallocate_to_mean_value
        self.score = score_settings
        self.global_stats = global_stats


class AnnAttention(nn.Module):
    """Generic ANN with local windowing and masking."""

    def __init__(self, settings: Settings, n_kv_heads: int, head_size: int):
        super().__init__()
        self.settings = settings
        self.score: nn.Module
        if isinstance(settings.score, LowRank.Settings):
            self.score = LowRank(settings.score, n_kv_heads, head_size)
        elif isinstance(settings.score, SparseQ.Settings):
            self.score = SparseQ(settings.score)
        else:
            raise ValueError(f"Unexpected settings.score = {settings.score}")
        # Set to an empty list to turn on ANN index logging
        self.debug_indices: Optional[List[Tensor]] = None

    def _attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        logmask: Tensor,
        threshold: float = 0.9,  # 累积权重阈值
        global_stats: Optional[dict] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Dense attention, with left-over weight reallocation and threshold pruning.

        query -- (batch, n_kv_heads, n_heads_per_kv, n_query, head_size)
        key -- (batch, n_kv_heads, 1, n_kv, head_size)
        value -- (batch, n_kv_heads, 1, n_kv, head_size)
        logmask -- (batch, n_kv_heads, n_heads_per_kv, n_query, n_kv)
        threshold -- float, cumulative weight threshold for token selection
        """
        scores = (query.div(query.shape[-1] ** 0.5) @ key.transpose(-1, -2)).add_(
            logmask
        )
        weights = torch.softmax(scores, -1, dtype=torch.float32).to(value.dtype)

        # 对 weights 做排序，选出累加值大于 threshold 的前 n 个 token
        orig_shape = weights.shape
        batch, n_kv_heads, n_heads_per_kv, n_query, n_kv = weights.shape
        assert n_query == 1, "Only support n_query == 1 for simplicity"
        flat_weights = weights.reshape(-1, n_kv)  # [B*H*Hk*Q, K]
        mask = torch.zeros_like(flat_weights, dtype=torch.bool)
        selected_weight_sum = torch.zeros(
            flat_weights.size(0), device=weights.device, dtype=weights.dtype
        )
        selected_indices = []
        header_selected_len = []
        for i in range(flat_weights.size(0)):
            w = flat_weights[i]
            sorted_w, idx = torch.sort(w, descending=True)
            cumsum = torch.cumsum(sorted_w, dim=0)
            n = (cumsum >= threshold).nonzero(as_tuple=True)[0]
            n = n[0].item() + 1 if len(n) > 0 else len(sorted_w)
            header_selected_len.append(n)
            if global_stats is not None:
                global_stats["n_selected"] += n
                global_stats["total_tokens"] += len(sorted_w)

            mask[i, idx[:n]] = True
            selected_weight_sum[i] = sorted_w[:n].sum()
            selected_indices.append(idx[:n])

        # 还原mask形状
        mask = mask.view(orig_shape)
        pruned_weights = weights * mask
        pruned_weights_sum = pruned_weights.sum(dim=-1, keepdim=True) + 1e-8
        # print(pruned_weights_sum)
        # print(selected_weight_sum)
        # todo, don't normalize.
        pruned_weights = pruned_weights / pruned_weights_sum

        # 计算 mean_value: (batch, n_kv_heads, n_heads_per_kv, n_query, head_size)
        # value: (batch, n_kv_heads, 1, n_kv, head_size)
        # logmask: (batch, n_kv_heads, n_heads_per_kv, n_query, n_kv)
        # 取 logmask 的 exp 得到 mask
        value_mask = (
            logmask[:, :, :1].squeeze(-2).unsqueeze(-1).exp()
        )  # (batch, n_kv_heads, 1, seq, 1)
        mean_value = (
            (value * value_mask)
            .sum(-2, dtype=torch.float32, keepdim=True)
            .div_(value_mask.sum(-2, dtype=torch.float32, keepdim=True))
            .to(value.dtype)
        )  # (batch, n_kv_heads, 1, 1, 1)

        # kv_weight: (batch, n_kv_heads, n_heads_per_kv, n_query)
        kv_weight = selected_weight_sum.view(batch, n_kv_heads, n_heads_per_kv, n_query)

        # Value-mixing with reallocation
        # pruned_weights = pruned_weights * kv_weight[..., None]
        output = pruned_weights @ value
        # output += (1 - kv_weight[..., None]) * mean_value
        return output, pruned_weights

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        logmask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Preprocess (key, value, mask) for ANN attention.

        query -- (batch, n_heads, 1, head_size)

        key -- (batch, n_kv_heads, seq, head_size)

        value -- (batch, n_kv_heads, seq, head_size)

        logmask -- (batch, n_heads, 1, seq)

        returns -- (output, weights)
                   output -- (batch, n_heads, 1, head_size)
                   weights -- (batch, n_heads, 1, seq)
        """
        batch, n_kv_heads, seq, head_size = key.shape
        n_heads_per_kv = query.shape[1] // n_kv_heads
        # print("logmask shape:", logmask.shape)
        # Group by KV head
        query, key, value, logmask = map(
            partial(torch.unflatten, dim=1, sizes=(n_kv_heads, -1)),
            [query, key, value, logmask],
        )
        # print("logmask shape:", logmask.shape)
        # print("logmask:", logmask)

        assert query.shape == (batch, n_kv_heads, n_heads_per_kv, 1, head_size)
        assert key.shape == (batch, n_kv_heads, 1, seq, head_size)
        assert value.shape == (batch, n_kv_heads, 1, seq, head_size)
        assert logmask.shape == (batch, n_kv_heads, n_heads_per_kv, 1, seq)

        # Calculate an approximate score for each (query, key) pair
        # shape -- (batch, n_kv_heads, n_heads_per_kv, 1, seq)
        # score = (self.score(query, key) + logmask).float()

        # def get_weights(query, key, value, logmask):
        #     scores = (query.div(query.shape[-1] ** 0.5) @ key.transpose(-1, -2)).add_(
        #         logmask
        #     )
        #     weights = torch.softmax(scores, -1, dtype=torch.float32).to(value.dtype)
        #     return weights

        # weights = get_weights(query, key, value, logmask)
        # assert weights.shape == (batch, n_kv_heads, n_heads_per_kv, 1, seq)
        # print("Weights shape:", weights.shape)
        # seq_len = key.shape[-2]

        # sparsity = self.settings.sparsity
        # valid_len = int((1 - sparsity) * seq_len)
        # if valid_len % 2 == 1:
        #     valid_len += 1
        # half_valid_len = int(valid_len / 2)
        # Set the score of local keys (+1 current) to max
        # causal_index = sparse_attention.causal_index(logmask)
        # print("causal_index:", causal_index)
        # is_local = (0 <= causal_index) & (causal_index < half_valid_len + 1)
        # print(is_local)
        # print("score shape:", score.shape)
        # topk_score = score.masked_fill(is_local, torch.finfo(score.dtype).max).sum(
        #     dim=2, keepdim=True
        # )
        # print("Topk_score shape:", topk_score.shape)
        # # print("half_valid_len", half_valid_len)
        # # print("valid_len", valid_len)
        # # Find max-score keys (note: +1 because the current token's k comes "for free")
        # indices = topk_score.topk(
        #     min(valid_len, score.shape[-1]), -1
        # ).indices  # (batch, n_kv_heads, 1, 1, k+1)
        # print("Indices shape:", indices.shape)
        # print("Indices:", indices)
        # if self.debug_indices is not None:
        #     self.debug_indices.append(indices)
        # Score shape: torch.Size([1, 16, 1, 1, 1327])
        # Indices shape: torch.Size([1, 16, 1, 1, 266])
        # Optional "mean_value" kv
        # Note: assumes same logmask for all heads
        # value_mask = (
        #     logmask[:, :, :1].squeeze(-2).unsqueeze(-1).exp()
        # )  # (batch, n_kv_heads, 1, seq, 1)
        # mean_value = (
        #     (value * value_mask)
        #     .sum(-2, dtype=torch.float32, keepdim=True)
        #     .div_(value_mask.sum(-2, dtype=torch.float32, keepdim=True))
        #     .to(value.dtype)
        # )  # (batch, n_kv_heads, 1, 1, 1)
        # kv_weight = torch.tensor(1.0, device=query.device)
        # if self.settings.reallocate_to_mean_value:
        #     kv_weight = (
        #         gather(torch.softmax(score, -1), -1, indices)  # no need to expand here
        #         .sum(-1)
        #         .to(value.dtype)
        #     )  # (batch, n_kv_heads, n_heads_per_kv, 1)

        # Slice key, value, logmask for attention
        # kv_indices = indices.squeeze(-2).unsqueeze(-1)  # (batch, n_kv_heads, 1, k+1, 1)
        output, weights = self._attention(
            query,
            key,
            value,
            logmask,
            threshold=self.settings.sparsity,
            global_stats=self.settings.global_stats,
        )

        # Note: expand indices as scatter does not broadcast (!)
        return output.flatten(1, 2), weights.flatten(1, 2)


Model = Union[
    GPTNeoXForCausalLM, LlamaForCausalLM, MistralForCausalLM, GemmaForCausalLM
]


class GPTNeoXAttentionWithANN(GPTNeoXAttention):  # type:ignore[misc]
    def __init__(self, config: GPTNeoXConfig, settings: Settings):
        utility.check_transformers_version(type(self))
        super().__init__(config)
        self.ann = AnnAttention(settings, self.num_attention_heads, self.head_size)

    def _attn(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        assert attention_mask is not None
        assert head_mask is None

        # query shape -- (batch, n_heads, token_length, head_size)
        # key shape -- (batch, n_heads, kv_length, head_size)
        # value shape -- (batch, n_heads, kv_length, head_size)
        # attention_mask shape -- (batch, n_heads, token_length, kv_length)
        # Only enable ANN during autoregressive generation
        if query.shape[-2] == 1:
            return self.ann(  # type:ignore[no-any-return]
                query,
                key,
                value,
                attention_mask.broadcast_to(key.unsqueeze(-3).shape[:-1]),
            )

        return super()._attn(  # type:ignore[no-any-return]
            query, key, value, attention_mask, head_mask
        )


class LlamaAttentionWithANN(llama_attention.LlamaAttention):
    def __init__(
        self, config: LlamaConfig, layer_idx: Optional[int], settings: Settings
    ):
        utility.check_transformers_version(type(self))
        super().__init__(config, layer_idx)
        self.settings = settings
        self.ann = AnnAttention(settings, self.num_key_value_heads, self.head_dim)

    def _attn(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        logmask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if query.shape[-2] == 1:
            print("Using ANN for LlamaAttentionWithANN")
            print("Query shape:", query.shape)
            print("Key shape:", key.shape)
            print("Value shape:", value.shape)
            print("Logmask shape:", logmask.shape)

            return self.ann(  # type:ignore[no-any-return]
                query,
                key,
                value,
                # reshape to (batch, n_heads, 1, seq_len)
                logmask.broadcast_to(*query.shape[:-1], key.shape[-2]),
            )
        return super()._attn(query, key, value, logmask)


class GemmaAttentionWithANN(gemma_attention.GemmaAttention):
    def __init__(
        self, config: GemmaConfig, layer_idx: Optional[int], settings: Settings
    ):
        utility.check_transformers_version(type(self))
        super().__init__(config, layer_idx)
        self.settings = settings
        self.ann = AnnAttention(settings, self.num_heads, self.head_dim)

    def _attn(
        self, query: Tensor, key: Tensor, value: Tensor, logmask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        if query.shape[-2] == 1:
            return self.ann(  # type:ignore[no-any-return]
                query,
                key,
                value,
                # reshape to (batch, n_heads, 1, seq_len)
                logmask.broadcast_to(*query.shape[:-1], key.shape[-2]),
            )
        return super()._attn(query, key, value, logmask)


class MistralAttentionWithANN(mistral_attention.MistralAttention):
    def __init__(
        self, config: MistralConfig, layer_idx: Optional[int], settings: Settings
    ):
        utility.check_transformers_version(type(self))
        super().__init__(config, layer_idx)
        self.settings = settings
        self.ann = AnnAttention(settings, self.num_key_value_heads, self.head_dim)

    def _attn(
        self, query: Tensor, key: Tensor, value: Tensor, logmask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        if query.shape[-2] == 1:
            return self.ann(  # type:ignore[no-any-return]
                query,
                key,
                value,
                # reshape to (batch, n_heads, 1, seq_len)
                logmask.broadcast_to(*query.shape[:-1], key.shape[-2]),
            )
        return super()._attn(query, key, value, logmask)


def convert(model: Model, settings: Settings) -> Model:
    """Convert a model to use KV cache compression using ANN."""
    print("Dynamic!!!")

    def _replace(m: nn.Module) -> Optional[nn.Module]:
        if isinstance(m, GPTNeoXAttention):
            return GPTNeoXAttentionWithANN(model.config, settings)
        if isinstance(m, LlamaAttention):
            return LlamaAttentionWithANN(model.config, m.layer_idx, settings)
        if isinstance(m, MistralAttention):
            return MistralAttentionWithANN(model.config, m.layer_idx, settings)
        if isinstance(m, GemmaAttention):
            return GemmaAttentionWithANN(model.config, m.layer_idx, settings)

    return utility.convert_module(model, _replace)
