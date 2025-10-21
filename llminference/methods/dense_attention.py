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
            assert (
                not args
            ), "ann_attention.Setting only accepts **args when `score` is a string"
            score_settings = score
        self.k = k
        self.local_k = local_k
        self.sparsity = sparsity
        self.reallocate_to_mean_value = reallocate_to_mean_value
        self.score = score_settings
        self.global_stats = global_stats


Model = Union[
    GPTNeoXForCausalLM, LlamaForCausalLM, MistralForCausalLM, GemmaForCausalLM
]


class GPTNeoXAttentionWithANN(GPTNeoXAttention):  # type:ignore[misc]
    def __init__(self, config: GPTNeoXConfig):
        utility.check_transformers_version(type(self))
        super().__init__(config)

    def _modified_attn(self, query, key, value, attention_mask=None, head_mask=None):
        # q, k, v: [bs, num_attention_heads, seq_len, attn_head_size]
        # compute causal mask from causal mask buffer
        raw_value_file = "dense_raw_value_debug.pt"
        # import os
        # if not os.path.exists(raw_value_file):
        #     torch.save(value, raw_value_file)
        # raw_query_file = "dense_raw_query_debug.pt"
        # if not os.path.exists(raw_query_file):
        #     torch.save(query, raw_query_file)
        # raw_key_file = "dense_raw_key_debug.pt"
        # if not os.path.exists(raw_key_file):
        #     torch.save(key, raw_key_file)

        batch_size, num_attention_heads, query_length, attn_head_size = query.size()
        key_length = key.size(-2)

        # dynamically increase the causal mask with the key length, if needed.
        if key_length > self.bias.shape[-1]:
            self._init_bias(key_length, device=key.device)
        causal_mask = self.bias[
            :, :, key_length - query_length : key_length, :key_length
        ]

        query = query.view(
            batch_size * num_attention_heads, query_length, attn_head_size
        )
        key = key.view(batch_size * num_attention_heads, key_length, attn_head_size)
        attn_scores = torch.zeros(
            batch_size * num_attention_heads,
            query_length,
            key_length,
            dtype=query.dtype,
            device=key.device,
        )
        attn_scores = torch.baddbmm(
            attn_scores,
            query,
            key.transpose(1, 2),
            beta=1.0,
            alpha=self.norm_factor,
        )
        attn_scores = attn_scores.view(
            batch_size, num_attention_heads, query_length, key_length
        )

        mask_value = torch.finfo(attn_scores.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.tensor(mask_value, dtype=attn_scores.dtype).to(
            attn_scores.device
        )
        attn_scores = torch.where(causal_mask, attn_scores, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_scores = attn_scores + attention_mask

        attn_weights = nn.functional.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights.to(value.dtype)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_weights = self.attention_dropout(attn_weights)
        # file_name = "dense_weights_debug.pt"
        # # save only when this file does not exist
        # import os

        # if not os.path.exists(file_name):
        #     torch.save(attn_weights, file_name)
        attn_output = torch.matmul(attn_weights, value)
        # value_file = "dense_value_debug.pt"
        # save only when this file does not exist
        # if not os.path.exists(value_file):
            # torch.save(value, value_file)

        # output_name = "dense_output_debug.pt"
        # if not os.path.exists(output_name):
            # torch.save(attn_output, output_name)
        # exit(1)
        return attn_output, attn_weights

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
            return self._modified_attn(  # type:ignore[no-any-return]
                query,
                key,
                value,
                attention_mask,
                head_mask,
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


def convert(model: Model) -> Model:
    """Convert a model to use KV cache compression using ANN."""
    print("Dynamic!!!")

    def _replace(m: nn.Module) -> Optional[nn.Module]:
        if isinstance(m, GPTNeoXAttention):
            return GPTNeoXAttentionWithANN(
                model.config,
            )
        if isinstance(m, LlamaAttention):
            return LlamaAttentionWithANN(
                model.config,
                m.layer_idx,
            )
        if isinstance(m, MistralAttention):
            return MistralAttentionWithANN(
                model.config,
                m.layer_idx,
            )
        if isinstance(m, GemmaAttention):
            return GemmaAttentionWithANN(
                model.config,
                m.layer_idx,
            )

    return utility.convert_module(model, _replace)
