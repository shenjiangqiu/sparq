# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""Approximate nearest neighbour methods that approximate `Q @ K.T`."""

from dataclasses import dataclass
from functools import partial
from typing import Any, List, Optional, Tuple, Union

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
import torch.nn.functional as F
from .. import utility
from ..models import gemma_attention, llama_attention, mistral_attention


def gather(t: Tensor, dim: int, i: Tensor) -> Tensor:
    """A broadcasting version of torch.gather."""
    dim += (dim < 0) * t.ndim
    return t.gather(dim, i.expand(*t.shape[:dim], i.shape[dim], *t.shape[dim + 1 :]))


def truncate_f32_tensor(tensor: torch.Tensor, num_valid_bits: int) -> torch.Tensor:
    """
    Truncate a f32 tensor, keeping only the exponent when num_valid_bits=0,
    or keeping num_valid_bits from the mantissa if num_valid_bits >= 1.

    Args:
        tensor (torch.Tensor): Input tensor with dtype=torch.float32.
        num_valid_bits (int): Number of valid bits to keep in the mantissa.

    Returns:
        torch.Tensor: Transformed tensor.
    """
    assert tensor.dtype == torch.float32, "Input tensor must be of dtype torch.float32"

    # Decompose into mantissa and exponent
    mantissa, exponent = torch.frexp(
        tensor
    )  # Returns mantissa in [-1,1) and exponent as int

    if num_valid_bits == 0:
        # Only keep the exponent (set mantissa to 1.0 or -1.0)
        mantissa = torch.sign(mantissa)
    else:
        # Scale mantissa to [1, 2), then truncate to keep num_valid_bits
        mantissa_scaled = mantissa * 2  # Shift range from [0.5,1) to [1,2)
        truncated_mantissa = torch.floor(
            mantissa_scaled * (2 ** (num_valid_bits - 1))
        ) / (2 ** (num_valid_bits - 1))

        # Scale back to original range
        mantissa = truncated_mantissa / 2

    # Reconstruct the number
    return torch.ldexp(mantissa, exponent)


class ExpScore(nn.Module):
    """Gather the top (absolute) components of Q from Q & K."""

    @dataclass
    class Settings:
        valid_bits: int
        last_score: int

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
        # topk = query.abs().sum(dim=2, keepdim=True).topk(dim=-1, k=self.settings.rank)

        # query_proj = gather(query, -1, topk.indices)
        # key_proj = gather(key, -1, topk.indices)

        # Scale could be:
        #  - sqrt(head_size) -- if we think our approximation is exact
        #  - sqrt(rank)      -- if our approximation is no better than random
        #  - sqrt(q_coverage * head_size) -- used below
        #       q_coverage estimates the variance of Q K^T from the approximated
        #       product, and the L1 coverage of Q by the topk components
        # scale = (
        #     query_proj.abs()
        #     .sum(-1)
        #     .div_(query.abs().sum(-1))
        #     .mul_(head_size)
        #     .pow_(0.5)
        #     .unsqueeze(-1)
        # )
        query_proj = truncate_f32_tensor(query, self.settings.valid_bits)
        key_proj = truncate_f32_tensor(key, self.settings.valid_bits)

        return query_proj @ key_proj.transpose(-1, -2)


ScoreSettings = ExpScore.Settings


@dataclass
class Settings:
    k: int
    local_k: int
    reallocate_to_mean_value: bool
    score: ScoreSettings
    sparsity: float

    def __init__(
        self,
        k: int,
        local_k: int,
        reallocate_to_mean_value: bool,
        score: Union[ScoreSettings, str],
        sparsity: float,
        **args: Any,
    ):
        if isinstance(score, str):
            ctor: Any = ExpScore.Settings
            score_settings: ScoreSettings = ctor(**args)
        else:
            assert not args, (
                "ann_attention.Setting only accepts **args when `score` is a string"
            )
            score_settings = score
        self.k = k
        self.local_k = local_k
        self.reallocate_to_mean_value = reallocate_to_mean_value
        self.sparsity = sparsity
        self.score = score_settings


class AlisaAttention(nn.Module):
    """Generic ANN with local windowing and masking."""

    def __init__(self, settings: Settings, n_kv_heads: int, head_size: int):
        super().__init__()
        self.settings = settings
        self.score: nn.Module

        if isinstance(settings.score, ExpScore.Settings):
            self.score = ExpScore(settings.score)
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
        mean_value: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Dense attention, with left-over weight reallocation.

        query -- (batch, n_kv_heads, n_heads_per_kv, n_query, head_size)

        key -- (batch, n_kv_heads, 1, n_kv, head_size)

        value -- (batch, n_kv_heads, 1, n_heads, n_kv, head_size)

        logmask -- (batch, n_kv_heads, n_heads_per_kv, n_query, n_kv)

        kv_weight -- (batch, n_kv_heads, n_heads_per_kv, n_query) | ()
                  -- 1.0 for regular attention (no reallocation)

        mean_value -- (batch, n_kv_heads, n_heads_per_kv, n_query, head_size)
        """
        scores = (query.div(query.shape[-1] ** 0.5) @ key.transpose(-1, -2)).add_(
            logmask
        )
        weights = torch.softmax(scores, -1, dtype=torch.float32).to(value.dtype)
        # Value-mixing with reallocation
        # weights *= kv_weight[..., None]
        output = weights @ value
        # output += (1 - kv_weight[..., None]) * mean_value
        return output, weights

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        logmask: Tensor,
        last_weight: Tensor,
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

        sparsity = self.settings.sparsity
        valid_part = 1 - sparsity
        valid_len = int(valid_part * key.shape[-2])
        if valid_len % 2 == 1:
            valid_len += 1

        batch, n_kv_heads, seq, head_size = key.shape
        n_heads_per_kv = query.shape[1] // n_kv_heads

        # Group by KV head
        query, key, value, logmask = map(
            partial(torch.unflatten, dim=1, sizes=(n_kv_heads, -1)),
            [query, key, value, logmask],
        )

        assert query.shape == (batch, n_kv_heads, n_heads_per_kv, 1, head_size)
        assert key.shape == (batch, n_kv_heads, 1, seq, head_size)
        assert value.shape == (batch, n_kv_heads, 1, seq, head_size)
        assert logmask.shape == (batch, n_kv_heads, n_heads_per_kv, 1, seq)

        # Calculate an approximate score for each (query, key) pair
        # shape -- (batch, n_kv_heads, n_heads_per_kv, 1, seq)
        # score = (self.score(query, key) + logmask).float()

        # # Set the score of local keys (+1 current) to max
        # causal_index = sparse_attention.causal_index(logmask)
        # is_local = (0 <= causal_index) & (causal_index < self.settings.local_k + 1)
        # topk_score = score.masked_fill(is_local, torch.finfo(score.dtype).max).sum(
        #     dim=2, keepdim=True
        # )
        # Find max-score keys (note: +1 because the current token's k comes "for free")
        half_valid_len = int(valid_len / 2)
        valid_len_int = int(valid_len)
        # print("valid len: ", valid_len_int)
        last_weight_sum = last_weight[:, :, -half_valid_len:, : key.shape[-2]].sum(
            -2, keepdim=True
        )
        last_weight_sum[:, :, :, -half_valid_len:] += 1
        indices = last_weight_sum.topk(valid_len_int, dim=-1).indices
        # print("indices_shape: ", indices.shape)
        # print("indices: ", indices[0, 0])
        # if self.debug_indices is not None:
        #     self.debug_indices.append(indices)

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
        kv_indices = indices.unsqueeze(-1)  # (batch, n_kv_heads, 1, k+1, 1)
        output, weights = self._attention(
            query,
            gather(key, -2, kv_indices),
            gather(value, -2, kv_indices),
            gather(logmask, -1, kv_indices.transpose(-1, -2)),
            mean_value=0,
        )
        # Note: expand indices as scatter does not broadcast (!)
        return output.flatten(1, 2), torch.zeros_like(logmask).scatter(
            -1, indices.unsqueeze(-2).expand_as(weights), weights
        ).flatten(1, 2)


Model = Union[
    GPTNeoXForCausalLM, LlamaForCausalLM, MistralForCausalLM, GemmaForCausalLM
]


class GPTNeoXAttentionWithANN(GPTNeoXAttention):  # type:ignore[misc]
    def __init__(self, config: GPTNeoXConfig, settings: Settings):
        utility.check_transformers_version(type(self))
        super().__init__(config)
        self.expatt = AlisaAttention(settings, self.num_attention_heads, self.head_size)
        # print("Using AlisaAttention init!")
        self.weights = None
        self.last_score = settings.score.last_score
        self.settings = settings

    def _attn(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        # print("Using AlisaAttention _attn!")
        # print("query shape: ", query.shape)
        # print("key shape: ", key.shape)
        # print("value shape: ", value.shape)

        assert attention_mask is not None
        assert head_mask is None

        # Only enable ANN during autoregressive generation
        if query.shape[-2] == 1:
            output, weight = self.expatt(  # type:ignore[no-any-return]
                query,
                key,
                value,
                attention_mask.broadcast_to(key.unsqueeze(-3).shape[:-1]),
                last_weight=self.weights,
            )
            # print("weight_shape: ", weight.shape)
            current_size = self.weights.shape[-1]
            target_size = weight.shape[-1]

            # Pad `self.weights` with zeros if needed
            if current_size < target_size:
                pad_size = 100
                self.weights = F.pad(self.weights, (0, pad_size))
            new_size = self.weights.shape[-1]
            # pad target weight
            if target_size < new_size:
                weight = F.pad(weight, (0, new_size - target_size))
            self.weights = torch.cat([self.weights[:, :, :, :], weight.clone()], dim=-2)
            # print("self.weights_shape: ", self.weights.shape)
            weight_sum = self.weights.sum(dim=-1)
            # print("weightsum:", weight_sum[0, 0])
            return output, weight
        else:
            output, weight = super()._attn(  # type:ignore[no-any-return]
                query, key, value, attention_mask, head_mask
            )
            # print("weight_shape: ", weight.shape)
            att_len = weight.shape[-2]
            valid_len = int((1 - self.settings.sparsity) * att_len / 2) + 4
            self.weights = weight[:, :, -valid_len:, :].clone()
            # print("self.weights_shape: ", self.weights.shape)
            return output, weight


class LlamaAttentionWithANN(llama_attention.LlamaAttention):
    def __init__(
        self, config: LlamaConfig, layer_idx: Optional[int], settings: Settings
    ):
        utility.check_transformers_version(type(self))
        super().__init__(config, layer_idx)
        self.settings = settings
        self.expatt = AlisaAttention(settings, self.num_key_value_heads, self.head_dim)

    def _attn(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        logmask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if query.shape[-2] == 1:
            return self.expatt(  # type:ignore[no-any-return]
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
        self.expatt = AlisaAttention(settings, self.num_heads, self.head_dim)

    def _attn(
        self, query: Tensor, key: Tensor, value: Tensor, logmask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        if query.shape[-2] == 1:
            return self.expatt(  # type:ignore[no-any-return]
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
        self.expatt = AlisaAttention(settings, self.num_key_value_heads, self.head_dim)

    def _attn(
        self, query: Tensor, key: Tensor, value: Tensor, logmask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        if query.shape[-2] == 1:
            return self.expatt(  # type:ignore[no-any-return]
                query,
                key,
                value,
                # reshape to (batch, n_heads, 1, seq_len)
                logmask.broadcast_to(*query.shape[:-1], key.shape[-2]),
            )
        return super()._attn(query, key, value, logmask)


def convert(model: Model, settings: Settings) -> Model:
    """Convert a model to use KV cache compression using ANN."""

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
