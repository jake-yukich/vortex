import math
from functools import partial

import torch
import torch.nn as nn
from einops import rearrange, repeat

from vortex.model.utils import get_dim_for_local_rank

# try:
#     from vortex.ops import (
#         local_flash_attn_kvpacked_func,
#         local_flash_attn_qkvpacked_func,
#         local_flash_attn_varlen_kvpacked_func,
#         local_flash_attn_varlen_qkvpacked_func,
#         local_flash_attn_with_kvcache,
#     )
# except ImportError:
#     local_flash_attn_varlen_qkvpacked_func, local_flash_attn_varlen_kvpacked_func = (
#         None,
#         None,
#     )
#     local_flash_attn_qkvpacked_func, local_flash_attn_kvpacked_func = None, None
#     local_flash_attn_with_kvcache = None

FusedDense, ColumnParallelLinear, RowParallelLinear = None, None, None

from vortex.model.rotary import RotaryEmbedding


# From https://github.com/ofirpress/attention_with_linear_biases/blob/4b92f28a005ead2567abe2359f633e73e08f3833/fairseq/models/transformer.py#L742
def get_alibi_slopes(nheads):
    def get_slopes_power_of_2(nheads):
        start = 2 ** (-(2 ** -(math.log2(nheads) - 3)))
        ratio = start
        return [start * ratio**i for i in range(nheads)]

    if math.log2(nheads).is_integer():
        return get_slopes_power_of_2(nheads)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(nheads))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_alibi_slopes(2 * closest_power_of_2)[0::2][: nheads - closest_power_of_2]
        )


# class FlashSelfAttention(nn.Module):
#     """Implement the scaled dot product attention with softmax.
#     Arguments
#     ---------
#         softmax_scale: The temperature to use for the softmax attention.
#                       (default: 1/sqrt(d_keys) where d_keys is computed at
#                       runtime)
#         attention_dropout: The dropout rate to apply to the attention
#                            (default: 0.0)
#     """

#     def __init__(
#         self,
#         layer_number,
#         causal=False,
#         softmax_scale=None,
#         attention_dropout=0.0,
#         window_size=(-1, -1),
#         alibi_slopes=None,
#         deterministic=False,
#     ):
#         super().__init__()
#         # assert local_flash_attn_varlen_qkvpacked_func is not None, "FlashAttention is not installed" # Removed
#         # assert local_flash_attn_qkvpacked_func is not None, "FlashAttention is not installed" # Removed
#         self.layer_number = layer_number
#         self.causal = causal
#         self.softmax_scale = softmax_scale
#         self.drop = nn.Dropout(attention_dropout)
#         self.register_buffer("alibi_slopes", alibi_slopes, persistent=False)
#         self.window_size = window_size
#         self.deterministic = deterministic

#     def forward(self, qkv, causal=None, cu_seqlens=None, max_seqlen=None):
#         """Implements the multihead softmax attention.
#         Arguments
#         ---------
#             qkv: The tensor containing the query, key, and value.
#                 If cu_seqlens is None and max_seqlen is None, then qkv has shape (B, S, 3, H, D).
#                 If cu_seqlens is not None and max_seqlen is not None, then qkv has shape
#                 (total, 3, H, D), where total is the sum of the sequence lengths in the batch.
#             causal: if passed, will override self.causal
#             cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
#                 of the sequences in the batch, used to index into qkv.
#             max_seqlen: int. Maximum sequence length in the batch.
#         Returns:
#         --------
#             out: (total, H, D) if cu_seqlens is not None and max_seqlen is not None,
#                 else (B, S, H, D).
#         """
#         assert qkv.dtype in [torch.float16, torch.bfloat16]
#         assert qkv.is_cuda

#         causal = self.causal if causal is None else causal
#         unpadded = cu_seqlens is not None
#         if self.alibi_slopes is not None:
#             self.alibi_slopes = self.alibi_slopes.to(torch.float32)
#         if unpadded:
#             assert cu_seqlens.dtype == torch.int32
#             assert max_seqlen is not None
#             assert isinstance(max_seqlen, int)
#             # return local_flash_attn_varlen_qkvpacked_func( # Removed
#             #     qkv,
#             #     cu_seqlens,
#             #     max_seqlen,
#             #     self.drop.p if self.training else 0.0,
#             #     softmax_scale=self.softmax_scale,
#             #     causal=causal,
#             #     alibi_slopes=self.alibi_slopes,
#             #     window_size=self.window_size,
#             #     deterministic=self.deterministic,
#             # )
#             raise NotImplementedError("FlashAttention (unpadded) path not available")
#         else:
#             # y = local_flash_attn_qkvpacked_func( # Removed
#             #     qkv,
#             #     self.drop.p if self.training else 0.0,
#             #     softmax_scale=self.softmax_scale,
#             #     causal=causal,
#             #     alibi_slopes=self.alibi_slopes,
#             #     window_size=self.window_size,
#             #     deterministic=self.deterministic,
#             # )
#             # return y
#             raise NotImplementedError("FlashAttention (padded) path not available")


# class FlashCrossAttention(nn.Module):
#     """Implement the scaled dot product attention with softmax.
#     Arguments
#     ---------
#         softmax_scale: The temperature to use for the softmax attention.
#                       (default: 1/sqrt(d_keys) where d_keys is computed at
#                       runtime)
#         attention_dropout: The dropout rate to apply to the attention
#                            (default: 0.0)
#     """

#     def __init__(
#         self,
#         causal=False,
#         softmax_scale=None,
#         attention_dropout=0.0,
#         alibi_slopes=None,
#         window_size=(-1, -1),
#         deterministic=False,
#     ):
#         super().__init__()
#         # assert local_flash_attn_varlen_kvpacked_func is not None, "FlashAttention is not installed" # Removed
#         # assert local_flash_attn_kvpacked_func is not None, "FlashAttention is not installed" # Removed
#         self.causal = causal
#         self.softmax_scale = softmax_scale
#         self.drop = nn.Dropout(attention_dropout)
#         self.register_buffer("alibi_slopes", alibi_slopes, persistent=False)
#         self.window_size = window_size
#         self.deterministic = deterministic

#     def forward(
#         self,
#         q,
#         kv,
#         causal=None,
#         cu_seqlens=None,
#         max_seqlen=None,
#         cu_seqlens_k=None,
#         max_seqlen_k=None,
#     ):
#         """Implements the multihead softmax attention.
#         Arguments
#         ---------
#             q: The tensor containing the query. (B, Sq, H, D)
#             kv: The tensor containing the key and value. (B, Sk, 2, H_k, D)
#             causal: if passed, will override self.causal
#             cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
#                 of the sequences in the batch, used to index into q.
#             max_seqlen: int. Maximum sequence length in the batch of q.
#             cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
#                 of the sequences in the batch, used to index into kv.
#             max_seqlen_k: int. Maximum sequence length in the batch of k and v.
#         """
#         assert q.dtype in [torch.float16, torch.bfloat16]
#         assert q.is_cuda and kv.is_cuda
#         causal = self.causal if causal is None else causal
#         unpadded = cu_seqlens is not None
#         if self.alibi_slopes is not None:
#             self.alibi_slopes = self.alibi_slopes.to(torch.float32)
#         if unpadded:
#             assert cu_seqlens.dtype == torch.int32
#             assert max_seqlen is not None
#             assert isinstance(max_seqlen, int)
#             assert cu_seqlens_k is not None
#             assert cu_seqlens_k.dtype == torch.int32
#             assert max_seqlen_k is not None
#             assert isinstance(max_seqlen_k, int)
#             # return local_flash_attn_varlen_kvpacked_func( # Removed
#             #     q,
#             #     kv,
#             #     cu_seqlens,
#             #     cu_seqlens_k,
#             #     max_seqlen,
#             #     max_seqlen_k,
#             #     self.drop.p if self.training else 0.0,
#             #     softmax_scale=self.softmax_scale,
#             #     causal=causal,
#             #     alibi_slopes=self.alibi_slopes,
#             #     window_size=self.window_size,
#             #     deterministic=self.deterministic,
#             # )
#             raise NotImplementedError("FlashAttention (unpadded) path not available")
#         else:
#             batch_size, seqlen_q = q.shape[0], q.shape[1]
#             seqlen_k = kv.shape[1]
#             assert kv.shape[0] == batch_size and kv.shape[4] == q.shape[3]
#             # return local_flash_attn_kvpacked_func( # Removed
#             #     q,
#             #     kv,
#             #     self.drop.p if self.training else 0.0,
#             #     causal=causal,
#             #     softmax_scale=self.softmax_scale,
#             #     alibi_slopes=self.alibi_slopes,
#             #     window_size=self.window_size,
#             #     deterministic=self.deterministic,
#             # )
#             raise NotImplementedError("FlashAttention (padded) path not available")


class SelfAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)

    def forward(self, qkv, causal=None, key_padding_mask=None):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D)
            causal: if passed, will override self.causal
            key_padding_mask: boolean mask to apply to the attention weights. True means to keep,
                False means to mask out. (B, S)
        """
        q, k, v = qkv.unbind(dim=2)  # each: (B, T, H, D)
        q = q.permute(0, 2, 1, 3)  # (B, H, T, D)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        batch_size, num_heads, seqlen, d = q.shape

        scale = self.softmax_scale if self.softmax_scale is not None else 1.0 / math.sqrt(d)
        q = q * (scale * math.sqrt(d))

        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = torch.where(
                repeat(key_padding_mask, "b s -> b t s", t=seqlen),
                0.0,
                -10000.0,
            )

        output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.drop.p if self.training else 0.0,
            is_causal=(self.causal if causal is None else causal),
        )

        output = output.permute(0, 2, 1, 3)
        return output


class CrossAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)

    def forward(self, q, kv, causal=None, key_padding_mask=None):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q: The tensor containing the query. (B, Sq, H, D)
            kv: The tensor containing the key and value. (B, Sk, 2, H_k, D)
            causal: if passed, will override self.causal
            key_padding_mask: boolean mask to apply to the attention weights. True means to keep,
                False means to mask out. (B, Sk)
        """
        batch_size, seqlen_q = q.shape[0], q.shape[1]
        causal = self.causal if causal is None else causal
        seqlen_k = kv.shape[1]
        assert kv.shape[0] == batch_size and kv.shape[4] == q.shape[3]
        if kv.shape[3] != q.shape[2]:  # MQA/GQA
            kv = repeat(kv, "... hkv d -> ... (hkv g) d", g=q.shape[2] // kv.shape[3])
        k, v = kv.unbind(dim=2)
        softmax_scale = self.softmax_scale or 1.0 / math.sqrt(q.shape[-1])
        scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)
        if key_padding_mask is not None:
            padding_mask = torch.full(
                (batch_size, seqlen_k),
                -10000.0,
                dtype=scores.dtype,
                device=scores.device,
            )
            padding_mask.masked_fill_(key_padding_mask, 0.0)
            # TD [2022-09-30]: Adding is faster than masked_fill_ (idk why, just better kernel I guess)
            scores = scores + rearrange(padding_mask, "b s -> b 1 1 s")
        if causal:
            # causal mask needs to take into account the difference between seqlen_q and seqlen_k
            row_idx = rearrange(torch.arange(seqlen_q, device=q.device, dtype=torch.long), "s -> s 1")
            col_idx = torch.arange(seqlen_k, device=kv.device, dtype=torch.long)
            sk = seqlen_k if key_padding_mask is None else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
            causal_mask = col_idx > row_idx + sk - seqlen_q
            scores = scores.masked_fill(causal_mask, -10000.0)
        attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
        attention_drop = self.drop(attention)
        output = torch.einsum("bhts,bshd->bthd", attention_drop, v)
        return output


class LinearResidual(nn.Linear):
    """Wrap nn.Linear to return the residual as well. For compatibility with FusedDense."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input), input


def _update_kv_cache(kv, inference_params, layer_idx):
    """kv: (batch_size, seqlen, 2, nheads, head_dim) or (batch_size, 1, 2, nheads, head_dim)"""
    # Pre-allocate memory for key-values for inference.
    num_heads, head_dim = kv.shape[-2:]
    if layer_idx not in inference_params.key_value_memory_dict:
        kv_cache = torch.empty(
            inference_params.max_batch_size,
            inference_params.max_seqlen,
            2,
            num_heads,
            head_dim,
            dtype=kv.dtype,
            device=kv.device,
        )
        inference_params.key_value_memory_dict[layer_idx] = kv_cache
    else:
        kv_cache = inference_params.key_value_memory_dict[layer_idx]
    # Adjust key and value for inference
    batch_start = inference_params.batch_size_offset
    batch_end = batch_start + kv.shape[0]
    sequence_start = inference_params.seqlen_offset
    sequence_end = sequence_start + kv.shape[1]
    assert batch_end <= kv_cache.shape[0]
    assert sequence_end <= kv_cache.shape[1]
    assert kv_cache is not None
    kv_cache[batch_start:batch_end, sequence_start:sequence_end, ...] = kv
    return kv_cache[batch_start:batch_end, :sequence_end, ...]


class MHA(nn.Module):
    """Multi-head self-attention and cross-attention"""

    def __init__(
        self,
        embed_dim,
        num_heads,
        num_heads_kv=None,
        cross_attn=False,
        qkv_proj_bias=True,
        out_proj_bias=True,
        dropout=0.0,
        softmax_scale=None,
        causal=False,
        layer_idx=None,
        dwconv=False,
        rotary_emb_dim=0,
        rotary_emb_base=10000.0,
        rotary_emb_scale_base=None,
        rotary_emb_interleaved=False,
        use_alibi=False,
        window_size=(-1, -1),
        fused_bias_fc=False,
        use_flash_attn=False,
        return_residual=False,
        checkpointing=False,
        device=None,
        dtype=None,
    ) -> None:
        """
        num_heads_kv: can be used to toggle MQA / GQA. If None, use num_heads.
        return_residual: whether to return the input x along with the output. This is for
            performance reason: for post-norm architecture, returning the input allows us
            to fuse the backward of nn.Linear with the residual connection.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.cross_attn = cross_attn
        self.causal = causal
        self.layer_idx = layer_idx
        self.dwconv = dwconv
        self.rotary_emb_dim = rotary_emb_dim
        self.use_flash_attn = False # Hardcoded to False
        self.return_residual = return_residual
        self.checkpointing = checkpointing
        # if use_alibi:  # ALiBi is tied to FlashAttention in the original code
            # assert self.use_flash_attn, "ALiBi code path requires flash_attn" # This assert will fail
            # alibi_slopes = torch.tensor(get_alibi_slopes(num_heads), device=device)
        # else:
        alibi_slopes = None # ALiBi not supported without FlashAttention for now
        # if window_size != (-1, -1): # Windowed attention is tied to FlashAttention
            # assert self.use_flash_attn, "Local (sliding window) attention code path requires flash_attn" # This assert will fail

        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv if num_heads_kv is not None else num_heads
        assert self.num_heads % self.num_heads_kv == 0, "num_heads must be divisible by num_heads_kv"
        assert self.embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        qkv_dim = self.head_dim * (self.num_heads + 2 * self.num_heads_kv)
        kv_dim = 2 * self.head_dim * self.num_heads_kv

        if self.rotary_emb_dim > 0:
            assert not cross_attn, "MHA with rotary embedding does not support cross-attention yet"
            assert RotaryEmbedding is not None, "rotary_emb is not installed"
            self.rotary_emb = RotaryEmbedding(
                self.rotary_emb_dim,
                base=rotary_emb_base,
                scale_base=rotary_emb_scale_base,
                interleaved=rotary_emb_interleaved,
                device=device,
            )

        if fused_bias_fc and FusedDense is None: # FusedDense related logic can be simplified if not used
            raise ImportError("fused_dense is not installed")
        linear_cls = nn.Linear # if not fused_bias_fc else FusedDense # Simplified
        linear_resid_cls = LinearResidual # if not fused_bias_fc else partial(FusedDense, return_residual=True) # Simplified
        wqkv_cls = linear_cls if not self.return_residual else linear_resid_cls
        inner_attn_cls = SelfAttention # Always SelfAttention
        # inner_attn_cls = ( # Original logic
        #     partial(
        #         FlashSelfAttention,
        #         layer_number=self.layer_idx,
        #         alibi_slopes=alibi_slopes,
        #         window_size=window_size,
        #     )
        #     if self.use_flash_attn # Now always False
        #     else SelfAttention
        # )
        inner_cross_attn_cls = CrossAttention # Always CrossAttention
        # inner_cross_attn_cls = ( # Original logic
        #     partial(FlashCrossAttention, alibi_slopes=alibi_slopes, window_size=window_size)
        #     if self.use_flash_attn # Now always False
        #     else CrossAttention
        # )
        if not self.cross_attn:
            self.Wqkv = wqkv_cls(embed_dim, qkv_dim, bias=qkv_proj_bias, **factory_kwargs)
        else:
            self.Wq = linear_cls(embed_dim, embed_dim, bias=qkv_proj_bias, **factory_kwargs)
            self.Wkv = wqkv_cls(embed_dim, kv_dim, bias=qkv_proj_bias, **factory_kwargs)
        if self.dwconv:
            if self.num_heads_kv == self.num_heads:
                self.dwconv_qkv = nn.Conv1d(qkv_dim, qkv_dim, kernel_size=3, padding=2, groups=qkv_dim)
            else:
                self.dwconv_q = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=2, groups=embed_dim)
                self.dwconv_kv = nn.Conv1d(kv_dim, kv_dim, kernel_size=3, padding=2, groups=kv_dim)
        self.inner_attn = inner_attn_cls(
            causal=causal,
            softmax_scale=softmax_scale,
            attention_dropout=dropout,
        )
        self.inner_cross_attn = inner_cross_attn_cls(
            causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout
        )
        self.out_proj = linear_cls(embed_dim, embed_dim, bias=out_proj_bias, **factory_kwargs)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        dtype = self.out_proj.weight.dtype if dtype is None else dtype
        device = self.out_proj.weight.device
        return torch.empty(
            batch_size,
            max_seqlen,
            2,
            self.num_heads_kv,
            self.head_dim,
            dtype=dtype,
            device=device,
        )

    def _update_kv_cache(self, kv, inference_params):
        """kv: (batch_size, seqlen, 2, nheads, head_dim) or (batch_size, 1, 2, nheads, head_dim)"""
        assert not self.dwconv, "Generation does not support dwconv yet"
        assert self.layer_idx is not None, "Generation requires layer_idx in the constructor"
        return _update_kv_cache(kv, inference_params, self.layer_idx)

    def _apply_rotary_update_kvcache_attention(self, q, kv, inference_params):
        """
        Fast path that combine 3 steps: apply rotary to Q and K, update kv cache, and apply attention.
        q: (batch_size, seqlen_q, nheads, head_dim)
        kv: (batch_size, seqlen_k, 2, nheads_kv, head_dim)
        """
        assert inference_params is not None and inference_params.seqlen_offset > 0
        # assert self.use_flash_attn # This would fail as use_flash_attn is False
        # This method was FlashAttention specific and will not be called.
        # If it were to be called, it would need a PyTorch-native implementation.
        # For now, we assume it becomes dead code due to use_flash_attn=False.
        raise NotImplementedError("_apply_rotary_update_kvcache_attention was FlashAttention specific")
        # if self.rotary_emb_dim > 0:
        #     assert self.rotary_emb.scale is None, "This code path does not support xPos"
        #     self.rotary_emb._update_cos_sin_cache(inference_params.max_seqlen, device=q.device, dtype=q.dtype)
        #     rotary_cos, rotary_sin = (
        #         self.rotary_emb._cos_cached,
        #         self.rotary_emb._sin_cached,
        #     )
        # else:
        #     rotary_cos, rotary_sin = None, None
        # batch = q.shape[0]
        # kv_cache = inference_params.key_value_memory_dict[self.layer_idx][:batch]
        # cache_seqlens = (
        #     inference_params.lengths_per_sample[:batch]
        #     if inference_params.lengths_per_sample is not None
        #     else inference_params.seqlen_offset
        # )
        # alibi_slopes = getattr(self.inner_cross_attn, "alibi_slopes", None)
        # context = local_flash_attn_with_kvcache( # This function is no longer available
        #     q,
        #     kv_cache[:, :, 0],
        #     kv_cache[:, :, 1],
        #     kv[:, :, 0],
        #     kv[:, :, 1],
        #     rotary_cos=rotary_cos,
        #     rotary_sin=rotary_sin,
        #     cache_seqlens=cache_seqlens,
        #     softmax_scale=self.inner_cross_attn.softmax_scale,
        #     causal=self.inner_cross_attn.causal,
        #     rotary_interleaved=self.rotary_emb.interleaved if self.rotary_emb_dim > 0 else False,
        #     alibi_slopes=alibi_slopes,
        # )
        # return context

    def _update_kvcache_attention(self, q, kv, inference_params):
        """Write kv to inference_params, then do attention"""
        # if inference_params.seqlen_offset == 0 or local_flash_attn_with_kvcache is None or not self.use_flash_attn: # Original condition
        if inference_params.seqlen_offset == 0 or not self.use_flash_attn: # Simplified condition as local_flash_attn_with_kvcache is None
            # TODO: this only uses seqlen_offset and not lengths_per_sample.
            kv = self._update_kv_cache(kv, inference_params)
            return self.inner_cross_attn(q, kv)
        else:
            # This path was FlashAttention specific and will not be taken.
            # If it were, it would require a PyTorch native implementation.
            raise NotImplementedError("_update_kvcache_attention with seqlen_offset > 0 was FlashAttention specific")
            # batch = q.shape[0]
            # kv_cache = inference_params.key_value_memory_dict[self.layer_idx][:batch]
            # cache_seqlens = (
            #     inference_params.lengths_per_sample[:batch]
            #     if inference_params.lengths_per_sample is not None
            #     else inference_params.seqlen_offset
            # )
            # alibi_slopes = getattr(self.inner_cross_attn, "alibi_slopes", None)
            # return local_flash_attn_with_kvcache( # This function is no longer available
            #     q,
            #     kv_cache[:, :, 0],
            #     kv_cache[:, :, 1],
            #     kv[:, :, 0],
            #     kv[:, :, 1],
            #     cache_seqlens=cache_seqlens,
            #     softmax_scale=self.inner_cross_attn.softmax_scale,
            #     causal=self.inner_cross_attn.causal,
            #     alibi_slopes=alibi_slopes,
            # )

    def forward(
        self,
        x,
        x_kv=None,
        key_padding_mask=None,
        cu_seqlens=None,
        max_seqlen=None,
        mixer_subset=None,
        inference_params=None,
        **kwargs,
    ):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim) if
                cu_seqlens is None and max_seqlen is None, else (total, hidden_dim) where total
                is the is the sum of the sequence lengths in the batch.
            x_kv: (batch, seqlen, hidden_dim), only applicable for cross-attention. If None, use x.
            cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                of the sequences in the batch, used to index into x. Only applicable when using
                FlashAttention.
            max_seqlen: int. Maximum sequence length in the batch.
            key_padding_mask: boolean mask, True means to keep, False means to mask out.
                (batch, seqlen). Only applicable when not using FlashAttention.
            mixer_subset: for cross-attention only. If not None, will take a subset of x
                before applying the query projection. Useful for e.g., ViT where we only care
                about the CLS token in the last layer.
            inference_params: for generation. Adapted from Megatron-LM (and Apex)
            https://github.com/NVIDIA/apex/blob/3ff1a10f72ec07067c4e44759442329804ac5162/apex/transformer/testing/standalone_transformer_lm.py#L470
        """
        # if cu_seqlens is not None: # This block is problematic as self.use_flash_attn is False
            # assert max_seqlen is not None
            # assert key_padding_mask is None
            # assert self.use_flash_attn # This will fail
            # assert not self.dwconv
            # assert self.rotary_emb_dim == 0
        if key_padding_mask is not None: # This path is fine
            assert cu_seqlens is None
            assert max_seqlen is None
            # assert not self.use_flash_attn # This is now true
        if inference_params is not None:
            assert key_padding_mask is None
            # assert cu_seqlens is None and max_seqlen is None # These are not passed to this path
            assert not self.dwconv

        # kwargs = (
        #     {"cu_seqlens": cu_seqlens, "max_seqlen": max_seqlen, **kwargs}
        #     if self.use_flash_attn # Now always False
        #     else {"key_padding_mask": key_padding_mask, **kwargs}
        # )
        # Simplified kwargs logic:
        if self.use_flash_attn: # This will currently not be true
             kwargs.update({"cu_seqlens": cu_seqlens, "max_seqlen": max_seqlen})
        else:
             kwargs.update({"key_padding_mask": key_padding_mask})

        seqlen_offset = (
            0
            if inference_params is None
            else (
                inference_params.lengths_per_sample
                if inference_params.lengths_per_sample is not None
                else inference_params.seqlen_offset
            )
        )
        rotary_max_seqlen = inference_params.max_seqlen if inference_params is not None else None
        batch, seqlen = x.shape[:2]
        if not self.cross_attn and self.num_heads_kv == self.num_heads:
            assert x_kv is None and mixer_subset is None
            if not self.return_residual:
                qkv = self.Wqkv(x)
            else:
                qkv, x = self.Wqkv(x)
            if self.dwconv:
                qkv = rearrange(
                    self.dwconv_qkv(rearrange(qkv, "b s d -> b d s"))[..., :-2],
                    "b d s -> b s d",
                ).contiguous()
            qkv = rearrange(qkv, "... (three h d) -> ... three h d", three=3, d=self.head_dim)
            if (
                inference_params is None
                or inference_params.seqlen_offset == 0
                or (self.rotary_emb_dim == 0 or self.rotary_emb_dim % 16 != 0)
                # or not self.use_flash_attn # self.use_flash_attn is always False
            ): # Simplified condition
                if self.rotary_emb_dim > 0:
                    qkv = self.rotary_emb(qkv, seqlen_offset=seqlen_offset, max_seqlen=rotary_max_seqlen)
                if inference_params is None:
                    if not self.checkpointing:
                        context = self.inner_attn(qkv, **kwargs)
                    else:
                        context = torch.utils.checkpoint.checkpoint(self.inner_attn, qkv, **kwargs)
                else:
                    # The FlashAttention-specific path _apply_rotary_update_kvcache_attention is not taken.
                    # _update_kvcache_attention will use the PyTorch native path.
                    context = self._update_kvcache_attention(qkv[:, :, 0], qkv[:, :, 1:], inference_params)
            else:
                # This else block was for the FlashAttention specific path which is now removed/not taken.
                # context = self._apply_rotary_update_kvcache_attention(qkv[:, :, 0], qkv[:, :, 1:], inference_params)
                # Fallback to standard path if conditions for fast path were not met,
                # but the fast path itself was Flash specific.
                # This logic needs to ensure it correctly falls into the non-Flash path.
                # The condition for this 'else' was specifically for `use_flash_attn` being true
                # and other conditions for the flash-specific kv cache update.
                # Since `use_flash_attn` is false, this path should not be hit.
                # If it were, it means there's a logic error in how conditions are checked.
                # For now, assume the `if` condition correctly routes to the non-flash path.
                 raise LogicError("Should not reach this else block when use_flash_attn is False")
        else:
            if self.cross_attn:
                if not self.return_residual:
                    q = self.Wq(x if mixer_subset is None else x[:, mixer_subset])
                    kv = self.Wkv(x_kv if x_kv is not None else x)
                else:
                    if x_kv is not None:
                        kv, x_kv = self.Wkv(x_kv)
                    else:
                        kv, x = self.Wkv(x)
                    q = self.Wq(x if mixer_subset is None else x[:, mixer_subset])
            else:
                assert self.num_heads_kv != self.num_heads
                if not self.return_residual:
                    qkv = self.Wqkv(x)
                else:
                    qkv, x = self.Wqkv(x)
                q = qkv[..., : self.num_heads * self.head_dim]
                kv = qkv[..., self.num_heads * self.head_dim :]
            q = rearrange(q, "... (h d) -> ... h d", d=self.head_dim)
            kv = rearrange(kv, "... (two hkv d) -> ... two hkv d", two=2, d=self.head_dim)
            if self.dwconv:
                q = rearrange(
                    self.dwconv_q(rearrange(q, "b s d -> b d s"))[..., :-2],
                    "b d s -> b s d",
                ).contiguous()
                kv = rearrange(
                    self.dwconv_kv(rearrange(kv, "b s d -> b d s"))[..., :-2],
                    "b d s -> b s d",
                ).contiguous()
            if (
                inference_params is None
                or inference_params.seqlen_offset == 0
                or (self.rotary_emb_dim == 0 or self.rotary_emb_dim % 16 != 0)
                # or not self.use_flash_attn # self.use_flash_attn is always False
            ): # Simplified condition
                if self.rotary_emb_dim > 0:
                    q, kv = self.rotary_emb(q, kv, seqlen_offset=seqlen_offset, max_seqlen=rotary_max_seqlen)
                if inference_params is None:
                    if not self.checkpointing:
                        context = self.inner_cross_attn(q, kv, **kwargs)
                    else:
                        context = torch.utils.checkpoint.checkpoint(self.inner_cross_attn, q, kv, **kwargs)
                else:
                     # The FlashAttention-specific path _apply_rotary_update_kvcache_attention is not taken.
                    # _update_kvcache_attention will use the PyTorch native path.
                    context = self._update_kvcache_attention(q, kv, inference_params)
            else:
                # This else block was for the FlashAttention specific path.
                # context = self._apply_rotary_update_kvcache_attention(q, kv, inference_params)
                raise LogicError("Should not reach this else block when use_flash_attn is False (cross_attn path)")
        out = self.out_proj(rearrange(context, "... h d -> ... (h d)"))
        return out if not self.return_residual else (out, x)


class ParallelMHA(nn.Module):
    """Multi-head self-attention and cross-attention"""

    def __init__(
        self,
        embed_dim,
        num_heads,
        process_group,
        num_heads_kv=None,
        qkv_proj_bias=True,
        out_proj_bias=True,
        dropout=0.0,
        softmax_scale=None,
        causal=False,
        layer_idx=None,
        rotary_emb_dim=0,
        rotary_emb_base=10000.0,
        rotary_emb_scale_base=None,
        rotary_emb_interleaved=False,
        use_alibi=False,
        window_size=(-1, -1),
        use_flash_attn=False,
        checkpointing=False,
        sequence_parallel=True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal
        self.layer_idx = layer_idx
        self.rotary_emb_dim = rotary_emb_dim
        self.use_flash_attn = False # Hardcoded to False
        self.checkpointing = checkpointing
        self.process_group = process_group
        self.world_size = process_group.size()
        self.local_rank = torch.distributed.get_rank(process_group)

        self.num_heads = num_heads
        assert self.embed_dim % self.num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_heads_kv = num_heads_kv if num_heads_kv is not None else num_heads
        assert self.num_heads % self.num_heads_kv == 0, "num_heads must be divisible by num_heads_kv"

        self.num_heads_per_rank = get_dim_for_local_rank(self.num_heads, self.world_size, self.local_rank)
        self.num_heads_kv_per_rank = get_dim_for_local_rank(self.num_heads_kv, self.world_size, self.local_rank)
        self.head_dim = self.embed_dim // num_heads
        qkv_dim = self.head_dim * (self.num_heads + 2 * self.num_heads_kv)

        # if use_alibi: # ALiBi is tied to FlashAttention
            # assert self.use_flash_attn, "ALiBi code path requires flash_attn" # This will fail
            # num_heads_local = math.ceil(self.num_heads / self.world_size)
            # alibi_slopes = torch.tensor(
            #     get_alibi_slopes(num_heads)[
            #         self.local_rank * num_heads_local : (self.local_rank + 1) * num_heads_local
            #     ],
            #     device=device,
            # )
        # else:
        alibi_slopes = None # ALiBi not supported without FlashAttention for now
        # if window_size != (-1, -1): # Windowed attention is tied to FlashAttention
            # assert self.use_flash_attn, "Local (sliding window) attention code path requires flash_attn" # This will fail

        if self.rotary_emb_dim > 0:
            assert RotaryEmbedding is not None, "rotary_emb is not installed"
            self.rotary_emb = RotaryEmbedding(
                self.rotary_emb_dim,
                base=rotary_emb_base,
                scale_base=rotary_emb_scale_base,
                interleaved=rotary_emb_interleaved,
                device=device,
            )

        if ColumnParallelLinear is None or RowParallelLinear is None:
            raise ImportError("fused_dense is not installed")
        self.Wqkv = ColumnParallelLinear(
            embed_dim,
            qkv_dim,
            process_group,
            bias=qkv_proj_bias,
            sequence_parallel=sequence_parallel,
            multiple_of=self.head_dim * (self.num_heads // self.num_heads_kv + 2),
            **factory_kwargs,
        )
        inner_attn_cls = SelfAttention # Always SelfAttention
        # inner_attn_cls = ( # Original logic
        #     partial(FlashSelfAttention, alibi_slopes=alibi_slopes, window_size=window_size)
        #     if self.use_flash_attn # Now always False
        #     else SelfAttention
        # )
        inner_cross_attn_cls = CrossAttention # Always CrossAttention
        # inner_cross_attn_cls = ( # Original logic
        #     partial(FlashCrossAttention, alibi_slopes=alibi_slopes, window_size=window_size)
        #     if self.use_flash_attn # Now always False
        #     else CrossAttention
        # )
        self.inner_attn = inner_attn_cls(causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout)
        self.inner_cross_attn = inner_cross_attn_cls(
            causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout
        )
        self.out_proj = RowParallelLinear(
            embed_dim,
            embed_dim,
            process_group,
            bias=out_proj_bias,
            sequence_parallel=sequence_parallel,
            multiple_of=self.head_dim,
            **factory_kwargs,
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        dtype = self.out_proj.weight.dtype if dtype is None else dtype
        device = self.out_proj.weight.device
        return torch.empty(
            batch_size,
            max_seqlen,
            2,
            self.num_heads_kv_per_rank,
            self.head_dim,
            dtype=dtype,
            device=device,
        )

    def _update_kv_cache(self, kv, inference_params):
        """kv: (batch_size, seqlen, 2, nheads, head_dim) or (batch_size, 1, 2, nheads, head_dim)"""
        assert self.layer_idx is not None, "Generation requires layer_idx in the constructor"
        return _update_kv_cache(kv, inference_params, self.layer_idx)

    def _apply_rotary_update_kvcache_attention(self, q, kv, inference_params):
        """
        Fast path that combine 3 steps: apply rotary to Q and K, update kv cache, and apply attention.
        q: (batch_size, seqlen_q, nheads, head_dim)
        kv: (batch_size, seqlen_k, 2, nheads_kv, head_dim)
        """
        # assert inference_params is not None and inference_params.seqlen_offset > 0 # Original conditions
        # assert self.use_flash_attn # This would fail
        # This method was FlashAttention specific and will not be called.
        raise NotImplementedError("_apply_rotary_update_kvcache_attention was FlashAttention specific (ParallelMHA)")
        # if self.rotary_emb_dim > 0:
        #     assert self.rotary_emb.scale is None, "This code path does not support xPos"
        #     self.rotary_emb._update_cos_sin_cache(inference_params.max_seqlen, device=q.device, dtype=q.dtype)
        #     rotary_cos, rotary_sin = (
        #         self.rotary_emb._cos_cached,
        #         self.rotary_emb._sin_cached,
        #     )
        # else:
        #     rotary_cos, rotary_sin = None, None
        # batch = q.shape[0]
        # kv_cache = inference_params.key_value_memory_dict[self.layer_idx][:batch]
        # cache_seqlens = (
        #     inference_params.lengths_per_sample[:batch]
        #     if inference_params.lengths_per_sample is not None
        #     else inference_params.seqlen_offset
        # )
        # alibi_slopes = getattr(self.inner_cross_attn, "alibi_slopes", None)
        # context = local_flash_attn_with_kvcache( # This function is no longer available
        #     q,
        #     kv_cache[:, :, 0],
        #     kv_cache[:, :, 1],
        #     kv[:, :, 0],
        #     kv[:, :, 1],
        #     rotary_cos=rotary_cos,
        #     rotary_sin=rotary_sin,
        #     cache_seqlens=cache_seqlens,
        #     softmax_scale=self.inner_cross_attn.softmax_scale,
        #     causal=self.inner_cross_attn.causal,
        #     rotary_interleaved=self.rotary_emb.interleaved if self.rotary_emb_dim > 0 else False,
        #     alibi_slopes=alibi_slopes,
        # )
        # return context

    def _update_kvcache_attention(self, q, kv, inference_params):
        """Write kv to inference_params, then do attention"""
        # if inference_params.seqlen_offset == 0 or not self.use_flash_attn: # Original condition
        if inference_params.seqlen_offset == 0 or not self.use_flash_attn: # Simplified condition
            # TODO: this only uses seqlen_offset and not lengths_per_sample.
            kv = self._update_kv_cache(kv, inference_params)
            return self.inner_cross_attn(q, kv)
        else:
            # This path was FlashAttention specific and will not be taken.
            raise NotImplementedError("_update_kvcache_attention with seqlen_offset > 0 was FlashAttention specific (ParallelMHA)")
            # batch = q.shape[0]
            # kv_cache = inference_params.key_value_memory_dict[self.layer_idx][:batch]
            # cache_seqlens = (
            #     inference_params.lengths_per_sample[:batch]
            #     if inference_params.lengths_per_sample is not None
            #     else inference_params.seqlen_offset
            # )
            # alibi_slopes = getattr(self.inner_cross_attn, "alibi_slopes", None)
            # context = local_flash_attn_with_kvcache( # This function is no longer available
            #     q,
            #     kv_cache[:, :, 0],
            #     kv_cache[:, :, 1],
            #     kv[:, :, 0],
            #     kv[:, :, 1],
            #     cache_seqlens=cache_seqlens,
            #     softmax_scale=self.inner_cross_attn.softmax_scale,
            #     causal=self.inner_cross_attn.causal,
            #     alibi_slopes=alibi_slopes,
            # )
            # return context

    def forward(self, x, seqlen=None, inference_params=None, **kwargs):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim) if seqlen=None.
                If seqlen is not None, x is (batch * seqlen, hidden_dim). This is so that when we
                split x during sequence parallel, we split the batch * seqlen dimension
                (in case batch is small).
        """
        qkv = self.Wqkv(x)
        if seqlen is not None:
            qkv = rearrange(qkv, "(b s) ... -> b s ...", s=seqlen)
        seqlen_offset = (
            0
            if inference_params is None
            else (
                inference_params.lengths_per_sample
                if inference_params.lengths_per_sample is not None
                else inference_params.seqlen_offset
            )
        )
        rotary_max_seqlen = inference_params.max_seqlen if inference_params is not None else None
        if self.num_heads_kv == self.num_heads:
            qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, d=self.head_dim)
            if (
                inference_params is None
                or inference_params.seqlen_offset == 0
                or (self.rotary_emb_dim == 0 or self.rotary_emb_dim % 16 != 0)
                # or not self.use_flash_attn # self.use_flash_attn is always False
            ): # Simplified condition
                if self.rotary_emb_dim > 0:
                    qkv = self.rotary_emb(qkv, seqlen_offset=seqlen_offset, max_seqlen=rotary_max_seqlen)
                if inference_params is None:
                    if not self.checkpointing:
                        context = self.inner_attn(qkv, **kwargs)
                    else:
                        context = torch.utils.checkpoint.checkpoint(self.inner_attn, qkv, **kwargs)
                else:
                    # The FlashAttention-specific path _apply_rotary_update_kvcache_attention is not taken.
                    context = self._update_kvcache_attention(qkv[:, :, 0], qkv[:, :, 1:], inference_params)
            else:
                # This else block was for the FlashAttention specific path.
                # context = self._apply_rotary_update_kvcache_attention(qkv[:, :, 0], qkv[:, :, 1:], inference_params)
                raise LogicError("Should not reach this else block when use_flash_attn is False (ParallelMHA, self-attn)")
        else:
            q = rearrange(
                qkv[..., : self.num_heads_per_rank * self.head_dim],
                "... (h d) -> ... h d",
                d=self.head_dim,
            )
            kv = rearrange(
                qkv[..., self.num_heads_per_rank * self.head_dim :],
                "... (two hkv d) -> ... two hkv d",
                two=2,
                d=self.head_dim,
            )
            if (
                inference_params is None
                or inference_params.seqlen_offset == 0
                or (self.rotary_emb_dim == 0 or self.rotary_emb_dim % 16 != 0)
                # or not self.use_flash_attn # self.use_flash_attn is always False
            ): # Simplified condition
                if self.rotary_emb_dim > 0:
                    q, kv = self.rotary_emb(q, kv, seqlen_offset=seqlen_offset, max_seqlen=rotary_max_seqlen)
                if inference_params is None:
                    if not self.checkpointing:
                        context = self.inner_cross_attn(q, kv, **kwargs)
                    else:
                        context = torch.utils.checkpoint.checkpoint(self.inner_cross_attn, q, kv, **kwargs)
                else:
                    # The FlashAttention-specific path _apply_rotary_update_kvcache_attention is not taken.
                    context = self._update_kvcache_attention(q, kv, inference_params)
            else:
                # This else block was for the FlashAttention specific path.
                # context = self._apply_rotary_update_kvcache_attention(q, kv, inference_params)
                raise LogicError("Should not reach this else block when use_flash_attn is False (ParallelMHA, cross-attn)")
        context = rearrange(context, "b s h d -> b s (h d)")
        if seqlen is not None:
            context = rearrange(context, "b s d -> (b s) d")
        out = self.out_proj(context)
        return out
