import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    rope_theta: float = 10000.0
    rotary_percentage: float = 1.0 # 控制旋转前xx维
    rope_condense_ratio: float = 1.0 # 位置压缩，1.0表示不压缩
    rope_interleaved: bool = False # 控制旋转位置编码是交错还是对半
    max_seq_len: int = 2048
    max_batch_size: int = 32
    norm_eps: float = 1e-5


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # 保持 FP32 精度进行 norm，再转回原类型
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output.type_as(x) * self.weight


def build_rope_cos_sin(rotary_dim: int, seq_len: int, theta: float = 10000.0, condense_ratio: float = 1.0, device=None, dtype=torch.float32):
    assert rotary_dim % 2 == 0, "rotary_dim must be even"
    inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, device=device, dtype=dtype) / rotary_dim))
    t = torch.arange(seq_len, device=device, dtype=dtype) / condense_ratio
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def rope_broadcast(cos_or_sin: torch.Tensor, x_slice: torch.Tensor):
    T, D2 = cos_or_sin.shape
    return cos_or_sin.view(1, T, 1, D2).to(dtype=x_slice.dtype, device=x_slice.device)


def apply_rotary_emb(
    xq: torch.Tensor, 
    xk: torch.Tensor, 
    cos: torch.Tensor, 
    sin: torch.Tensor, 
    rotary_dim: int,
    interleaved: bool,
):
    xq_rot, xq_pass = xq[..., :rotary_dim], xq[..., rotary_dim:]
    xk_rot, xk_pass = xk[..., :rotary_dim], xk[..., rotary_dim:]

    if interleaved:
        xq_1, xq_2 = xq_rot[..., ::2], xq_rot[..., 1::2]
        xk_1, xk_2 = xk_rot[..., ::2], xk_rot[..., 1::2]

        cos_b, sin_b = rope_broadcast(cos, xq_1), rope_broadcast(sin, xq_1)

        xq_rotated_1 = xq_1 * cos_b - xq_2 * sin_b
        xq_rotated_2 = xq_1 * sin_b + xq_2 * cos_b
        xk_rotated_1 = xk_1 * cos_b - xk_2 * sin_b
        xk_rotated_2 = xk_1 * sin_b + xk_2 * cos_b

        xq_rotated = torch.empty_like(xq_rot)
        xk_rotated = torch.empty_like(xk_rot)
        xq_rotated[..., ::2], xq_rotated[..., 1::2] = xq_rotated_1, xq_rotated_2
        xk_rotated[..., ::2], xk_rotated[..., 1::2] = xk_rotated_1, xk_rotated_2
    else:
        xq_1, xq_2 = xq_rot.chunk(2, dim=-1)
        xk_1, xk_2 = xk_rot.chunk(2, dim=-1)

        cos_b, sin_b = rope_broadcast(cos, xq_1), rope_broadcast(sin, xq_1)

        xq_rotated = torch.cat([xq_1 * cos_b - xq_2 * sin_b, xq_1 * sin_b + xq_2 * cos_b], dim=-1)
        xk_rotated = torch.cat([xk_1 * cos_b - xk_2 * sin_b, xk_1 * sin_b + xk_2 * cos_b], dim=-1)

    return torch.cat([xq_rotated, xq_pass], dim=-1), torch.cat([xk_rotated, xk_pass], dim=-1)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    bs, slen, n_kv_heads, head_dim = x.shape
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads = args.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.rotary_dim = int(self.head_dim * args.rotary_percentage)

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.register_buffer(
            "cache_k",
            torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)),
            persistent=False,
        )
        self.register_buffer(
            "cache_v",
            torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(
            xq, xk, cos=cos, sin=sin, rotary_dim=self.rotary_dim, interleaved=False
        )

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores + mask
        
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, ffn_dim_multiplier: Optional[float]):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, cos: torch.Tensor, sin: torch.Tensor, mask: Optional[torch.Tensor]):
        h = x + self.attention(self.attention_norm(x), start_pos, cos, sin, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        
        self.layers = torch.nn.ModuleList([TransformerBlock(i, params) for i in range(params.n_layers)])
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        head_dim = params.dim // params.n_heads
        rotary_dim = int(head_dim * params.rotary_percentage)
        rope_cos, rope_sin = build_rope_cos_sin(
            rotary_dim=rotary_dim,
            seq_len=params.max_seq_len * 2,
            theta=params.rope_theta,
            condense_ratio=params.rope_condense_ratio,
        )

        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int = 0, attention_mask: Optional[torch.Tensor] = None):
        if isinstance(start_pos, torch.Tensor):
            start_pos = start_pos.item()
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)

        pos_indices = torch.arange(seqlen, device=tokens.device) + start_pos
        cos = self.rope_cos[pos_indices].to(h.dtype)
        sin = self.rope_sin[pos_indices].to(h.dtype)

        mask = None
        if seqlen > 1:
            # 标准因果掩码 (Causal Mask)
            mask = torch.full((seqlen, seqlen), float("-inf"), device=h.device, dtype=h.dtype)
            mask = torch.triu(mask, diagonal=1)
            # 如果是分块 prefill (start_pos > 0)，需要让当前 token 可以看到之前的全部 token
            if start_pos > 0:
                mask = torch.hstack([torch.zeros((seqlen, start_pos), device=h.device, dtype=h.dtype), mask])
        
        if attention_mask is not None:
            if mask is None:
                mask = torch.zeros((seqlen, start_pos + seqlen), device=h.device, dtype=h.dtype)
            
            pad_mask = (attention_mask == 0).view(_bsz, 1, 1, seqlen)
            
            if start_pos > 0:
                past_valid = torch.zeros((_bsz, 1, 1, start_pos), device=h.device, dtype=torch.bool)
                pad_mask = torch.cat([past_valid, pad_mask], dim=-1)
                
            mask = mask.masked_fill(pad_mask, float("-inf"))

        for layer in self.layers:
            h = layer(h, start_pos, cos, sin, mask)

        h = self.norm(h)
        return self.output(h).float()