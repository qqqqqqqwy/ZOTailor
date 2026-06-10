import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

try:
    from .llama import (
        apply_rotary_emb,
        repeat_kv,
    )
except ImportError:
    from llama import (  # type: ignore[no-redef]
        apply_rotary_emb,
        repeat_kv,
    )


def _lookup_config_value(
    config: Dict[str, Any],
    keys: Tuple[str, ...],
    *,
    default: Any = None,
    required: bool = False,
) -> Any:
    for key in keys:
        value = config.get(key)
        if value is not None:
            return value

    for nested_key in ("text_config", "llm_config", "language_config", "model_config"):
        nested = config.get(nested_key)
        if not isinstance(nested, dict):
            continue
        for key in keys:
            value = nested.get(key)
            if value is not None:
                return value

    if required:
        available = ", ".join(sorted(str(key) for key in config.keys()))
        raise KeyError(
            f"Missing required config field; tried {keys}. "
            f"Available top-level keys: {available}"
        )
    return default


@dataclass(frozen=True)
class QwenModelArgs:
    model_type: str
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    vocab_size: int
    intermediate_size: int
    norm_eps: float
    rope_theta: float
    max_seq_len: int
    rope_scaling: Optional[Dict[str, Any]]
    tie_word_embeddings: bool
    attention_bias: bool
    mlp_bias: bool
    use_qk_norm: bool
    rotary_percentage: float = 1.0

    @classmethod
    def from_hf_config(
        cls,
        config: Dict[str, Any],
        *,
        min_seq_len: int,
    ) -> "QwenModelArgs":
        model_type = str(config.get("model_type", ""))
        supported_model_types = {"qwen2", "qwen3", "qwen3_5"}
        if model_type not in supported_model_types:
            raise ValueError(
                "Expected HF config model_type qwen2/qwen3/qwen3_5, "
                f"got {model_type!r}"
            )
        if any(config.get(key) not in (None, 0) for key in ("num_experts", "moe_intermediate_size")):
            raise ValueError("MoE Qwen checkpoints are not supported by this dense wrapper")
        max_position_embeddings = int(
            _lookup_config_value(
                config,
                (
                    "max_position_embeddings",
                    "max_seq_len",
                    "max_sequence_length",
                    "seq_length",
                    "n_positions",
                    "context_length",
                ),
                default=min_seq_len,
            )
        )
        dim = int(
            _lookup_config_value(
                config,
                ("hidden_size", "dim", "n_embd", "embedding_size"),
                required=True,
            )
        )
        n_heads = int(
            _lookup_config_value(
                config,
                ("num_attention_heads", "n_heads", "num_heads", "n_head"),
                required=True,
            )
        )
        head_dim = int(
            _lookup_config_value(config, ("head_dim", "kv_channels"), default=dim // n_heads)
        )
        n_layers = int(
            _lookup_config_value(
                config,
                ("num_hidden_layers", "n_layers", "num_layers", "n_layer"),
                required=True,
            )
        )
        n_kv_heads = int(
            _lookup_config_value(
                config,
                (
                    "num_key_value_heads",
                    "n_kv_heads",
                    "num_kv_heads",
                    "kv_heads",
                    "multi_query_group_num",
                ),
                default=n_heads,
            )
        )
        vocab_size = int(
            _lookup_config_value(config, ("vocab_size", "padded_vocab_size"), required=True)
        )
        intermediate_size = int(
            _lookup_config_value(
                config,
                ("intermediate_size", "hidden_dim", "ffn_hidden_size", "ffn_dim", "n_inner"),
                required=True,
            )
        )
        return cls(
            model_type=model_type,
            dim=dim,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            vocab_size=vocab_size,
            intermediate_size=intermediate_size,
            norm_eps=float(
                _lookup_config_value(
                    config,
                    ("rms_norm_eps", "norm_eps", "layer_norm_eps", "layer_norm_epsilon"),
                    default=1.0e-5,
                )
            ),
            rope_theta=float(
                _lookup_config_value(
                    config,
                    ("rope_theta", "rope_base", "rotary_emb_base"),
                    default=10000.0,
                )
            ),
            max_seq_len=max(max_position_embeddings, min_seq_len),
            rope_scaling=_lookup_config_value(config, ("rope_scaling",), default=None),
            tie_word_embeddings=bool(
                _lookup_config_value(config, ("tie_word_embeddings",), default=False)
            ),
            attention_bias=bool(
                _lookup_config_value(config, ("attention_bias", "bias"), default=False)
            ),
            mlp_bias=bool(_lookup_config_value(config, ("mlp_bias",), default=False)),
            use_qk_norm=bool(
                _lookup_config_value(
                    config,
                    ("qk_layernorm",),
                    default=_lookup_config_value(
                        config,
                        ("use_qk_norm",),
                        default=model_type in {"qwen3", "qwen3_5"},
                    ),
                )
            ),
            rotary_percentage=float(
                _lookup_config_value(
                    config,
                    ("rotary_percentage", "partial_rotary_factor"),
                    default=1.0,
                )
            ),
        )


def _qwen_inv_freq(
    rotary_dim: int,
    theta: float,
    rope_scaling: Optional[Dict[str, Any]],
    *,
    device: torch.device | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    inv_freq = 1.0 / (
        theta
        ** (torch.arange(0, rotary_dim, 2, device=device, dtype=dtype) / rotary_dim)
    )
    if rope_scaling is None:
        return inv_freq

    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
    if rope_type is None:
        return inv_freq
    if rope_type == "linear":
        return inv_freq / float(rope_scaling["factor"])
    if rope_type in {"dynamic", "yarn", "default"}:
        return inv_freq
    raise ValueError(f"Unsupported Qwen RoPE scaling type: {rope_type!r}")


def build_qwen_rope_cos_sin(
    *,
    rotary_dim: int,
    seq_len: int,
    theta: float,
    rope_scaling: Optional[Dict[str, Any]],
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if rotary_dim % 2 != 0:
        raise ValueError("rotary_dim must be even")
    inv_freq = _qwen_inv_freq(
        rotary_dim,
        theta,
        rope_scaling,
        device=device,
        dtype=dtype,
    )
    positions = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.outer(positions, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


class StableRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        normed = x_fp32 * torch.rsqrt(
            x_fp32.pow(2).mean(-1, keepdim=True) + self.eps
        )
        return normed.to(dtype=x.dtype) * self.weight


@dataclass(frozen=True)
class LoraBSpec:
    name: str
    out_features: int
    rank: int
    offset: int

    @property
    def numel(self) -> int:
        return self.out_features * self.rank


class LoraBLayout:
    def __init__(self) -> None:
        self._specs: List[LoraBSpec] = []
        self._total_numel = 0

    def add(self, name: str, out_features: int, rank: int) -> LoraBSpec:
        spec = LoraBSpec(
            name=name,
            out_features=out_features,
            rank=rank,
            offset=self._total_numel,
        )
        self._specs.append(spec)
        self._total_numel += spec.numel
        return spec

    @property
    def total_numel(self) -> int:
        return self._total_numel

    @property
    def specs(self) -> Tuple[LoraBSpec, ...]:
        return tuple(self._specs)

    def to_jsonable(self) -> List[Dict[str, int | str]]:
        return [
            {
                "name": spec.name,
                "out_features": spec.out_features,
                "rank": spec.rank,
                "offset": spec.offset,
                "numel": spec.numel,
            }
            for spec in self._specs
        ]


class LoraFALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        *,
        rank: int,
        scaling: float,
        layout: LoraBLayout,
        name: str,
        generator: torch.Generator,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = scaling
        self.b_spec = layout.add(name, out_features, rank)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

        lora_a = torch.empty(rank, in_features, dtype=torch.float32)
        nn.init.kaiming_uniform_(lora_a, a=math.sqrt(5), generator=generator)
        self.register_buffer("lora_a", lora_a, persistent=True)

    def forward(self, x: torch.Tensor, lora_b_flat: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        b = lora_b_flat.narrow(0, self.b_spec.offset, self.b_spec.numel)
        b = b.view(self.out_features, self.rank)
        b = b.to(dtype=x.dtype)
        lora_hidden = F.linear(x, self.lora_a)
        lora_out = F.linear(lora_hidden, b)
        return base + lora_out * self.scaling


class AttentionLoraFA(nn.Module):
    def __init__(
        self,
        layer_id: int,
        args: QwenModelArgs,
        *,
        rank: int,
        scaling: float,
        layout: LoraBLayout,
        generator: torch.Generator,
    ) -> None:
        super().__init__()
        self.n_kv_heads = args.n_kv_heads
        if args.n_heads % self.n_kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        self.n_heads = args.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.head_dim
        self.rotary_dim = int(self.head_dim * args.rotary_percentage)
        self.q_norm = (
            StableRMSNorm(self.head_dim, eps=args.norm_eps) if args.use_qk_norm else None
        )
        self.k_norm = (
            StableRMSNorm(self.head_dim, eps=args.norm_eps) if args.use_qk_norm else None
        )

        prefix = f"layers.{layer_id}.attention"
        self.wq = LoraFALinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=args.attention_bias,
            rank=rank,
            scaling=scaling,
            layout=layout,
            name=f"{prefix}.wq",
            generator=generator,
        )
        self.wk = LoraFALinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=args.attention_bias,
            rank=rank,
            scaling=scaling,
            layout=layout,
            name=f"{prefix}.wk",
            generator=generator,
        )
        self.wv = LoraFALinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=args.attention_bias,
            rank=rank,
            scaling=scaling,
            layout=layout,
            name=f"{prefix}.wv",
            generator=generator,
        )
        self.wo = LoraFALinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=args.attention_bias,
            rank=rank,
            scaling=scaling,
            layout=layout,
            name=f"{prefix}.wo",
            generator=generator,
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor,
        lora_b_flat: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        xq = self.wq(x, lora_b_flat)
        xk = self.wk(x, lora_b_flat)
        xv = self.wv(x, lora_b_flat)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        if self.q_norm is not None:
            xq = self.q_norm(xq)
        if self.k_norm is not None:
            xk = self.k_norm(xk)

        xq, xk = apply_rotary_emb(
            xq,
            xk,
            cos=cos,
            sin=sin,
            rotary_dim=self.rotary_dim,
            interleaved=False,
        )

        keys = repeat_kv(xk, self.n_rep)
        values = repeat_kv(xv, self.n_rep)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = torch.clamp(scores.float() + mask, min=-80.0, max=80.0)
        scores = F.softmax(scores, dim=-1).to(dtype=xq.dtype)
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output, lora_b_flat)


class FeedForwardLoraFA(nn.Module):
    def __init__(
        self,
        layer_id: int,
        args: QwenModelArgs,
        *,
        rank: int,
        scaling: float,
        layout: LoraBLayout,
        generator: torch.Generator,
    ) -> None:
        super().__init__()
        dim = args.dim
        hidden_dim = args.intermediate_size

        prefix = f"layers.{layer_id}.feed_forward"
        self.w1 = LoraFALinear(
            dim,
            hidden_dim,
            bias=args.mlp_bias,
            rank=rank,
            scaling=scaling,
            layout=layout,
            name=f"{prefix}.w1",
            generator=generator,
        )
        self.w2 = LoraFALinear(
            hidden_dim,
            dim,
            bias=args.mlp_bias,
            rank=rank,
            scaling=scaling,
            layout=layout,
            name=f"{prefix}.w2",
            generator=generator,
        )
        self.w3 = LoraFALinear(
            dim,
            hidden_dim,
            bias=args.mlp_bias,
            rank=rank,
            scaling=scaling,
            layout=layout,
            name=f"{prefix}.w3",
            generator=generator,
        )

    def forward(self, x: torch.Tensor, lora_b_flat: torch.Tensor) -> torch.Tensor:
        gate = self.w1(x, lora_b_flat)
        up = self.w3(x, lora_b_flat)
        return self.w2(gate * torch.sigmoid(gate) * up, lora_b_flat)


class TransformerBlockLoraFA(nn.Module):
    def __init__(
        self,
        layer_id: int,
        args: QwenModelArgs,
        *,
        rank: int,
        scaling: float,
        layout: LoraBLayout,
        generator: torch.Generator,
    ) -> None:
        super().__init__()
        self.attention = AttentionLoraFA(
            layer_id,
            args,
            rank=rank,
            scaling=scaling,
            layout=layout,
            generator=generator,
        )
        self.feed_forward = FeedForwardLoraFA(
            layer_id,
            args,
            rank=rank,
            scaling=scaling,
            layout=layout,
            generator=generator,
        )
        self.attention_norm = StableRMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = StableRMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor,
        lora_b_flat: torch.Tensor,
    ) -> torch.Tensor:
        h = x + self.attention(
            self.attention_norm(x), cos, sin, mask, lora_b_flat
        )
        return h + self.feed_forward(self.ffn_norm(h), lora_b_flat)


class QwenLoraFAForPrefillLoss(nn.Module):
    def __init__(
        self,
        params: QwenModelArgs,
        *,
        batch_size: int = 4,
        seq_len: int = 32,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        adapter_scale: float = 1.0,
        lora_seed: int = 0,
        loss_mode: str = "full",
    ) -> None:
        super().__init__()
        if loss_mode not in {"full", "single_label"}:
            raise ValueError(f"Unsupported loss_mode: {loss_mode}")
        self.params = params
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.adapter_scale = adapter_scale
        self.loss_mode = loss_mode
        self.scaling = (lora_alpha / lora_rank) * adapter_scale
        self.layout = LoraBLayout()

        generator = torch.Generator(device="cpu")
        generator.manual_seed(lora_seed)

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.layers = nn.ModuleList(
            [
                TransformerBlockLoraFA(
                    i,
                    params,
                    rank=lora_rank,
                    scaling=self.scaling,
                    layout=self.layout,
                    generator=generator,
                )
                for i in range(params.n_layers)
            ]
        )
        self.norm = StableRMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        head_dim = params.head_dim
        rotary_dim = int(head_dim * params.rotary_percentage)
        rope_cos, rope_sin = build_qwen_rope_cos_sin(
            rotary_dim=rotary_dim,
            seq_len=params.max_seq_len,
            theta=params.rope_theta,
            rope_scaling=params.rope_scaling,
        )
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

        mask = torch.full((seq_len, seq_len), -10000.0, dtype=torch.float32)
        mask = torch.triu(mask, diagonal=1).view(1, 1, seq_len, seq_len)
        self.register_buffer("prefill_mask", mask, persistent=False)

    @property
    def lora_b_numel(self) -> int:
        return self.layout.total_numel

    def lora_b_specs_jsonable(self) -> List[Dict[str, int | str]]:
        return self.layout.to_jsonable()

    def forward(
        self,
        tokens: torch.Tensor,
        labels: torch.Tensor,
        lora_b_flat: torch.Tensor,
    ) -> torch.Tensor:
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)

        cos = self.rope_cos[:seqlen].to(dtype=h.dtype)
        sin = self.rope_sin[:seqlen].to(dtype=h.dtype)
        mask = self.prefill_mask[:, :, :seqlen, :seqlen]

        for layer in self.layers:
            h = layer(h, cos, sin, mask, lora_b_flat)

        h = self.norm(h)
        if self.loss_mode == "single_label":
            return single_label_cross_entropy_from_hidden(h, self.output.weight, labels)
        logits = F.linear(h.float(), self.output.weight.float())
        return next_token_cross_entropy(logits, labels)


def initialize_lora_a_kaiming_uniform(
    model: QwenLoraFAForPrefillLoss,
    *,
    seed: int = 42,
) -> int:
    import numpy as np

    targets: List[Tuple[str, LoraFALinear]] = []
    for layer_id, layer in enumerate(model.layers):
        targets.extend(
            [
                (f"blk.{layer_id}.attn_q.weight", layer.attention.wq),
                (f"blk.{layer_id}.attn_k.weight", layer.attention.wk),
                (f"blk.{layer_id}.attn_v.weight", layer.attention.wv),
                (f"blk.{layer_id}.attn_output.weight", layer.attention.wo),
                (f"blk.{layer_id}.ffn_gate.weight", layer.feed_forward.w1),
                (f"blk.{layer_id}.ffn_up.weight", layer.feed_forward.w3),
                (f"blk.{layer_id}.ffn_down.weight", layer.feed_forward.w2),
            ]
        )

    rng = np.random.default_rng(seed=seed)
    total_numel = 0
    with torch.no_grad():
        for _name, module in sorted(targets, key=lambda item: item[0]):
            bound = math.sqrt(6.0 / float(module.in_features))
            lora_a = rng.uniform(
                -bound,
                bound,
                size=(module.rank, module.in_features),
            ).astype(np.float16)
            source = torch.from_numpy(lora_a).to(
                device=module.lora_a.device,
                dtype=module.lora_a.dtype,
            )
            module.lora_a.copy_(source)
            total_numel += int(source.numel())
    return total_numel


initialize_lora_a_like_llama_cpp_generate_lora = initialize_lora_a_kaiming_uniform


def next_token_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits = torch.where(logits == logits, logits, torch.zeros_like(logits))
    logits = torch.clamp(logits.float(), min=-80.0, max=80.0)
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, :-1]
    valid = shift_labels.ne(-100)
    safe_labels = torch.where(valid, shift_labels, torch.zeros_like(shift_labels))

    max_logits = torch.amax(shift_logits, dim=-1, keepdim=True)
    shifted = shift_logits - max_logits
    log_probs = shifted - torch.log(torch.sum(torch.exp(shifted), dim=-1, keepdim=True))
    gathered = torch.gather(log_probs, dim=-1, index=safe_labels.unsqueeze(-1))
    nll = -gathered.squeeze(-1)

    valid_float = valid.to(dtype=logits.dtype)
    loss_sum = torch.sum(torch.where(valid, nll, torch.zeros_like(nll)))
    token_count = torch.sum(valid_float)
    return loss_sum / (token_count + 1.0e-6)


def single_label_cross_entropy_from_hidden(
    hidden: torch.Tensor,
    output_weight: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    valid = labels.ne(-100)

    hidden_fp32 = hidden.float()
    zero_hidden = torch.zeros_like(hidden_fp32)
    selected_hidden = torch.sum(
        torch.where(valid.unsqueeze(-1), hidden_fp32, zero_hidden),
        dim=1,
    )
    selected_hidden = torch.where(
        selected_hidden == selected_hidden,
        selected_hidden,
        torch.zeros_like(selected_hidden),
    )
    selected_hidden = torch.clamp(selected_hidden, min=-1000.0, max=1000.0)

    safe_labels = torch.where(valid, labels, torch.zeros_like(labels))
    selected_labels = torch.sum(safe_labels, dim=1)
    valid_float = valid.to(dtype=hidden_fp32.dtype)
    valid_rows = torch.sum(valid_float, dim=1).gt(0)
    selected_labels = torch.where(
        valid_rows, selected_labels, torch.zeros_like(selected_labels)
    )

    logits = F.linear(selected_hidden, output_weight.float())
    logits = torch.where(logits == logits, logits, torch.zeros_like(logits))
    logits = torch.clamp(logits, min=-80.0, max=80.0)
    max_logits = torch.amax(logits, dim=-1, keepdim=True)
    shifted = logits - max_logits
    log_probs = shifted - torch.log(torch.sum(torch.exp(shifted), dim=-1, keepdim=True))
    gathered = torch.gather(log_probs, dim=-1, index=selected_labels.unsqueeze(-1))
    nll = -gathered.squeeze(-1)

    valid_row_float = valid_rows.to(dtype=logits.dtype)
    loss_sum = torch.sum(torch.where(valid_rows, nll, torch.zeros_like(nll)))
    token_count = torch.sum(valid_row_float)
    return loss_sum / (token_count + 1.0e-6)


HF_TO_META_KEY_MAP = {
    "model.embed_tokens.weight": "tok_embeddings.weight",
    "model.norm.weight": "norm.weight",
    "lm_head.weight": "output.weight",
}


def hf_key_to_lora_fa_key(key: str) -> Optional[str]:
    if key in HF_TO_META_KEY_MAP:
        return HF_TO_META_KEY_MAP[key]

    parts = key.split(".")
    if len(parts) < 5 or parts[0] != "model" or parts[1] != "layers":
        return None
    layer = parts[2]
    block = parts[3]

    attn = {
        "q_proj.weight": "attention.wq.weight",
        "q_proj.bias": "attention.wq.bias",
        "k_proj.weight": "attention.wk.weight",
        "k_proj.bias": "attention.wk.bias",
        "v_proj.weight": "attention.wv.weight",
        "v_proj.bias": "attention.wv.bias",
        "o_proj.weight": "attention.wo.weight",
        "o_proj.bias": "attention.wo.bias",
        "q_norm.weight": "attention.q_norm.weight",
        "k_norm.weight": "attention.k_norm.weight",
    }
    mlp = {
        "gate_proj.weight": "feed_forward.w1.weight",
        "gate_proj.bias": "feed_forward.w1.bias",
        "down_proj.weight": "feed_forward.w2.weight",
        "down_proj.bias": "feed_forward.w2.bias",
        "up_proj.weight": "feed_forward.w3.weight",
        "up_proj.bias": "feed_forward.w3.bias",
    }
    norms = {
        "input_layernorm.weight": "attention_norm.weight",
        "post_attention_layernorm.weight": "ffn_norm.weight",
    }

    tail = ".".join(parts[4:])
    if block == "self_attn" and tail in attn:
        return f"layers.{layer}.{attn[tail]}"
    if block == "mlp" and tail in mlp:
        return f"layers.{layer}.{mlp[tail]}"
    norm_tail = ".".join(parts[3:])
    if norm_tail in norms:
        return f"layers.{layer}.{norms[norm_tail]}"
    return None


def load_hf_state_dict_into_lora_fa_model(
    model: QwenLoraFAForPrefillLoss,
    hf_state_dict: Dict[str, torch.Tensor],
    *,
    dtype: torch.dtype,
) -> Tuple[List[str], List[str]]:
    mapped: Dict[str, torch.Tensor] = {}
    ignored: List[str] = []
    model_state_keys = set(model.state_dict().keys())
    for key, tensor in hf_state_dict.items():
        target_key = hf_key_to_lora_fa_key(key)
        if target_key is None or target_key not in model_state_keys:
            ignored.append(key)
            continue
        mapped[target_key] = tensor.detach().to(dtype=dtype, device="cpu")
    if (
        "output.weight" not in mapped
        and model.params.tie_word_embeddings
        and "tok_embeddings.weight" in mapped
    ):
        mapped["output.weight"] = mapped["tok_embeddings.weight"].clone()

    missing, unexpected = model.load_state_dict(mapped, strict=False)
    expected_missing = {
        name
        for name in model.state_dict().keys()
        if name.endswith(".lora_a")
        or name in {"rope_cos", "rope_sin", "prefill_mask"}
    }
    real_missing = [name for name in missing if name not in expected_missing]
    real_unexpected = list(unexpected)
    if real_missing:
        raise RuntimeError(f"Missing mapped checkpoint keys: {real_missing[:20]}")
    if real_unexpected:
        raise RuntimeError(f"Unexpected mapped checkpoint keys: {real_unexpected[:20]}")
    return ignored, sorted(expected_missing)
