import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import torch

try:
    from transformers import AutoConfig
except ImportError as exc:
    raise RuntimeError(
        "transformers is not installed. Install transformers, safetensors, "
        "huggingface_hub, sentencepiece, and accelerate."
    ) from exc

try:
    from .export_lora_fa_llama import (
        apply_xnnpack_int4_quantization,
        export_to_executorch,
        load_hf_state_dict,
        make_dummy_batch,
        run_eager_smoke,
        run_executorch_smoke,
        setup_logging,
    )
    from .lora_fa_gemma import (
        GemmaLoraFAForPrefillLoss,
        GemmaModelArgs,
        initialize_lora_a_kaiming_uniform,
        load_hf_state_dict_into_lora_fa_model,
    )
except ImportError:
    from export_lora_fa_llama import (  # type: ignore[no-redef]
        apply_xnnpack_int4_quantization,
        export_to_executorch,
        load_hf_state_dict,
        make_dummy_batch,
        run_eager_smoke,
        run_executorch_smoke,
        setup_logging,
    )
    from lora_fa_gemma import (  # type: ignore[no-redef]
        GemmaLoraFAForPrefillLoss,
        GemmaModelArgs,
        initialize_lora_a_kaiming_uniform,
        load_hf_state_dict_into_lora_fa_model,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Gemma3 text-only LoRA-FA prefill loss model to ExecuTorch."
    )
    parser.add_argument(
        "--hf_model",
        default="/home/fuyongjian/qwy/models/gemma-3-4b-it-textonly",
        help="Local Gemma3 text-only Hugging Face model directory.",
    )
    parser.add_argument("--output_dir", default="mytest/out")
    parser.add_argument(
        "--log_path",
        default="mytest/out/gemma3_4b_textonly_export_log.txt",
        help="Write Python export/smoke logs to this text file.",
    )
    parser.add_argument(
        "--pte_name",
        default="gemma3_4b_textonly_lora_fa.pte",
        help="Output ExecuTorch program filename.",
    )
    parser.add_argument(
        "--weights_name",
        default="gemma3_4b_textonly_lora_fa_weights.ptd",
        help="Output external constants filename.",
    )
    parser.add_argument(
        "--layout_name",
        default="gemma3_4b_textonly_lora_fa_layout.json",
        help="Output LoRA layout JSON filename.",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--adapter_scale", type=float, default=1.0)
    parser.add_argument(
        "--loss_mode",
        choices=("full", "single_label"),
        default="single_label",
        help=(
            "full computes CE over every labeled token position; single_label "
            "computes the LM head only for one labeled position in each row."
        ),
    )
    parser.add_argument(
        "--lora_a_init",
        choices=("kaiming_uniform", "torch_linear", "llama_cpp", "torch"),
        default="kaiming_uniform",
        help=(
            "LoRA-A initialization. kaiming_uniform uses "
            "U(-sqrt(6/fan_in), sqrt(6/fan_in)); torch_linear keeps the "
            "constructor initialization. llama_cpp/torch are legacy aliases."
        ),
    )
    parser.add_argument("--lora_seed", type=int, default=42)
    parser.add_argument("--dummy_seed", type=int, default=1234)
    parser.add_argument(
        "--smoke_epsilon",
        type=float,
        default=1.0e-2,
        help="LoRA-B perturbation scale used by eager/ExecuTorch smoke tests.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--skip_eager_smoke",
        action="store_true",
        help="Skip the eager finite-loss smoke test before export.",
    )
    parser.add_argument(
        "--run_executorch_smoke",
        action="store_true",
        help="Load the exported .pte/.ptd with Python pybindings and run one forward.",
    )
    parser.add_argument(
        "--use_xnnpack",
        action="store_true",
        help="Lower supported fp16/fp32 CPU ops to the XNNPACK delegate.",
    )
    parser.add_argument(
        "--quantization",
        choices=("none", "xnnpack_8da4w"),
        default="none",
        help=(
            "Quantization mode. xnnpack_8da4w uses int8 dynamic activations "
            "and int4 weights for LoRA-FA base linear layers."
        ),
    )
    parser.add_argument(
        "--int4_group_size",
        type=int,
        default=32,
        help="Group size for xnnpack_8da4w int4 base linear weights.",
    )
    parser.add_argument(
        "--quantize_lm_head",
        action="store_true",
        help="Also quantize the output lm_head linear with xnnpack_8da4w.",
    )
    parser.add_argument(
        "--xnnpack_dynamic_quant",
        action="store_true",
        help="Use the XNNPACK dynamic quantization partitioner.",
    )
    parser.add_argument(
        "--xnnpack_per_op_mode",
        action="store_true",
        help="Emit one XNNPACK delegate per matched op, useful for slow baselines.",
    )
    parser.add_argument(
        "--xnnpack_verbose",
        action="store_true",
        help="Enable verbose XNNPACK partitioner logs during export.",
    )
    parser.add_argument(
        "--xnnpack_force_non_static_weights",
        action="store_true",
        help=(
            "Allow XNNPACK to accept non-static linear weights. Leave this off "
            "first so static base weights can be packed; enable only when "
            "debugging dynamic LoRA-B linear partitioning."
        ),
    )
    parser.add_argument(
        "--min_xnnpack_delegates",
        type=int,
        default=1,
        help="With --use_xnnpack, fail export if fewer delegate calls are produced.",
    )
    return parser.parse_args()


def load_model_args(
    hf_model: str,
    cache_dir: str | None,
    *,
    seq_len: int,
) -> Tuple[GemmaModelArgs, Dict[str, object]]:
    hf_config = AutoConfig.from_pretrained(hf_model, cache_dir=cache_dir)
    hf_config_dict = hf_config.to_dict()
    model_args = GemmaModelArgs.from_hf_config(
        hf_config_dict,
        min_seq_len=seq_len,
    )
    return model_args, hf_config_dict


def save_lora_layout(
    output_dir: Path,
    model: GemmaLoraFAForPrefillLoss,
    args: argparse.Namespace,
) -> None:
    layout = {
        "model_family": "gemma3_textonly",
        "hf_model": args.hf_model,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "loss_mode": args.loss_mode,
        "vocab_size": model.params.vocab_size,
        "hidden_size": model.params.dim,
        "intermediate_size": model.params.intermediate_size,
        "num_hidden_layers": model.params.n_layers,
        "num_attention_heads": model.params.n_heads,
        "num_key_value_heads": model.params.n_kv_heads,
        "head_dim": model.params.head_dim,
        "rope_theta": model.params.rope_theta,
        "local_rope_theta": model.params.local_rope_theta,
        "rope_scaling": model.params.rope_scaling,
        "sliding_window": model.params.sliding_window,
        "layer_types": list(model.params.layer_types),
        "use_qk_norm": model.params.use_qk_norm,
        "qk_norm_before_rope": model.params.qk_norm_before_rope,
        "query_pre_attn_scalar": model.params.query_pre_attn_scalar,
        "embedding_scale_factor": model.params.embedding_scale_factor,
        "tie_word_embeddings": model.params.tie_word_embeddings,
        "attention_bias": model.params.attention_bias,
        "mlp_bias": model.params.mlp_bias,
        "rank": args.rank,
        "alpha": args.alpha,
        "adapter_scale": args.adapter_scale,
        "scaling": model.scaling,
        "lora_a_init": args.lora_a_init,
        "lora_seed": args.lora_seed,
        "use_xnnpack": args.use_xnnpack,
        "quantization": args.quantization,
        "int4_group_size": args.int4_group_size,
        "quantize_lm_head": args.quantize_lm_head,
        "xnnpack_dynamic_quant": args.xnnpack_dynamic_quant,
        "xnnpack_per_op_mode": args.xnnpack_per_op_mode,
        "xnnpack_force_non_static_weights": args.xnnpack_force_non_static_weights,
        "lora_b_numel": model.lora_b_numel,
        "specs": model.lora_b_specs_jsonable(),
    }
    layout_path = output_dir / args.layout_name
    with open(layout_path, "w", encoding="utf-8") as f:
        json.dump(layout, f, indent=2)
    logging.info("Saved LoRA layout to %s", layout_path)


def main() -> None:
    args = parse_args()
    if args.quantization == "xnnpack_8da4w" and not args.use_xnnpack:
        raise RuntimeError("--quantization xnnpack_8da4w requires --use_xnnpack")
    if args.lora_a_init == "llama_cpp":
        args.lora_a_init = "kaiming_uniform"
    elif args.lora_a_init == "torch":
        args.lora_a_init = "torch_linear"

    setup_logging(args.log_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_args, hf_config = load_model_args(
        args.hf_model,
        args.cache_dir,
        seq_len=args.seq_len,
    )
    logging.info(
        "HF config: model_type=%s hidden_size=%d intermediate_size=%d layers=%d "
        "heads=%d kv_heads=%d head_dim=%d vocab_size=%d rope_theta=%s "
        "local_rope_theta=%s sliding_window=%d use_qk_norm=%s "
        "tie_word_embeddings=%s",
        hf_config.get("model_type"),
        model_args.dim,
        model_args.intermediate_size,
        model_args.n_layers,
        model_args.n_heads,
        model_args.n_kv_heads,
        model_args.head_dim,
        model_args.vocab_size,
        model_args.rope_theta,
        model_args.local_rope_theta,
        model_args.sliding_window,
        model_args.use_qk_norm,
        model_args.tie_word_embeddings,
    )
    logging.info("Gemma layer_types=%s", ",".join(model_args.layer_types))

    model = GemmaLoraFAForPrefillLoss(
        model_args,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lora_rank=args.rank,
        lora_alpha=args.alpha,
        adapter_scale=args.adapter_scale,
        lora_seed=args.lora_seed,
        loss_mode=args.loss_mode,
    )
    model.to(dtype=torch.float16)
    if args.lora_a_init == "kaiming_uniform":
        lora_a_numel = initialize_lora_a_kaiming_uniform(
            model,
            seed=args.lora_seed,
        )
        logging.info(
            "LoRA-A init: kaiming_uniform seed=%d numel=%d",
            args.lora_seed,
            lora_a_numel,
        )
    else:
        logging.info("LoRA-A init: torch_linear constructor seed=%d", args.lora_seed)

    hf_state = load_hf_state_dict(args.hf_model, args.cache_dir)
    ignored, expected_missing = load_hf_state_dict_into_lora_fa_model(
        model,
        hf_state,
        dtype=torch.float16,
    )
    del hf_state
    model.output.to(dtype=torch.float32)
    model.eval()

    logging.info("Ignored %d HF checkpoint keys", len(ignored))
    logging.info("Expected missing LoRA/buffer keys: %d", len(expected_missing))
    logging.info("LM head dtype: %s", model.output.weight.dtype)
    logging.info("LoRA scaling: %.6f", model.scaling)
    logging.info("LoRA-B flat numel: %d", model.lora_b_numel)

    tokens, labels = make_dummy_batch(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=model_args.vocab_size,
        seed=args.dummy_seed,
        loss_mode=args.loss_mode,
    )
    lora_b_flat = torch.zeros(model.lora_b_numel, dtype=torch.float16)

    if not args.skip_eager_smoke:
        run_eager_smoke(
            model,
            tokens,
            labels,
            lora_b_flat,
            smoke_epsilon=args.smoke_epsilon,
            seed=args.dummy_seed + 1,
        )

    if args.quantization == "xnnpack_8da4w":
        model = apply_xnnpack_int4_quantization(
            model,
            group_size=args.int4_group_size,
            quantize_lm_head=args.quantize_lm_head,
        )
        model.eval()
        if not args.skip_eager_smoke:
            logging.info("Running post-quantization eager finite-loss smoke test")
            run_eager_smoke(
                model,
                tokens,
                labels,
                lora_b_flat,
                smoke_epsilon=args.smoke_epsilon,
                seed=args.dummy_seed + 1,
            )

    save_lora_layout(output_dir, model, args)
    pte_path, weights_path = export_to_executorch(
        model,
        (tokens, labels, lora_b_flat),
        output_dir=output_dir,
        pte_name=args.pte_name,
        weights_name=args.weights_name,
        use_xnnpack=args.use_xnnpack,
        quantization=args.quantization,
        xnnpack_dynamic_quant=args.xnnpack_dynamic_quant,
        xnnpack_per_op_mode=args.xnnpack_per_op_mode,
        xnnpack_verbose=args.xnnpack_verbose,
        xnnpack_force_non_static_weights=args.xnnpack_force_non_static_weights,
        min_xnnpack_delegates=args.min_xnnpack_delegates,
    )

    if args.run_executorch_smoke:
        run_executorch_smoke(
            pte_path,
            weights_path,
            (tokens, labels, lora_b_flat),
            smoke_epsilon=args.smoke_epsilon,
            seed=args.dummy_seed + 1,
        )
    logging.info("Export completed: pte=%s weights=%s", pte_path, weights_path)


if __name__ == "__main__":
    with torch.no_grad():
        main()
