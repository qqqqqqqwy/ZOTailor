import argparse
from collections import Counter
import json
import logging
from pathlib import Path
import sys
from typing import Dict, Tuple

import torch
from torch.export import export

try:
    from executorch.exir import EdgeCompileConfig, to_edge, to_edge_transform_and_lower
    from executorch.exir.capture import ExecutorchBackendConfig
    from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
except ImportError as exc:
    raise RuntimeError(
        "ExecuTorch Python package is not installed. Run install_executorch.bat "
        "and pip install -e . --no-build-isolation first."
    ) from exc

try:
    from transformers import AutoConfig, AutoModelForCausalLM
except ImportError as exc:
    raise RuntimeError(
        "transformers is not installed. Install transformers, safetensors, "
        "huggingface_hub, sentencepiece, and accelerate."
    ) from exc

try:
    from .lora_fa_llama import (
        LlamaLoraFAForPrefillLoss,
        LlamaModelArgs,
        initialize_lora_a_kaiming_uniform,
        load_hf_state_dict_into_lora_fa_model,
    )
except ImportError:
    from lora_fa_llama import (  # type: ignore[no-redef]
        LlamaLoraFAForPrefillLoss,
        LlamaModelArgs,
        initialize_lora_a_kaiming_uniform,
        load_hf_state_dict_into_lora_fa_model,
    )


LOG_FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"


def setup_logging(log_path: str) -> None:
    log_file = Path(log_path)
    if log_file.parent != Path("."):
        log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        ],
        force=True,
    )
    logging.info("Writing export log to %s", log_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Llama-family LoRA-FA prefill loss model to ExecuTorch."
    )
    parser.add_argument(
        "--hf_model",
        default="/home/fuyongjian/qwy/models/Llama-3.2-3B",
        help="Hugging Face model id or local model directory.",
    )
    parser.add_argument("--output_dir", default="mytest/out")
    parser.add_argument(
        "--log_path",
        default="mytest/out/llama32_3b_export_log.txt",
        help="Write Python export/smoke logs to this text file.",
    )
    parser.add_argument(
        "--pte_name",
        default="llama32_3b_lora_fa.pte",
        help="Output ExecuTorch program filename.",
    )
    parser.add_argument(
        "--weights_name",
        default="llama32_3b_lora_fa_weights.ptd",
        help="Output external constants filename.",
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
            "U(-sqrt(6/fan_in), sqrt(6/fan_in)); torch_linear uses "
            "PyTorch Linear's reset-parameter variant. llama_cpp/torch are "
            "legacy aliases for these two modes."
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
) -> Tuple[LlamaModelArgs, Dict[str, object]]:
    hf_config = AutoConfig.from_pretrained(hf_model, cache_dir=cache_dir)
    hf_config_dict = hf_config.to_dict()
    model_args = LlamaModelArgs.from_hf_config(
        hf_config_dict,
        min_seq_len=seq_len,
    )
    return model_args, hf_config_dict


def make_dummy_batch(
    *,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    seed: int,
    loss_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    tokens = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_len),
        dtype=torch.int64,
        generator=generator,
    )
    if loss_mode == "single_label":
        labels = torch.full_like(tokens, -100)
        labels[:, seq_len - 2] = tokens[:, seq_len - 1]
    else:
        labels = torch.empty_like(tokens)
        labels[:, :-1] = tokens[:, 1:]
        labels[:, -1] = -100
    return tokens, labels


def load_hf_state_dict(hf_model: str, cache_dir: str | None) -> Dict[str, torch.Tensor]:
    logging.info("Loading HF checkpoint: %s", hf_model)
    model = AutoModelForCausalLM.from_pretrained(
        hf_model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        cache_dir=cache_dir,
    )
    state_dict = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }
    del model
    return state_dict


def save_lora_layout(
    output_dir: Path,
    model: LlamaLoraFAForPrefillLoss,
    args: argparse.Namespace,
) -> None:
    layout = {
        "model_family": "llama",
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
        "rope_theta": model.params.rope_theta,
        "rope_scaling": model.params.rope_scaling,
        "tie_word_embeddings": model.params.tie_word_embeddings,
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
    layout_path = output_dir / "llama32_3b_lora_fa_layout.json"
    with open(layout_path, "w", encoding="utf-8") as f:
        json.dump(layout, f, indent=2)
    logging.info("Saved LoRA layout to %s", layout_path)


def summarize_graph(edge_program, stage: str) -> int:
    graph = edge_program._edge_programs["forward"].graph
    nodes = list(graph.nodes)
    call_functions = [node for node in nodes if node.op == "call_function"]
    delegate_count = sum(
        "executorch_call_delegate" in str(node.target)
        or "executorch_call_delegate" in node.name
        for node in nodes
    )
    targets = Counter(str(node.target) for node in call_functions)
    logging.info(
        "%s graph: nodes=%d call_function=%d delegate_calls=%d",
        stage,
        len(nodes),
        len(call_functions),
        delegate_count,
    )
    for target, count in targets.most_common(20):
        logging.info("%s graph op[%d]: %s", stage, count, target)
    return delegate_count


def apply_xnnpack_int4_quantization(
    model: LlamaLoraFAForPrefillLoss,
    *,
    group_size: int,
    quantize_lm_head: bool,
) -> LlamaLoraFAForPrefillLoss:
    if group_size <= 0 or group_size % 32 != 0:
        raise ValueError(
            "--int4_group_size must be a positive multiple of 32 for XNNPACK int4"
        )
    try:
        from torchao.quantization.granularity import PerGroup
        try:
            from torchao.quantization.quant_api import (
                Int8DynamicActivationIntxWeightConfig,
                quantize_,
            )
        except ImportError:
            from torchao.quantization import (  # type: ignore[no-redef]
                Int8DynamicActivationIntxWeightConfig,
                quantize_,
            )
        from torchao.utils import unwrap_tensor_subclass
    except ImportError as exc:
        raise RuntimeError(
            "TorchAO is required for --quantization xnnpack_8da4w. "
            "Install the ExecuTorch Python dependencies or pip install torchao."
        ) from exc

    quantized_fqns: set[str] = set()

    def filter_fn(module: torch.nn.Module, fqn: str) -> bool:
        if not isinstance(module, torch.nn.Linear):
            return False
        if not fqn.endswith(".base") and not (quantize_lm_head and fqn == "output"):
            return False
        if module.in_features % group_size != 0:
            raise ValueError(
                f"Linear {fqn} in_features={module.in_features} is not divisible "
                f"by int4 group size {group_size}"
            )
        quantized_fqns.add(fqn)
        return True

    logging.info(
        "Applying XNNPACK 8da4w quantization: group_size=%d quantize_lm_head=%s",
        group_size,
        quantize_lm_head,
    )
    quantize_(
        model,
        Int8DynamicActivationIntxWeightConfig(
            weight_dtype=torch.int4,
            weight_granularity=PerGroup(group_size),
        ),
        filter_fn=filter_fn,
    )
    unwrapped = unwrap_tensor_subclass(model)
    if unwrapped is not None:
        model = unwrapped
    expected = model.params.n_layers * 7 + (1 if quantize_lm_head else 0)
    if len(quantized_fqns) != expected:
        raise RuntimeError(
            "XNNPACK int4 quantization matched "
            f"{len(quantized_fqns)} base linear modules, expected {expected}"
        )
    logging.info("Quantized %d base linear modules", len(quantized_fqns))
    return model


def export_to_executorch(
    model: LlamaLoraFAForPrefillLoss,
    example_inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    output_dir: Path,
    pte_name: str,
    weights_name: str,
    use_xnnpack: bool,
    quantization: str,
    xnnpack_dynamic_quant: bool,
    xnnpack_per_op_mode: bool,
    xnnpack_verbose: bool,
    xnnpack_force_non_static_weights: bool,
    min_xnnpack_delegates: int,
) -> Tuple[Path, Path]:
    external_tag = weights_name[:-4] if weights_name.endswith(".ptd") else weights_name
    logging.info("Capturing torch.export graph")
    exported_program = export(model, example_inputs, strict=False)

    if use_xnnpack:
        from executorch.backends.xnnpack.utils.configs import (
            get_xnnpack_edge_compile_config,
        )
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
            XnnpackPartitioner,
        )

        edge_compile_config = get_xnnpack_edge_compile_config(skip_dim_order=True)
        logging.info("Lowering to edge dialect and XNNPACK")
        if quantization == "xnnpack_8da4w":
            if not xnnpack_dynamic_quant:
                raise RuntimeError(
                    "--quantization xnnpack_8da4w requires --xnnpack_dynamic_quant"
                )
            from executorch.backends.xnnpack.partition.config.xnnpack_config import (
                ConfigPrecisionType,
            )

            partitioner = XnnpackPartitioner(
                config_precisions=ConfigPrecisionType.DYNAMIC_QUANT,
                per_op_mode=xnnpack_per_op_mode,
                verbose=xnnpack_verbose,
            )
        else:
            partitioner = XnnpackPartitioner(
                verbose=xnnpack_verbose,
                force_non_static_weights_for_f32_linear=(
                    xnnpack_force_non_static_weights
                ),
            )
        edge_program = to_edge_transform_and_lower(
            exported_program,
            partitioner=[partitioner],
            compile_config=edge_compile_config,
        )
        delegate_count = summarize_graph(edge_program, "xnnpack")
        if delegate_count < min_xnnpack_delegates:
            raise RuntimeError(
                "XNNPACK lowering produced "
                f"{delegate_count} delegate calls, expected at least "
                f"{min_xnnpack_delegates}."
            )
    else:
        edge_compile_config = EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        )
        logging.info("Lowering to edge dialect")
        edge_program = to_edge(
            exported_program,
            compile_config=edge_compile_config,
        )
        summarize_graph(edge_program, "edge")

    logging.info("Converting to ExecuTorch with external constants")
    backend_config = ExecutorchBackendConfig(
        external_constants=lambda _: external_tag,
        memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
    )
    executorch_program = edge_program.to_executorch(config=backend_config)

    pte_path = output_dir / pte_name
    with open(pte_path, "wb") as f:
        executorch_program.write_to_file(f)
    executorch_program.write_tensor_data_to_file(outdir=str(output_dir))

    weights_path = output_dir / (
        external_tag if external_tag.endswith(".ptd") else f"{external_tag}.ptd"
    )
    logging.info("Saved PTE to %s", pte_path)
    logging.info("Saved external weights to %s", weights_path)
    return pte_path, weights_path


def run_eager_smoke(
    model: LlamaLoraFAForPrefillLoss,
    tokens: torch.Tensor,
    labels: torch.Tensor,
    lora_b_flat: torch.Tensor,
    *,
    smoke_epsilon: float,
    seed: int,
) -> None:
    logging.info("Running eager finite-loss smoke test")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    perturbed_lora_b = (
        torch.randn(
            lora_b_flat.shape,
            dtype=torch.float32,
            generator=generator,
        )
        * smoke_epsilon
    ).to(dtype=lora_b_flat.dtype)
    with torch.no_grad():
        for name, smoke_lora_b in (
            ("zero_b", lora_b_flat),
            ("perturbed_b", perturbed_lora_b),
        ):
            loss = model(tokens, labels, smoke_lora_b)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Eager smoke test {name} produced non-finite loss: {loss}"
                )
            logging.info("Eager smoke loss[%s]: %.6f", name, loss.item())


def run_executorch_smoke(
    pte_path: Path,
    weights_path: Path,
    example_inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    smoke_epsilon: float,
    seed: int,
) -> None:
    logging.info("Running ExecuTorch pybinding smoke test")
    from executorch.extension.pybindings import portable_lib

    module = portable_lib._load_for_executorch(str(pte_path), str(weights_path))
    tokens, labels, lora_b_flat = example_inputs
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    perturbed_lora_b = (
        torch.randn(
            lora_b_flat.shape,
            dtype=torch.float32,
            generator=generator,
        )
        * smoke_epsilon
    ).to(dtype=lora_b_flat.dtype)
    for name, smoke_lora_b in (
        ("zero_b", lora_b_flat),
        ("perturbed_b", perturbed_lora_b),
    ):
        out = module.forward((tokens, labels, smoke_lora_b))
        loss = out[0]
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"ExecuTorch smoke test {name} produced non-finite loss: {loss}"
            )
        logging.info("ExecuTorch smoke loss[%s]: %.6f", name, loss.item())


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
        "heads=%d kv_heads=%d vocab_size=%d rope_theta=%s rope_scaling=%s "
        "tie_word_embeddings=%s",
        hf_config.get("model_type"),
        model_args.dim,
        model_args.intermediate_size,
        model_args.n_layers,
        model_args.n_heads,
        model_args.n_kv_heads,
        model_args.vocab_size,
        model_args.rope_theta,
        model_args.rope_scaling,
        model_args.tie_word_embeddings,
    )
    model = LlamaLoraFAForPrefillLoss(
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
        logging.info(
            "LoRA-A init: torch_linear constructor seed=%d",
            args.lora_seed,
        )

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
