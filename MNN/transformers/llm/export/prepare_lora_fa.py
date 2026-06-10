#!/usr/bin/env python3
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


TARGET_MODULES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}

GGUF_MODULE_SUFFIX = {
    "q_proj": "attn_q.weight",
    "k_proj": "attn_k.weight",
    "v_proj": "attn_v.weight",
    "o_proj": "attn_output.weight",
    "gate_proj": "ffn_gate.weight",
    "up_proj": "ffn_up.weight",
    "down_proj": "ffn_down.weight",
}

GGUF_OP_TO_MODULE = {
    "attn_q": ("self_attn", "q_proj"),
    "attn_k": ("self_attn", "k_proj"),
    "attn_v": ("self_attn", "v_proj"),
    "attn_output": ("self_attn", "o_proj"),
    "ffn_gate": ("mlp", "gate_proj"),
    "ffn_up": ("mlp", "up_proj"),
    "ffn_down": ("mlp", "down_proj"),
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, obj, pretty=True):
    with open(path, "w", encoding="utf-8") as f:
        if pretty:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        else:
            json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")


def tensor_names(model):
    if "tensorName" in model:
        return model["tensorName"]
    if "tensors" in model:
        return model["tensors"]
    raise RuntimeError("MNN JSON does not contain tensorName/tensors")


def add_tensor(model, name):
    names = tensor_names(model)
    index = len(names)
    names.append(name)
    return index


def as_index(indexes):
    if not indexes:
        raise RuntimeError("empty tensor index list")
    return indexes[0]


def clean_name(name):
    return name.strip("/").replace("/", "_").replace(".", "_")


def parse_target_name(name):
    parts = name.strip("/").split("/")
    if len(parts) == 4:
        layer, group, module, linear = parts
        if not layer.startswith("layers.") or group not in ("self_attn", "mlp") or linear != "Linear":
            return None
        if module not in TARGET_MODULES:
            return None
        try:
            layer_id = int(layer.split(".")[-1])
        except ValueError:
            return None
        return layer_id, group, module

    dot_parts = name.split(".")
    if len(dot_parts) == 3 and dot_parts[0] == "blk":
        try:
            layer_id = int(dot_parts[1])
        except ValueError:
            return None
        mapped = GGUF_OP_TO_MODULE.get(dot_parts[2])
        if mapped is None:
            return None
        group, module = mapped
        return layer_id, group, module

    return None


def get_conv_dims(op):
    main = op.get("main", {})
    common = main.get("common", {})
    ic = int(common.get("inputCount", 0))
    oc = int(common.get("outputCount", 0))
    if ic <= 0 or oc <= 0:
        raise RuntimeError(f"cannot read Conv dims from {op.get('name', '<unnamed>')}")
    return ic, oc


def build_tensor_maps(ops):
    producers = {}
    consumers = {}
    for op in ops:
        for idx in op.get("outputIndexes", []) or []:
            producers[idx] = op
        for idx in op.get("inputIndexes", []) or []:
            consumers.setdefault(idx, []).append(op)
    return producers, consumers


def reshape_dims(op, fallback):
    if op is None:
        return fallback
    dims = op.get("main", {}).get("dims", None)
    if not isinstance(dims, list) or len(dims) == 0:
        return fallback
    return [int(dim) for dim in dims]


def find_pre_reshape(name, conv_input, by_name, producers):
    named = by_name.get(f"{name}/pre_reshape")
    if named is not None:
        return named
    producer = producers.get(conv_input)
    if producer is not None and producer.get("type") == "Reshape":
        return producer
    return None


def find_post_reshape(name, conv_output, by_name, consumers):
    named = by_name.get(f"{name}/post_reshape")
    if named is not None:
        return named
    reshape_consumers = [
        consumer for consumer in consumers.get(conv_output, [])
        if consumer.get("type") == "Reshape"
    ]
    for consumer in reshape_consumers:
        consumer_name = consumer.get("name", "")
        if consumer_name.startswith(name) or name in consumer_name:
            return consumer
    if len(reshape_consumers) == 1:
        return reshape_consumers[0]

    convert_consumers = [
        consumer for consumer in consumers.get(conv_output, [])
        if is_convert_tensor(consumer, "NC4HW4", "NCHW")
    ]
    convert_reshape_consumers = []
    for convert in convert_consumers:
        convert_output = as_index(convert.get("outputIndexes", []))
        convert_reshape_consumers.extend([
            consumer for consumer in consumers.get(convert_output, [])
            if consumer.get("type") == "Reshape"
        ])
    for consumer in convert_reshape_consumers:
        consumer_name = consumer.get("name", "")
        if consumer_name.startswith(name) or name in consumer_name:
            return consumer
    if len(convert_reshape_consumers) == 1:
        return convert_reshape_consumers[0]
    return None


def is_convert_tensor(op, source=None, dest=None):
    if op is None or op.get("type") != "ConvertTensor":
        return False
    main = op.get("main", {})
    if source is not None and main.get("source") != source:
        return False
    if dest is not None and main.get("dest") != dest:
        return False
    return True


def find_post_convert(name, conv_output, post_op, by_name, producers, consumers):
    named = by_name.get(f"{name}/post_convert")
    if (is_convert_tensor(named, "NC4HW4", "NCHW") and
            as_index(named.get("inputIndexes", [])) == conv_output):
        return named

    if post_op is not None:
        post_input = as_index(post_op.get("inputIndexes", []))
        producer = producers.get(post_input)
        if (is_convert_tensor(producer, "NC4HW4", "NCHW") and
                as_index(producer.get("inputIndexes", [])) == conv_output):
            return producer

    convert_consumers = [
        consumer for consumer in consumers.get(conv_output, [])
        if is_convert_tensor(consumer, "NC4HW4", "NCHW")
    ]
    if len(convert_consumers) == 1:
        return convert_consumers[0]
    return None


def resolve_lora_hidden_input(conv_input, pre_op, producers):
    if pre_op is not None:
        return as_index(pre_op.get("inputIndexes", [])), "own_pre_reshape"

    producer = producers.get(conv_input)
    if is_convert_tensor(producer, "NCHW", "NC4HW4"):
        convert_input = as_index(producer.get("inputIndexes", []))
        convert_input_producer = producers.get(convert_input)
        if convert_input_producer is not None and convert_input_producer.get("type") == "Reshape":
            return as_index(convert_input_producer.get("inputIndexes", [])), "shared_pre_convert_pre_reshape"
        return convert_input, "shared_pre_convert_input"

    return conv_input, "fallback_conv_input"


def find_targets(model):
    ops = model.get("oplists", [])
    by_name = {op.get("name", ""): op for op in ops}
    producers, consumers = build_tensor_maps(ops)
    targets = []
    for op in ops:
        if op.get("type") != "Convolution":
            continue
        name = op.get("name", "")
        parsed = parse_target_name(name)
        if parsed is None:
            continue
        if "lm_head" in name or "embed" in name:
            continue
        ic, oc = get_conv_dims(op)
        conv_input = as_index(op.get("inputIndexes", []))
        conv_output = as_index(op.get("outputIndexes", []))
        pre_op = find_pre_reshape(name, conv_input, by_name, producers)
        post_op = find_post_reshape(name, conv_output, by_name, consumers)
        post_convert_op = find_post_convert(name, conv_output, post_op, by_name, producers, consumers)
        hidden_input, hidden_input_source = resolve_lora_hidden_input(conv_input, pre_op, producers)
        if post_op is not None and post_convert_op is not None:
            base_output = as_index(post_convert_op["inputIndexes"])
            insert_after_op = name
            lora_output_shape = [-1, oc, 1, 1]
            lora_delta_format = "NC4HW4"
            injection_point = "pre_post_convert_nc4hw4"
        elif post_op is not None:
            base_output = as_index(post_op["inputIndexes"])
            base_producer = producers.get(base_output)
            insert_after_op = base_producer.get("name", "") if base_producer is not None else name
            lora_output_shape = [-1, oc, 1, 1]
            lora_delta_format = "NCHW"
            injection_point = "pre_post_reshape"
        else:
            base_output = conv_output
            insert_after_op = name
            lora_output_shape = [-1, oc] if pre_op is not None else [1, -1, oc]
            lora_delta_format = "NCHW"
            injection_point = "post_linear_output"
        targets.append({
            "target_op": name,
            "pre_op": pre_op.get("name", "") if pre_op is not None else "",
            "post_op": post_op.get("name", "") if post_op is not None else "",
            "post_convert_op": post_convert_op.get("name", "") if post_convert_op is not None else "",
            "insert_after_op": insert_after_op,
            "hidden_input": hidden_input,
            "hidden_input_source": hidden_input_source,
            "base_output": base_output,
            "lora_output_shape": lora_output_shape,
            "lora_delta_format": lora_delta_format,
            "injection_point": injection_point,
            "ic": ic,
            "oc": oc,
            "layer": parsed[0],
            "group": parsed[1],
            "module": parsed[2],
        })
    return targets


def canonical_target_key(target):
    # Match examples/zoo/generate_lora.py and the ExecuTorch exporter, which
    # consume the Kaiming RNG after sorting LoRA targets by GGUF-style name.
    return f"blk.{target['layer']}.{GGUF_MODULE_SUFFIX[target['module']]}"


def prepare_target_layout(targets, rank, alpha, rng):
    scale = float(alpha) / float(rank)
    offset = 0
    for index, target in enumerate(sorted(targets, key=canonical_target_key)):
        target["lora_index"] = index
        target["offset"] = offset
        target["a_values"] = kaiming_a_values(rng, target["ic"], rank, scale)
        target["lora_scale"] = scale
        target["canonical_key"] = canonical_target_key(target)
        offset += rank * target["oc"]
    return offset


def build_input_op(name, output_index, shape):
    return {
        "name": name,
        "type": "Input",
        "inputIndexes": [],
        "outputIndexes": [output_index],
        "main_type": "Input",
        "main": {
            "dims": shape,
            "dtype": "DT_FLOAT",
            "dformat": "NCHW",
        },
        "defaultDimentionFormat": "NCHW",
    }


def build_const_op(name, output_index, shape, values):
    return {
        "name": name,
        "type": "Const",
        "inputIndexes": [],
        "outputIndexes": [output_index],
        "main_type": "Blob",
        "main": {
            "dims": shape,
            "dataFormat": "NCHW",
            "dataType": "DT_FLOAT",
            "float32s": values,
        },
        "defaultDimentionFormat": "NCHW",
    }


def build_reshape_op(name, input_index, output_index, dims):
    return {
        "name": name,
        "type": "Reshape",
        "inputIndexes": [input_index],
        "outputIndexes": [output_index],
        "main_type": "Reshape",
        "main": {
            "dims": dims,
            "dimType": "NCHW",
        },
        "defaultDimentionFormat": "NHWC",
    }


def build_convert_op(name, input_index, output_index, source, dest):
    return {
        "name": name,
        "type": "ConvertTensor",
        "inputIndexes": [input_index],
        "outputIndexes": [output_index],
        "main_type": "TensorConvertInfo",
        "main": {
            "source": source,
            "dest": dest,
        },
        "defaultDimentionFormat": "NHWC",
    }


def build_matmul_op(name, inputs, output_index):
    return {
        "name": name,
        "type": "MatMul",
        "inputIndexes": inputs,
        "outputIndexes": [output_index],
        "main_type": "MatMul",
        "main": {
            "T": "DT_FLOAT",
            "transposeA": False,
            "transposeB": False,
        },
        "defaultDimentionFormat": "NHWC",
    }


def build_add_op(name, left, right, output_index):
    return {
        "name": name,
        "type": "BinaryOp",
        "inputIndexes": [left, right],
        "outputIndexes": [output_index],
        "main_type": "BinaryOp",
        "main": {
            "opType": "ADD",
            "T": "DT_FLOAT",
            "activationType": 0,
        },
        "defaultDimentionFormat": "NHWC",
    }


def kaiming_a_values(rng, ic, rank, scale):
    # Match examples/zoo/generate_lora.py: A is generated as FP16 [rank, in_feat]
    # with U(-sqrt(6/fan_in), sqrt(6/fan_in)). MNN stores [in_feat, rank].
    bound = math.sqrt(6.0 / float(ic))
    cpu_a = rng.uniform(-bound, bound, size=(rank, ic)).astype(np.float16).astype(np.float32)
    return [float(cpu_a[r, in_idx] * scale) for in_idx in range(ic) for r in range(rank)]


def inject_branch(model, target, rank):
    target_op = target["target_op"]
    prefix = f"{target_op}/lora_fa"
    safe = clean_name(target_op)
    ic = target["ic"]
    oc = target["oc"]
    index = target["lora_index"]
    offset = target["offset"]
    scale = target["lora_scale"]
    a_values = target["a_values"]

    x2d = add_tensor(model, f"{prefix}/x2d")
    a_const = add_tensor(model, f"{prefix}/A")
    b_input = add_tensor(model, f"lora_fa_B_{index:03d}")
    xa = add_tensor(model, f"{prefix}/xA")
    xb = add_tensor(model, f"{prefix}/xAB")
    lora_out = add_tensor(model, f"{prefix}/lora_out")
    lora_add_input = lora_out
    lora_packed = None
    if target.get("lora_delta_format") == "NC4HW4":
        lora_packed = add_tensor(model, f"{prefix}/lora_out_nc4hw4")
        lora_add_input = lora_packed
    add_out = add_tensor(model, f"{prefix}/add")

    b_name = f"lora_fa_B_{index:03d}"
    ops = [
        build_input_op(b_name, b_input, [rank, oc]),
        build_const_op(f"{prefix}/A_const", a_const, [ic, rank], a_values),
        build_reshape_op(f"{prefix}/reshape_in", target["hidden_input"], x2d, [-1, ic]),
        build_matmul_op(f"{prefix}/matmul_A", [x2d, a_const], xa),
        build_matmul_op(f"{prefix}/matmul_B", [xa, b_input], xb),
        build_reshape_op(f"{prefix}/reshape_out", xb, lora_out, target["lora_output_shape"]),
    ]
    if lora_packed is not None:
        ops.append(build_convert_op(f"{prefix}/pack_nc4hw4", lora_out, lora_packed, "NCHW", "NC4HW4"))
    ops.append(build_add_op(f"{prefix}/add", target["base_output"], lora_add_input, add_out))

    meta = {
        "index": index,
        "target_op": target_op,
        "layer": target["layer"],
        "group": target["group"],
        "module": target["module"],
        "input_size": ic,
        "output_size": oc,
        "a_shape": [ic, rank],
        "a_init": "kaiming_uniform_sqrt6_f16_cpu_order",
        "lora_scale_folded_into_a": scale,
        "b_input": b_name,
        "b_shape": [rank, oc],
        "lora_output_shape": target["lora_output_shape"],
        "injection_point": target["injection_point"],
        "post_convert_op": target.get("post_convert_op", ""),
        "lora_delta_format": target.get("lora_delta_format", "NCHW"),
        "offset": offset,
        "size": rank * oc,
        "canonical_key": target["canonical_key"],
        "hidden_input_tensor": target["hidden_input"],
        "hidden_input_source": target.get("hidden_input_source", ""),
        "base_output_tensor": target["base_output"],
        "lora_delta_tensor": lora_add_input,
        "lora_output_tensor": add_out,
        "safe_name": safe,
    }
    return ops, meta, (target["base_output"], add_out)


def replace_consumers(model, replacements, protected_op_names):
    for op in model.get("oplists", []):
        if op.get("name", "") in protected_op_names:
            continue
        inputs = op.get("inputIndexes")
        if not inputs:
            continue
        op["inputIndexes"] = [replacements.get(idx, idx) for idx in inputs]


def read_vocab_size(model_dir, default):
    for name in ("llm_config.json", "config.json"):
        path = model_dir / name
        if not path.exists():
            continue
        try:
            cfg = load_json(path)
        except Exception:
            continue
        for key in ("vocab_size", "vocabSize"):
            if key in cfg:
                return int(cfg[key])
    return default


def update_config(model_dir, output_config, output_mnn):
    config_path = model_dir / "config.json"
    if config_path.exists():
        cfg = load_json(config_path)
    else:
        cfg = {}
    cfg["llm_model"] = output_mnn.name
    cfg.setdefault("llm_weight", "llm.mnn.weight")
    if (model_dir / "tokenizer.mtok").exists():
        cfg["tokenizer_file"] = "tokenizer.mtok"
    elif (model_dir / "tokenizer.txt").exists():
        cfg["tokenizer_file"] = "tokenizer.txt"
    cfg["backend_type"] = "cpu"
    cfg["all_logits"] = True
    cfg["reuse_kv"] = False
    cfg["async"] = False
    save_json(output_config, cfg, pretty=True)


def call_mnnconvert(args, mnnconvert):
    if mnnconvert:
        subprocess.check_call([mnnconvert] + args[1:])
        return
    try:
        from MNN.tools import mnnconvert as pymnnconvert
    except Exception as exc:
        raise RuntimeError("MNNConvert was not found; pass --mnnconvert or install pymnn") from exc
    if hasattr(pymnnconvert, "convert"):
        pymnnconvert.convert(args)
        return
    old_argv = sys.argv[:]
    try:
        sys.argv = args
        pymnnconvert.main()
    finally:
        sys.argv = old_argv


def ensure_json(input_mnn, input_json, mnnconvert):
    if input_json.exists():
        if not input_mnn.exists():
            return
        if input_json.stat().st_mtime >= input_mnn.stat().st_mtime:
            return
        print(f"Regenerate stale MNN JSON: {input_json}")
    call_mnnconvert(["", "-f", "MNN", "--modelFile", str(input_mnn), "--JsonFile", str(input_json)], mnnconvert)


def convert_json_to_mnn(output_json, output_mnn, mnnconvert):
    # --convertMatmulToConv=0: keep our injected LoRA matmul_A as MatMul instead
    # of letting the optimizer rewrite it to a Convolution. The rewrite makes
    # matmul_A and matmul_B traverse completely different runtime kernels
    # (Conv weight-pack vs MatMul dynamic-pack), which appears to perturb the
    # ZO loss landscape in a way that prevents convergence on Arm CPU. Keeping
    # both LoRA matmuls as MatMul ops matches the behavior of llama.cpp /
    # executorch where the LoRA branch is a single uniform code path.
    call_mnnconvert([
        "",
        "-f", "JSON",
        "--modelFile", str(output_json),
        "--MNNModel", str(output_mnn),
        "--convertMatmulToConv=0",
    ], mnnconvert)


def main():
    parser = argparse.ArgumentParser(description="Inject LoRA-FA B inputs into an exported MNN LLM graph.")
    parser.add_argument("--model-dir", type=Path, required=True, help="Directory produced by llmexport.py")
    parser.add_argument("--input-mnn", type=Path, default=None, help="Input llm.mnn path")
    parser.add_argument("--input-json", type=Path, default=None, help="Input llm.mnn.json path")
    parser.add_argument("--output-mnn", type=Path, default=None, help="Output MNN path")
    parser.add_argument("--output-json", type=Path, default=None, help="Output JSON path")
    parser.add_argument("--metadata", type=Path, default=None, help="Output LoRA-FA metadata path")
    parser.add_argument("--output-config", type=Path, default=None, help="Output runtime config path")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--expected-targets", type=int, default=0)
    parser.add_argument("--mnnconvert", type=str, default=None, help="Path to MNNConvert; falls back to pymnn")
    parser.add_argument("--no-convert", action="store_true", help="Only write JSON and metadata")
    args = parser.parse_args()

    model_dir = args.model_dir
    input_mnn = args.input_mnn or (model_dir / "llm.mnn")
    input_json = args.input_json or (model_dir / "llm.mnn.json")
    output_mnn = args.output_mnn or (model_dir / "llm_lora_fa.mnn")
    output_json = args.output_json or (model_dir / "llm_lora_fa.mnn.json")
    metadata_path = args.metadata or (model_dir / "lora_fa.json")
    output_config = args.output_config or (model_dir / "config_lora_fa.json")
    if args.rank <= 0 or args.alpha <= 0:
        raise RuntimeError("rank and alpha must be positive")

    mnnconvert = args.mnnconvert
    if mnnconvert is not None:
        resolved = shutil.which(mnnconvert) or mnnconvert
        if not os.path.exists(resolved):
            raise RuntimeError(f"MNNConvert not found: {mnnconvert}")
        mnnconvert = resolved

    ensure_json(input_mnn, input_json, mnnconvert)
    model = load_json(input_json)
    targets = find_targets(model)
    if args.expected_targets and len(targets) != args.expected_targets:
        raise RuntimeError(f"expected {args.expected_targets} LoRA targets, found {len(targets)}")
    if not targets:
        raise RuntimeError("no transformer block linear targets found")

    insert_to_target = {target["insert_after_op"]: target for target in targets}
    rng = np.random.default_rng(seed=args.seed)
    total_b_size = prepare_target_layout(targets, args.rank, args.alpha, rng)
    new_ops = []
    metas_by_index = [None] * len(targets)
    replacements = {}
    protected_op_names = set()
    for op in model.get("oplists", []):
        new_ops.append(op)
        target = insert_to_target.get(op.get("name", ""))
        if target is None:
            continue
        branch_ops, meta, replacement = inject_branch(model, target, args.rank)
        new_ops.extend(branch_ops)
        metas_by_index[meta["index"]] = meta
        replacements[replacement[0]] = replacement[1]
        protected_op_names.add(branch_ops[-1].get("name", ""))

    model["oplists"] = new_ops
    replace_consumers(model, replacements, protected_op_names)
    if any(meta is None for meta in metas_by_index):
        raise RuntimeError("internal error: not all LoRA targets were injected")
    metas = metas_by_index

    vocab_size = read_vocab_size(model_dir, args.vocab_size)
    metadata = {
        "format": "mnn_lora_fa_zo_v1",
        "rank": args.rank,
        "alpha": args.alpha,
        "seed": args.seed,
        "vocab_size": vocab_size,
        "target_modules": sorted(TARGET_MODULES),
        "input_names": [meta["b_input"] for meta in metas],
        "total_b_size": total_b_size,
        "targets": metas,
    }

    save_json(output_json, model, pretty=False)
    save_json(metadata_path, metadata, pretty=True)
    update_config(model_dir, output_config, output_mnn)

    if not args.no_convert:
        convert_json_to_mnn(output_json, output_mnn, mnnconvert)

    print(f"lora_fa_targets={len(metas)}")
    print(f"lora_fa_total_b_size={total_b_size}")
    print(f"lora_fa_metadata={metadata_path}")
    print(f"lora_fa_config={output_config}")


if __name__ == "__main__":
    main()
