"""
Usage:
    python generate_lora.py <base_model.gguf> [output_lora.gguf] [--rank R] [--alpha A]

The script reads the base model to discover architecture, layer count, and
tensor shapes, then writes a LoRA adapter where:
  - A matrices are Kaiming-uniform initialized and will be FROZEN (LoRA-FA)
  - B matrices are zero-initialized and will be the ONLY trainable part
"""

import sys
import os
import argparse
import numpy as np

# Allow running from anywhere – put gguf-py on the path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GGUF_PY_DIR = os.path.join(SCRIPT_DIR, "..", "..", "gguf-py")
if os.path.isdir(GGUF_PY_DIR):
    sys.path.insert(0, GGUF_PY_DIR)

import gguf
from gguf import GGUFReader, GGUFWriter
from gguf.constants import Keys

# Layer suffixes to inject LoRA into
TARGET_SUFFIXES = [
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_output.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_down.weight",
]


def read_base_model_info(model_path: str):
    """Read architecture and tensor metadata from the base model GGUF."""
    reader = GGUFReader(model_path)

    # Get architecture string
    arch = None
    if "general.architecture" in reader.fields:
        field = reader.fields["general.architecture"]
        # field.parts is a list of numpy arrays; for string type,
        # parts[-1] contains the UTF-8 bytes
        raw = bytes(field.parts[-1])
        arch = raw.decode("utf-8")

    if not arch:
        raise RuntimeError("Cannot determine architecture from base model GGUF")

    # Collect target tensors: name -> (ne[0], ne[1])
    # In GGUF, shape is stored as ggml dims: [ne0, ne1, ...]
    # ne0 = in_features (columns), ne1 = out_features (rows) for 2D weight
    targets = {}
    for t in reader.tensors:
        name = t.name
        for suffix in TARGET_SUFFIXES:
            if name.endswith(suffix):
                # reader.shape is stored as (ne[0], ne[1], ...) in ggml order
                shape = tuple(t.shape)
                if len(shape) >= 2:
                    targets[name] = (int(shape[0]), int(shape[1]))  # (in_feat, out_feat)
                break

    return arch, targets


def generate_lora_gguf(model_path: str, output_path: str, rank: int, alpha: float):
    arch, targets = read_base_model_info(model_path)

    if not targets:
        print("ERROR: no target tensors found in the base model", file=sys.stderr)
        sys.exit(1)

    print(f"Architecture : {arch}")
    print(f"LoRA rank    : {rank}")
    print(f"LoRA alpha   : {alpha}")
    print(f"Target layers: {len(targets)}")

    writer = GGUFWriter(output_path, arch)

    # -- metadata --
    writer.add_type(gguf.GGUFType.ADAPTER)
    writer.add_string(Keys.Adapter.TYPE, "lora")
    writer.add_float32(Keys.Adapter.LORA_ALPHA, alpha)

    # -- tensors --
    rng = np.random.default_rng(seed=42)
    total_params = 0

    for name in sorted(targets.keys()):
        in_feat, out_feat = targets[name]

        # lora_a : numpy shape (rank, in_feat) -> ggml ne[0]=in_feat, ne[1]=rank
        # Kaiming uniform: U(-bound, bound), bound = sqrt(6 / fan_in)
        bound = np.sqrt(6.0 / in_feat)
        lora_a = rng.uniform(-bound, bound, size=(rank, in_feat)).astype(np.float16)

        # lora_b : numpy shape (out_feat, rank) -> ggml ne[0]=rank, ne[1]=out_feat
        # Zero-initialized (standard LoRA init, ensures initial output = 0)
        lora_b = np.zeros((out_feat, rank), dtype=np.float16)

        writer.add_tensor(name + ".lora_a", lora_a)
        writer.add_tensor(name + ".lora_b", lora_b)

        total_params += lora_a.size + lora_b.size
        print(f"  {name:50s}  A({rank},{in_feat})  B({out_feat},{rank})")

    print(f"\nTotal LoRA params : {total_params:,}  ({total_params * 2 / 1024 / 1024:.2f} MiB in FP16)")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"Written: {output_path} ({size_mb:.2f} MiB)")


def main():
    parser = argparse.ArgumentParser(description="Generate LoRA-FA adapter GGUF")
    parser.add_argument("base_model", help="Path to the base model GGUF file")
    parser.add_argument("output", nargs="?", default=None,
                        help="Output LoRA GGUF path (default: <base_model>-lora-r<rank>.gguf)")
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank (default: 8)")
    parser.add_argument("--alpha", type=float, default=16.0, help="LoRA alpha (default: 16.0)")
    args = parser.parse_args()

    if not os.path.isfile(args.base_model):
        print(f"ERROR: base model not found: {args.base_model}", file=sys.stderr)
        sys.exit(1)

    output = args.output
    if output is None:
        base = os.path.splitext(args.base_model)[0]
        output = f"{base}-lora-r{args.rank}.gguf"

    generate_lora_gguf(args.base_model, output, args.rank, args.alpha)


if __name__ == "__main__":
    main()
