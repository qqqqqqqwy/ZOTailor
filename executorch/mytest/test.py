# compare_logits.py
import os
import json
import math
import torch
import torch.nn.functional as F

HF_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
META_PTH = "./llama/tinyllama_meta_format.pth"
META_CFG = "./llama/tinyllama_config.json"

from llama.model import ModelArgs, Transformer
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_llama_meta_model(cfg_json_path, pth_path, device, dtype=torch.float32):
    with open(cfg_json_path, "r") as f:
        params = json.load(f)
    args = ModelArgs(**params)
    model = Transformer(args)
    sd = torch.load(pth_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=True)
    print(f"[Meta->LLaMA] missing={missing}, unexpected={unexpected}")
    model.to(device=device, dtype=dtype).eval()
    return model

@torch.no_grad()
def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    # 1) HF 侧
    tok = AutoTokenizer.from_pretrained(HF_MODEL, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    hf = AutoModelForCausalLM.from_pretrained(HF_MODEL, torch_dtype=dtype).to(device).eval()

    # 2) LLaMA 仓库侧
    meta = load_llama_meta_model(META_CFG, META_PTH, device=device, dtype=dtype)

    # 3) 构造同一批输入
    B, T = 1, 64
    sample_text = "TinyLlama sanity check for logits alignment between HF and LLaMA repo."
    enc = tok(sample_text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"][:, :T]  # 截断到 T
    if input_ids.size(1) < T:  # 不足则重复填充到 T
        pad = input_ids.new_full((input_ids.size(0), T - input_ids.size(1)), tok.pad_token_id)
        input_ids = torch.cat([input_ids, pad], dim=1)
    attn_mask = (input_ids != tok.pad_token_id).long()

    input_ids = input_ids.to(device)
    attn_mask = attn_mask.to(device)

    # 4) 前向：HF
    out_hf = hf(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
    logits_hf = out_hf.logits.to(dtype)

    # 5) 前向：LLaMA 仓库
    logits_meta = meta(input_ids.to(torch.int64), start_pos=torch.tensor([0], dtype=torch.long), attention_mask=attn_mask)  # [B, T, V]
    logits_meta = logits_meta.to(dtype)

    # 6) 对齐与断言
    assert logits_hf.shape == logits_meta.shape, f"shape mismatch: HF {logits_hf.shape} vs META {logits_meta.shape}"
    B, T, V = logits_hf.shape
    print(f"logits shape: {B} x {T} x {V}")

    # 7) 误差度量
    valid_pos = attn_mask.bool().unsqueeze(-1).expand_as(logits_hf)  # [B,T,V]
    diff = (logits_hf - logits_meta).float()
    # 拉平有效位置做统计
    diff_valid = diff.masked_select(valid_pos)
    hf_valid   = logits_hf.masked_select(valid_pos)
    meta_valid = logits_meta.masked_select(valid_pos)
    max_abs  = diff_valid.abs().max().item()
    mean_abs = diff_valid.abs().mean().item()
    max_rel  = (diff_valid.abs() / hf_valid.abs().clamp_min(1e-8)).max().item()
    # 余弦相似度（把有效位置的向量拼到一起）
    cos = F.cosine_similarity(hf_valid.view(1, -1), meta_valid.view(1, -1), dim=1).item()
    print(f"[Valid tokens only] max_abs={max_abs:.6e} | mean_abs={mean_abs:.6e} | max_rel={max_rel:.6e} | cos={cos:.6f}")

    # 8) 进一步：对比一步交叉熵（语言建模常用 next-token loss）
    # labels: 右移一位，pad 为 -100 以屏蔽
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = tok.pad_token_id
    labels = labels.masked_fill(labels == tok.pad_token_id, -100)

    loss_hf = F.cross_entropy(
        logits_hf[:, :-1, :].contiguous().view(-1, V),
        labels[:, :-1].contiguous().view(-1),
        ignore_index=-100,
        reduction="mean",
    ).item()

    loss_meta = F.cross_entropy(
        logits_meta[:, :-1, :].contiguous().view(-1, V),
        labels[:, :-1].contiguous().view(-1),
        ignore_index=-100,
        reduction="mean",
    ).item()

    print(f"[Loss] HF CE={loss_hf:.6f} | META CE={loss_meta:.6f} | Δ={abs(loss_hf - loss_meta):.6e}")

    # 9) 打印最后一个位置 top-5 token 对齐情况
    k = 5
    last_idx = attn_mask[0].nonzero(as_tuple=False).max().item()
    top_hf   = torch.topk(logits_hf[0, last_idx], k)
    top_meta = torch.topk(logits_meta[0, last_idx], k)
    print("[Top-k @ last pos] HF:", list(zip(top_hf.indices.tolist(), [f"{v:.4f}" for v in top_hf.values.tolist()])))
    print("[Top-k @ last pos] META:", list(zip(top_meta.indices.tolist(), [f"{v:.4f}" for v in top_meta.values.tolist()])))

if __name__ == "__main__":
    main()