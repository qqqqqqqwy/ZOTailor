import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

try:
    from transformers import AutoTokenizer
except ImportError as exc:
    raise RuntimeError(
        "transformers is required. Install transformers, sentencepiece, and "
        "huggingface_hub before preparing SST2 tokens."
    ) from exc


FORMAT_NAME = "sst2_tokens_v3"


@dataclass(frozen=True)
class EncodedSample:
    label: int
    negative_prompt_len: int
    negative_label_token_id: int
    negative_tokens: List[int]
    positive_prompt_len: int
    positive_label_token_id: int
    positive_tokens: List[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tokenize SST2 TSV files for the LoRA-FA Android runner."
    )
    parser.add_argument("--train_tsv", required=True, help="Raw SST2 training TSV.")
    parser.add_argument("--eval_tsv", required=True, help="Raw SST2 validation TSV.")
    parser.add_argument("--output_dir", default="mytest/sst2_tokens")
    parser.add_argument(
        "--hf_model",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Hugging Face model id or local tokenizer/model directory.",
    )
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True to Hugging Face AutoTokenizer.",
    )
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument(
        "--model_tag",
        default=None,
        help=(
            "Optional filename tag. For example, llama32_3b writes "
            "llama32_3b_sst2_train_tokens.tsv and "
            "llama32_3b_sst2_dev_tokens.tsv unless explicit names are passed."
        ),
    )
    parser.add_argument("--train_name", default=None)
    parser.add_argument("--eval_name", default=None)
    return parser.parse_args()


def choose_pad_token_id(tokenizer: object) -> int:
    for name in ("pad_token_id", "eos_token_id", "bos_token_id"):
        token_id = getattr(tokenizer, name, None)
        if token_id is not None:
            return int(token_id)
    return 0


def encode_verbalizer(tokenizer: object, text: str) -> List[int]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        raise RuntimeError(f"Tokenizer produced no ids for verbalizer text: {text!r}")
    return [int(token_id) for token_id in token_ids]


def default_output_name(model_tag: str | None, split: str) -> str:
    if model_tag:
        return f"{model_tag}_sst2_{split}_tokens.tsv"
    return f"sst2_{split}_tokens.tsv"


def read_raw_sst2_tsv(path: Path) -> Iterable[tuple[str, int]]:
    with open(path, "r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        if reader.fieldnames is None or not {"sentence", "label"}.issubset(
            reader.fieldnames
        ):
            raise RuntimeError(
                f"{path} must have a TSV header containing sentence and label columns"
            )

        for row in reader:
            sentence = (row.get("sentence") or "").strip()
            label_text = (row.get("label") or "").strip()
            try:
                label = int(label_text)
            except ValueError:
                continue
            if not sentence or label not in (0, 1):
                continue
            yield sentence, label


def encode_class_prompt(
    sentence_tokens: List[int],
    verbalizer_tokens: List[int],
    *,
    seq_len: int,
    pad_token_id: int,
) -> tuple[int, int, List[int]]:
    if not verbalizer_tokens:
        raise RuntimeError("Empty SST2 verbalizer token sequence")

    suffix_prefix_tokens = verbalizer_tokens[:-1]
    label_token_id = verbalizer_tokens[-1]
    max_sentence_tokens = seq_len - len(suffix_prefix_tokens)
    if max_sentence_tokens <= 0:
        raise RuntimeError(
            "seq_len is too small for the SST2 verbalizer prefix tokens"
        )

    prompt_tokens = sentence_tokens[:max_sentence_tokens] + suffix_prefix_tokens
    if not prompt_tokens:
        raise RuntimeError("Tokenizer produced an empty SST2 prompt")

    fixed_tokens = prompt_tokens + [pad_token_id] * (seq_len - len(prompt_tokens))
    return len(prompt_tokens), label_token_id, fixed_tokens


def encode_samples(
    raw_tsv: Path,
    tokenizer: object,
    *,
    seq_len: int,
    pad_token_id: int,
    max_samples: int,
    negative_verbalizer_tokens: List[int],
    positive_verbalizer_tokens: List[int],
) -> List[EncodedSample]:
    if seq_len <= 1:
        raise ValueError("seq_len must leave room for next-token classification")

    samples: List[EncodedSample] = []
    for sentence, label in read_raw_sst2_tsv(raw_tsv):
        sentence_tokens = [
            int(token_id)
            for token_id in tokenizer.encode(sentence, add_special_tokens=True)
        ]
        if not sentence_tokens:
            continue

        (
            negative_prompt_len,
            negative_label_token_id,
            negative_tokens,
        ) = encode_class_prompt(
            sentence_tokens,
            negative_verbalizer_tokens,
            seq_len=seq_len,
            pad_token_id=pad_token_id,
        )
        (
            positive_prompt_len,
            positive_label_token_id,
            positive_tokens,
        ) = encode_class_prompt(
            sentence_tokens,
            positive_verbalizer_tokens,
            seq_len=seq_len,
            pad_token_id=pad_token_id,
        )
        samples.append(
            EncodedSample(
                label=label,
                negative_prompt_len=negative_prompt_len,
                negative_label_token_id=negative_label_token_id,
                negative_tokens=negative_tokens,
                positive_prompt_len=positive_prompt_len,
                positive_label_token_id=positive_label_token_id,
                positive_tokens=positive_tokens,
            )
        )
        if len(samples) >= max_samples:
            break
    return samples


def write_token_tsv(
    path: Path,
    samples: List[EncodedSample],
    *,
    seq_len: int,
    pad_token_id: int,
    negative_verbalizer_tokens: List[int],
    positive_verbalizer_tokens: List[int],
) -> None:
    negative_token_columns = [f"negative_token_{index}" for index in range(seq_len)]
    positive_token_columns = [f"positive_token_{index}" for index in range(seq_len)]
    negative_label_token_id = negative_verbalizer_tokens[-1]
    positive_label_token_id = positive_verbalizer_tokens[-1]
    with open(path, "w", encoding="utf-8", newline="\n") as output:
        output.write(f"# {FORMAT_NAME}\n")
        output.write(
            "# negative_suffix_token_ids\t"
            f"{','.join(str(token_id) for token_id in negative_verbalizer_tokens)}\n"
        )
        output.write(
            "# positive_suffix_token_ids\t"
            f"{','.join(str(token_id) for token_id in positive_verbalizer_tokens)}\n"
        )
        output.write(f"seq_len\t{seq_len}\n")
        output.write(f"pad_token_id\t{pad_token_id}\n")
        output.write(f"negative_label_token_id\t{negative_label_token_id}\n")
        output.write(f"positive_label_token_id\t{positive_label_token_id}\n")

        writer = csv.writer(output, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "label",
                "negative_prompt_len",
                "negative_label_token_id",
                "positive_prompt_len",
                "positive_label_token_id",
                *negative_token_columns,
                *positive_token_columns,
            ]
        )
        for sample in samples:
            writer.writerow(
                [
                    sample.label,
                    sample.negative_prompt_len,
                    sample.negative_label_token_id,
                    sample.positive_prompt_len,
                    sample.positive_label_token_id,
                    *sample.negative_tokens,
                    *sample.positive_tokens,
                ]
            )


def main() -> None:
    args = parse_args()
    if args.max_samples <= 0:
        raise ValueError("max_samples must be positive")

    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_model,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )
    pad_token_id = choose_pad_token_id(tokenizer)
    negative_verbalizer_tokens = encode_verbalizer(tokenizer, " It was terrible")
    positive_verbalizer_tokens = encode_verbalizer(tokenizer, " It was great")

    train_samples = encode_samples(
        Path(args.train_tsv),
        tokenizer,
        seq_len=args.seq_len,
        pad_token_id=pad_token_id,
        max_samples=args.max_samples,
        negative_verbalizer_tokens=negative_verbalizer_tokens,
        positive_verbalizer_tokens=positive_verbalizer_tokens,
    )
    eval_samples = encode_samples(
        Path(args.eval_tsv),
        tokenizer,
        seq_len=args.seq_len,
        pad_token_id=pad_token_id,
        max_samples=args.max_samples,
        negative_verbalizer_tokens=negative_verbalizer_tokens,
        positive_verbalizer_tokens=positive_verbalizer_tokens,
    )
    if not train_samples or not eval_samples:
        raise RuntimeError("Tokenization produced an empty training or evaluation split")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_name = args.train_name or default_output_name(args.model_tag, "train")
    eval_name = args.eval_name or default_output_name(args.model_tag, "dev")
    train_path = output_dir / train_name
    eval_path = output_dir / eval_name
    write_token_tsv(
        train_path,
        train_samples,
        seq_len=args.seq_len,
        pad_token_id=pad_token_id,
        negative_verbalizer_tokens=negative_verbalizer_tokens,
        positive_verbalizer_tokens=positive_verbalizer_tokens,
    )
    write_token_tsv(
        eval_path,
        eval_samples,
        seq_len=args.seq_len,
        pad_token_id=pad_token_id,
        negative_verbalizer_tokens=negative_verbalizer_tokens,
        positive_verbalizer_tokens=positive_verbalizer_tokens,
    )

    print(f"train_tokens={train_path} samples={len(train_samples)}")
    print(f"eval_tokens={eval_path} samples={len(eval_samples)}")
    print(f"vocab_size={len(tokenizer)}")
    print(
        "seq_len={} pad_token_id={} negative_suffix_token_ids={} "
        "negative_label_token_id={} positive_suffix_token_ids={} "
        "positive_label_token_id={}".format(
            args.seq_len,
            pad_token_id,
            negative_verbalizer_tokens,
            negative_verbalizer_tokens[-1],
            positive_verbalizer_tokens,
            positive_verbalizer_tokens[-1],
        )
    )


if __name__ == "__main__":
    main()
