from datasets import load_dataset
from pathlib import Path

out_dir = Path("sst2")
out_dir.mkdir(exist_ok=True)

ds = load_dataset("glue", "sst2")

def export(split, out_name):
    with open(out_dir / out_name, "w", encoding="utf-8", newline="\n") as f:
        f.write("sentence\tlabel\n")
        for ex in ds[split]:
            sentence = ex["sentence"].replace("\t", " ").replace("\n", " ").strip()
            label = int(ex["label"])
            f.write(f"{sentence}\t{label}\n")

export("train", "train.tsv")
export("validation", "dev.tsv")