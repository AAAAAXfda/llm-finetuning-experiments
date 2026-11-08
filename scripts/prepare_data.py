"""Build a training dataset with a single "text" column.

Usage:
    python scripts/prepare_data.py --dataset codealpaca --max_samples 20000
"""
import argparse
import os

from datasets import load_dataset

from prompts import build_prompt

EOS = "<|end_of_text|>"


def format_codealpaca(ex):
    prompt = build_prompt(ex["instruction"], ex.get("input", ""))
    return {"text": prompt + ex["output"] + EOS}


def format_ossinstruct(ex):
    prompt = build_prompt(ex["problem"])
    return {"text": prompt + ex["solution"] + EOS}


DATASETS = {
    "codealpaca": ("sahil2801/CodeAlpaca-20k", format_codealpaca),
    "ossinstruct": ("ise-uiuc/Magicoder-OSS-Instruct-75K", format_ossinstruct),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, required=True)
    ap.add_argument("--max_samples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="data")
    args = ap.parse_args()

    hf_name, fmt = DATASETS[args.dataset]
    ds = load_dataset(hf_name, split="train").shuffle(seed=args.seed)
    ds = ds.select(range(min(args.max_samples, len(ds))))
    ds = ds.map(fmt, remove_columns=ds.column_names)

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, args.dataset)
    ds.save_to_disk(path)
    print(f"saved {len(ds)} examples to {path}")


if __name__ == "__main__":
    main()
