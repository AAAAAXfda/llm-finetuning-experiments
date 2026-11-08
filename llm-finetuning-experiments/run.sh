#!/usr/bin/env bash
# Runs the full pipeline. Needs a GPU, a Hugging Face login (Llama is gated),
# and ideally a sandboxed machine since evaluate.py runs generated code.
set -e

python scripts/prepare_data.py --dataset codealpaca
python scripts/prepare_data.py --dataset ossinstruct

python scripts/evaluate.py --model meta-llama/Llama-3.2-3B --benchmark humaneval
python scripts/evaluate.py --model meta-llama/Llama-3.2-3B --benchmark mbpp

for cfg in qlora_codealpaca qlora_ossinstruct full_finetune; do
  python scripts/train.py --config configs/$cfg.yaml
done

for run in qlora-codealpaca qlora-ossinstruct; do
  python scripts/evaluate.py --model meta-llama/Llama-3.2-3B --adapter outputs/$run --benchmark humaneval
  python scripts/evaluate.py --model meta-llama/Llama-3.2-3B --adapter outputs/$run --benchmark mbpp
done

# full fine-tune saves a whole model, so point --model at it, no adapter
python scripts/evaluate.py --model outputs/full-codealpaca --benchmark humaneval
python scripts/evaluate.py --model outputs/full-codealpaca --benchmark mbpp
