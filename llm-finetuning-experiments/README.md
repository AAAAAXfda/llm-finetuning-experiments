# qlora-code-llm

Fine-tuning Llama-3.2-3B for code generation with QLoRA (4-bit quantized low-rank adaptation). I want to see how close parameter-efficient training gets to full fine-tuning, and whether it's practical on consumer GPUs.

## Goal

Benchmark QLoRA against full fine-tuning on HumanEval and MBPP, using Llama-3.2-3B as the base model. All weights and training logs will be released publicly on Hugging Face.

## Experiments

1. **Baseline:** Llama-3.2-3B, zero-shot
2. **QLoRA on CodeAlpaca-20k**
3. **QLoRA on a subset of OSS-Instruct**
4. **Full fine-tune** on CodeAlpaca-20k, for comparison with run 2

Evaluation is pass@1 on HumanEval and pass@10 on MBPP.

## Hardware

- **QLoRA runs:** these should fit on a single 24GB consumer GPU.
- **Full fine-tune:** this needs 40-80GB of VRAM. I'm using a single A100 80GB.

Each training run writes its peak VRAM to `peak_vram.txt` in the output folder, and I'll replace the "should fit" claim with the real number.

## Quickstart

```bash
pip install -r requirements.txt
huggingface-cli login          # Llama 3.2 is gated, request access first

python scripts/prepare_data.py --dataset codealpaca
python scripts/train.py --config configs/qlora_codealpaca.yaml
python scripts/evaluate.py --model meta-llama/Llama-3.2-3B \
    --adapter outputs/qlora-codealpaca --benchmark humaneval
```

`run.sh` runs the whole pipeline end to end.

**Heads up:** `evaluate.py` executes model-generated code. Run it in a container or throwaway VM.

## Repo layout

```
configs/    one YAML per experiment
scripts/    data prep, training, evaluation
results/    raw eval JSON and the results table
logs/       training logs
```

## Deliverables

- [ ] Training scripts
- [ ] Public model weights on Hugging Face
- [ ] Benchmark results table
- [ ] Technical writeup

## Stack

- transformers
- peft
- bitsandbytes
- trl
- datasets

## License

MIT for the code. The Llama 3.2 weights fall under Meta's own license.
