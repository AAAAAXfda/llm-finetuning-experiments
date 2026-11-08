# qlora-code-llm

Fine-tuning Llama-3.2-3B for code generation with QLoRA (4-bit quantized low-rank adaptation). I want to see how close parameter-efficient training gets to full fine-tuning, and whether it's practical on consumer GPUs.

## Goal

Benchmark QLoRA against full fine-tuning on HumanEval and MBPP, using Llama-3.2-3B as the base model. All weights and training logs will be released publicly on Hugging Face.

## Experiments

1. **Baseline:** Llama-3.2-3B, zero-shot
2. **QLoRA on CodeAlpaca-20k**
3. **QLoRA on a subset of OSS-Instruct**
4. **Full fine-tune** on the same data, for comparison

Evaluation is pass@1 on HumanEval and pass@10 on MBPP.

## Hardware

- **QLoRA runs:** these should fit on a single 24GB consumer GPU.
- **Full fine-tune and the quantization config comparisons:** these need 40-80GB of VRAM. I'm using a single A100 80GB.

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
