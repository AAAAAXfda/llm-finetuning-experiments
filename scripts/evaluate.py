"""Evaluate on HumanEval (pass@1) or MBPP (pass@10).

Usage:
    # baseline
    python scripts/evaluate.py --model meta-llama/Llama-3.2-3B --benchmark humaneval
    # fine-tuned QLoRA adapter
    python scripts/evaluate.py --model meta-llama/Llama-3.2-3B \
        --adapter outputs/qlora-codealpaca --benchmark mbpp --n 10

WARNING: this executes model-generated code. Run it in a container or
throwaway VM, not on a machine with anything you care about.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompts import build_prompt


def pass_at_k(n, c, k):
    """Unbiased estimator from the Codex paper."""
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def extract_code(text):
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1)
    # model may keep going after the answer, cut at the next section header
    return text.split("### Instruction:")[0]


def run_program(program, timeout=10):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(program)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, timeout=timeout)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(path)


def load_model(model_name, adapter):
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map={"": 0}
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    model.eval()
    return model, tok


@torch.no_grad()
def generate(model, tok, prompt, n, temperature, max_new_tokens):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    sampling = temperature > 0
    out = model.generate(
        **inputs,
        do_sample=sampling,
        temperature=temperature if sampling else None,
        top_p=0.95 if sampling else None,
        num_return_sequences=n,
        max_new_tokens=max_new_tokens,
        pad_token_id=tok.pad_token_id,
    )
    gen = out[:, inputs["input_ids"].shape[1]:]
    return tok.batch_decode(gen, skip_special_tokens=True)


def humaneval_tasks():
    ds = load_dataset("openai/openai_humaneval", split="test")
    for ex in ds:
        instr = "Complete the following Python function.\n\n" + ex["prompt"]

        def make_program(completion, ex=ex):
            code = extract_code(completion)
            if f"def {ex['entry_point']}" not in code:
                code = ex["prompt"] + code
            return f"{code}\n\n{ex['test']}\n\ncheck({ex['entry_point']})\n"

        yield ex["task_id"], build_prompt(instr), make_program


def mbpp_tasks():
    ds = load_dataset("google-research-datasets/mbpp", "full", split="test")
    for ex in ds:
        instr = (
            f"{ex['text']}\nYour code should pass this test:\n{ex['test_list'][0]}"
        )

        def make_program(completion, ex=ex):
            code = extract_code(completion)
            tests = "\n".join(ex["test_list"])
            return f"{ex['test_setup_code']}\n{code}\n\n{tests}\n"

        yield ex["task_id"], build_prompt(instr), make_program


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--benchmark", choices=["humaneval", "mbpp"], required=True)
    ap.add_argument("--n", type=int, default=None, help="samples per task")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None, help="only first N tasks (debug)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    # defaults match the planned setup: HumanEval pass@1 (greedy), MBPP pass@10
    if args.benchmark == "humaneval":
        n = args.n or 1
        k = args.k or 1
        temp = 0.0 if args.temperature is None else args.temperature
        tasks = humaneval_tasks()
    else:
        n = args.n or 10
        k = args.k or 10
        temp = 0.8 if args.temperature is None else args.temperature
        tasks = mbpp_tasks()

    model, tok = load_model(args.model, args.adapter)

    scores, records = [], []
    for i, (task_id, prompt, make_program) in enumerate(tasks):
        if args.limit and i >= args.limit:
            break
        completions = generate(model, tok, prompt, n, temp, args.max_new_tokens)
        passed = [run_program(make_program(c)) for c in completions]
        scores.append(pass_at_k(n, sum(passed), k))
        records.append({"task_id": task_id, "passed": passed, "completions": completions})
        print(f"[{i + 1}] {task_id}: {sum(passed)}/{n} passed")

    result = float(np.mean(scores))
    print(f"\n{args.benchmark} pass@{k} = {result:.4f} ({len(scores)} tasks)")

    os.makedirs(args.out, exist_ok=True)
    tag = os.path.basename(args.adapter.rstrip("/")) if args.adapter else "baseline"
    with open(os.path.join(args.out, f"{args.benchmark}_{tag}.json"), "w") as f:
        json.dump(
            {"benchmark": args.benchmark, "model": args.model, "adapter": args.adapter,
             "n": n, "k": k, "temperature": temp, "score": result, "records": records},
            f, indent=2,
        )


if __name__ == "__main__":
    main()
