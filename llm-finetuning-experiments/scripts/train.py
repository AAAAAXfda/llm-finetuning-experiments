"""Train with QLoRA or full fine-tuning, depending on the config.

Usage:
    python scripts/train.py --config configs/qlora_codealpaca.yaml
"""
import argparse
import os

import torch
import yaml
from datasets import load_from_disk
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    tok.pad_token = tok.pad_token or tok.eos_token

    data_path = os.path.join("data", cfg["dataset"])
    train_ds = load_from_disk(data_path)

    peft_config = None
    if cfg["method"] == "qlora":
        q = cfg["quantization"]
        bnb = BitsAndBytesConfig(
            load_in_4bit=q["load_in_4bit"],
            bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=getattr(torch, q["bnb_4bit_compute_dtype"]),
            bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_name"], quantization_config=bnb, device_map={"": 0}
        )
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg["training"]["gradient_checkpointing"]
        )
        l = cfg["lora"]
        peft_config = LoraConfig(
            r=l["r"],
            lora_alpha=l["alpha"],
            lora_dropout=l["dropout"],
            target_modules=l["target_modules"],
            bias="none",
            task_type="CAUSAL_LM",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_name"], torch_dtype=torch.bfloat16, device_map={"": 0}
        )

    t = cfg["training"]
    sft_args = SFTConfig(
        output_dir=cfg["output_dir"],
        run_name=cfg["run_name"],
        dataset_text_field="text",
        max_seq_length=t["max_seq_length"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        logging_steps=t["logging_steps"],
        save_steps=t["save_steps"],
        bf16=t["bf16"],
        gradient_checkpointing=t["gradient_checkpointing"],
        optim=t["optim"],
        seed=t["seed"],
        report_to=t["report_to"],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        tokenizer=tok,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    tok.save_pretrained(cfg["output_dir"])

    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"peak VRAM allocated: {peak:.1f} GB")
    with open(os.path.join(cfg["output_dir"], "peak_vram.txt"), "w") as f:
        f.write(f"{peak:.2f} GB\n")


if __name__ == "__main__":
    main()
