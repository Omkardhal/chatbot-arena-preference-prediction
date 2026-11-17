"""
LoRA fine-tune of a causal LLM as a 3-class preference classifier
(a wins / b wins / tie) for LLM Classification Finetuning.

Two modes (the competition has no internet, so train and infer separately):

  1) TRAIN  (notebook WITH internet, or locally):
       python llm_pref_lora.py --mode train --model google/gemma-2-9b-it --out lora_out
     -> saves LoRA adapter + score head to ./lora_out

  2) INFER  (Kaggle submission notebook, internet OFF):
       Add the base model and your lora_out folder as Kaggle datasets, then
       python llm_pref_lora.py --mode infer --model /kaggle/input/<base-model-path> \
              --adapter /kaggle/input/<your-lora-dataset> --data /kaggle/input/llm-classification-finetuning
     -> writes submission.csv (with a<->b swap test-time augmentation)

Needs: pip install transformers peft accelerate bitsandbytes datasets scipy scikit-learn
Pick the model to fit your GPU: e.g. Qwen/Qwen2.5-1.5B-Instruct or meta-llama/Llama-3.2-3B-Instruct
for a T4/16GB; gemma-2-9b-it (4-bit) for bigger GPUs (A100 / 2x T4 with patience).
"""
import argparse
import inspect
import json

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from sklearn.metrics import log_loss
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          BitsAndBytesConfig, DataCollatorWithPadding,
                          Trainer, TrainingArguments)

P_LEN, R_LEN = 128, 256  # token budget: prompt / each response  (~640 total; shorter = much faster on a T4)


def clean(x):
    try:
        return " ".join(s for s in json.loads(x) if s is not None)
    except Exception:
        return str(x)


def load(path):
    df = pd.read_csv(path)
    for c in ("prompt", "response_a", "response_b"):
        df[c] = df[c].map(clean)
    return df


def encode(tok, prompt, ra, rb):
    """[bos] <prompt>: ... <response_a>: ... <response_b>: ... [eos], each part truncated."""
    p = tok("<prompt>: " + prompt, add_special_tokens=False).input_ids[:P_LEN]
    a = tok("\n\n<response_a>: " + ra, add_special_tokens=False).input_ids[:R_LEN]
    b = tok("\n\n<response_b>: " + rb, add_special_tokens=False).input_ids[:R_LEN]
    ids = ([tok.bos_token_id] if tok.bos_token_id is not None else []) + p + a + b
    return ids + ([tok.eos_token_id] if tok.eos_token_id is not None else [])


def load_tok(model_name):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def load_model(model_name, tok, four_bit=True):
    kw = {}
    if four_bit:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto", **kw,
    )
    model.config.pad_token_id = tok.pad_token_id
    model.config.use_cache = False
    return model


# ----------------------------------------------------------------------------- train
def train(args):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    tok = load_tok(args.model)
    df = load(f"{args.data}/train.csv")
    if args.max_train and args.max_train < len(df):
        df = df.sample(args.max_train, random_state=42).reset_index(drop=True)
    print(f"training rows: {len(df)}")
    y = df[["winner_model_a", "winner_model_b", "winner_tie"]].values.argmax(1)

    # random a<->b swap on half the rows (label 0<->1) to kill position bias
    rng = np.random.RandomState(42)
    swap = rng.rand(len(df)) < 0.5
    ids, labels = [], []
    for i, (p, a, b) in enumerate(zip(df.prompt, df.response_a, df.response_b)):
        if swap[i]:
            a, b = b, a
            lab = {0: 1, 1: 0, 2: 2}[y[i]]
        else:
            lab = y[i]
        ids.append(encode(tok, p, a, b))
        labels.append(int(lab))

    perm = rng.permutation(len(ids))
    n_val = min(300, max(100, len(ids) // 5))  # small validation set: every eval pass costs minutes
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    mk = lambda idx: [{"input_ids": ids[i], "labels": labels[i]} for i in idx]
    train_ds, val_ds = mk(tr_idx), mk(val_idx)

    model = load_model(args.model, tok, four_bit=not args.no_4bit)
    if not args.no_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lcfg = LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, bias="none",
        task_type="SEQ_CLS",  # also trains + saves the classification head
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    # Zero-init the trainable classification head. A random head on an LLM's large final
    # hidden states gives huge logits (train loss ~10); starting at zero gives loss ~1.1.
    zeroed = 0
    for n, p in model.named_parameters():
        if "score" in n and "modules_to_save" in n and n.endswith("weight"):
            torch.nn.init.zeros_(p.data)
            zeroed += 1
    print(f"zero-initialised {zeroed} score-head tensor(s)  (expect 1)")

    def metrics(ev):
        logits, lab = ev
        return {"log_loss": log_loss(lab, softmax(logits, axis=1), labels=[0, 1, 2])}

    # newer transformers removed warmup_ratio (warmup_steps accepts a float ratio < 1 instead)
    if "warmup_ratio" in inspect.signature(TrainingArguments.__init__).parameters:
        warmup_kw = {"warmup_ratio": 0.05}
    else:
        warmup_kw = {"warmup_steps": 0.05}

    targs = TrainingArguments(
        output_dir=args.out + "_ckpt", num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs, per_device_eval_batch_size=args.bs,
        gradient_accumulation_steps=args.accum, learning_rate=args.lr,
        lr_scheduler_type="cosine", weight_decay=0.01, **warmup_kw,
        bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True, optim="paged_adamw_8bit" if not args.no_4bit else "adamw_torch",
        eval_strategy="steps", eval_steps=args.eval_steps, save_strategy="steps", save_steps=args.eval_steps,
        save_total_limit=2, logging_steps=25, report_to="none",
        load_best_model_at_end=True, metric_for_best_model="log_loss", greater_is_better=False,
    )
    trainer = Trainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics,
    )
    trainer.train()
    print("best val log loss:", trainer.state.best_metric)  # no second evaluation pass
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"saved adapter to {args.out}")


# ----------------------------------------------------------------------------- infer
@torch.no_grad()
def predict(model, tok, seqs, bs):
    order = np.argsort([len(s) for s in seqs])
    out = np.zeros((len(seqs), 3), dtype=np.float32)
    dev = next(model.parameters()).device
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        batch = tok.pad({"input_ids": [seqs[j] for j in idx]}, return_tensors="pt").to(dev)
        logits = model(**batch).logits.float().cpu().numpy()
        out[idx] = softmax(logits, axis=1)
    return out


def infer(args):
    from peft import PeftModel

    tok = load_tok(args.model)
    test = load(f"{args.data}/test.csv")
    model = load_model(args.model, tok, four_bit=not args.no_4bit)
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    seq = [encode(tok, p, a, b) for p, a, b in zip(test.prompt, test.response_a, test.response_b)]
    seq_sw = [encode(tok, p, b, a) for p, a, b in zip(test.prompt, test.response_a, test.response_b)]
    p = predict(model, tok, seq, args.bs)
    p_sw = predict(model, tok, seq_sw, args.bs)[:, [1, 0, 2]]  # swap back a<->b
    probs = (p + p_sw) / 2

    sub = pd.DataFrame({
        "id": test["id"], "winner_model_a": probs[:, 0],
        "winner_model_b": probs[:, 1], "winner_tie": probs[:, 2],
    })
    sub.to_csv("submission.csv", index=False)
    print(sub.head())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "infer"], required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default="lora_out")
    ap.add_argument("--out", default="lora_out")
    ap.add_argument("--data", default="/kaggle/input/competitions/llm-classification-finetuning")
    ap.add_argument("--max_train", type=int, default=0, help="train on a random subset of N rows (0 = all)")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--eval_steps", type=int, default=50)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--no_4bit", action="store_true", help="load in fp16/bf16 instead of 4-bit")
    a = ap.parse_args()
    train(a) if a.mode == "train" else infer(a)
