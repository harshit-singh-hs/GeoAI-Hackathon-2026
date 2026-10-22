"""LoRA fine-tune of Qwen3.5-2B (text-only, 1.88B params) on the two tasks the assistant uses the SLM for:
query parsing (question -> JSON) and grounded answer writing (fact sheet -> answer).

Prompts are the zero-shot runtime prompts (no few-shot examples), so the fine-tuned model is also faster.
Loss is computed on the assistant turn only.

Usage: python finetune/train_lora.py --out finetune/out/geoai-qwen3.5-2b-lora
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.parse import PARSE_SYSTEM  # noqa: E402

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj", "gate_proj", "up_proj", "down_proj"]


def is_heldout(key: str, frac: float = 0.1) -> bool:
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % 1000 < frac * 1000


def load_examples(parse_path: str, narrate_path: str) -> tuple[list[dict], list[dict]]:
    train, held = [], []
    for line in open(parse_path):
        d = json.loads(line)
        msgs = [{"role": "system", "content": PARSE_SYSTEM}, {"role": "user", "content": d["query"]},
                {"role": "assistant", "content": json.dumps(d["label"], ensure_ascii=False)}]
        (held if is_heldout(d["query"]) else train).append({"task": "parse", "messages": msgs})
    for line in open(narrate_path):
        d = json.loads(line)
        (held if is_heldout(d["query"] + d["messages"][1]["content"][:200]) else train).append(
            {"task": "narrate", "messages": d["messages"]})
    return train, held


def encode(tok, msgs: list[dict], max_len: int) -> dict | None:
    prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    target = msgs[-1]["content"] + "<|im_end|>"
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    t_ids = tok(target, add_special_tokens=False)["input_ids"]
    if len(p_ids) + len(t_ids) > max_len:
        return None
    return {"input_ids": p_ids + t_ids, "labels": [-100] * len(p_ids) + t_ids}


def batches(items: list[dict], bs: int, pad_id: int, rng: random.Random):
    # Bucket by length to cut padding, then shuffle the buckets.
    items = sorted(items, key=lambda x: len(x["input_ids"]))
    groups = [items[i:i + bs] for i in range(0, len(items), bs)]
    rng.shuffle(groups)
    for g in groups:
        L = max(len(x["input_ids"]) for x in g)
        ids = torch.full((len(g), L), pad_id)
        lab = torch.full((len(g), L), -100)
        att = torch.zeros((len(g), L), dtype=torch.long)
        for i, x in enumerate(g):
            n = len(x["input_ids"])
            ids[i, :n] = torch.tensor(x["input_ids"])
            lab[i, :n] = torch.tensor(x["labels"])
            att[i, :n] = 1
        yield ids, lab, att


def lm_loss(model, ids, att, lab):
    """Cross-entropy on the assistant tokens only, computing logits just for the trailing positions that hold
    targets: with a ~248k-token vocabulary, full-sequence logits would dominate GPU memory."""
    first = (lab != -100).int().argmax(dim=1)  # first target position per row
    k = int(ids.shape[1] - first.min().item()) + 1
    logits = model(input_ids=ids, attention_mask=att, logits_to_keep=k).logits  # positions L-k .. L-1
    tgt = lab[:, ids.shape[1] - k + 1:]  # tokens predicted by those positions
    return torch.nn.functional.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                             tgt.reshape(-1), ignore_index=-100)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--parse", default="finetune/data/parse.jsonl")
    ap.add_argument("--narrate", default="finetune/data/narrate.jsonl")
    ap.add_argument("--out", default="finetune/out/geoai-qwen3.5-2b-lora")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=2304)
    args = ap.parse_args()
    torch.manual_seed(0)
    rng = random.Random(0)

    tok = AutoTokenizer.from_pretrained(args.base)
    train, held = load_examples(args.parse, args.narrate)
    enc = [e for e in (encode(tok, x["messages"], args.max_len) for x in train) if e]
    print(f"train examples: {len(enc)} (parse {sum(x['task'] == 'parse' for x in train)}, "
          f"narrate {sum(x['task'] == 'narrate' for x in train)}); held-out: {len(held)}")
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "heldout.jsonl"), "w") as f:
        for x in held:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16).cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    cfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, target_modules=TARGETS, task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

    steps_per_epoch = math.ceil(len(enc) / args.bs / args.accum)
    total = int(steps_per_epoch * args.epochs)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.04 * total)), total)
    model.train()
    step, micro, t0, run_loss = 0, 0, time.time(), []
    log = open(os.path.join(args.out, "train_log.jsonl"), "w")
    while step < total:
        for ids, lab, att in batches(enc, args.bs, tok.pad_token_id or tok.eos_token_id, rng):
            loss = lm_loss(model, ids.cuda(), att.cuda(), lab.cuda())
            (loss / args.accum).backward()
            run_loss.append(loss.item())
            micro += 1
            if micro % args.accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0 or step == total:
                avg = sum(run_loss) / len(run_loss)
                run_loss.clear()
                rec = {"step": step, "of": total, "loss": round(avg, 4), "lr": sched.get_last_lr()[0],
                       "elapsed_s": round(time.time() - t0)}
                print(rec, flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
            if step >= total:
                break
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    json.dump({"base_model": args.base, "epochs": args.epochs, "lr": args.lr, "batch": args.bs * args.accum,
               "lora_rank": args.rank, "lora_alpha": 2 * args.rank, "targets": TARGETS, "train_examples": len(enc),
               "heldout_examples": len(held), "minutes": round((time.time() - t0) / 60, 1)},
              open(os.path.join(args.out, "train_config.json"), "w"), indent=2)
    print(f"saved adapter -> {args.out} in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
