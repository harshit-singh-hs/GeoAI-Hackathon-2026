"""Compare the base SLM (few-shot prompts) with base + LoRA adapter (zero-shot prompts) on held-out data.

Parsing:  JSON validity and per-field accuracy against the synthetic labels.
Writing:  pass rate of the live system's grounding check and decision check, length and latency.

Usage: python finetune/eval.py --adapter finetune/out/geoai-qwen3.5-2b-lora [--n_parse 200 --n_narrate 60]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.narrate import check_decision, check_grounding, narrate_messages  # noqa: E402
from app.parse import _extract_json, parse_messages  # noqa: E402


def generate(model, tok, msgs, max_new_tokens):
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def norm(x):
    return x.strip().lower() if isinstance(x, str) else x


def eval_parse(model, tok, items, few_shot):
    keys = ["activity", "location", "profiles", "intent", "out_of_scope", "radius_km", "commute"]
    hit = {k: 0 for k in keys}
    valid = allok = 0
    t0 = time.time()
    for it in items:
        q = it["messages"][1]["content"]
        gold = json.loads(it["messages"][2]["content"])
        pred = _extract_json(generate(model, tok, parse_messages(q, few_shot=few_shot), 160)) or {}
        valid += bool(pred)
        ok = True
        for k in keys:
            g, p = gold.get(k), pred.get(k)
            if k == "profiles":
                same = sorted(g or []) == sorted(p or [])
            elif k == "radius_km":
                same = (g is None and p is None) or (g is not None and p is not None and float(g) == float(p))
            else:
                same = norm(g) == norm(p)
            hit[k] += same
            if k in ("activity", "location", "profiles", "out_of_scope"):
                ok &= same
        allok += ok
    n = len(items)
    return {"n": n, "json_valid": round(valid / n, 3), **{f"acc_{k}": round(v / n, 3) for k, v in hit.items()},
            "all_key_fields": round(allok / n, 3), "sec_per_query": round((time.time() - t0) / n, 2)}


def eval_narrate(model, tok, items, finetuned, show=2):
    g_ok = d_ok = both = 0
    lens, t0 = [], time.time()
    for i, it in enumerate(items):
        facts = it["messages"][1]["content"].split("FACTS:\n", 1)[1].rsplit("\n\nWrite the reply.", 1)[0]
        out = generate(model, tok, narrate_messages(facts, finetuned), 360)
        bad, wrong = check_grounding(out, facts), check_decision(out, facts)
        g_ok += not bad
        d_ok += wrong is None
        both += (not bad) and wrong is None
        lens.append(len(out.split()))
        if i < show:
            print(f"--- sample ({'ok' if not bad and not wrong else bad + [wrong]})\n{out}\n")
    n = len(items)
    return {"n": n, "grounding_pass": round(g_ok / n, 3), "decision_pass": round(d_ok / n, 3),
            "both_pass": round(both / n, 3), "avg_words": round(sum(lens) / n), "sec_per_answer": round((time.time() - t0) / n, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--n_parse", type=int, default=200)
    ap.add_argument("--n_narrate", type=int, default=60)
    args = ap.parse_args()
    held = [json.loads(l) for l in open(os.path.join(args.adapter, "heldout.jsonl"))]
    parse_items = [h for h in held if h["task"] == "parse"][: args.n_parse]
    narr_items = [h for h in held if h["task"] == "narrate"][: args.n_narrate]
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16).cuda().eval()
    report = {"base_model": args.base, "adapter": args.adapter}
    print("== base model, few-shot prompts")
    report["base"] = {"parse": eval_parse(model, tok, parse_items, True), "narrate": eval_narrate(model, tok, narr_items, False)}
    print(json.dumps(report["base"], indent=1))
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter).eval()
    print("== base + LoRA, zero-shot prompts")
    report["finetuned"] = {"parse": eval_parse(model, tok, parse_items, False), "narrate": eval_narrate(model, tok, narr_items, True)}
    print(json.dumps(report["finetuned"], indent=1))
    json.dump(report, open(os.path.join(args.adapter, "eval_report.json"), "w"), indent=2)
    print("saved", os.path.join(args.adapter, "eval_report.json"))


if __name__ == "__main__":
    main()
