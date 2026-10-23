---
license: apache-2.0
base_model: Qwen/Qwen3.5-2B
library_name: peft
pipeline_tag: text-generation
language:
- en
tags:
- lora
- geoai
- outdoor-activities
- grounded-generation
- json-extraction
---

# GeoAI Outdoor Planner: Qwen3.5-2B LoRA adapter

A LoRA adapter for [`Qwen/Qwen3.5-2B`](https://huggingface.co/Qwen/Qwen3.5-2B), the small language model inside the
**GeoAI Outdoor Planner**. The planner answers questions like *"Is Saturday or Sunday better for a hike near
Lonavala?"* by scoring nearby places for every hour of the requested window, using live OpenStreetMap, weather and
air-quality data. The model does two jobs in that system:

1. **Parse** a free-form question into JSON: activity, place (copied verbatim), radius, time phrase, health profile
   (asthma, senior, child, pregnancy, heart), question type, commute flag, out-of-scope flag.
2. **Write** the reply from a *fact sheet* produced by the planner. It leads with the computed decision (go / not
   recommended / yes / no / which day), then gives the options, what to carry, what to watch for and when to avoid
   going. It uses only numbers and places that appear in the fact sheet.

It is not a general weather or travel model and holds no world knowledge the planner relies on; all facts come from data fetched at query time.

* Code, data generation and evaluation: https://github.com/harshit-singh-hs/GeoAI-Hackathon-2026
* Live endpoint: listed in the GitHub README

## Base model and changes

| | |
|---|---|
| Base | `Qwen/Qwen3.5-2B` (Apache-2.0), loaded text-only via `AutoModelForCausalLM` → `Qwen3_5ForCausalLM`, **1.88B parameters** |
| Method | LoRA, r=16, alpha=32, dropout 0.05 |
| Target modules | `q_proj, k_proj, v_proj, o_proj` (full attention), `in_proj_qkv, in_proj_z, out_proj` (Gated DeltaNet linear attention), `gate_proj, up_proj, down_proj` (MLP) |
| Training | 2 epochs, lr 2e-4 cosine, effective batch 16, bf16, gradient checkpointing; loss on assistant tokens only |
| Hardware | 1x NVIDIA RTX 4000 Ada (20 GB), 25 minutes |
| Prompts | zero-shot: the system prompts in the code (`app/parse.py: PARSE_SYSTEM`, `app/narrate.py: NARRATE_SYSTEM`) |

## Training data (generated programmatically, no other LLM involved)

* **Parsing, ~2,160 examples.** Questions assembled from templates: 11 activities × several phrasings, 150+ Indian
  neighbourhoods, cities and landmarks, 30+ time phrases, distances, health profiles and five question types, plus noise
  (lowercase, slang such as "tmrw evng", missing place or time, unsupported activities, off-topic questions).
  Labels are known by construction.
* **Answer writing, ~390 examples.** Fact sheets from real runs of the planner (live Open-Meteo forecasts, CAMS air
  quality, OSRM travel times, Nominatim geocoding; OpenStreetMap features from a Geofabrik extract). Weather scenarios
  (monsoon rain, afternoon thunderstorms, heatwave, winter fog, smog, clean air, strong wind) are applied to the fetched
  values before scoring, so the model sees conditions a single season doesn't offer. Fact sheets stay internally
  consistent because the planner computes them from the modified values. Target answers come from the planner's
  deterministic reference writer and had to pass the same grounding and decision checks used at run time.
* The hackathon rulebook's example questions are excluded.

## Evaluation (held-out, same checks as production)

**Held-out synthetic data** (200 parsing, 37 answer-writing examples; same checks as production):

| | Base Qwen3.5-2B, few-shot prompts | Base + LoRA, zero-shot prompts |
|---|---|---|
| Parse: valid JSON | 100.0% | 100.0% |
| Parse: activity correct | 77.0% | 100.0% |
| Parse: place correct | 99.0% | 100.0% |
| Parse: health profile correct | 93.5% | 100.0% |
| Parse: question type correct | 58.0% | 100.0% |
| Parse: all key fields correct | 71.5% | 100.0% |
| Answer: grounding check passed | 75.7% | 100.0% |
| Answer: decision check passed | 86.5% | 100.0% |
| Answer: both passed | 64.9% | 100.0% |

The held-out set comes from the same generators as the training data, so it measures how well the model learned the
task format, not robustness to arbitrary phrasing. The end-to-end test below is the more honest measure.

**End-to-end, 30 varied real-world questions** through the live system (10 rulebook examples, 14 other cities and phrasings, 6 messy or out-of-scope),
`scripts/eval_queries.py`:

| | Base, few-shot | Base + LoRA |
|---|---|---|
| Questions answered without error | 30 / 30 | 30 / 30 |
| Recommendation questions answered by the SLM *and* passing all checks | 20 / 24 | 24 / 24 |
| Clarify / decline / beyond-horizon handled gracefully | 6 / 6 | 6 / 6 |
| Median latency (RTX 4000 Ada) | 12.2 s | 8.6 s |

## Intended use

Inside the GeoAI Outdoor Planner pipeline, with the planner's fact sheets as input. In production, every answer is
re-checked: numbers, places, day words and air-quality labels must match the fact sheet, and the opening must carry
the computed decision. Answers that fail fall back to the deterministic writer.

**Out of scope:** answering weather, air-quality or travel questions *without* a fact sheet. The model would have to
rely on memory, which is exactly what the system is designed to avoid. Not medical advice.

## How to use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B")
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-2B", dtype="bfloat16")
model = PeftModel.from_pretrained(base, "NamasteWorld/geoai-outdoor-qwen3.5-2b-lora")

msgs = [{"role": "system", "content": PARSE_SYSTEM},  # from app/parse.py
        {"role": "user", "content": "my 70 yr old dad wants a walk near cubbon park tmrw morning"}]
prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
out = model.generate(**tok(prompt, return_tensors="pt"), max_new_tokens=160, do_sample=False)
```

## Limitations

* Trained on English questions about Indian places; other phrasings and regions work less reliably.
* Answer style follows the reference writer; it can still choose an imperfect comparison word, which is why the
  production system re-checks every answer.
* Inherits the base model's limitations and biases.

## License

Apache-2.0 (same as the base model). Training data derived from OpenStreetMap (ODbL, © OpenStreetMap contributors)
and Open-Meteo (CC BY 4.0).
