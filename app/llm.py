"""The small language model (<= 3B params, open weights, run locally).

Backends:
  transformers  -- GPU or CPU, loads the HF repo (and an optional LoRA adapter) with from_pretrained
  llamacpp      -- CPU-friendly GGUF via llama-cpp-python (used for low-cost hosting)
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time

log = logging.getLogger("geoai.llm")

MODEL_ID = os.environ.get("GEOAI_MODEL", "Qwen/Qwen3.5-2B")
ADAPTER = os.environ.get("GEOAI_ADAPTER", "")  # optional LoRA adapter repo/path
BACKEND = os.environ.get("GEOAI_LLM_BACKEND", "transformers")
GGUF_PATH = os.environ.get("GEOAI_GGUF", "")


class LLM:
    def __init__(self):
        self.ready = False
        self.error: str | None = None
        self.model_id = MODEL_ID
        self.adapter = ADAPTER or None
        self.backend = BACKEND
        self._lock = threading.Lock()
        self._model = self._tok = None

    # -------------------------------------------------------------- loading
    def load(self) -> None:
        t0 = time.time()
        try:
            if self.backend == "llamacpp":
                self._load_llamacpp()
            else:
                self._load_transformers()
            self.ready = True
            log.info("LLM ready: %s (%s) in %.1fs", self.model_id, self.backend, time.time() - t0)
        except Exception as e:  # keep serving with rule-based parsing + templated answers
            self.error = f"{type(e).__name__}: {e}"
            log.exception("LLM failed to load; continuing without it")

    def _load_transformers(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dev = os.environ.get("GEOAI_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if dev == "cuda" else torch.float32
        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        try:
            model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=dtype)
        except (ValueError, KeyError):
            from transformers import AutoModelForImageTextToText  # multimodal checkpoints (e.g. Qwen3.5)
            model = AutoModelForImageTextToText.from_pretrained(self.model_id, dtype=dtype)
        if self.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, self.adapter)
            model = model.merge_and_unload()
        self._model = model.to(dev).eval()
        self._device = dev

    def _load_llamacpp(self) -> None:
        from llama_cpp import Llama

        self._model = Llama(model_path=GGUF_PATH, n_ctx=4096, n_threads=os.cpu_count(), verbose=False)
        self.model_id = f"{self.model_id} (GGUF: {os.path.basename(GGUF_PATH)})"

    # -------------------------------------------------------------- generation
    def _generate(self, messages: list[dict], max_new_tokens: int, temperature: float) -> str:
        with self._lock:
            if self.backend == "llamacpp":
                out = self._model.create_chat_completion(messages=messages, max_tokens=max_new_tokens,
                                                         temperature=max(temperature, 0.0))
                return out["choices"][0]["message"]["content"]
            import torch

            prompt = self._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                   enable_thinking=False)
            ids = self._tok(prompt, return_tensors="pt").to(self._device)
            kw = dict(max_new_tokens=max_new_tokens, pad_token_id=self._tok.eos_token_id)
            if temperature > 0:
                kw.update(do_sample=True, temperature=temperature, top_p=0.9)
            else:
                kw.update(do_sample=False)
            with torch.inference_mode():
                out = self._model.generate(**ids, **kw)
            return self._tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    async def chat(self, messages: list[dict], max_new_tokens: int = 400, temperature: float = 0.0) -> str | None:
        if not self.ready:
            return None
        try:
            return await asyncio.to_thread(self._generate, messages, max_new_tokens, temperature)
        except Exception:
            log.exception("generation failed")
            return None

    def info(self) -> dict:
        return {"model": self.model_id, "adapter": self.adapter, "backend": self.backend, "ready": self.ready,
                "error": self.error}
