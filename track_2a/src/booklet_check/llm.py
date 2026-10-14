"""Thin adapter for an OpenAI-compatible Apertus endpoint.

Endpoint and key come only from the environment, first match wins:
  BASE_URL / API_KEY                    (what the official evaluation injects)
  LLM_BASE_URL / LLM_API_KEY            (Hack Apertus project template names)
  APERTUS_BASE_URL / APERTUS_API_KEY    (local development)
Model: LLM_NAME or APERTUS_MODEL; if unset, the first Apertus v1.5 8B instruct id that
the endpoint lists under /models, else `swiss-ai/apertus-v1.5-8b`.

Without a key the adapter runs in DRY MODE: no network call, zero tokens, and every
response is marked `dry=True` so no dry number can be mistaken for a model result.

Spend guard (local runs only): when BUDGET_LEDGER points to a file, every call appends its
token counts and estimated USD cost there, and calls stop once BUDGET_USD (default 1.50) is
reached across all runs.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = "swiss-ai/apertus-v1.5-8b"
# USD per million tokens (input, output), from the Public AI /models listing on 2026-10-14.
PRICES = {"8b": (0.10, 0.20), "70b": (0.82, 2.92)}


def _env(*names: str) -> str | None:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return None


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Reply:
    text: str
    input_tokens: int
    output_tokens: int
    ms: int
    dry: bool = False


class Apertus:
    def __init__(self):
        self.base_url = (_env("BASE_URL", "LLM_BASE_URL", "APERTUS_BASE_URL") or "").rstrip("/")
        self._key = _env("API_KEY", "LLM_API_KEY", "APERTUS_API_KEY")
        self.dry = not (self.base_url and self._key)
        self.model = _env("LLM_NAME", "APERTUS_MODEL")
        self.ledger = Path(os.environ["BUDGET_LEDGER"]) if os.environ.get("BUDGET_LEDGER") else None
        self.budget = float(os.environ.get("BUDGET_USD", "1.50"))
        self.run_tag = os.environ.get("RUN_TAG", "")
        if self.dry:
            print("[booklet-check] DRY MODE: no endpoint/key in the environment, no model calls are made; "
                  "labels come from a placeholder rule and are NOT model results.", file=sys.stderr)
            self.model = self.model or DEFAULT_MODEL
        elif not self.model:
            self.model = self._pick_model()

    # -- helpers -----------------------------------------------------------------------------
    def _request(self, path: str, payload: dict | None, timeout: int = 180) -> dict:
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(self.base_url + path, data=data, method="POST" if data else "GET",
                                     headers={"Authorization": f"Bearer {self._key}",
                                              "Content-Type": "application/json",
                                              "User-Agent": "booklet-check/0.2"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)

    def _pick_model(self) -> str:
        try:
            ids = [m["id"] for m in self._request("/models", None, timeout=30).get("data", [])]
        except Exception:
            return DEFAULT_MODEL
        # Only Apertus v1.5 is allowed; prefer it explicitly, then the 8B size, never "thinking".
        cands = [i for i in ids if "apertus" in i.lower() and "thinking" not in i.lower()]
        if cands:
            def score(i: str) -> tuple:
                li = i.lower()
                return ("1.5" in li or "v1_5" in li or "1-5" in li, "8b" in li, "instruct" in li)
            best = max(cands, key=score)
            if score(best)[0]:
                return best
        return DEFAULT_MODEL

    def price(self) -> tuple[float, float]:
        return PRICES["70b"] if "70b" in self.model.lower() else PRICES["8b"]

    def spent(self) -> float:
        if not self.ledger or not self.ledger.exists():
            return 0.0
        total = 0.0
        for line in self.ledger.read_text(encoding="utf-8").splitlines():
            if line.strip():
                total += json.loads(line).get("usd", 0.0)
        return total

    def _log(self, reply: Reply):
        if not self.ledger:
            return
        pin, pout = self.price()
        usd = (reply.input_tokens * pin + reply.output_tokens * pout) / 1e6
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "run": self.run_tag, "model": self.model,
                                "in": reply.input_tokens, "out": reply.output_tokens, "usd": round(usd, 7)}) + "\n")

    # -- main call ---------------------------------------------------------------------------
    def chat(self, messages: list[dict], max_tokens: int = 8) -> Reply:
        if self.dry:
            return Reply("", 0, 0, 0, dry=True)
        if self.ledger and self.spent() >= self.budget:
            raise BudgetExceeded(f"spend cap US${self.budget:.2f} reached (ledger {self.ledger})")
        payload = {"model": self.model, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
        t0 = time.time()
        last = None
        for attempt in range(7):
            try:
                out = self._request("/chat/completions", payload)
                break
            except urllib.error.HTTPError as e:
                last = e
                if e.code in (429, 500, 502, 503, 504):
                    time.sleep(min(2 ** attempt * 2, 60))
                    continue
                body = e.read().decode(errors="replace")[:300]
                raise RuntimeError(f"HTTP {e.code}: {body}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                last = e
                time.sleep(min(2 ** attempt * 2, 60))
        else:
            raise RuntimeError(f"endpoint failed after retries: {last}")
        ms = int((time.time() - t0) * 1000)
        usage = out.get("usage") or {}
        text = (out["choices"][0]["message"].get("content") or "").strip()
        reply = Reply(text, int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)), ms)
        self._log(reply)
        return reply
