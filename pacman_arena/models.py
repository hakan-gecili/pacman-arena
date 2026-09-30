"""Decision models. Each one receives the same Payload, wrapped in the model's native input format.

- Laya (laya-mlx, Apple MLX): a typed `choice` question; option probabilities come from its
  calibrated decision head.
- tev1 and nimble (Ollama decision models): a prompt whose answer is one option letter; option
  probabilities come from the letter token's log-probabilities.
- greedy: no model; picks the path with the best heuristic value (a baseline).
"""
from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import requests

from .payload import LETTERS, Payload

LAYA_MODEL = "aac6fef/laya-mlx"
LAYA_REVISION = "20aed815fc6acde75733882e7ec0e3f28aeb9717"  # Hugging Face commit used for the published runs
OLLAMA_URL = "http://localhost:11434"
TEV_MODEL = "tev1:4b"
NIMBLE_MODEL = "nimble:latest"
# Ollama manifest digests of the models used for the published runs (see `python -m pacman_arena --check`).
KNOWN_DIGESTS = {
    "tev1:4b": "cef45ef93cf6df8bf32bdd689b0a8fd01f88ae9034d33ce890c54f77e4cd981e",
    "tev1:0.8b": "d45e875d63fed9465390a4eb9e55f51f470390a446667b55d0a075a15e0336bf",
    "nimble:latest": "24e550a16a7081881be2f1f0d91e8cc13a597472735c04119f035a0a85c67e0c",
}

Answer = dict[str, Any]  # letter, probabilities, latency_ms, input_tokens, prompt


@dataclass
class Contestant:
    key: str                       # laya | tev | nimble | greedy
    name: str                      # shown on screen and used for file names
    choose: Callable[[Payload], Answer]
    info: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------- native input formats

def tev_prompt(p: Payload) -> str:
    """tev1's decision layout: Task / State / lettered Options / Answer."""
    return "\n".join([f"Task: {p.instructions}", "State:", p.state, "Options:",
                      *[f"{k}) {v}" for k, v in p.criteria.items()], "Answer:"])


def nimble_prompt(p: Payload) -> str:
    """nimble's decision layout: a context plus a schema field whose choices have one-letter codes."""
    return "\n".join([f"Context:\n{p.state}", "", "Schema:", f"field: path - {p.instructions}", "choices:",
                      *[f"  {k}: {v}" for k, v in p.criteria.items()], "", "Requested field: path", "Answer:"])


def laya_question(p: Payload) -> dict[str, Any]:
    return {"path": {"type": "choice", "instructions": p.instructions, "criteria": p.criteria}}


# ---------------------------------------------------------------- Ollama

def ollama_models(url: str) -> dict[str, str]:
    """Pulled model name -> manifest digest."""
    try:
        tags = requests.get(f"{url}/api/tags", timeout=10).json().get("models", [])
    except requests.RequestException as error:
        raise SystemExit(f"Cannot reach Ollama at {url} ({error}). Start it with `ollama serve`.")
    return {m["name"]: m.get("digest", "") for m in tags}


def ollama_digest(url: str, model: str) -> str:
    models = ollama_models(url)
    for name in (model, f"{model}:latest"):
        if name in models:
            return models[name]
    raise SystemExit(f"Ollama model {model!r} is not pulled. Run `ollama pull {model}`.")


def ask_ollama(url: str, model: str, prompt: str, n_options: int) -> Answer:
    started = time.perf_counter()
    reply = requests.post(f"{url}/api/chat", json={
        "model": model, "messages": [{"role": "user", "content": prompt}], "stream": False,
        "think": False, "logprobs": True, "top_logprobs": 20, "keep_alive": "30m",
        "options": {"temperature": 0, "seed": 0, "num_predict": 2},
    }, timeout=300).json()
    latency_ms = (time.perf_counter() - started) * 1000.0
    if reply.get("error"):
        raise SystemExit(f"Ollama error ({model}): {reply['error']}")
    valid = set(LETTERS[:n_options])
    raw: dict[str, float] = {}
    for entry in (reply.get("logprobs") or [{}])[0].get("top_logprobs", []):
        token = entry["token"].strip()
        if token in valid and token not in raw:
            raw[token] = math.exp(entry["logprob"])
    total = sum(raw.values())
    probabilities = {k: v / total for k, v in sorted(raw.items())} if total else {}
    text = reply.get("message", {}).get("content", "").strip()
    letter = text[:1] if text[:1] in valid else (max(probabilities, key=probabilities.get) if probabilities else None)
    return {"letter": letter, "probabilities": probabilities, "latency_ms": latency_ms,
            "input_tokens": reply.get("prompt_eval_count", 0), "prompt": prompt}


def ollama_contestant(key: str, model: str, url: str, render: Callable[[Payload], str]) -> Contestant:
    digest = ollama_digest(url, model)

    def choose(p: Payload) -> Answer:
        return ask_ollama(url, model, render(p), len(p.criteria))

    return Contestant(key, model, choose, {"runtime": "ollama", "model": model, "digest": digest,
                                          "matches_published_digest": KNOWN_DIGESTS.get(model) == digest})


# ---------------------------------------------------------------- Laya

def load_laya(model: str, revision: str | None, dtype: str) -> Any:
    import laya_mlx
    with warnings.catch_warnings():
        # The checkpoint ships one calibration temperature outside laya-mlx's accepted range; the
        # library clamps it and warns. It only affects questions with 11+ options; fitted payloads have fewer.
        warnings.simplefilter("ignore", RuntimeWarning)
        return laya_mlx.load(model, dtype=dtype, revision=revision)


def laya_contestant(agent: Any, model: str, revision: str | None) -> Contestant:
    def choose(p: Payload) -> Answer:
        question = laya_question(p)
        started = time.perf_counter()
        result = agent.predict(p.state, question)
        latency_ms = (time.perf_counter() - started) * 1000.0
        answer = result["answers"]["path"]
        return {"letter": answer["choice"], "probabilities": answer["probabilities"], "latency_ms": latency_ms,
                "input_tokens": result["usage"]["input_tokens"],
                "prompt": f"state:\n{p.state}\nquestion: {question}"}

    return Contestant("laya", "Laya", choose, {"runtime": "laya-mlx", "model": model, "revision": revision})


# ---------------------------------------------------------------- baseline

def greedy_contestant() -> Contestant:
    def choose(p: Payload) -> Answer:
        best = max(range(len(p.options)), key=lambda i: p.options[i].value)
        letter = LETTERS[best]
        return {"letter": letter, "probabilities": {letter: 1.0}, "latency_ms": 0.0, "input_tokens": 0,
                "prompt": "(no model: highest heuristic path value)"}

    return Contestant("greedy", "greedy baseline", choose, {"runtime": "none", "model": "paths.path_value"})
