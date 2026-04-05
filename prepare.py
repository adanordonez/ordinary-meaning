"""
Ordinary Meaning — provider helpers and embedding math.

Contains:
  - load_input() — reads TERM and CONTEXT from input.md
  - Provider helpers (call_anthropic, call_openai, call_perplexity)
  - Embedding math (embed_text, cosine_sim, term_alignment, context_alignment, consensus_score)
"""

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
load_dotenv()

import anthropic
import openai

# ---------------------------------------------------------------------------
# Read term and context from input.md (DO NOT hardcode legal scenarios)
# ---------------------------------------------------------------------------

INPUT_FILE = Path(__file__).parent / "input.md"


def load_input() -> tuple[str, str]:
    if not INPUT_FILE.exists():
        return "", ""

    text = INPUT_FILE.read_text()
    term = ""
    context = ""
    current_section = None

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower() == "## term":
            current_section = "term"
            continue
        elif stripped.lower() == "## context":
            current_section = "context"
            continue
        elif stripped.startswith("## ") or stripped.startswith("# "):
            current_section = None
            continue

        if current_section == "term" and stripped:
            term = stripped
        elif current_section == "context":
            if context and stripped:
                context += " " + stripped
            elif stripped:
                context = stripped

    return term, context


TERM, CONTEXT = load_input()

# ---------------------------------------------------------------------------
# Provider helpers — used to generate candidate definitions
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    model: str
    provider: str
    text: str
    input_tokens: int
    output_tokens: int


def call_anthropic(
    prompt: str,
    system: str,
    temperature: float = 0.3,
    model: str = "claude-sonnet-4-6",
) -> LLMResponse:
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return LLMResponse(
        model=model,
        provider="anthropic",
        text=resp.content[0].text,
        input_tokens=resp.usage.input_tokens,
        output_tokens=resp.usage.output_tokens,
    )


def call_openai(
    prompt: str,
    system: str,
    temperature: float = 0.3,
    model: str = "gpt-5.4-nano",
) -> LLMResponse:
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_completion_tokens=1024,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    return LLMResponse(
        model=model,
        provider="openai",
        text=resp.choices[0].message.content,
        input_tokens=resp.usage.prompt_tokens,
        output_tokens=resp.usage.completion_tokens,
    )


def call_perplexity(
    prompt: str,
    system: str,
    temperature: float = 0.3,
    model: str = "sonar-pro",
) -> LLMResponse:
    client = openai.OpenAI(
        api_key=os.environ.get("PERPLEXITY_API_KEY"),
        base_url="https://api.perplexity.ai",
    )
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    return LLMResponse(
        model=model,
        provider="perplexity",
        text=resp.choices[0].message.content,
        input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
        output_tokens=resp.usage.completion_tokens if resp.usage else 0,
    )


# ---------------------------------------------------------------------------
# Embedding math — deterministic, no LLM opinions
# ---------------------------------------------------------------------------

_embed_cache: dict[str, np.ndarray] = {}


def embed_text(text: str) -> np.ndarray:
    if text in _embed_cache:
        return _embed_cache[text]

    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=[text],
    )
    vec = np.array(resp.data[0].embedding)
    _embed_cache[text] = vec
    return vec


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def term_alignment(definition: str, term: str) -> float:
    return cosine_sim(embed_text(definition), embed_text(term))


def context_alignment(definition: str, context: str) -> float:
    return cosine_sim(embed_text(definition), embed_text(context))


def consensus_score(definition: str, other_definitions: list[str]) -> float:
    if not other_definitions:
        return 0.0
    def_vec = embed_text(definition)
    sims = [cosine_sim(def_vec, embed_text(other)) for other in other_definitions]
    return sum(sims) / len(sims)
