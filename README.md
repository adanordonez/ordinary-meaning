# Ordinary Meaning Tool

A transparent, reproducible protocol for determining the plain-English meaning of legal terms using multiple AI models and mathematical evaluation.

![How it works](ordinary-meaning-diagram.png)

Inspired by Andrej Karpathy's [AutoResearch](https://github.com/karpathy/autoresearch) loop — a system that lets an AI agent autonomously iterate on machine-learning code, keeping only the improvements. This tool adapts that pattern for legal interpretation: instead of optimizing training code, it optimizes definitions of words.

## What it does

You give it a word and the sentence from the contract where it appears. The tool asks three AI models — Claude (Anthropic), GPT (OpenAI), and Sonar Pro (Perplexity) — to define that word in plain English.

**The models never see the contract.** They define the word blind, based only on their training. The contract clause is used purely for measurement — to see how well the ordinary meaning naturally maps onto the document.

## How the loop works

```
For each model (Claude, GPT, Perplexity):
    For each strategy (bare, dictionary, context, examples, contrastive):
        For each round (1 to N):
            1. Ask the model: "What does this word mean?" (model sees only the word)
            2. Convert the definition to a vector (1,536 numbers)
            3. Convert the contract clause to a vector (the model never saw this)
            4. Compute cosine similarity between them
            5. If this score > current best for this strategy → keep it
            6. If not → discard it
        Best-of-N definition survives for this strategy
    All strategies shown — nothing hidden, no winner declared
```

## What it measures

| Metric | What it compares | What it tells you |
|---|---|---|
| **Context Alignment** | Blind definition vs. the contract clause | Does the ordinary meaning naturally fit the document? |
| **Term Alignment** | Blind definition vs. the bare word | Does the definition match the word's general meaning? |
| **Gap** | Difference between term and context alignment | Is the contract using the word normally or unusually? |
| **Consensus** | Same strategy across different models | Do independent models agree on the meaning? |

### Reading the gap

- **Small gap** — the ordinary meaning naturally covers the contract's usage. The word means what it normally means.
- **Large gap** — the contract may be using the word in a specific, narrowed, or unusual way that doesn't match how ordinary people understand it.

## Quick start

**Requirements:** Python 3.10+, API keys for OpenAI, Anthropic, and Perplexity.

```bash
git clone https://github.com/adanordonez/ordinary-meaning.git
cd ordinary-meaning

pip install -r requirements.txt

# Create .env with your API keys
echo 'OPENAI_API_KEY=sk-your-key' >> .env
echo 'ANTHROPIC_API_KEY=sk-ant-your-key' >> .env
echo 'PERPLEXITY_API_KEY=pplx-your-key' >> .env

streamlit run app.py
```

## Prompting strategies

All strategies ask the model to define the word without any document context. The model defines the word from its training alone.

| Strategy | What it asks |
|---|---|
| **Bare** | "What does this word mean?" — no framing at all |
| **Dictionary** | "Define this like a dictionary editor" |
| **Context** | "If someone saw this term in a written agreement, what would they understand?" |
| **Examples** | "Give examples of what it includes and doesn't" |
| **Contrastive** | "What does it include and exclude?" |

## The math

Evaluation uses [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) on vectors from OpenAI's `text-embedding-3-small` model:

```
similarity = dot(A, B) / (‖A‖ × ‖B‖)
```

Both texts are converted to 1,536-dimensional vectors. The formula measures how close they are. Same inputs always produce the same output.

## Design choices

- **Blind generation.** Models never see the contract. They define the word purely from training. The contract is the measuring stick, not the input.
- **Three providers, not one.** A single model's output is an anecdote. Three independent models converging is evidence.
- **No LLM-based scoring.** Models generate definitions. Math evaluates them.
- **All strategies shown.** No strategy is hidden or discarded. Every prompt type's best result is visible.
- **Autoresearch rounds within each strategy.** Multiple rounds smooth out randomness and find each strategy's best output.
- **The gap is the finding.** The difference between term alignment and context alignment tells you whether the contract uses the word normally or unusually.

## Inspiration

This tool adapts [Karpathy's AutoResearch](https://github.com/karpathy/autoresearch) pattern:

| | Karpathy's AutoResearch | This tool |
|---|---|---|
| **Domain** | ML training code | Legal word definitions |
| **What changes each round** | Model architecture, hyperparameters | Prompting strategy, LLM randomness |
| **What's locked** | `prepare.py` (evaluation harness) | Cosine similarity (embedding math) |
| **Metric** | val_bpb (lower is better) | Context alignment (higher is better) |
| **Loop** | Edit code → train → measure → keep or revert | Generate blind definition → embed → measure → keep or discard |
| **Key difference** | Model sees the problem | Model is blind — it never sees what it's being compared to |

## Limitations

- All scoring runs through one embedding model (`text-embedding-3-small`).
- Cosine similarity measures relatedness, not definition quality.
- AI training data is not curated for legal use.
- Words change meaning over time; the models reflect current usage.

## License

MIT
