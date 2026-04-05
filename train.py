"""
Ordinary Meaning — definition generator. Single run, single score.
The agent modifies this file. Everything here is fair game.
Usage: uv run train.py
"""

from prepare import (
    TERM, CONTEXT,
    call_anthropic, call_openai, call_perplexity,
    evaluate_definition,
)

# ---------------------------------------------------------------------------
# Editable config (agent modifies these)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a neutral language analyst. Your task is to explain what words "
    "and phrases mean in plain English as understood by ordinary people. "
    "Do not advocate for any interpretation. Do not take sides. "
    "Do not ask clarifying questions. Just provide the definition."
)

USER_PROMPT = (
    'What is the ordinary, everyday meaning of "{term}"? '
    "Provide a clear, plain-English explanation that a normal person "
    "would understand."
)

MODEL = "anthropic"  # "anthropic", "openai", or "perplexity"
TEMPERATURE = 0.3

# ---------------------------------------------------------------------------
# Generate definition using current config
# ---------------------------------------------------------------------------

def generate_definition() -> str:
    providers = {
        "anthropic": call_anthropic,
        "openai": call_openai,
        "perplexity": call_perplexity,
    }
    call_fn = providers[MODEL]
    final_prompt = USER_PROMPT.format(term=TERM)
    resp = call_fn(final_prompt, SYSTEM_PROMPT, TEMPERATURE)
    print(f"model_used:       {resp.model}")
    print(f"provider:         {resp.provider}")
    print(f"input_tokens:     {resp.input_tokens}")
    print(f"output_tokens:    {resp.output_tokens}")
    print()
    print("--- DEFINITION ---")
    print(resp.text)
    print("--- END DEFINITION ---")
    print()
    return resp.text.strip()

# ---------------------------------------------------------------------------
# Main — generate one definition and score it
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not TERM or not CONTEXT:
        print("ERROR: input.md is missing or empty. Create it with a ## Term and ## Context section.")
        raise SystemExit(1)
    print(f'term:             {TERM}')
    print(f'model_config:     {MODEL}')
    print(f'temperature:      {TEMPERATURE}')
    print()

    candidate = generate_definition()

    scores = evaluate_definition(candidate)

    print("---")
    print(f"composite_score:  {scores.composite}")
    print(f"fidelity:         {scores.fidelity}")
    print(f"readability:      {scores.readability}")
    print(f"completeness:     {scores.completeness}")
    print(f"neutrality:       {scores.neutrality}")
    print(f"model:            {MODEL}")
    print(f"temperature:      {TEMPERATURE}")
    print()
    print("fidelity_reason:    ", scores.fidelity_reason)
    print("readability_reason: ", scores.readability_reason)
    print("completeness_reason:", scores.completeness_reason)
    print("neutrality_reason:  ", scores.neutrality_reason)
