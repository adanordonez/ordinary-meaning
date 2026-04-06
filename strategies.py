SYSTEM_PROMPT = (
    "You are a neutral language analyst. Your task is to explain what words "
    "and phrases mean in plain English as understood by ordinary people. "
    "Do not advocate for any interpretation. Do not take sides. "
    "Do not ask clarifying questions. Just provide the definition."
)

STRATEGY_INFO: dict[str, dict] = {
    "bare": {
        "description": "Ask for the plain meaning with no framing or context. Tests what the model produces from training alone.",
        "template": (
            'What is the ordinary, everyday meaning of "{term}"? '
            "Provide a clear, plain-English explanation that a normal person would understand."
        ),
    },
    "dictionary": {
        "description": "Ask the model to act as a dictionary editor. Tests whether a structured, sense-ordered definition scores differently.",
        "template": (
            'Define the word "{term}" as a dictionary editor would. '
            "Give the primary sense first, then note any secondary senses. "
            "Use plain language."
        ),
    },
    "context": {
        "description": "Give the model a generic framing — define the term as it would be used in a written agreement. No specific document is shown.",
        "template": (
            'If someone encountered the term "{term}" in a written agreement, '
            "what would they understand it to mean in plain English? "
            "Explain simply, as a normal person would."
        ),
    },
    "examples": {
        "description": "Ask for concrete examples of what the term includes and excludes. Tests whether example-driven definitions are more precise.",
        "template": (
            'What does "{term}" mean? Explain in plain English and give '
            "5 concrete, everyday examples of things that would and would not "
            "fall under this term."
        ),
    },
    "contrastive": {
        "description": "Ask the model to define the boundaries — what the term includes and what it does not. Tests edge-case awareness.",
        "template": (
            'What does "{term}" ordinarily include and exclude? '
            "Explain the boundaries of this term as a normal English speaker "
            "would understand them."
        ),
    },
}

STRATEGIES: dict[str, str] = {k: v["template"] for k, v in STRATEGY_INFO.items()}


def build_prompt(strategy: str, term: str) -> str:
    return STRATEGIES[strategy].format(term=term)


def strategy_names() -> list[str]:
    return list(STRATEGIES.keys())
