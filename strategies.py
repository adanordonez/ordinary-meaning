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
            "{document_addendum}"
        ),
    },
    "dictionary": {
        "description": "Ask the model to act as a dictionary editor. Tests whether a structured, sense-ordered definition scores differently.",
        "template": (
            'Define the word "{term}" as a dictionary editor would. '
            "Give the primary sense first, then note any secondary senses. "
            "Use plain language."
            "{document_addendum}"
        ),
    },
    "context": {
        "description": "Give the model the passage and ask what the term means in that passage. Tests context-grounded definitions.",
        "template": (
            'In the following passage, what does "{term}" mean in plain English?\n\n'
            "Passage:\n{context}\n\n"
            "Explain the ordinary meaning of that word as used here."
            "{document_addendum}"
        ),
    },
    "examples": {
        "description": "Ask for concrete examples of what the term includes and excludes. Tests whether example-driven definitions are more precise.",
        "template": (
            'What does "{term}" mean? Explain in plain English and give '
            "5 concrete, everyday examples of things that would and would not "
            "fall under this term."
            "{document_addendum}"
        ),
    },
    "contrastive": {
        "description": "Ask the model to define the boundaries -- what the term includes and what it does not. Tests edge-case awareness.",
        "template": (
            'What does "{term}" ordinarily include and exclude? '
            "Explain the boundaries of this term as a normal English speaker "
            "would understand them."
            "{document_addendum}"
        ),
    },
    "refine": {
        "description": "Give the model a current definition and ask it to improve it. Used for iterative refinement.",
        "template": (
            'Here is a current plain-English definition of "{term}":\n\n'
            "{current_definition}\n\n"
            "Improve this definition. Make it clearer, more readable, and more "
            "faithful to how ordinary people use this word. "
            "Keep it neutral and plain. Return only the improved definition."
            "{document_addendum}"
        ),
    },
    "full_document": {
        "description": "Give the model the clause and the full document. Ask what the term means in the clause using the document as background.",
        "template": (
            'The term "{term}" appears in the following contract clause:\n\n'
            "--- CLAUSE ---\n{context}\n--- END CLAUSE ---\n\n"
            "The full document this clause comes from is provided below for reference.\n\n"
            "--- FULL DOCUMENT ---\n{full_document}\n--- END DOCUMENT ---\n\n"
            'What is the ordinary, everyday meaning of "{term}" as used in the clause above? '
            "Use the full document only as background context to understand how the term "
            "is used. Provide a clear, plain-English definition."
        ),
    },
    "document_scope": {
        "description": "Give the model the entire document and ask what the term means based on how it is used throughout. Tests document-wide usage patterns.",
        "template": (
            'The term "{term}" appears in the document below.\n\n'
            "--- DOCUMENT ---\n{full_document}\n--- END DOCUMENT ---\n\n"
            'Based on how "{term}" is used throughout this document, what does it ordinarily '
            "mean in plain English? Note any consistent usage patterns you observe. "
            "Do not interpret it legally — just explain what a normal person would understand."
        ),
    },
    "document_contrastive": {
        "description": "Give the model the full document and ask what the term includes and excludes, citing specific parts of the document.",
        "template": (
            'Read the following document:\n\n'
            "--- DOCUMENT ---\n{full_document}\n--- END DOCUMENT ---\n\n"
            'Based on this document, what does "{term}" ordinarily include and exclude? '
            "Use specific references from the document to explain the boundaries of this term "
            "as a normal English speaker would understand them."
        ),
    },
}

STRATEGIES: dict[str, str] = {k: v["template"] for k, v in STRATEGY_INFO.items()}

DOCUMENT_ADDENDUM = (
    "\n\nFor reference, the term appears in this clause:\n"
    "--- CLAUSE ---\n{context}\n--- END CLAUSE ---\n\n"
    "And the full document is provided below:\n"
    "--- FULL DOCUMENT ---\n{full_document}\n--- END DOCUMENT ---\n\n"
    "Use the document as background context for how the term is used, "
    "but define the term in plain English as an ordinary person would understand it."
)


def build_prompt(
    strategy: str,
    term: str,
    context: str,
    current_definition: str = "",
    full_document: str = "",
) -> str:
    doc_only_strategies = {"full_document", "document_scope", "document_contrastive"}

    if full_document and strategy not in doc_only_strategies:
        addendum = DOCUMENT_ADDENDUM.format(context=context, full_document=full_document)
    else:
        addendum = ""

    template = STRATEGIES[strategy]
    return template.format(
        term=term,
        context=context,
        current_definition=current_definition,
        full_document=full_document,
        document_addendum=addendum,
    )


def strategy_names(has_document: bool = False) -> list[str]:
    doc_strategies = {"full_document", "document_scope", "document_contrastive"}
    if has_document:
        return list(STRATEGIES.keys())
    return [k for k in STRATEGIES if k not in doc_strategies]
