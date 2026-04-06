# Ordinary Meaning Tool

This tool determines the plain-English meaning of any word or phrase by
querying multiple AI models and evaluating the results with deterministic math.

## Key design principle

Models define the term **blind** — they never see the contract clause or
document. The contract is used only as a comparison target. This ensures
definitions reflect the word's ordinary meaning from training data, not
a paraphrase of any specific document.

## How it works

1. The user provides a term and the clause where it appears.
2. Three models (Claude, GPT, Perplexity) define the term using five
   prompting strategies. Models see only the word — never the clause.
3. Each strategy runs multiple rounds (autoresearch loop). The definition
   closest to the contract clause (by cosine similarity) survives.
4. All strategies are shown with two scores: context alignment (how well
   the blind definition maps onto the document) and term alignment (how
   close it is to the word's general meaning).
5. The gap between the two scores is the finding: small gap means the
   ordinary meaning fits the contract; large gap means the contract may
   use the term unusually.

## Files

- `app.py` — Streamlit web interface (entry point)
- `prepare.py` — LLM API helpers and embedding math (locked evaluator)
- `strategies.py` — Prompting strategies and prompt construction
- `extract.py` — PDF/DOCX/TXT text extraction
- `input.md` — Default term and context

## Running

```bash
streamlit run app.py
```
