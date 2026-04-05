# Ordinary Meaning Tool

This tool determines the plain-English meaning of any word or phrase by
querying multiple AI models and evaluating the results with deterministic math.

## How it works

1. The user provides a term, the clause it appears in, and optionally a full document.
2. The tool asks three models (Claude, GPT, Perplexity) to define the term using
   eight different prompting strategies.
3. Each definition is converted to a vector embedding and compared to the contract
   clause using cosine similarity (context alignment).
4. The definition with the highest context alignment survives. All others are discarded.
5. The surviving definitions are shown side by side with three independent measurements:
   context alignment (document meaning), term alignment (world meaning), and consensus
   (cross-model agreement).

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
