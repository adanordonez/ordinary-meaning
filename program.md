# Ordinary Meaning

This is an experiment to have an LLM autonomously research the ordinary,
everyday meaning of a legal term or phrase.

Inspired by Judge Newsom's concurrence in Snell v. United Specialty Insurance
Co., 102 F.4th 1208 (11th Cir. 2024), which explored whether LLMs can help
determine the ordinary meaning of words used in legal instruments.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr4`). The branch `ordinary-meaning/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b ordinary-meaning/<tag>` from the current branch.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `program.md` — these instructions (you are reading this now).
   - `input.md` — the term and context paragraph. Set by the user. Do not modify.
   - `prepare.py` — locked evaluator, provider helpers, judge. Do not modify.
   - `train.py` — the file you modify. Prompt config, model selection, generation logic.
4. **Verify API keys**: Check that a `.env` file exists with at least `ANTHROPIC_API_KEY` set (required for the judge). `OPENAI_API_KEY` and `PERPLEXITY_API_KEY` enable additional models.
5. **Verify input.md**: Confirm that `input.md` contains a term and context paragraph. If not, ask the user to fill it in.
6. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
7. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Input format

The user provides the term and context in `input.md`. The format is:

```markdown
# Input

## Term

landscaping

## Context

The policy covers bodily injury arising from the insured's performance
of landscaping.
```

This file is read by `prepare.py` at runtime. You do NOT modify it.
The term and context can be anything — any word, any clause, any contract,
any statute. The tool is not specific to any legal scenario.

## Experimentation

Each experiment calls LLM APIs to generate a candidate definition, then
the locked judge scores it. You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: system prompt, user prompt, model selection, temperature, prompt structure, multi-step generation, chain-of-thought, examples, anything.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the locked judge, provider helpers, and input loading.
- Modify `input.md`. It contains the user's term and context. It is read-only.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the judge. The `evaluate_definition` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the highest composite_score.** The composite score is the average of four axes (fidelity, readability, completeness, neutrality), each scored 0-10 by the locked judge. Higher is better. A perfect score is 10.0.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

**The first run**: Your very first run should always be to establish the baseline, so you will run train.py as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
composite_score:  8.50
fidelity:         9
readability:      8
completeness:     8
neutrality:       9
model:            anthropic
temperature:      0.3
---
```

You can extract the key metric from the log file:

```
grep "^composite_score:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 5 columns:

```
commit	composite_score	status	model	description
```

1. git commit hash (short, 7 chars)
2. composite_score achieved (e.g. 8.50) — use 0.00 for crashes
3. status: `keep`, `discard`, or `crash`
4. model used (e.g. anthropic, openai, perplexity)
5. short text description of what this experiment tried

Example:

```
commit	composite_score	status	model	description
a1b2c3d	8.50	keep	anthropic	baseline
b2c3d4e	8.75	keep	openai	added context to prompt
c3d4e5f	8.25	discard	perplexity	tried chain-of-thought
d4e5f6g	0.00	crash	anthropic	bad prompt format
```

## The experiment loop

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^composite_score:\|^fidelity:\|^readability:\|^completeness:\|^neutrality:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up on that idea.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If composite_score improved (higher), you "advance" the branch, keeping the git commit
9. If composite_score is equal or worse, you git reset back to where you started

**Ideas to try** (not exhaustive — be creative):

- Change `MODEL` between "anthropic", "openai", and "perplexity"
- Rewrite `SYSTEM_PROMPT` — try different personas, constraints, instructions
- Rewrite `USER_PROMPT` — try dictionary-style, context-heavy, example-rich, contrastive
- Change `TEMPERATURE` — try 0.0, 0.1, 0.3, 0.5, 0.7, 1.0
- Add a multi-step approach — generate a draft, then refine it
- Include the context in the prompt so the model knows how the term is used
- Try asking for what the term includes AND excludes
- Try asking the model to avoid legal jargon entirely
- Try asking for a one-sentence definition vs. a detailed explanation
- Combine strategies that scored well individually

**Crashes**: If a run crashes (API error, bad JSON, etc.), use your judgment: If it's a typo or simple fix, fix it and re-run. If the idea is fundamentally broken, log "crash" and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be away and expects you to continue working indefinitely until you are manually stopped. You are autonomous. If you run out of ideas, think harder — try combining previous near-misses, try more radical prompt changes, try different models. The loop runs until the human interrupts you.
