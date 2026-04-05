# How the Ordinary Meaning Tool Works

## What is this?

This tool figures out the plain-English, everyday meaning of a word or phrase as it appears in a legal document. It does this by asking multiple AI models to define the word, then using math to measure and compare the results.

No AI grades another AI. No human picks a winner. The math is the math.

---

## The Core Idea

Large language models (like GPT, Claude, and Perplexity) have been trained on billions of documents written by ordinary people: news articles, books, emails, forums, contracts, social media posts, legal filings, and more. When you ask one of these models "what does this word mean?", its answer reflects the collective usage of that word across all of that text. It is, in a meaningful sense, the average understanding of millions of English speakers.

This tool harnesses that. It asks multiple models -- each trained on different data, by different companies, using different methods -- to independently define the same word. Then it measures how those definitions relate to each other and to the word itself using embedding math.

If three independently trained models all converge on the same meaning, that is strong evidence you have found the ordinary meaning.

---

## Step by Step

### What You Provide

1. **A term** -- the word or phrase you want defined (e.g., "landscaping").
2. **A context passage** -- the sentence or clause from the document where the term appears.
3. **Optionally, the full document** -- the complete contract or filing, uploaded as a PDF, Word doc, or text file. This gives the models additional background but does not control the scoring.

### What the Tool Does

The experiment runs in four phases.

---

### Phase 1: The Autoresearch Loop

This is the core engine, inspired by Andrej Karpathy's AutoResearch pattern. For each model (Claude, GPT, Perplexity), the tool does the following:

1. It tries multiple **prompting strategies** -- different ways of asking the model to define the term. For example:
   - "What does this word mean?" (bare)
   - "Define this word as a dictionary editor would." (dictionary)
   - "What does this word mean in this passage?" (context)
   - "Give examples of what this word includes and excludes." (examples / contrastive)
   - And more, including document-aware strategies if you uploaded a full contract.

2. Each strategy produces a definition. That definition is then **embedded** -- converted into a numerical vector (a long list of numbers) by a separate embedding model (`text-embedding-3-small`).

3. The bare term itself (just the word, e.g., "landscaping") is also embedded into that same vector space.

4. The tool computes the **cosine similarity** between the definition's vector and the term's vector. This is a standard mathematical formula that measures how close two vectors are to each other, on a scale where 1.0 means identical and 0.0 means completely unrelated.

5. **Why the bare term?** The embedding of the bare word represents where that word naturally sits in the model's learned understanding of all of English. It is not narrowed to one contract or one clause. It reflects the word's meaning across the entire training corpus -- the "world's meaning." A definition that lands close to the bare word in this space is faithfully capturing what ordinary people understand that word to mean.

6. The tool keeps a running "best definition" for each model. If a new strategy produces a definition with higher term alignment than the current best, it replaces it. Otherwise, it is discarded. This is the autoresearch loop: try many approaches, keep only improvements.

At the end of Phase 1, each model has one surviving definition -- the one that was closest to the word's natural position in vector space.

---

### Phase 2: Results

Each model's surviving definition is displayed side by side with three independent measurements:

- **Term Alignment**: cosine similarity between the definition and the bare term. This is the primary measurement. It answers: "How well does this definition capture the word's meaning in the world?"

- **Context Alignment**: cosine similarity between the definition and the context passage. This answers: "How relevant is this definition to the specific document?" This is shown for reference. It is not used to pick or rank anything, because we do not want to narrow the meaning to one contract.

- **Consensus**: average cosine similarity between this model's definition and the other models' definitions. This answers: "Do the models agree with each other?" High consensus across independently trained models is strong evidence of ordinary meaning.

These three numbers are presented as-is. No formula combines them. No winner is declared. You read the definitions and the numbers and decide.

---

### Phase 3: Cross-Model Similarity Matrix

A table showing the pairwise cosine similarity between every pair of surviving definitions. For example, if Claude's definition and GPT's definition have a similarity of 0.94, they are saying nearly the same thing. If one pair is notably lower, those two models disagree about some aspect of the word's meaning.

The tool also reports the average pairwise similarity and flags whether the models show strong convergence, moderate agreement, or notable divergence.

---

### Phase 4: Qualitative Comparison

One single AI call (not for scoring) describes, in plain English, where the definitions agree and where they differ. This is a readability aid for the human -- it translates the math into sentences. It does not rank or pick a winner.

---

## What is NOT happening

- **No AI scores another AI.** There is no "judge model" assigning numerical ratings. All evaluation is cosine similarity -- a deterministic math formula.
- **No arbitrary weighting formula.** The tool does not blend the three measurements into a single "composite score." You see all three numbers independently.
- **No dictionary lookup.** The tool does not consult any dictionary. The definitions come purely from what the models learned during training.
- **No winner is declared.** The tool presents the results. You interpret them.

---

## Why This Approach

The ordinary meaning doctrine asks: what does this word mean to a normal person? Traditional practice sends judges to dictionaries -- but dictionaries are written by a handful of editors exercising personal judgment on a limited sample of text. Judges then pick among dictionaries, which itself involves discretion that is rarely explained.

This tool replaces that with:

1. **Broader evidence base.** Each model's training data is orders of magnitude larger than any dictionary's citation files.
2. **Multiple independent sources.** Three models from three companies with different training data and methods.
3. **Transparent process.** Every prompt, every raw output, every model ID, every temperature setting, and every mathematical score is visible and reproducible.
4. **Deterministic evaluation.** Cosine similarity is a formula. Given the same inputs, it returns the same number every time. There is no discretion in the evaluation step.

The models provide the linguistic evidence. The math measures it. You decide what it means.
