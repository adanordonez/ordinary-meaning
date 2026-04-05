## The Experiment: An Ordinary Meaning Protocol

The arguments above are interesting in theory. But without a working demonstration, they remain exactly that—theory. This Section presents a functioning prototype: an open-source tool, built for this paper, that asks multiple AI models to define a word and then uses math—not opinions—to measure and compare their answers.

[**Figure 1**: High-level diagram of the Ordinary Meaning tool. See `ordinary-meaning-diagram.png`.]

### A. Inspiration: Karpathy's AutoResearch Loop

The tool is modeled on Andrej Karpathy's AutoResearch—an open-source system that automates machine-learning research.[^karpathy] Karpathy's system works as follows. There are three files. The first, `train.py`, contains the code that trains a small language model. An AI agent is allowed to edit this file—changing the model architecture, the optimizer, the learning rate, or anything else it thinks might help. The second, `prepare.py`, contains the evaluation harness: the code that measures how well the model performs. This file is locked. The agent cannot touch it. The third, `program.md`, contains plain-English instructions telling the agent what to try and what to prioritize.

The loop is simple. The agent reads its instructions, makes a change to the training code, and runs the experiment. The evaluation harness measures the result using a fixed metric called "validation bits per byte"—a number that captures how well the model predicts text it has never seen. Lower is better. If the change improved the score, it is kept. If not, the code is reverted to its previous state. Then the agent tries again. Karpathy ran this loop for two days. The agent conducted roughly 700 experiments and found 20 genuine improvements, producing an 11% efficiency gain on code that was already well-tuned.[^karpathy-results]

[^karpathy]: Andrej Karpathy, AutoResearch (2025), https://github.com/karpathy/autoresearch.
[^karpathy-results]: *Id.* (reporting results from a two-day autonomous run).

The core insight is the separation between *generation* and *evaluation*. The agent generates ideas. A locked, deterministic metric evaluates them. The agent never grades its own work. The metric never changes. This separation is what makes the loop trustworthy: the evaluation cannot be gamed, because the agent cannot modify the evaluator.

**What we borrowed.** Our tool adopts this same separation. The AI models generate definitions—that is their job, and the only job they are given. A separate, locked mathematical formula evaluates the definitions. No model scores another model's output. The evaluation is cosine similarity: a deterministic formula that, given the same inputs, always returns the same number. Like Karpathy's validation bits per byte, it cannot be influenced by the thing being evaluated.

We also borrowed the iterative loop structure. Each model does not generate just one definition. It generates many, using different prompting strategies—different ways of asking the same question. After each attempt, the tool measures the result and keeps only the best. This is the same try-measure-keep-or-discard cycle that Karpathy's agent runs on training code.

**Where we differ.** Karpathy's system optimizes a single number: validation bits per byte. Lower is better. There is one right direction. Our tool does not reduce the result to a single number. Instead, it presents three independent measurements—context alignment, term alignment, and consensus—without combining them into a composite score. The interpreter sees all three and decides what weight to give each one.

Karpathy's system also uses a single model (the AI agent) making changes to a single file. Our tool uses three different models from three different companies, each running independently. This is deliberate: a single model's definition is an anecdote; three models converging is evidence.

Finally, Karpathy's system operates in a domain—machine learning—where there is an objective ground truth (does the model predict text better or worse?). Ordinary meaning has no such ground truth. The tool cannot tell you the "right" definition. It can only show you what three independent models produced, how similar those definitions are to the document and to each other, and where they agree and disagree. The interpretation remains with the human.

### B. What the Tool Does

The user provides three things: a word or phrase to define, the sentence from the contract where it appears, and optionally the full contract itself. The tool then asks three AI models—Claude (Anthropic), GPT (OpenAI), and Sonar Pro (Perplexity)—to define that word in plain English.

But it does not just ask once. It asks each model the same question in several different ways, because how you ask a question affects the answer you get. These different ways of asking are called "prompting strategies." Each one approaches the definition from a different angle. Here are the main ones, shown with the actual prompt text the model receives.

Every prompt begins with the same instruction, sent to every model before any question:

> "You are a neutral language analyst. Your task is to explain what words and phrases mean in plain English as understood by ordinary people. Do not advocate for any interpretation. Do not take sides. Do not ask clarifying questions. Just provide the definition."

Then the tool sends one of the following questions:

**Bare** — Ask for the meaning with no context at all. This tests what the model produces purely from its training.

> What is the ordinary, everyday meaning of "storage areas"? Provide a clear, plain-English explanation that a normal person would understand.

**Dictionary** — Ask the model to act like a dictionary editor.

> Define the word "storage areas" as a dictionary editor would. Give the primary sense first, then note any secondary senses. Use plain language.

**Context** — Give the model the actual sentence from the contract and ask what the word means there.

> In the following passage, what does "storage areas" mean in plain English?
>
> Passage: Contractor shall provide Subcontractor with suitable storage areas at the Project site for Subcontractor's materials and equipment, and shall afford Subcontractor reasonable access to the Project site for performance of the Work.
>
> Explain the ordinary meaning of that word as used here.

**Examples** — Ask for concrete examples of what the word covers and what it does not.

> What does "storage areas" mean? Explain in plain English and give 5 concrete, everyday examples of things that would and would not fall under this term.

**Contrastive** — Ask the model to draw boundaries around the word.

> What does "storage areas" ordinarily include and exclude? Explain the boundaries of this term as a normal English speaker would understand them.

When the user uploads a full contract, the tool appends the entire document to every one of these prompts. That way, even when the model is asked a simple question like "what does this word mean?", it knows the word comes from a construction contract—not a lease, not an insurance policy, not a news article. The model always has the full picture.

There are also three additional strategies that are only available when a document is uploaded. One asks the model to define the term based on how it is used throughout the document. Another asks the model to define the term specifically as used in the contract clause, with the full document as background. A third asks the model to explain what the term includes and excludes by pointing to specific parts of the document.

### C. How the Tool Picks the Best Definition

Each model generates one definition per strategy. That means each model produces five to eight definitions, depending on whether a document was uploaded. The tool needs to figure out which one is the best for each model.

It does this using a technique from computational linguistics. Every piece of text—whether it is a single word, a sentence, or a full paragraph—can be converted into a list of numbers called a "vector." This conversion is done by a separate AI model (OpenAI's text-embedding-3-small) that was trained on a massive amount of written English. The key property of this conversion is that texts which mean similar things end up with similar numbers. Texts which mean different things end up with different numbers.

Once the definition and the contract passage have both been converted into numbers, the tool computes how similar those two lists of numbers are. This is done with a standard formula called cosine similarity, which has been used in search engines, recommendation systems, and text analysis for decades. The formula produces a number between 0 and 1. A score of 1 means the two texts are identical in meaning. A score of 0 means they are completely unrelated. Most related texts fall somewhere between 0.3 and 0.9.

The tool compares each definition to the contract passage where the term appears. The definition that is most similar to the passage—the one that best captures what the word means in the context of that document—becomes the model's surviving definition. All others are discarded.

This is the "autoresearch" part. The tool automatically tries every prompting strategy, measures the result, and keeps only the best. No human needs to manually read and compare eight definitions per model.

### D. What the Tool Shows at the End

After the loop finishes, each model has one surviving definition. The tool then shows all three definitions side by side, with three measurements for each:

**Context alignment** measures how close the definition is to the contract passage. This is the number that drove the selection process. A higher number means the definition more closely captures what the term means in this particular document.

**Term alignment** measures how close the definition is to the bare word itself—just "storage areas" with nothing around it. The bare word, when converted to numbers, represents where that word naturally sits in the AI's learned understanding of all of English. This is not narrowed to one contract. It reflects how millions of people use that word across billions of documents. A higher number means the definition captures the word's general, everyday meaning—the "world meaning."

**Consensus** measures how much the three models agree with each other. If Claude, GPT, and Perplexity—three models built by different companies, trained on different data—all produce definitions that end up in the same region of the number space, that is strong evidence that the ordinary meaning has been found.

No formula combines these three numbers into a single score. No winner is declared. They are presented as-is. The interpreter reads the definitions, reads the numbers, and decides.

The tool also produces a similarity matrix showing how close each model's definition is to every other model's definition, and a plain-English comparison that describes where the definitions agree and where they differ.

### E. Results: "Storage Areas" in a Construction Contract

The tool was run against a standard contractor–subcontractor agreement. The term was "storage areas." The context passage was Section 3.3. The full contract was uploaded.

| Model | Surviving Strategy | Context Alignment | Term Alignment | Consensus |
|---|---|---|---|---|
| Claude Sonnet 4.6 | document_contrastive | 0.6295 | 0.5236 | 0.8913 |
| GPT-5.4 Nano | document_scope | 0.6855 | 0.5105 | 0.8966 |
| Perplexity Sonar Pro | document_scope | 0.6936 | 0.5941 | 0.8960 |

All three models converged on the same core meaning: storage areas are physical spaces on the job site, provided by the contractor, where the subcontractor can keep its materials and equipment during the project. The average similarity between the three definitions was 0.8946—moderate-to-strong agreement.

The differences were at the margins. Claude was the only model to explicitly define what storage areas *exclude*—active work areas, office space, and space for other parties' materials. GPT was the only model to note that the term carries no implication of legal rights or ownership; it is purely operational. Perplexity was the only model to introduce a security dimension ("safe," "without getting damaged") that is not explicitly stated in the contract, and the only one to offer concrete physical examples like sheds, open zones, and lumber stacked to the side.

Notable word choices differed as well. Claude used "keep its stuff"—the most colloquial phrasing. GPT used "belongings," which carries mild personal-property connotations not present in the contract. Perplexity introduced "safe" and "without getting damaged," implying a protective duty the contract does not explicitly impose.

### F. Limitations

This is a prototype, and it has limitations that should be stated plainly.

All of the scoring runs through a single embedding model. If that model has blind spots—say, it does not handle highly technical construction terminology well—the scores will reflect that distortion. A stronger version of this protocol would use multiple embedding models in parallel, the way the tool already uses multiple generative models.

The similarity measurement captures how related two texts are, not whether one is a better definition than the other. A definition that simply rephrases the contract clause might score high on context alignment without actually being a good definition. The tool addresses this by also showing term alignment—a definition that scores well on both is genuinely capturing the meaning, not just echoing the passage—but the tension exists.

The AI training data is not curated for legal interpretation. Unlike a dictionary, where editors deliberately choose what to include, LLMs are trained on whatever text was available at scale. This includes high-quality published material, but also informal writing and content that may not reflect careful usage. The models reflect how people actually talk. Whether that is a strength or a weakness depends on one's theory of ordinary meaning.[^curation]

[^curation]: This is arguably an advantage. The ordinary meaning doctrine asks how a *normal person* would understand a word, not how an editor at Merriam-Webster would define it. But the lack of curation also means the training data includes non-standard usage, which could pull definitions in unexpected directions.

Different prompting strategies produce different definitions, and the surviving definition depends on which strategies were tried. The tool runs every available strategy and shows the full log, but it cannot test every possible way of asking. A standardized prompting protocol—agreed upon by a rulemaking body or professional standards organization—would strengthen the evidentiary value of the results.

Finally, words change meaning over time. The models reflect current usage. If the question is what a word meant in 1987, the tool in its current form cannot answer that.

### G. What This Demonstrates

Three independently trained models, queried with transparent and reproducible prompts, converge on the same ordinary meaning of a term in a construction contract. The degree of their agreement is quantified. The points where they diverge—exclusion analysis, legal-rights framing, security implications—are surfaced rather than hidden. Every prompt, every output, and every score is visible.

That is something a dictionary entry does not provide. A dictionary gives one definition, written by one editor, with no explanation of the process that produced it. This tool gives three definitions from three independent sources, with a mathematical measurement of how much they agree, and a complete record of how each definition was generated. The interpreter does not have to trust the tool. The interpreter can verify every step.

The tool does not solve the ordinary meaning problem. But it demonstrates that a transparent, reproducible, multi-model protocol for extracting ordinary meaning from LLMs is buildable—and it provides a concrete artifact around which courts, scholars, and rulemakers could develop standards.
