"""
Ordinary Meaning — Streamlit interface.
Usage: uv run streamlit run app.py

Each model defines the term given the full document context.
The autoresearch loop keeps the definition closest to the context passage.
At the end, both document meaning and world meaning are shown side by side.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

import streamlit as st

from extract import extract_text
from prepare import (
    call_anthropic,
    call_openai,
    call_perplexity,
    embed_text,
    cosine_sim,
    term_alignment,
    context_alignment,
    consensus_score,
    LLMResponse,
)
from strategies import SYSTEM_PROMPT, STRATEGY_INFO, build_prompt, strategy_names

PROVIDERS = {
    "anthropic": {"fn": call_anthropic, "label": "Claude Sonnet 4.6", "model": "claude-sonnet-4-6", "key_env": "ANTHROPIC_API_KEY"},
    "openai": {"fn": call_openai, "label": "GPT-5.4 Nano", "model": "gpt-5.4-nano", "key_env": "OPENAI_API_KEY"},
    "perplexity": {"fn": call_perplexity, "label": "Perplexity Sonar Pro", "model": "sonar-pro", "key_env": "PERPLEXITY_API_KEY"},
}


def get_available():
    return {k: v for k, v in PROVIDERS.items() if os.environ.get(v["key_env"])}


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Ordinary Meaning", page_icon="OM", layout="wide")

st.markdown(
    "<style>"
    ".block-container { max-width: 1200px; }"
    "div[data-testid='stExpander'] details summary span { font-weight: 600; }"
    "</style>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Ordinary Meaning")
    st.caption("Autoresearch loop for the ordinary meaning of legal terms")
    st.caption("Inspired by Judge Newsom, *Snell v. United Specialty* (11th Cir. 2024)")
    st.divider()

    term = st.text_input("Term to define", placeholder="e.g. landscaping")

    context = st.text_area(
        "Contract clause / context paragraph",
        height=120,
        placeholder="Paste the clause where the term appears...",
    )

    st.divider()
    st.subheader("Full Document (optional)")
    uploaded_file = st.file_uploader(
        "Upload the full contract",
        type=["pdf", "docx", "doc", "txt"],
        help="When provided, ALL prompting strategies include the full document so the model always knows the domain. PDF, Word, or plain text.",
    )

    doc_text = ""
    if uploaded_file is not None:
        try:
            raw = uploaded_file.read()
            doc_text = extract_text(uploaded_file.name, raw)
            word_count = len(doc_text.split())
            st.success(f"Extracted {word_count:,} words from {uploaded_file.name}")
            with st.expander("Preview extracted text", expanded=False):
                st.text(doc_text[:3000] + ("\n..." if len(doc_text) > 3000 else ""))
        except Exception as e:
            st.error(f"Could not read file: {e}")

    st.divider()
    st.subheader("Parameters")
    rounds = st.slider(
        "Rounds per strategy",
        min_value=1, max_value=5, value=1,
        help="Each strategy generates this many definitions per model.",
    )
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.1)

    st.divider()
    st.subheader("Models")
    st.caption("Each model defines the term independently using every strategy.")

    available = get_available()
    if not available:
        st.error("No API keys found. Add them to your .env file.")

    selected = []
    for pid, info in available.items():
        if st.checkbox(info["label"], value=True, key=f"model_{pid}"):
            selected.append(pid)

    missing = [v["label"] for k, v in PROVIDERS.items() if k not in available]
    if missing:
        st.caption(f"Unavailable (no API key): {', '.join(missing)}")

    has_doc = bool(doc_text)
    all_strategies = [s for s in strategy_names(has_document=has_doc) if s != "refine"]
    n_models = len(selected)
    n_strats = len(all_strategies)
    n_total = n_models * n_strats * rounds

    if selected:
        st.info(
            f"{n_models} models x {n_strats} strategies x {rounds} rounds = "
            f"**{n_total} experiments**"
        )

    st.divider()
    st.subheader("How it works")
    st.caption(
        "1. Each model generates definitions using different prompting strategies. "
        "When a document is uploaded, every strategy includes the full document.\n\n"
        "2. The autoresearch loop keeps the definition closest to the **context passage** "
        "in embedding space -- the definition that best captures the meaning of the term "
        "as used in the document.\n\n"
        "3. At the end, all surviving definitions are shown with two measurements: "
        "**context alignment** (meaning in the document) and **term alignment** "
        "(meaning in the world). Plus cross-model consensus.\n\n"
        "No arbitrary scoring formula. No winner picked. You read the results."
    )
    st.markdown("**Embedding model:** `text-embedding-3-small`")

    st.divider()

    start = st.button(
        "Start Experiment",
        type="primary",
        use_container_width=True,
        disabled=(not term or not context or not selected),
    )

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

if "experiment_log" not in st.session_state:
    st.session_state.experiment_log = []

# ---------------------------------------------------------------------------
# Landing page
# ---------------------------------------------------------------------------

if not start:
    st.header("Ordinary Meaning")
    st.markdown(
        "Configure your term, context, and parameters in the sidebar, "
        "then press **Start Experiment**."
    )
    st.markdown("---")
    st.markdown(
        "Each selected model defines the term using multiple prompting strategies. "
        "When a document is uploaded, the full document is included in every prompt "
        "so the model always knows the domain. The autoresearch loop keeps the "
        "definition that best matches the context passage. At the end, you see "
        "all definitions with both **document meaning** and **world meaning** measurements."
    )

    if st.session_state.experiment_log:
        st.header("Previous Results")
        df = pd.DataFrame(st.session_state.experiment_log)
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.stop()

# ---------------------------------------------------------------------------
# Experiment run
# ---------------------------------------------------------------------------

Path("input.md").write_text(
    f"# Input\n\n## Term\n\n{term}\n\n## Context\n\n{context}\n"
)

doc_truncated = doc_text[:12000] if doc_text else ""

st.header(f'Defining: "{term}"')

mc1, mc2, mc3 = st.columns(3)
mc1.metric("Models", n_models)
mc2.metric("Strategies", n_strats)
mc3.metric("Total Experiments", n_total)

with st.expander("Context passage", expanded=False):
    st.markdown(context)

if has_doc:
    with st.expander(f"Full document ({len(doc_text.split()):,} words)", expanded=False):
        st.text(doc_text[:5000] + ("\n..." if len(doc_text) > 5000 else ""))

with st.expander("System prompt (same for all experiments)", expanded=False):
    st.code(SYSTEM_PROMPT, language=None)

# -- Strategy reference
with st.expander(f"Prompting strategies ({n_strats} active)", expanded=True):
    st.markdown(
        "Each strategy asks the model to define the term in a different way. "
        "The autoresearch loop tries every strategy and keeps the best result."
    )
    if has_doc:
        st.info("A full document was uploaded -- it is appended to every strategy below.")

    for strat in all_strategies:
        info = STRATEGY_INFO[strat]
        preview_prompt = build_prompt(
            strategy=strat,
            term=term,
            context=context,
            full_document=doc_truncated,
        )
        st.markdown(f"**{strat}** -- {info['description']}")
        with st.expander(f"Full prompt for \"{strat}\"", expanded=False):
            st.code(preview_prompt, language=None)

st.divider()

# =========================================================================
# PHASE 1: Autoresearch Loop (context alignment for keep/discard)
# =========================================================================

st.header("Phase 1: Autoresearch Loop")
st.markdown(
    "Each model runs through every strategy. Each definition is measured by "
    "**context alignment** -- cosine similarity between the definition and the "
    "context passage in embedding space. This keeps the definition that best captures "
    f'what "{term}" means as used in the document. If a new definition scores higher '
    "than the current best for that model, it takes over."
)

if has_doc:
    st.info("Full document is included in every prompt so all strategies are document-aware.")

progress = st.progress(0, text="Starting...")

survivors = {}
challenge_logs = {}
log_rows = []
step = 0

for pid in selected:
    prov_info = PROVIDERS[pid]
    current_best_text = None
    current_best_score = -1.0
    current_best_strategy = None
    model_log = []

    for strat in all_strategies:
        strat_desc = STRATEGY_INFO[strat]["description"]
        for rd in range(1, rounds + 1):
            step += 1
            progress.progress(
                step / n_total,
                text=f"{step}/{n_total} -- {prov_info['label']} -- {strat}" + (f" r{rd}" if rounds > 1 else ""),
            )

            prompt = build_prompt(
                strategy=strat,
                term=term,
                context=context,
                full_document=doc_truncated,
            )

            resp_text = ""
            resp_meta = ""
            c_align = 0.0
            error = None

            try:
                resp = prov_info["fn"](prompt, SYSTEM_PROMPT, temperature)
                resp_text = resp.text.strip()
                resp_meta = (
                    f"{resp.model} -- "
                    f"{resp.input_tokens} input, {resp.output_tokens} output tokens"
                )
            except Exception as e:
                error = f"Generation error: {e}"

            if resp_text and not error:
                try:
                    c_align = context_alignment(resp_text, context)
                except Exception as e:
                    error = f"Embedding error: {e}"

            if current_best_text is None and resp_text and not error:
                kept = True
                current_best_text = resp_text
                current_best_score = c_align
                current_best_strategy = strat
            elif c_align > current_best_score and not error:
                kept = True
                current_best_text = resp_text
                current_best_score = c_align
                current_best_strategy = strat
            else:
                kept = False

            entry = {
                "strategy": strat,
                "strategy_description": strat_desc,
                "round": rd,
                "prompt": prompt,
                "response": resp_text,
                "response_meta": resp_meta,
                "context_alignment": round(c_align, 6),
                "best_so_far": round(current_best_score, 6),
                "kept": kept,
                "error": error,
            }
            model_log.append(entry)

            log_rows.append({
                "Model": prov_info["label"],
                "Strategy": strat,
                "Round": rd,
                "Context Alignment": round(c_align, 6),
                "Best So Far": round(current_best_score, 6),
                "Decision": "KEEP" if kept else ("ERROR" if error else "DISCARD"),
            })

    survivors[pid] = {
        "definition": current_best_text or "",
        "context_alignment": current_best_score,
        "strategy": current_best_strategy or "none",
    }
    challenge_logs[pid] = model_log

progress.progress(1.0, text="Done")

st.session_state.experiment_log = log_rows

for pid in selected:
    prov_info = PROVIDERS[pid]
    surv = survivors[pid]
    logs = challenge_logs[pid]

    with st.expander(
        f"{prov_info['label']} -- best: {surv['strategy']} "
        f"(context alignment {surv['context_alignment']:.4f})",
        expanded=False,
    ):
        st.markdown(f"**Model:** {prov_info['label']} (`{prov_info['model']}`)")
        st.markdown(f"**Surviving strategy:** {surv['strategy']}")
        st.metric("Context Alignment", f"{surv['context_alignment']:.6f}")

        log_df = pd.DataFrame([
            {
                "Strategy": e["strategy"],
                "Round": e["round"],
                "Context Align": e["context_alignment"],
                "Best": e["best_so_far"],
                "Decision": "KEEP" if e["kept"] else ("ERROR" if e["error"] else "DISCARD"),
            }
            for e in logs
        ])
        st.dataframe(log_df, use_container_width=True, hide_index=True)

        for e in logs:
            label = f"{e['strategy']}"
            if rounds > 1:
                label += f" r{e['round']}"
            decision = "KEEP" if e["kept"] else ("ERROR" if e["error"] else "DISCARD")
            with st.expander(f"{label} -- {e['context_alignment']:.4f} -- {decision}", expanded=False):
                if e["error"]:
                    st.error(e["error"])
                    continue
                st.caption(f"**Strategy:** {e['strategy']} -- {e['strategy_description']}")
                with st.expander("Full prompt sent to model", expanded=False):
                    st.code(e["prompt"], language=None)
                st.markdown("**Response:**")
                st.markdown(e["response"])
                st.caption(e["response_meta"])

# =========================================================================
# PHASE 2: Results -- document meaning vs world meaning
# =========================================================================

st.divider()
st.header("Phase 2: Results")
st.markdown(
    "Each model's surviving definition, shown with both **document meaning** "
    "(context alignment) and **world meaning** (term alignment), plus cross-model consensus."
)

active = {p: s for p, s in survivors.items() if s["definition"]}

measurements = {}
for pid, surv in active.items():
    others = [s["definition"] for p, s in active.items() if p != pid]
    c_a = surv["context_alignment"]
    t_a = term_alignment(surv["definition"], term)
    cons = consensus_score(surv["definition"], others) if others else None
    measurements[pid] = {
        "context_alignment": round(c_a, 6),
        "term_alignment": round(t_a, 6),
        "consensus": round(cons, 6) if cons is not None else None,
    }

score_rows = []
for pid in selected:
    if pid not in measurements:
        continue
    prov_info = PROVIDERS[pid]
    surv = survivors[pid]
    m = measurements[pid]
    score_rows.append({
        "Model": prov_info["label"],
        "Strategy": surv["strategy"],
        "Context Alignment (document)": m["context_alignment"],
        "Term Alignment (world)": m["term_alignment"],
        "Consensus": m["consensus"] if m["consensus"] is not None else "N/A",
    })

if score_rows:
    st.dataframe(pd.DataFrame(score_rows), use_container_width=True, hide_index=True)

st.markdown(
    "**Context Alignment** = cos(definition, passage) -- how well the definition "
    "captures the meaning of the term as used in *this document*.\n\n"
    "**Term Alignment** = cos(definition, bare term) -- how close the definition is "
    "to the word's general meaning in the world, across all of the training data.\n\n"
    "**Consensus** = average cos(definition, other models' definitions) -- how much "
    "the models agree with each other."
)

st.divider()

for pid in selected:
    if pid not in active:
        continue
    prov_info = PROVIDERS[pid]
    surv = survivors[pid]
    m = measurements[pid]

    st.subheader(f"{prov_info['label']}")
    st.caption(f"Model: `{prov_info['model']}` | Strategy: {surv['strategy']}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Context Alignment", f"{m['context_alignment']:.4f}", help="Meaning in the document")
    c2.metric("Term Alignment", f"{m['term_alignment']:.4f}", help="Meaning in the world")
    if m["consensus"] is not None:
        c3.metric("Consensus", f"{m['consensus']:.4f}", help="Agreement with other models")
    else:
        c3.metric("Consensus", "N/A")

    st.markdown(surv["definition"])
    st.divider()

# =========================================================================
# PHASE 3: Cross-model similarity matrix
# =========================================================================

if len(active) >= 2:
    st.header("Phase 3: Cross-Model Similarity")
    st.markdown(
        "Pairwise cosine similarity between surviving definitions. "
        "High values mean the models converge on the same meaning."
    )

    pids = list(active.keys())
    labels = [PROVIDERS[p]["label"] for p in pids]
    vecs = {p: embed_text(active[p]["definition"]) for p in pids}

    sim_matrix = []
    for i, pid_i in enumerate(pids):
        row = {"": labels[i]}
        for j, pid_j in enumerate(pids):
            score = cosine_sim(vecs[pid_i], vecs[pid_j])
            row[labels[j]] = round(score, 4)
        sim_matrix.append(row)

    st.dataframe(pd.DataFrame(sim_matrix), use_container_width=True, hide_index=True)

    pair_scores = []
    for i in range(len(pids)):
        for j in range(i + 1, len(pids)):
            s = cosine_sim(vecs[pids[i]], vecs[pids[j]])
            pair_scores.append(s)
    avg_sim = sum(pair_scores) / len(pair_scores) if pair_scores else 0

    if avg_sim >= 0.95:
        st.success(f"Average pairwise similarity: {avg_sim:.4f} -- Strong convergence")
    elif avg_sim >= 0.85:
        st.info(f"Average pairwise similarity: {avg_sim:.4f} -- Moderate agreement")
    else:
        st.warning(f"Average pairwise similarity: {avg_sim:.4f} -- Notable divergence")

# =========================================================================
# PHASE 4: Qualitative comparison (one LLM call to describe, not score)
# =========================================================================

if len(active) >= 2:
    st.divider()
    st.header("Phase 4: Qualitative Comparison")
    st.markdown(
        "One LLM call describes where the definitions agree and diverge. "
        "This explains the math above in plain English. It does not score or rank."
    )

    pids = list(active.keys())
    def_list = "\n\n".join(
        f"**{PROVIDERS[pid]['label']}:**\n{active[pid]['definition']}"
        for pid in pids
    )

    comparison_prompt = (
        f'Multiple AI models each produced a plain-English definition of "{term}" '
        f"given this context:\n\n{context}\n\n"
        f"Their definitions:\n\n{def_list}\n\n"
        "Describe:\n"
        "1. Where do they AGREE? What core meaning do they share?\n"
        "2. Where do they DIFFER? What does one include that others omit?\n"
        "3. Are there any notable word choices or framings that differ?\n"
        "Be specific and concise. Do not pick a winner or rank them."
    )

    comp_system = (
        "You are a neutral language analyst. Describe similarities and differences "
        "between definitions. Do not score or rank them."
    )

    with st.expander("Comparison prompt", expanded=False):
        st.code(comparison_prompt, language=None)

    try:
        comparison_resp = call_anthropic(comparison_prompt, comp_system, 0.0)
        st.markdown(comparison_resp.text)
        st.caption(
            f"{comparison_resp.model} -- "
            f"{comparison_resp.input_tokens} input, {comparison_resp.output_tokens} output tokens"
        )
    except Exception as e:
        st.error(f"Comparison failed: {e}")

# =========================================================================
# Save results
# =========================================================================

if active:
    lines = [f"# Ordinary Meaning: {term}\n"]
    for pid in selected:
        if pid not in active:
            continue
        prov_info = PROVIDERS[pid]
        surv = survivors[pid]
        m = measurements[pid]
        lines.append(f"\n## {prov_info['label']} (strategy: {surv['strategy']})\n")
        lines.append(f"{surv['definition']}\n")
        lines.append(f"\nContext Alignment (document): {m['context_alignment']:.6f}")
        lines.append(f"Term Alignment (world): {m['term_alignment']:.6f}")
        if m["consensus"] is not None:
            lines.append(f"Consensus: {m['consensus']:.6f}")
        lines.append("")

    Path("definition.md").write_text("\n".join(lines))
