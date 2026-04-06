"""
Ordinary Meaning — Streamlit interface.
Usage: uv run streamlit run app.py

Models define the term BLIND — they never see the contract or clause.
The autoresearch loop runs rounds within each strategy, optimizing for
context alignment (closeness to the clause the model never saw).
All strategies are shown with both term alignment and context alignment.
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
    st.caption("Transparent protocol for the ordinary meaning of legal terms")
    st.divider()

    term = st.text_input("Term to define", placeholder="e.g. storage areas")

    context = st.text_area(
        "Contract clause (for comparison — models will NOT see this)",
        height=120,
        placeholder="Paste the clause where the term appears...",
    )

    st.divider()
    st.subheader("Full Document (optional)")
    st.caption("Used only for your reference. Models never see it.")
    uploaded_file = st.file_uploader(
        "Upload the full contract",
        type=["pdf", "docx", "doc", "txt"],
        help="Shown alongside results for comparison. NOT sent to any model.",
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
        min_value=1, max_value=5, value=3,
        help="Each strategy runs this many times per model. The best-scoring definition from each strategy survives.",
    )
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.1)

    st.divider()
    st.subheader("Models")

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

    all_strategies = strategy_names()
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
        "**1. Blind generation** — models define the term using only the word itself. "
        "They never see the contract clause or document.\n\n"
        "**2. Autoresearch loop** — each strategy runs multiple rounds. "
        "The definition closest to the contract clause (by cosine similarity) survives. "
        "The model has no idea what it's being compared to.\n\n"
        "**3. Results** — every strategy's best definition is shown with two scores: "
        "term alignment (world meaning) and context alignment (document meaning).\n\n"
        "If a blind definition naturally matches the contract, the ordinary meaning fits. "
        "If it doesn't, the contract may use the term unusually."
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
        "Models define the term **blind** — they never see the contract. "
        "The autoresearch loop tries different prompting strategies and keeps "
        "the definition that naturally lands closest to the contract clause "
        "in embedding space. You see every strategy's result with both "
        "**world meaning** and **document meaning** scores."
    )

    if "experiment_log" in st.session_state and st.session_state.experiment_log:
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

st.header(f'Defining: "{term}"')

mc1, mc2, mc3 = st.columns(3)
mc1.metric("Models", n_models)
mc2.metric("Strategies", n_strats)
mc3.metric("Total Experiments", n_total)

st.info(
    "**Blind generation** — the contract clause below is used only for measurement. "
    "No model sees it during definition generation."
)

with st.expander("Contract clause (comparison target — not sent to models)", expanded=False):
    st.markdown(context)

if doc_text:
    with st.expander(f"Full document ({len(doc_text.split()):,} words) — for your reference only", expanded=False):
        st.text(doc_text[:5000] + ("\n..." if len(doc_text) > 5000 else ""))

with st.expander("System prompt (same for all experiments)", expanded=False):
    st.code(SYSTEM_PROMPT, language=None)

with st.expander(f"Prompting strategies ({n_strats})", expanded=False):
    for strat in all_strategies:
        info = STRATEGY_INFO[strat]
        preview = build_prompt(strategy=strat, term=term)
        st.markdown(f"**{strat}** — {info['description']}")
        with st.expander(f"Full prompt for \"{strat}\"", expanded=False):
            st.code(preview, language=None)

st.divider()

# =========================================================================
# PHASE 1: Autoresearch Loop — rounds within each strategy
# =========================================================================

st.header("Phase 1: Autoresearch Loop")
st.markdown(
    f"Each model defines \"{term}\" using {n_strats} strategies x {rounds} rounds. "
    "Models only see the word — never the contract. "
    "Within each strategy, the definition with the highest **context alignment** "
    "(closeness to the contract clause the model never saw) survives."
)

progress = st.progress(0, text="Starting...")

results = {}
all_logs = {}
log_rows = []
step = 0

for pid in selected:
    prov_info = PROVIDERS[pid]
    model_results = {}
    model_log = []

    for strat in all_strategies:
        strat_desc = STRATEGY_INFO[strat]["description"]
        best_text = None
        best_c_align = -1.0
        best_t_align = 0.0
        best_round = 0
        strat_log = []

        for rd in range(1, rounds + 1):
            step += 1
            progress.progress(
                step / n_total,
                text=f"{step}/{n_total} — {prov_info['label']} — {strat} r{rd}",
            )

            prompt = build_prompt(strategy=strat, term=term)

            resp_text = ""
            resp_meta = ""
            c_align = 0.0
            t_align = 0.0
            error = None

            try:
                resp = prov_info["fn"](prompt, SYSTEM_PROMPT, temperature)
                resp_text = resp.text.strip()
                resp_meta = (
                    f"{resp.model} — "
                    f"{resp.input_tokens} input, {resp.output_tokens} output tokens"
                )
            except Exception as e:
                error = f"Generation error: {e}"

            if resp_text and not error:
                try:
                    c_align = context_alignment(resp_text, context)
                    t_align = term_alignment(resp_text, term)
                except Exception as e:
                    error = f"Embedding error: {e}"

            kept = False
            if resp_text and not error and c_align > best_c_align:
                kept = True
                best_text = resp_text
                best_c_align = c_align
                best_t_align = t_align
                best_round = rd

            entry = {
                "strategy": strat,
                "round": rd,
                "prompt": prompt,
                "response": resp_text,
                "response_meta": resp_meta,
                "context_alignment": round(c_align, 6),
                "term_alignment": round(t_align, 6),
                "best_c_align": round(best_c_align, 6),
                "kept": kept,
                "error": error,
            }
            strat_log.append(entry)
            model_log.append(entry)

            log_rows.append({
                "Model": prov_info["label"],
                "Strategy": strat,
                "Round": rd,
                "Context Align": round(c_align, 6),
                "Term Align": round(t_align, 6),
                "Best (ctx)": round(best_c_align, 6),
                "Decision": "KEEP" if kept else ("ERROR" if error else "DISCARD"),
            })

        model_results[strat] = {
            "definition": best_text or "",
            "context_alignment": best_c_align,
            "term_alignment": best_t_align,
            "best_round": best_round,
            "rounds_log": strat_log,
        }

    results[pid] = model_results
    all_logs[pid] = model_log

progress.progress(1.0, text="Done")
st.session_state.experiment_log = log_rows

# -- Detailed logs (collapsed)
with st.expander("Detailed experiment log", expanded=False):
    for pid in selected:
        prov_info = PROVIDERS[pid]
        st.subheader(prov_info["label"])

        log_df = pd.DataFrame([
            {
                "Strategy": e["strategy"],
                "Round": e["round"],
                "Context Align": e["context_alignment"],
                "Term Align": e["term_alignment"],
                "Best (ctx)": e["best_c_align"],
                "Decision": "KEEP" if e["kept"] else ("ERROR" if e["error"] else "DISCARD"),
            }
            for e in all_logs[pid]
        ])
        st.dataframe(log_df, use_container_width=True, hide_index=True)

        for e in all_logs[pid]:
            label = f"{e['strategy']} r{e['round']}"
            decision = "KEEP" if e["kept"] else ("ERROR" if e["error"] else "DISCARD")
            with st.expander(f"{label} — ctx:{e['context_alignment']:.4f} term:{e['term_alignment']:.4f} — {decision}", expanded=False):
                if e["error"]:
                    st.error(e["error"])
                    continue
                with st.expander("Full prompt", expanded=False):
                    st.code(e["prompt"], language=None)
                st.markdown("**Response:**")
                st.markdown(e["response"])
                st.caption(e["response_meta"])

# =========================================================================
# PHASE 2: Results — all strategies per model
# =========================================================================

st.divider()
st.header("Phase 2: Results")
st.markdown(
    "Every strategy's best definition (after autoresearch rounds), shown with both scores. "
    "The models defined the term **blind** — they never saw the contract clause. "
    "Context alignment measures how well the blind definition naturally maps onto the document."
)

for pid in selected:
    prov_info = PROVIDERS[pid]
    model_res = results[pid]

    st.subheader(prov_info["label"])

    summary_rows = []
    for strat in all_strategies:
        r = model_res[strat]
        if not r["definition"]:
            continue
        summary_rows.append({
            "Strategy": strat,
            "Context Alignment": round(r["context_alignment"], 4),
            "Term Alignment": round(r["term_alignment"], 4),
            "Best of Round": r["best_round"],
        })

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

    for strat in all_strategies:
        r = model_res[strat]
        if not r["definition"]:
            continue

        c_a = r["context_alignment"]
        t_a = r["term_alignment"]
        gap = abs(t_a - c_a)

        with st.expander(
            f"{strat} — context: {c_a:.4f} | term: {t_a:.4f}",
            expanded=False,
        ):
            m1, m2, m3 = st.columns(3)
            m1.metric("Context Alignment", f"{c_a:.4f}", help="Closeness to contract clause (model never saw this)")
            m2.metric("Term Alignment", f"{t_a:.4f}", help="Closeness to word's world meaning")
            m3.metric("Gap", f"{gap:.4f}", help="Difference between world meaning and document meaning")

            if gap < 0.05:
                st.success("Minimal gap — the ordinary meaning naturally fits the document.")
            elif gap < 0.12:
                st.info("Moderate gap — the meaning mostly fits, with some narrowing in the document.")
            else:
                st.warning("Large gap — the document may use this term in a specific or unusual way.")

            st.markdown(r["definition"])
            st.caption(f"Best of {rounds} rounds (round {r['best_round']})")

    st.divider()

# =========================================================================
# PHASE 3: Cross-Model Consensus
# =========================================================================

if len(selected) >= 2:
    st.header("Phase 3: Cross-Model Consensus")
    st.markdown(
        "For each strategy, how much do the models agree with each other? "
        "Same prompt, same question, different models — high similarity means convergence."
    )

    consensus_tabs = st.tabs(all_strategies)

    for idx, strat in enumerate(all_strategies):
        with consensus_tabs[idx]:
            pids_with_def = [p for p in selected if results[p][strat]["definition"]]
            if len(pids_with_def) < 2:
                st.caption("Not enough definitions to compare.")
                continue

            labels = [PROVIDERS[p]["label"] for p in pids_with_def]
            vecs = {p: embed_text(results[p][strat]["definition"]) for p in pids_with_def}

            sim_matrix = []
            for i, pid_i in enumerate(pids_with_def):
                row = {"": labels[i]}
                for j, pid_j in enumerate(pids_with_def):
                    row[labels[j]] = round(cosine_sim(vecs[pid_i], vecs[pid_j]), 4)
                sim_matrix.append(row)
            st.dataframe(pd.DataFrame(sim_matrix), use_container_width=True, hide_index=True)

            pair_scores = []
            for i in range(len(pids_with_def)):
                for j in range(i + 1, len(pids_with_def)):
                    pair_scores.append(cosine_sim(vecs[pids_with_def[i]], vecs[pids_with_def[j]]))
            avg = sum(pair_scores) / len(pair_scores) if pair_scores else 0

            if avg >= 0.95:
                st.success(f"Average similarity: {avg:.4f} — Strong convergence")
            elif avg >= 0.85:
                st.info(f"Average similarity: {avg:.4f} — Moderate agreement")
            else:
                st.warning(f"Average similarity: {avg:.4f} — Notable divergence")

# =========================================================================
# PHASE 4: Overall best per model + qualitative comparison
# =========================================================================

best_per_model = {}
for pid in selected:
    model_res = results[pid]
    best_strat = max(
        [s for s in all_strategies if model_res[s]["definition"]],
        key=lambda s: model_res[s]["context_alignment"],
        default=None,
    )
    if best_strat:
        best_per_model[pid] = {
            "strategy": best_strat,
            **model_res[best_strat],
        }

if len(best_per_model) >= 2:
    st.divider()
    st.header("Phase 4: Qualitative Comparison")
    st.markdown(
        "Using each model's highest context-alignment definition (the blind definition "
        "that naturally landed closest to the contract), one LLM call describes "
        "where they agree and diverge. It does not score or rank."
    )

    summary_rows = []
    for pid in selected:
        if pid not in best_per_model:
            continue
        b = best_per_model[pid]
        summary_rows.append({
            "Model": PROVIDERS[pid]["label"],
            "Best Strategy": b["strategy"],
            "Context Alignment": round(b["context_alignment"], 4),
            "Term Alignment": round(b["term_alignment"], 4),
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    pids = [p for p in selected if p in best_per_model]
    def_list = "\n\n".join(
        f"**{PROVIDERS[pid]['label']}** (strategy: {best_per_model[pid]['strategy']}):\n"
        f"{best_per_model[pid]['definition']}"
        for pid in pids
    )

    comparison_prompt = (
        f'Three AI models each defined "{term}" in plain English. '
        f"They did NOT see the contract — they defined the word blind, "
        f"based only on their training.\n\n"
        f"For reference, the contract clause is:\n{context}\n\n"
        f"Their definitions:\n\n{def_list}\n\n"
        "Describe:\n"
        "1. Where do they AGREE? What core meaning do they share?\n"
        "2. Where do they DIFFER? What does one include that others omit?\n"
        "3. How well do these blind definitions fit the contract clause above? "
        "Does the ordinary meaning naturally cover the contract's usage, or does "
        "the contract seem to use the term in a specific/unusual way?\n"
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
            f"{comparison_resp.model} — "
            f"{comparison_resp.input_tokens} input, {comparison_resp.output_tokens} output tokens"
        )
    except Exception as e:
        st.error(f"Comparison failed: {e}")

# =========================================================================
# Save results
# =========================================================================

any_results = any(
    any(results[p][s]["definition"] for s in all_strategies)
    for p in selected
)

if any_results:
    lines = [f"# Ordinary Meaning: {term}\n"]
    lines.append(f"Contract clause (comparison target — models never saw this):\n> {context}\n")

    for pid in selected:
        prov_info = PROVIDERS[pid]
        model_res = results[pid]
        lines.append(f"\n## {prov_info['label']}\n")

        for strat in all_strategies:
            r = model_res[strat]
            if not r["definition"]:
                continue
            lines.append(f"### {strat} (best of {rounds} rounds, round {r['best_round']})\n")
            lines.append(f"{r['definition']}\n")
            lines.append(f"Context Alignment: {r['context_alignment']:.6f}")
            lines.append(f"Term Alignment: {r['term_alignment']:.6f}\n")

    Path("definition.md").write_text("\n".join(lines))
