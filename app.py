"""
Ordinary Meaning — Streamlit interface.
Usage: uv run streamlit run app.py

Models define the term BLIND — they never see the contract or clause.
Phase 1: Fixed strategies with autoresearch rounds.
Phase 2: An agent rewrites prompts to improve scores. The agent sees
scores and definitions but NEVER the contract clause.
All results shown with full transparency.
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

AGENT_SYSTEM = (
    "You are a prompt engineer optimizing prompts for a word-definition task. "
    "You will see prompts that have already been tried, the definitions they "
    "produced, and their scores (higher is better). You do NOT know what the "
    "definitions are being compared to — the scoring target is hidden from you.\n\n"
    "Your job: generate ONE new prompt that might score higher. The prompt must "
    "ask for a plain-English definition of the given word. Do NOT include any "
    "specific document, contract, clause, or context in your prompt — the model "
    "must define the word based purely on general knowledge.\n\n"
    "Look at which prompts scored highest. What did they have in common? What "
    "framing, structure, or emphasis seemed to help? Use that to craft a better prompt.\n\n"
    "Return ONLY the prompt text. No explanation, no preamble, no markdown."
)


def get_available():
    return {k: v for k, v in PROVIDERS.items() if os.environ.get(v["key_env"])}


def call_agent(agent_prompt: str) -> str:
    if os.environ.get("ANTHROPIC_API_KEY"):
        resp = call_anthropic(agent_prompt, AGENT_SYSTEM, 0.7)
        return resp.text.strip()
    elif os.environ.get("OPENAI_API_KEY"):
        resp = call_openai(agent_prompt, AGENT_SYSTEM, 0.7)
        return resp.text.strip()
    return ""


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
        min_value=1, max_value=5, value=2,
        help="Each fixed strategy runs this many times. Best-of-N survives.",
    )
    agent_rounds = st.slider(
        "Agent rounds",
        min_value=0, max_value=10, value=5,
        help="After fixed strategies, an agent generates this many new prompts. "
             "The agent sees scores and definitions but NEVER the contract clause. "
             "Set to 0 to disable the agent.",
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
    n_fixed = n_models * n_strats * rounds
    n_agent = n_models * agent_rounds
    n_total = n_fixed + n_agent

    if selected:
        st.info(
            f"**Fixed:** {n_models} models x {n_strats} strategies x {rounds} rounds = {n_fixed}\n\n"
            f"**Agent:** {n_models} models x {agent_rounds} rounds = {n_agent}\n\n"
            f"**Total: {n_total} experiments**"
        )

    st.divider()
    st.subheader("How it works")
    st.caption(
        "**1. Blind generation** — models define the term using only the word. "
        "They never see the contract.\n\n"
        "**2. Fixed strategies** — 5 predetermined prompts, each run multiple rounds.\n\n"
        "**3. Agent optimization** — an agent sees which prompts scored best and "
        "generates new prompts to try. The agent never sees the contract clause — "
        "it only sees scores and definitions.\n\n"
        "**4. Results** — every prompt's best definition shown with term alignment "
        "(world meaning) and context alignment (document meaning)."
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
        "Fixed strategies run first, then an agent invents new prompts based on "
        "what scored well. The agent sees scores but never the contract clause. "
        "Everything is shown with full transparency."
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
mc2.metric("Fixed Experiments", n_fixed)
mc3.metric("Agent Rounds", n_agent)

st.info(
    "**Blind generation** — the contract clause below is used only for measurement. "
    "No model and no agent sees it."
)

with st.expander("Contract clause (comparison target — hidden from all models and the agent)", expanded=False):
    st.markdown(context)

if doc_text:
    with st.expander(f"Full document ({len(doc_text.split()):,} words) — for your reference only", expanded=False):
        st.text(doc_text[:5000] + ("\n..." if len(doc_text) > 5000 else ""))

with st.expander("System prompt (same for all definition generation)", expanded=False):
    st.code(SYSTEM_PROMPT, language=None)

with st.expander(f"Fixed prompting strategies ({n_strats})", expanded=False):
    for strat in all_strategies:
        info = STRATEGY_INFO[strat]
        preview = build_prompt(strategy=strat, term=term)
        st.markdown(f"**{strat}** — {info['description']}")
        with st.expander(f"Full prompt for \"{strat}\"", expanded=False):
            st.code(preview, language=None)

if agent_rounds > 0:
    with st.expander("Agent system prompt (used to generate new prompts)", expanded=False):
        st.code(AGENT_SYSTEM, language=None)

st.divider()

# =========================================================================
# PHASE 1: Fixed Strategies
# =========================================================================

st.header("Phase 1: Fixed Strategies")
st.markdown(
    f"Each model defines \"{term}\" using {n_strats} fixed strategies x {rounds} rounds. "
    "Models only see the word — never the contract."
)

progress = st.progress(0, text="Starting fixed strategies...")

fixed_results = {}
all_experiment_rows = []
step = 0
total_steps = n_fixed + n_agent

for pid in selected:
    prov_info = PROVIDERS[pid]
    model_fixed = {}

    for strat in all_strategies:
        best_text = None
        best_c_align = -1.0
        best_t_align = 0.0
        best_round = 0

        for rd in range(1, rounds + 1):
            step += 1
            progress.progress(step / total_steps, text=f"{step}/{total_steps} — {prov_info['label']} — {strat} r{rd}")

            prompt = build_prompt(strategy=strat, term=term)
            resp_text = ""
            c_align = 0.0
            t_align = 0.0
            error = None

            try:
                resp = prov_info["fn"](prompt, SYSTEM_PROMPT, temperature)
                resp_text = resp.text.strip()
            except Exception as e:
                error = str(e)

            if resp_text and not error:
                try:
                    c_align = context_alignment(resp_text, context)
                    t_align = term_alignment(resp_text, term)
                except Exception as e:
                    error = str(e)

            kept = False
            if resp_text and not error and c_align > best_c_align:
                kept = True
                best_text = resp_text
                best_c_align = c_align
                best_t_align = t_align
                best_round = rd

            all_experiment_rows.append({
                "Model": prov_info["label"],
                "Source": "Fixed",
                "Prompt Label": strat,
                "Round": rd,
                "Context Align": round(c_align, 4),
                "Term Align": round(t_align, 4),
                "Decision": "KEEP" if kept else ("ERROR" if error else "DISCARD"),
                "Prompt": prompt,
                "Definition": resp_text,
                "Error": error,
            })

        model_fixed[strat] = {
            "definition": best_text or "",
            "context_alignment": best_c_align,
            "term_alignment": best_t_align,
            "best_round": best_round,
            "prompt": build_prompt(strategy=strat, term=term),
        }

    fixed_results[pid] = model_fixed

with st.expander("Fixed strategy results", expanded=False):
    for pid in selected:
        prov_info = PROVIDERS[pid]
        st.markdown(f"**{prov_info['label']}**")
        rows = []
        for strat in all_strategies:
            r = fixed_results[pid][strat]
            if r["definition"]:
                rows.append({
                    "Strategy": strat,
                    "Context Align": round(r["context_alignment"], 4),
                    "Term Align": round(r["term_alignment"], 4),
                    "Best Round": r["best_round"],
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# =========================================================================
# PHASE 2: Agent Optimization
# =========================================================================

agent_results = {pid: {} for pid in selected}

if agent_rounds > 0:
    st.divider()
    st.header("Phase 2: Agent Optimization")
    st.markdown(
        "An agent sees which prompts scored best and generates new ones. "
        "The agent sees scores and definitions but **never the contract clause**. "
        "Each new prompt is run on all models."
    )

    agent_log_container = st.container()

    global_best_per_model = {}
    for pid in selected:
        best_strat = max(
            all_strategies,
            key=lambda s: fixed_results[pid][s]["context_alignment"] if fixed_results[pid][s]["definition"] else -1,
        )
        r = fixed_results[pid][best_strat]
        global_best_per_model[pid] = {
            "prompt": r["prompt"],
            "definition": r["definition"],
            "context_alignment": r["context_alignment"],
            "term_alignment": r["term_alignment"],
            "label": best_strat,
        }

    all_tried = []
    for strat in all_strategies:
        avg_score = np.mean([
            fixed_results[pid][strat]["context_alignment"]
            for pid in selected
            if fixed_results[pid][strat]["definition"]
        ]) if selected else 0
        all_tried.append({
            "label": strat,
            "prompt": build_prompt(strategy=strat, term=term),
            "avg_score": round(float(avg_score), 4),
            "definitions": {
                PROVIDERS[pid]["label"]: fixed_results[pid][strat]["definition"]
                for pid in selected if fixed_results[pid][strat]["definition"]
            },
        })

    for agent_rd in range(1, agent_rounds + 1):
        step += 1
        progress.progress(step / total_steps, text=f"{step}/{total_steps} — Agent round {agent_rd}")

        sorted_tried = sorted(all_tried, key=lambda x: x["avg_score"], reverse=True)
        top_entries = sorted_tried[:5]

        history_text = f'The word to define is: "{term}"\n\n'
        history_text += "Here are the prompts tried so far, ranked by score (higher is better):\n\n"
        for i, entry in enumerate(top_entries, 1):
            history_text += f"--- Prompt #{i} (avg score: {entry['avg_score']}) ---\n"
            history_text += f"{entry['prompt']}\n"
            for model_label, defn in entry["definitions"].items():
                preview = defn[:200] + "..." if len(defn) > 200 else defn
                history_text += f"  {model_label}: {preview}\n"
            history_text += "\n"

        history_text += (
            "Generate a new prompt that might score higher. "
            "Remember: the model will ONLY see your prompt and the word. "
            "No documents, no contracts, no clauses. "
            "Return ONLY the prompt text."
        )

        new_prompt = ""
        agent_error = None
        try:
            new_prompt = call_agent(history_text)
        except Exception as e:
            agent_error = str(e)

        if not new_prompt or agent_error:
            all_experiment_rows.append({
                "Model": "Agent",
                "Source": "Agent",
                "Prompt Label": f"agent_r{agent_rd}",
                "Round": agent_rd,
                "Context Align": 0,
                "Term Align": 0,
                "Decision": "ERROR",
                "Prompt": history_text[:500],
                "Definition": "",
                "Error": agent_error or "Empty response",
            })
            continue

        new_prompt_formatted = new_prompt.format(term=term) if "{term}" in new_prompt else new_prompt
        if term.lower() not in new_prompt_formatted.lower():
            new_prompt_formatted = new_prompt_formatted + f' The word is: "{term}".'

        round_scores = []
        for pid in selected:
            prov_info = PROVIDERS[pid]
            resp_text = ""
            c_align = 0.0
            t_align = 0.0
            error = None

            try:
                resp = prov_info["fn"](new_prompt_formatted, SYSTEM_PROMPT, temperature)
                resp_text = resp.text.strip()
            except Exception as e:
                error = str(e)

            if resp_text and not error:
                try:
                    c_align = context_alignment(resp_text, context)
                    t_align = term_alignment(resp_text, term)
                except Exception as e:
                    error = str(e)

            kept = False
            if resp_text and not error and c_align > global_best_per_model[pid]["context_alignment"]:
                kept = True
                global_best_per_model[pid] = {
                    "prompt": new_prompt_formatted,
                    "definition": resp_text,
                    "context_alignment": c_align,
                    "term_alignment": t_align,
                    "label": f"agent_r{agent_rd}",
                }

            agent_results[pid][f"agent_r{agent_rd}"] = {
                "definition": resp_text,
                "context_alignment": c_align,
                "term_alignment": t_align,
                "prompt": new_prompt_formatted,
                "kept": kept,
                "error": error,
            }

            round_scores.append(c_align)

            all_experiment_rows.append({
                "Model": prov_info["label"],
                "Source": "Agent",
                "Prompt Label": f"agent_r{agent_rd}",
                "Round": agent_rd,
                "Context Align": round(c_align, 4),
                "Term Align": round(t_align, 4),
                "Decision": "KEEP" if kept else ("ERROR" if error else "DISCARD"),
                "Prompt": new_prompt_formatted,
                "Definition": resp_text,
                "Error": error,
            })

        avg_round_score = float(np.mean(round_scores)) if round_scores else 0
        all_tried.append({
            "label": f"agent_r{agent_rd}",
            "prompt": new_prompt_formatted,
            "avg_score": round(avg_round_score, 4),
            "definitions": {
                PROVIDERS[pid]["label"]: agent_results[pid][f"agent_r{agent_rd}"]["definition"]
                for pid in selected
                if agent_results[pid].get(f"agent_r{agent_rd}", {}).get("definition")
            },
        })

    with agent_log_container:
        with st.expander("Agent-generated prompts", expanded=False):
            for agent_rd in range(1, agent_rounds + 1):
                key = f"agent_r{agent_rd}"
                any_result = any(
                    agent_results[pid].get(key, {}).get("definition")
                    for pid in selected
                )
                if not any_result:
                    continue

                sample_pid = next(
                    (p for p in selected if agent_results[p].get(key, {}).get("definition")),
                    None,
                )
                if not sample_pid:
                    continue

                ar = agent_results[sample_pid][key]
                scores = [
                    agent_results[pid][key]["context_alignment"]
                    for pid in selected
                    if agent_results[pid].get(key, {}).get("definition")
                ]
                avg = np.mean(scores) if scores else 0
                kept_any = any(
                    agent_results[pid].get(key, {}).get("kept", False)
                    for pid in selected
                )

                status = "NEW BEST" if kept_any else "no improvement"
                with st.expander(f"Round {agent_rd} — avg: {avg:.4f} — {status}", expanded=False):
                    st.markdown("**Agent-generated prompt:**")
                    st.code(ar["prompt"], language=None)
                    for pid in selected:
                        if not agent_results[pid].get(key, {}).get("definition"):
                            continue
                        a = agent_results[pid][key]
                        prov_info = PROVIDERS[pid]
                        st.markdown(f"**{prov_info['label']}** — ctx: {a['context_alignment']:.4f} | term: {a['term_alignment']:.4f} | {'KEEP' if a['kept'] else 'DISCARD'}")
                        with st.expander(f"Definition ({prov_info['label']})", expanded=False):
                            st.markdown(a["definition"])

progress.progress(1.0, text="Done")
st.session_state.experiment_log = all_experiment_rows

# =========================================================================
# PHASE 3: Results — all prompts per model
# =========================================================================

st.divider()
st.header("Phase 3: Results")
st.markdown(
    "Every prompt's best definition — both fixed strategies and agent-generated — "
    "shown with both scores. Models defined the term **blind**."
)

for pid in selected:
    prov_info = PROVIDERS[pid]

    st.subheader(prov_info["label"])

    summary_rows = []
    all_entries = []

    for strat in all_strategies:
        r = fixed_results[pid][strat]
        if r["definition"]:
            entry = {
                "source": "Fixed",
                "label": strat,
                "definition": r["definition"],
                "context_alignment": r["context_alignment"],
                "term_alignment": r["term_alignment"],
                "prompt": r["prompt"],
            }
            all_entries.append(entry)
            summary_rows.append({
                "Source": "Fixed",
                "Prompt": strat,
                "Context Align": round(r["context_alignment"], 4),
                "Term Align": round(r["term_alignment"], 4),
                "Gap": round(abs(r["term_alignment"] - r["context_alignment"]), 4),
            })

    for key, ar in agent_results.get(pid, {}).items():
        if ar.get("definition") and not ar.get("error"):
            entry = {
                "source": "Agent",
                "label": key,
                "definition": ar["definition"],
                "context_alignment": ar["context_alignment"],
                "term_alignment": ar["term_alignment"],
                "prompt": ar["prompt"],
            }
            all_entries.append(entry)
            summary_rows.append({
                "Source": "Agent",
                "Prompt": key,
                "Context Align": round(ar["context_alignment"], 4),
                "Term Align": round(ar["term_alignment"], 4),
                "Gap": round(abs(ar["term_alignment"] - ar["context_alignment"]), 4),
            })

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

    sorted_entries = sorted(all_entries, key=lambda e: e["context_alignment"], reverse=True)

    for entry in sorted_entries:
        c_a = entry["context_alignment"]
        t_a = entry["term_alignment"]
        gap = abs(t_a - c_a)
        source_tag = f"[{entry['source']}]"

        with st.expander(
            f"{source_tag} {entry['label']} — ctx: {c_a:.4f} | term: {t_a:.4f} | gap: {gap:.4f}",
            expanded=False,
        ):
            m1, m2, m3 = st.columns(3)
            m1.metric("Context Alignment", f"{c_a:.4f}", help="Closeness to contract (model never saw it)")
            m2.metric("Term Alignment", f"{t_a:.4f}", help="Closeness to word's world meaning")
            m3.metric("Gap", f"{gap:.4f}", help="Difference between world and document meaning")

            if gap < 0.05:
                st.success("Minimal gap — ordinary meaning naturally fits the document.")
            elif gap < 0.12:
                st.info("Moderate gap — meaning mostly fits, with some narrowing.")
            else:
                st.warning("Large gap — document may use this term unusually.")

            st.markdown(entry["definition"])

            with st.expander("Prompt used", expanded=False):
                st.code(entry["prompt"], language=None)

    st.divider()

# =========================================================================
# PHASE 4: Cross-Model Consensus (using overall best per model)
# =========================================================================

if len(selected) >= 2:
    st.header("Phase 4: Cross-Model Consensus")
    st.markdown(
        "Using each model's highest-scoring definition (from any source — fixed or agent), "
        "how much do the models agree with each other?"
    )

    pids_with_best = [p for p in selected if global_best_per_model.get(p, {}).get("definition")]

    if len(pids_with_best) >= 2:
        labels = [PROVIDERS[p]["label"] for p in pids_with_best]
        vecs = {p: embed_text(global_best_per_model[p]["definition"]) for p in pids_with_best}

        sim_matrix = []
        for i, pid_i in enumerate(pids_with_best):
            row = {"": labels[i]}
            for j, pid_j in enumerate(pids_with_best):
                row[labels[j]] = round(cosine_sim(vecs[pid_i], vecs[pid_j]), 4)
            sim_matrix.append(row)
        st.dataframe(pd.DataFrame(sim_matrix), use_container_width=True, hide_index=True)

        pair_scores = []
        for i in range(len(pids_with_best)):
            for j in range(i + 1, len(pids_with_best)):
                pair_scores.append(cosine_sim(vecs[pids_with_best[i]], vecs[pids_with_best[j]]))
        avg = sum(pair_scores) / len(pair_scores) if pair_scores else 0

        if avg >= 0.95:
            st.success(f"Average similarity: {avg:.4f} — Strong convergence")
        elif avg >= 0.85:
            st.info(f"Average similarity: {avg:.4f} — Moderate agreement")
        else:
            st.warning(f"Average similarity: {avg:.4f} — Notable divergence")

        for pid in pids_with_best:
            b = global_best_per_model[pid]
            with st.expander(f"{PROVIDERS[pid]['label']} — best: {b['label']} (ctx: {b['context_alignment']:.4f})", expanded=False):
                st.markdown(b["definition"])
                with st.expander("Prompt", expanded=False):
                    st.code(b["prompt"], language=None)

# =========================================================================
# PHASE 5: Qualitative Comparison
# =========================================================================

if len(selected) >= 2:
    pids_with_best = [p for p in selected if global_best_per_model.get(p, {}).get("definition")]

    if len(pids_with_best) >= 2:
        st.divider()
        st.header("Phase 5: Qualitative Comparison")
        st.markdown(
            "One LLM call describes where the best definitions agree and diverge, "
            "and whether the blind definitions naturally fit the contract."
        )

        def_list = "\n\n".join(
            f"**{PROVIDERS[pid]['label']}** (prompt: {global_best_per_model[pid]['label']}):\n"
            f"{global_best_per_model[pid]['definition']}"
            for pid in pids_with_best
        )

        comparison_prompt = (
            f'Three AI models each defined "{term}" in plain English. '
            f"They did NOT see the contract — they defined the word blind.\n\n"
            f"For reference, the contract clause is:\n{context}\n\n"
            f"Their best definitions:\n\n{def_list}\n\n"
            "Describe:\n"
            "1. Where do they AGREE? What core meaning do they share?\n"
            "2. Where do they DIFFER? What does one include that others omit?\n"
            "3. How well do these blind definitions fit the contract clause? "
            "Does the ordinary meaning naturally cover the contract's usage?\n"
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
    global_best_per_model.get(p, {}).get("definition")
    for p in selected
)

if any_results:
    lines = [f"# Ordinary Meaning: {term}\n"]
    lines.append(f"Contract clause (models never saw this):\n> {context}\n")

    for pid in selected:
        if pid not in global_best_per_model:
            continue
        prov_info = PROVIDERS[pid]
        b = global_best_per_model[pid]
        lines.append(f"\n## {prov_info['label']} (best prompt: {b['label']})\n")
        lines.append(f"{b['definition']}\n")
        lines.append(f"Context Alignment: {b['context_alignment']:.6f}")
        lines.append(f"Term Alignment: {b['term_alignment']:.6f}")
        lines.append(f"Gap: {abs(b['term_alignment'] - b['context_alignment']):.6f}\n")
        lines.append(f"Prompt used:\n> {b['prompt']}\n")

    Path("definition.md").write_text("\n".join(lines))
