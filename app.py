"""
Ordinary Meaning — Streamlit interface.
Usage: uv run streamlit run app.py

Each model defines the term using two tracks:
  World track  — non-document strategies, best picked by term alignment.
  Document track — document strategies, best picked by context alignment.
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

WORLD_STRATEGIES = ["bare", "dictionary", "examples", "contrastive"]
DOC_STRATEGIES = ["context", "full_document", "document_scope", "document_contrastive"]


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
        "Contract clause / context paragraph",
        height=120,
        placeholder="Paste the clause where the term appears...",
    )

    st.divider()
    st.subheader("Full Document (optional)")
    uploaded_file = st.file_uploader(
        "Upload the full contract",
        type=["pdf", "docx", "doc", "txt"],
        help="When provided, the full document is appended to every prompt.",
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
    world_strats = [s for s in WORLD_STRATEGIES]
    doc_strats = [s for s in DOC_STRATEGIES if has_doc or s == "context"]
    all_strategies = world_strats + doc_strats
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
        "Each model generates definitions using two tracks:\n\n"
        "**World track** — bare, dictionary, examples, contrastive. "
        "Best picked by **term alignment** (closeness to the word's general meaning).\n\n"
        "**Document track** — context, full document, document scope, document contrastive. "
        "Best picked by **context alignment** (closeness to the contract clause).\n\n"
        "You see both definitions side by side for each model. "
        "If they say the same thing, the meaning is clear. "
        "If they diverge, the contract may be using the word in a specific way."
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

with st.expander(f"Prompting strategies ({n_strats} active)", expanded=False):
    wcol, dcol = st.columns(2)
    with wcol:
        st.markdown("**World strategies** (selected by term alignment)")
        for strat in world_strats:
            info = STRATEGY_INFO[strat]
            st.markdown(f"- **{strat}** — {info['description']}")
    with dcol:
        st.markdown("**Document strategies** (selected by context alignment)")
        for strat in doc_strats:
            info = STRATEGY_INFO[strat]
            st.markdown(f"- **{strat}** — {info['description']}")

st.divider()

# =========================================================================
# PHASE 1: Autoresearch Loop — two tracks
# =========================================================================

st.header("Phase 1: Autoresearch Loop")
st.markdown(
    "Each model runs every strategy. Two independent tracks pick the best definition:\n\n"
    "- **World track** — best from bare, dictionary, examples, contrastive "
    "(picked by **term alignment**: closeness to the word's general meaning)\n"
    "- **Document track** — best from context, full document, document scope, document contrastive "
    "(picked by **context alignment**: closeness to the contract clause)"
)

progress = st.progress(0, text="Starting...")

world_survivors = {}
doc_survivors = {}
all_logs = {}
log_rows = []
step = 0

for pid in selected:
    prov_info = PROVIDERS[pid]

    world_best_text = None
    world_best_score = -1.0
    world_best_strategy = None

    doc_best_text = None
    doc_best_score = -1.0
    doc_best_strategy = None

    model_log = []

    for strat in all_strategies:
        strat_desc = STRATEGY_INFO[strat]["description"]
        is_world = strat in WORLD_STRATEGIES

        for rd in range(1, rounds + 1):
            step += 1
            progress.progress(
                step / n_total,
                text=f"{step}/{n_total} — {prov_info['label']} — {strat}" + (f" r{rd}" if rounds > 1 else ""),
            )

            prompt = build_prompt(
                strategy=strat,
                term=term,
                context=context,
                full_document=doc_truncated,
            )

            resp_text = ""
            resp_meta = ""
            score = 0.0
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
                    if is_world:
                        score = term_alignment(resp_text, term)
                    else:
                        score = context_alignment(resp_text, context)
                except Exception as e:
                    error = f"Embedding error: {e}"

            kept = False
            if is_world:
                if resp_text and not error and (world_best_text is None or score > world_best_score):
                    kept = True
                    world_best_text = resp_text
                    world_best_score = score
                    world_best_strategy = strat
                track = "world"
                best_so_far = world_best_score
            else:
                if resp_text and not error and (doc_best_text is None or score > doc_best_score):
                    kept = True
                    doc_best_text = resp_text
                    doc_best_score = score
                    doc_best_strategy = strat
                track = "document"
                best_so_far = doc_best_score

            entry = {
                "strategy": strat,
                "strategy_description": strat_desc,
                "track": track,
                "round": rd,
                "prompt": prompt,
                "response": resp_text,
                "response_meta": resp_meta,
                "score": round(score, 6),
                "metric": "term_alignment" if is_world else "context_alignment",
                "best_so_far": round(best_so_far, 6),
                "kept": kept,
                "error": error,
            }
            model_log.append(entry)

            log_rows.append({
                "Model": prov_info["label"],
                "Track": track.title(),
                "Strategy": strat,
                "Round": rd,
                "Score": round(score, 6),
                "Metric": "Term Align" if is_world else "Context Align",
                "Best So Far": round(best_so_far, 6),
                "Decision": "KEEP" if kept else ("ERROR" if error else "DISCARD"),
            })

    world_survivors[pid] = {
        "definition": world_best_text or "",
        "score": world_best_score,
        "strategy": world_best_strategy or "none",
    }
    doc_survivors[pid] = {
        "definition": doc_best_text or "",
        "score": doc_best_score,
        "strategy": doc_best_strategy or "none",
    }
    all_logs[pid] = model_log

progress.progress(1.0, text="Done")
st.session_state.experiment_log = log_rows

# -- Detailed logs per model (collapsed)
with st.expander("Detailed experiment log", expanded=False):
    for pid in selected:
        prov_info = PROVIDERS[pid]
        logs = all_logs[pid]

        st.subheader(prov_info["label"])

        log_df = pd.DataFrame([
            {
                "Track": e["track"].title(),
                "Strategy": e["strategy"],
                "Round": e["round"],
                "Score": e["score"],
                "Metric": e["metric"],
                "Best": e["best_so_far"],
                "Decision": "KEEP" if e["kept"] else ("ERROR" if e["error"] else "DISCARD"),
            }
            for e in logs
        ])
        st.dataframe(log_df, use_container_width=True, hide_index=True)

        for e in logs:
            label = f"{e['track'].title()} / {e['strategy']}"
            if rounds > 1:
                label += f" r{e['round']}"
            decision = "KEEP" if e["kept"] else ("ERROR" if e["error"] else "DISCARD")
            with st.expander(f"{label} — {e['score']:.4f} — {decision}", expanded=False):
                if e["error"]:
                    st.error(e["error"])
                    continue
                st.caption(f"{e['strategy']} — {e['strategy_description']}")
                with st.expander("Full prompt", expanded=False):
                    st.code(e["prompt"], language=None)
                st.markdown("**Response:**")
                st.markdown(e["response"])
                st.caption(e["response_meta"])

# =========================================================================
# PHASE 2: Results — side by side
# =========================================================================

st.divider()
st.header("Phase 2: Results")
st.markdown(
    "Each model's best **world meaning** definition (from non-document prompts, "
    "selected by term alignment) and best **document meaning** definition "
    "(from document prompts, selected by context alignment), shown side by side.\n\n"
    "If they converge, the ordinary meaning is clear. "
    "If they diverge, the contract may use the term in a specific way."
)

for pid in selected:
    prov_info = PROVIDERS[pid]
    ws = world_survivors[pid]
    ds = doc_survivors[pid]

    if not ws["definition"] and not ds["definition"]:
        continue

    st.subheader(prov_info["label"])

    wcol, dcol = st.columns(2)

    with wcol:
        st.markdown("**World Meaning**")
        st.caption(f"Strategy: `{ws['strategy']}` | Selected by: term alignment")
        if ws["definition"]:
            t_a = term_alignment(ws["definition"], term)
            c_a = context_alignment(ws["definition"], context)
            m1, m2 = st.columns(2)
            m1.metric("Term Alignment", f"{t_a:.4f}", help="Primary metric for this track")
            m2.metric("Context Alignment", f"{c_a:.4f}", help="Shown for comparison")
            st.markdown(ws["definition"])
        else:
            st.warning("No world definition produced.")

    with dcol:
        st.markdown("**Document Meaning**")
        st.caption(f"Strategy: `{ds['strategy']}` | Selected by: context alignment")
        if ds["definition"]:
            t_a_d = term_alignment(ds["definition"], term)
            c_a_d = context_alignment(ds["definition"], context)
            m1, m2 = st.columns(2)
            m1.metric("Context Alignment", f"{c_a_d:.4f}", help="Primary metric for this track")
            m2.metric("Term Alignment", f"{t_a_d:.4f}", help="Shown for comparison")
            st.markdown(ds["definition"])
        else:
            st.warning("No document definition produced.")

    if ws["definition"] and ds["definition"]:
        sim = cosine_sim(embed_text(ws["definition"]), embed_text(ds["definition"]))
        if sim >= 0.92:
            st.success(f"World ↔ Document similarity: **{sim:.4f}** — These definitions strongly agree. The ordinary meaning is clear.")
        elif sim >= 0.82:
            st.info(f"World ↔ Document similarity: **{sim:.4f}** — Moderate agreement. The core meaning is shared, with some differences in scope.")
        else:
            st.warning(f"World ↔ Document similarity: **{sim:.4f}** — Notable divergence. The contract may use this term in a specific or narrowed way.")

    st.divider()

# =========================================================================
# PHASE 3: Cross-Model Consensus
# =========================================================================

active_world = {p: s for p, s in world_survivors.items() if s["definition"]}
active_doc = {p: s for p, s in doc_survivors.items() if s["definition"]}

if len(active_world) >= 2 or len(active_doc) >= 2:
    st.header("Phase 3: Cross-Model Consensus")
    st.markdown(
        "How much do the different models agree with each other? "
        "Pairwise cosine similarity between definitions. "
        "High values mean convergence on the same meaning."
    )

    tabs = []
    tab_labels = []
    if len(active_world) >= 2:
        tab_labels.append("World Definitions")
    if len(active_doc) >= 2:
        tab_labels.append("Document Definitions")

    consensus_tabs = st.tabs(tab_labels)
    tab_idx = 0

    if len(active_world) >= 2:
        with consensus_tabs[tab_idx]:
            pids = list(active_world.keys())
            labels = [PROVIDERS[p]["label"] for p in pids]
            vecs = {p: embed_text(active_world[p]["definition"]) for p in pids}

            sim_matrix = []
            for i, pid_i in enumerate(pids):
                row = {"": labels[i]}
                for j, pid_j in enumerate(pids):
                    row[labels[j]] = round(cosine_sim(vecs[pid_i], vecs[pid_j]), 4)
                sim_matrix.append(row)
            st.dataframe(pd.DataFrame(sim_matrix), use_container_width=True, hide_index=True)

            pair_scores = []
            for i in range(len(pids)):
                for j in range(i + 1, len(pids)):
                    pair_scores.append(cosine_sim(vecs[pids[i]], vecs[pids[j]]))
            avg = sum(pair_scores) / len(pair_scores) if pair_scores else 0

            if avg >= 0.95:
                st.success(f"Average similarity: {avg:.4f} — Strong convergence")
            elif avg >= 0.85:
                st.info(f"Average similarity: {avg:.4f} — Moderate agreement")
            else:
                st.warning(f"Average similarity: {avg:.4f} — Notable divergence")
        tab_idx += 1

    if len(active_doc) >= 2:
        with consensus_tabs[tab_idx]:
            pids = list(active_doc.keys())
            labels = [PROVIDERS[p]["label"] for p in pids]
            vecs = {p: embed_text(active_doc[p]["definition"]) for p in pids}

            sim_matrix = []
            for i, pid_i in enumerate(pids):
                row = {"": labels[i]}
                for j, pid_j in enumerate(pids):
                    row[labels[j]] = round(cosine_sim(vecs[pid_i], vecs[pid_j]), 4)
                sim_matrix.append(row)
            st.dataframe(pd.DataFrame(sim_matrix), use_container_width=True, hide_index=True)

            pair_scores = []
            for i in range(len(pids)):
                for j in range(i + 1, len(pids)):
                    pair_scores.append(cosine_sim(vecs[pids[i]], vecs[pids[j]]))
            avg = sum(pair_scores) / len(pair_scores) if pair_scores else 0

            if avg >= 0.95:
                st.success(f"Average similarity: {avg:.4f} — Strong convergence")
            elif avg >= 0.85:
                st.info(f"Average similarity: {avg:.4f} — Moderate agreement")
            else:
                st.warning(f"Average similarity: {avg:.4f} — Notable divergence")

# =========================================================================
# PHASE 4: Qualitative Comparison
# =========================================================================

all_defs = {}
for pid in selected:
    ws = world_survivors.get(pid, {})
    ds = doc_survivors.get(pid, {})
    if ws.get("definition") or ds.get("definition"):
        all_defs[pid] = {"world": ws.get("definition", ""), "document": ds.get("definition", "")}

if len(all_defs) >= 2:
    st.divider()
    st.header("Phase 4: Qualitative Comparison")
    st.markdown(
        "One LLM call describes where the definitions agree and diverge. "
        "This explains the numbers above in plain English. It does not score or rank."
    )

    pids = list(all_defs.keys())
    def_parts = []
    for pid in pids:
        label = PROVIDERS[pid]["label"]
        w = all_defs[pid]["world"]
        d = all_defs[pid]["document"]
        block = f"**{label}**\n"
        if w:
            block += f"World meaning ({world_survivors[pid]['strategy']}):\n{w}\n\n"
        if d:
            block += f"Document meaning ({doc_survivors[pid]['strategy']}):\n{d}\n"
        def_parts.append(block)

    def_list = "\n---\n".join(def_parts)

    comparison_prompt = (
        f'Multiple AI models each produced two definitions of "{term}" — one based on '
        f"general English (world meaning) and one based on the contract (document meaning).\n\n"
        f"Context clause: {context}\n\n"
        f"Their definitions:\n\n{def_list}\n\n"
        "Describe:\n"
        "1. Where do the world meanings AGREE across models?\n"
        "2. Where do the document meanings AGREE across models?\n"
        "3. How do the world meanings DIFFER from the document meanings? "
        "Does the contract narrow or shift the term's meaning?\n"
        "4. Any notable word choices or framings that differ between models?\n"
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
    world_survivors.get(p, {}).get("definition") or doc_survivors.get(p, {}).get("definition")
    for p in selected
)

if any_results:
    lines = [f"# Ordinary Meaning: {term}\n"]
    for pid in selected:
        prov_info = PROVIDERS[pid]
        ws = world_survivors.get(pid, {})
        ds = doc_survivors.get(pid, {})

        lines.append(f"\n## {prov_info['label']}\n")

        if ws.get("definition"):
            t_a = term_alignment(ws["definition"], term)
            c_a = context_alignment(ws["definition"], context)
            lines.append(f"### World Meaning (strategy: {ws['strategy']})\n")
            lines.append(f"{ws['definition']}\n")
            lines.append(f"Term Alignment: {t_a:.6f}")
            lines.append(f"Context Alignment: {c_a:.6f}\n")

        if ds.get("definition"):
            t_a_d = term_alignment(ds["definition"], term)
            c_a_d = context_alignment(ds["definition"], context)
            lines.append(f"### Document Meaning (strategy: {ds['strategy']})\n")
            lines.append(f"{ds['definition']}\n")
            lines.append(f"Context Alignment: {c_a_d:.6f}")
            lines.append(f"Term Alignment: {t_a_d:.6f}\n")

    Path("definition.md").write_text("\n".join(lines))
