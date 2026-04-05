"""
Ordinary Meaning — interactive experiment loop.
Usage: uv run run.py
"""

import itertools
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from rich.prompt import Prompt, IntPrompt, FloatPrompt, Confirm
from rich.rule import Rule
from rich.panel import Panel
from rich.table import Table

from prepare import (
    call_anthropic, call_openai, call_perplexity,
    evaluate_definition, LLMResponse,
)
from strategies import SYSTEM_PROMPT, build_prompt, strategy_names
from display import (
    console,
    PROVIDER_DISPLAY,
    show_header,
    show_round_start,
    show_prompt,
    show_response,
    show_scores,
    show_decision,
    show_experiment_log,
    show_best_definition,
    show_final_report,
)


INPUT_FILE = Path("input.md")
DEFINITION_FILE = Path("definition.md")
RESULTS_FILE = Path("results.tsv")

PROVIDERS = {
    "anthropic": call_anthropic,
    "openai": call_openai,
    "perplexity": call_perplexity,
}

PROVIDER_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "perplexity": "PERPLEXITY_API_KEY",
}


def available_providers() -> list[str]:
    return [name for name, env in PROVIDER_KEYS.items() if os.environ.get(env)]


def write_input(term: str, context: str):
    INPUT_FILE.write_text(
        f"# Input\n\n## Term\n\n{term}\n\n## Context\n\n{context}\n"
    )


def write_definition(term: str, definition: str, score: float):
    DEFINITION_FILE.write_text(
        f"# Ordinary Meaning: {term}\n\n{definition}\n\nScore: {score:.2f}\n"
    )


def append_result(row: dict):
    header = "round\tprovider\tstrategy\tscore\tbest\tdecision\n"
    if not RESULTS_FILE.exists():
        RESULTS_FILE.write_text(header)
    with open(RESULTS_FILE, "a") as f:
        f.write(
            f"{row['round']}\t{row['provider']}\t{row['strategy']}\t"
            f"{row['score']:.2f}\t{row['best']:.2f}\t{row['decision']}\n"
        )


def generate(provider: str, prompt: str, system: str, temperature: float) -> LLMResponse:
    return PROVIDERS[provider](prompt, system, temperature)


def run_experiment(term: str, context: str, rounds: int, temperature: float, providers: list[str]):
    show_header(term, context, rounds, temperature, providers)

    all_strategies = strategy_names()
    provider_cycle = itertools.cycle(providers)
    strategy_cycle = itertools.cycle(all_strategies)

    best_score = 0.0
    best_definition = ""
    log_rows: list[dict] = []

    provider_totals: dict[str, dict] = {
        p: {"scores": [], "kept": 0, "rounds": 0} for p in providers
    }

    for round_num in range(1, rounds + 1):
        provider = next(provider_cycle)
        strategy = next(strategy_cycle)

        show_round_start(round_num, rounds, provider, strategy)

        prompt = build_prompt(
            strategy=strategy,
            term=term,
            context=context,
            current_definition=best_definition,
        )

        show_prompt(prompt, SYSTEM_PROMPT)

        try:
            resp = generate(provider, prompt, SYSTEM_PROMPT, temperature)
        except Exception as e:
            console.print(f"[red]API error ({provider}): {e}[/]")
            row = {
                "round": round_num, "provider": provider, "strategy": strategy,
                "score": 0.0, "best": best_score, "decision": "crash",
            }
            log_rows.append(row)
            append_result(row)
            continue

        show_response(resp)
        candidate = resp.text.strip()

        try:
            scores = evaluate_definition(candidate)
        except Exception as e:
            console.print(f"[red]Judge error: {e}[/]")
            row = {
                "round": round_num, "provider": provider, "strategy": strategy,
                "score": 0.0, "best": best_score, "decision": "crash",
            }
            log_rows.append(row)
            append_result(row)
            continue

        show_scores(scores, best_score)

        kept = scores.composite > best_score
        decision = "keep" if kept else "discard"

        if kept:
            best_score = scores.composite
            best_definition = candidate
            write_definition(term, best_definition, best_score)
            show_decision(True, f"New best: {best_score:.2f}")
        else:
            show_decision(False, f"Score {scores.composite:.2f} did not beat {best_score:.2f}")

        row = {
            "round": round_num, "provider": provider, "strategy": strategy,
            "score": scores.composite, "best": best_score, "decision": decision,
        }
        log_rows.append(row)
        append_result(row)

        provider_totals[provider]["scores"].append(scores.composite)
        provider_totals[provider]["rounds"] += 1
        if kept:
            provider_totals[provider]["kept"] += 1

        show_best_definition(best_definition, best_score)
        show_experiment_log(log_rows)

    provider_stats = {}
    for p, data in provider_totals.items():
        s = data["scores"]
        provider_stats[p] = {
            "avg": sum(s) / len(s) if s else 0,
            "best": max(s) if s else 0,
            "rounds": data["rounds"],
            "kept": data["kept"],
        }

    show_final_report(
        term=term,
        definition=best_definition,
        best_score=best_score,
        total_rounds=rounds,
        kept=sum(1 for r in log_rows if r["decision"] == "keep"),
        discarded=sum(1 for r in log_rows if r["decision"] == "discard"),
        provider_stats=provider_stats,
    )


def setup_interactive():
    console.print()
    console.print(Rule("ORDINARY MEANING", style="bold white"))
    console.print()
    console.print("[dim]Autoresearch loop for the ordinary meaning of legal terms[/]")
    console.print("[dim]Inspired by Judge Newsom, Snell v. United Specialty (11th Cir. 2024)[/]")
    console.print()

    providers = available_providers()
    if not providers:
        console.print("[red bold]No API keys found.[/] Set at least one of:")
        console.print("  ANTHROPIC_API_KEY, OPENAI_API_KEY, PERPLEXITY_API_KEY")
        console.print("in your .env file.")
        sys.exit(1)

    provider_names = ", ".join(PROVIDER_DISPLAY.get(p, p) for p in providers)
    console.print(f"  Models available: [bold cyan]{provider_names}[/]")
    console.print()

    console.print(Rule("Setup", style="dim white"))
    console.print()

    term = Prompt.ask("[bold white]Term to define[/]")
    console.print()

    console.print("[dim]Paste the contract clause or context paragraph (press Enter twice when done):[/]")
    context_lines = []
    while True:
        line = input()
        if line == "" and context_lines and context_lines[-1] == "":
            context_lines.pop()
            break
        context_lines.append(line)
    context = "\n".join(context_lines).strip()

    if not context:
        console.print("[red]No context provided. Exiting.[/]")
        sys.exit(1)

    console.print()
    console.print(Panel(context, title="Your context", border_style="dim white", width=90))
    console.print()

    rounds = IntPrompt.ask("[bold white]Number of rounds[/]", default=18)
    temperature = FloatPrompt.ask("[bold white]Temperature[/]", default=0.3)

    console.print()

    selected_providers = []
    for p in providers:
        name = PROVIDER_DISPLAY.get(p, p)
        use = Confirm.ask(f"  Use [cyan]{name}[/]?", default=True)
        if use:
            selected_providers.append(p)

    if not selected_providers:
        console.print("[red]No models selected. Exiting.[/]")
        sys.exit(1)

    console.print()
    console.print(Rule("Review", style="dim white"))
    console.print()

    review = Table(show_header=False, box=None, padding=(0, 2))
    review.add_column(style="bold white", width=14)
    review.add_column()
    review.add_row("Term", f"[bold cyan]{term}[/]")
    review.add_row("Context", context[:80] + ("..." if len(context) > 80 else ""))
    review.add_row("Rounds", str(rounds))
    review.add_row("Temperature", str(temperature))
    review.add_row("Models", ", ".join(PROVIDER_DISPLAY.get(p, p) for p in selected_providers))
    console.print(review)
    console.print()

    go = Confirm.ask("[bold white]Start experiment?[/]", default=True)
    if not go:
        console.print("[dim]Cancelled.[/]")
        sys.exit(0)

    write_input(term, context)

    return term, context, rounds, temperature, selected_providers


def main():
    term, context, rounds, temperature, selected_providers = setup_interactive()
    run_experiment(term, context, rounds, temperature, selected_providers)


if __name__ == "__main__":
    main()
