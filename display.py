from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule
from rich import box

from prepare import LLMResponse, JudgeScore

console = Console()

PROVIDER_DISPLAY = {
    "anthropic": "Claude Sonnet",
    "openai": "GPT-4o",
    "perplexity": "Perplexity Sonar",
}


def score_color(score: int) -> str:
    if score >= 8:
        return "green"
    if score >= 6:
        return "yellow"
    return "red"


def show_header(term: str, context: str, rounds: int, temperature: float, providers: list[str]):
    console.print()
    console.print(Rule("ORDINARY MEANING", style="bold white"))
    console.print()

    info = Table(show_header=False, box=None, padding=(0, 2))
    info.add_column(style="bold white", width=14)
    info.add_column()
    info.add_row("Term", f"[bold cyan]{term}[/]")
    info.add_row("Rounds", str(rounds))
    info.add_row("Temperature", str(temperature))
    info.add_row("Models", ", ".join(PROVIDER_DISPLAY.get(p, p) for p in providers))
    console.print(info)

    console.print()
    console.print(Panel(context, title="Context", border_style="dim white", width=90))
    console.print()


def show_round_start(round_num: int, total: int, provider: str, strategy: str):
    console.print()
    console.print(Rule(
        f"Round {round_num}/{total}  --  {PROVIDER_DISPLAY.get(provider, provider)}  --  {strategy}",
        style="bold white",
    ))


def show_prompt(prompt: str, system: str):
    console.print()
    console.print(Panel(
        system,
        title="System Prompt",
        border_style="dim blue",
        width=90,
    ))
    console.print(Panel(
        prompt,
        title="User Prompt",
        border_style="blue",
        width=90,
    ))


def show_response(resp: LLMResponse):
    console.print()
    tokens = f"[dim]{resp.input_tokens} in / {resp.output_tokens} out[/]"
    console.print(Panel(
        resp.text,
        title=f"Response  ({resp.model})  {tokens}",
        border_style="cyan",
        width=90,
    ))


def show_scores(scores: JudgeScore, prev_best: float):
    console.print()

    table = Table(
        title="Judge Evaluation",
        box=box.SIMPLE_HEAVY,
        title_style="bold white",
        width=90,
    )
    table.add_column("Axis", style="bold white", width=14)
    table.add_column("Score", justify="center", width=7)
    table.add_column("Reasoning", ratio=1)

    axes = [
        ("Fidelity", scores.fidelity, scores.fidelity_reason),
        ("Readability", scores.readability, scores.readability_reason),
        ("Completeness", scores.completeness, scores.completeness_reason),
        ("Neutrality", scores.neutrality, scores.neutrality_reason),
    ]
    for name, val, reason in axes:
        color = score_color(val)
        table.add_row(name, f"[{color}]{val}/10[/]", reason)

    console.print(table)

    comp_color = score_color(int(scores.composite))
    delta = scores.composite - prev_best
    direction = f"+{delta:.2f}" if delta > 0 else f"{delta:.2f}"
    delta_color = "green" if delta > 0 else ("yellow" if delta == 0 else "red")

    console.print(
        f"  Composite: [{comp_color} bold]{scores.composite:.2f}[/]"
        f"   (best: {prev_best:.2f}  /  delta: [{delta_color}]{direction}[/])"
    )


def show_decision(kept: bool, reason: str):
    console.print()
    if kept:
        console.print(Panel(
            f"[green bold]KEEP[/]  {reason}",
            border_style="green",
            width=90,
        ))
    else:
        console.print(Panel(
            f"[red bold]DISCARD[/]  {reason}",
            border_style="red",
            width=90,
        ))


def show_experiment_log(rows: list[dict]):
    console.print()
    table = Table(
        title="Experiment Log",
        box=box.SIMPLE_HEAVY,
        title_style="bold white",
        width=90,
    )
    table.add_column("#", justify="right", width=4)
    table.add_column("Model", width=18)
    table.add_column("Strategy", width=14)
    table.add_column("Score", justify="center", width=7)
    table.add_column("Best", justify="center", width=7)
    table.add_column("Decision", justify="center", width=10)

    for r in rows:
        sc = score_color(int(r["score"]))
        decision_style = "green" if r["decision"] == "keep" else "red"
        table.add_row(
            str(r["round"]),
            PROVIDER_DISPLAY.get(r["provider"], r["provider"]),
            r["strategy"],
            f"[{sc}]{r['score']:.2f}[/]",
            f"{r['best']:.2f}",
            f"[{decision_style}]{r['decision'].upper()}[/]",
        )
    console.print(table)


def show_best_definition(definition: str, score: float):
    console.print()
    sc = score_color(int(score))
    console.print(Panel(
        definition,
        title=f"Current Best Definition  [{sc}]{score:.2f}[/]",
        border_style=sc,
        width=90,
    ))


def show_final_report(
    term: str,
    definition: str,
    best_score: float,
    total_rounds: int,
    kept: int,
    discarded: int,
    provider_stats: dict[str, dict],
):
    console.print()
    console.print(Rule("FINAL REPORT", style="bold white"))
    console.print()

    console.print(Panel(
        definition,
        title=f'Ordinary meaning of "{term}"',
        border_style="bold green",
        width=90,
    ))

    stats = Table(box=box.SIMPLE_HEAVY, title="Summary", title_style="bold white", width=90)
    stats.add_column("", style="bold white", width=20)
    stats.add_column("")
    stats.add_row("Final Score", f"[bold]{best_score:.2f}[/]")
    stats.add_row("Total Rounds", str(total_rounds))
    stats.add_row("Kept", f"[green]{kept}[/]")
    stats.add_row("Discarded", f"[red]{discarded}[/]")
    console.print(stats)

    model_table = Table(
        box=box.SIMPLE_HEAVY,
        title="Model Comparison",
        title_style="bold white",
        width=90,
    )
    model_table.add_column("Model", width=20)
    model_table.add_column("Avg Score", justify="center", width=10)
    model_table.add_column("Best Score", justify="center", width=10)
    model_table.add_column("Rounds", justify="center", width=8)
    model_table.add_column("Kept", justify="center", width=8)

    for provider, data in provider_stats.items():
        avg_color = score_color(int(data["avg"]))
        best_color = score_color(int(data["best"]))
        model_table.add_row(
            PROVIDER_DISPLAY.get(provider, provider),
            f"[{avg_color}]{data['avg']:.2f}[/]",
            f"[{best_color}]{data['best']:.2f}[/]",
            str(data["rounds"]),
            str(data["kept"]),
        )
    console.print(model_table)
    console.print()
