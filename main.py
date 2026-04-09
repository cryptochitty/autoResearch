#!/usr/bin/env python3
"""
AutoResearch — Multi-Agent Research System
Usage:
    python main.py "AI for Crop Yield Prediction using weather and soil data"
    python main.py --topic "transformer models for time series forecasting"
    python main.py          # prompts interactively
"""

import sys
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from crew import run_research

console = Console()

BANNER = """
  █████╗ ██╗   ██╗████████╗ ██████╗ ██████╗ ███████╗███████╗ █████╗ ██████╗  ██████╗██╗  ██╗
 ██╔══██╗██║   ██║╚══██╔══╝██╔═══██╗██╔══██╗██╔════╝██╔════╝██╔══██╗██╔══██╗██╔════╝██║  ██║
 ███████║██║   ██║   ██║   ██║   ██║██████╔╝█████╗  ███████╗███████║██████╔╝██║     ███████║
 ██╔══██║██║   ██║   ██║   ██║   ██║██╔══██╗██╔══╝  ╚════██║██╔══██║██╔══██╗██║     ██╔══██║
 ██║  ██║╚██████╔╝   ██║   ╚██████╔╝██║  ██║███████╗███████║██║  ██║██║  ██║╚██████╗██║  ██║
 ╚═╝  ╚═╝ ╚═════╝    ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AutoResearch: 8-agent pipeline from topic → paper + code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "topic",
        nargs="?",
        help="Research topic (positional)",
    )
    parser.add_argument(
        "--topic", "-t",
        dest="topic_flag",
        help="Research topic (flag form)",
    )
    args = parser.parse_args()

    topic: str = args.topic or args.topic_flag or ""

    if not topic:
        console.print(Panel(BANNER, style="bold cyan", expand=False))
        topic = console.input("[bold yellow]Enter research topic:[/bold yellow] ").strip()

    if not topic:
        console.print("[red]Error:[/red] Research topic is required.")
        sys.exit(1)

    console.print(Panel(
        Text.from_markup(
            f"[bold green]Topic:[/bold green] {topic}\n"
            "[dim]Running 8 agents sequentially. This takes ~5–15 min.[/dim]"
        ),
        title="AutoResearch",
        border_style="green",
    ))

    results = run_research(topic)

    # Print each section header so the user can see what was produced
    console.rule("[bold green]Research Complete")
    for section in results:
        word_count = len(results[section].split())
        console.print(f"  [cyan]{section}[/cyan]  ({word_count} words)")

    console.print("\n[bold green]Done![/bold green] Check the [yellow]outputs/[/yellow] folder for the full report.\n")


if __name__ == "__main__":
    main()
