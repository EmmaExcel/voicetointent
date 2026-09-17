from __future__ import annotations

import json
from pathlib import Path

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from config import get_llm_provider
from core import transcriber
from core.mic_recorder import record_until_silence
from schemas import get_schema


app = typer.Typer(
    name="vtt",
    help="Voice-to-Intent: transcribe audio and extract structured financial intent.",
    rich_markup_mode="rich",
)

console = Console()


def _confidence_label(score: float) -> str:
    if score >= 0.85:
        return f"[green]confidence: {score:.0%}[/green]"
    if score >= 0.65:
        return f"[yellow]confidence: {score:.0%} (review before sending)[/yellow]"
    return f"[red]confidence: {score:.0%} (low — do not proceed automatically)[/red]"


def _print_result(transcript: str, confidence: float, schema_name: str, intent: dict) -> None:
    rprint(Panel(
        f"[bold]{transcript}[/bold]\n{_confidence_label(confidence)}",
        title="Transcript",
    ))

    json_str = json.dumps(intent, indent=2, ensure_ascii=False)
    syntax = Syntax(json_str, "json", theme="monokai", line_numbers=False)
    rprint(Panel(syntax, title=f"Extracted Intent  ({schema_name})"))


@app.command()
def process(
    audio_file: Path = typer.Argument(..., help="Path to an audio file (WAV, MP3, etc.)"),
    schema: str = typer.Option("financial", "--schema", "-s", help="Schema name."),
    system_prompt: str = typer.Option("", "--system-prompt", help="Override the system prompt."),
):
    if not audio_file.exists():
        rprint(f"[red]error:[/red] file not found: {audio_file}")
        raise typer.Exit(1)

    try:
        schema_cls = get_schema(schema)
    except KeyError as exc:
        rprint(f"[red]error:[/red] {exc}")
        raise typer.Exit(1)

    with console.status("transcribing audio..."):
        transcript, confidence = transcriber.transcribe_file(audio_file)

    if not transcript:
        rprint("[red]no speech detected in the audio file.[/red]")
        raise typer.Exit(1)

    with console.status("extracting intent..."):
        provider = get_llm_provider()
        intent = provider.extract(transcript, schema_cls, system_prompt)

    _print_result(transcript, confidence, schema, intent.model_dump())


@app.command()
def listen(
    schema: str = typer.Option("financial", "--schema", "-s", help="Schema name."),
    system_prompt: str = typer.Option("", "--system-prompt", help="Override the system prompt."),
    silence_seconds: float = typer.Option(
        1.5, "--silence", help="Seconds of silence before auto-stop."
    ),
):
    try:
        schema_cls = get_schema(schema)
    except KeyError as exc:
        rprint(f"[red]error:[/red] {exc}")
        raise typer.Exit(1)

    audio_bytes = record_until_silence(silence_seconds=silence_seconds)

    with console.status("transcribing..."):
        transcript, confidence = transcriber.transcribe_bytes(audio_bytes)

    if not transcript:
        rprint("[red]no speech detected.[/red]")
        raise typer.Exit(1)

    with console.status("extracting intent..."):
        provider = get_llm_provider()
        intent = provider.extract(transcript, schema_cls, system_prompt)

    _print_result(transcript, confidence, schema, intent.model_dump())


@app.command()
def extract(
    text: str = typer.Argument(..., help="Text to extract intent from."),
    schema: str = typer.Option("financial", "--schema", "-s", help="Schema name."),
    system_prompt: str = typer.Option("", "--system-prompt", help="Override the system prompt."),
):
    try:
        schema_cls = get_schema(schema)
    except KeyError as exc:
        rprint(f"[red]error:[/red] {exc}")
        raise typer.Exit(1)

    with console.status("extracting intent..."):
        provider = get_llm_provider()
        intent = provider.extract(text, schema_cls, system_prompt)

    _print_result(text, 1.0, schema, intent.model_dump())


@app.command()
def schemas():
    from schemas import SCHEMA_REGISTRY

    table = Table(title="Registered Schemas", show_header=True, header_style="bold")
    table.add_column("Name", style="cyan")
    table.add_column("Class", style="dim")

    for name, cls in SCHEMA_REGISTRY.items():
        table.add_row(name, f"{cls.__module__}.{cls.__name__}")

    console.print(table)


if __name__ == "__main__":
    app()
