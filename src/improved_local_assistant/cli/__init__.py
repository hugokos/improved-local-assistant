"""
Improved Local Assistant CLI
"""

import os
import platform
import sys
from typing import Optional

import typer
import uvicorn

# Check for unsupported platforms
if platform.system() == "Darwin":
    print("❌ Improved Local Assistant: macOS is currently unsupported.")
    print("   Please use Windows 10/11 or Linux (Ubuntu 20.04+ recommended).")
    print(
        "   See: https://github.com/hugokos/improved-local-assistant/blob/main/docs/installation.md"
    )
    sys.exit(1)

app = typer.Typer(help="Improved Local Assistant CLI")


@app.command()
def api(
    host: str = typer.Option("127.0.0.1", help="Host to bind to"),
    port: int = typer.Option(8000, help="Port to bind to"),
    reload: bool = typer.Option(False, help="Enable auto-reload"),
    config: Optional[str] = typer.Option(None, help="Config file path"),
):
    """Run FastAPI server."""
    if config:
        os.environ["ILA_CONFIG"] = config

    uvicorn.run(
        "improved_local_assistant.api.main:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True,
    )


@app.command()
def repl():
    """Run interactive GraphRAG REPL."""
    from improved_local_assistant.cli.graphrag_repl import main as repl_main

    repl_main()


@app.command()
def bench():
    """Run lightweight smoke/benchmark test."""
    typer.echo("Running quick GraphRAG benchmark...")
    # Import and run benchmark
    try:
        from improved_local_assistant.scripts.quick_graphrag_benchmark import main as bench_main

        bench_main()
    except ImportError:
        typer.echo("Benchmark module not found. Run: python scripts/quick_graphrag_benchmark.py")


@app.command()
def health():
    """Check system health and configuration."""
    typer.echo("Checking system health...")
    # Import and run health check
    try:
        from improved_local_assistant.scripts.system_health_check import main as health_main

        health_main()
    except ImportError:
        typer.echo("Health check module not found. Run: python scripts/system_health_check.py")


@app.command()
def download_graphs(
    action: str = typer.Argument("all", help="Graph to download: 'all', 'survivalist', etc."),
):
    """Download prebuilt knowledge graphs."""
    typer.echo(f"Downloading graphs: {action}")
    try:
        from improved_local_assistant.scripts.download_graphs import main as download_main

        download_main([action])
    except ImportError:
        typer.echo("Download script not found. Run: python scripts/download_graphs.py")


if __name__ == "__main__":
    app()
