"""CLI module for improved local assistant."""

def main():
    """Main CLI entry point."""
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from cli.graphrag_repl import main as repl_main
    repl_main()

if __name__ == "__main__":
    main()