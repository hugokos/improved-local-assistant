"""Main application entry point for the package."""


def main():
    """Main entry point for the improved local assistant."""
    import sys
    from pathlib import Path

    # Add the project root to Python path for development
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    # Import and run the main application
    from run_app import main as app_main

    app_main()


if __name__ == "__main__":
    main()
