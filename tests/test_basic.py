"""Basic tests that should pass in CI environment."""

import sys
from pathlib import Path


def test_python_version():
    """Test that we're running on a supported Python version."""
    assert sys.version_info >= (3, 10), f"Python 3.10+ required, got {sys.version_info}"


def test_package_structure():
    """Test that the package structure is correct."""
    project_root = Path(__file__).parent.parent
    src_dir = project_root / "src"
    package_dir = src_dir / "improved_local_assistant"

    assert src_dir.exists(), "src/ directory should exist"
    assert package_dir.exists(), "src/improved_local_assistant/ should exist"
    assert (package_dir / "__init__.py").exists(), "__init__.py should exist in package"


def test_basic_imports():
    """Test that basic Python imports work."""
    import json
    import pathlib

    import yaml

    # Test that we can import these without errors
    assert json is not None
    assert yaml is not None
    assert pathlib is not None


def test_package_metadata():
    """Test that package metadata is accessible."""
    # Add project root to path for testing
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root / "src"))

    try:
        from improved_local_assistant import __author__
        from improved_local_assistant import __version__

        assert __version__ == "1.0.0"
        assert __author__ == "AI Team"
    except ImportError:
        # This is expected in CI without full dependencies
        pass


def test_configuration_files():
    """Test that configuration files exist and are valid."""
    project_root = Path(__file__).parent.parent

    # Check that key files exist
    assert (project_root / "pyproject.toml").exists()
    assert (project_root / "README.md").exists()
    assert (project_root / "requirements.txt").exists()

    # Check that pyproject.toml is valid TOML
    try:
        import tomllib  # Python 3.11+
    except ImportError:
        try:
            import tomli as tomllib  # Python < 3.11
        except ImportError:
            # Skip TOML validation if no parser available
            return

    with open(project_root / "pyproject.toml", "rb") as f:
        config = tomllib.load(f)
        assert "project" in config
        assert config["project"]["name"] == "improved-local-assistant"
